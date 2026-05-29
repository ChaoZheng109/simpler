/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/**
 * @file l0_perf_collector_aicpu.cpp
 * @brief AICPU-side L0 marker producer (a5 onboard), buffer-pool transport.
 *
 * AICPU does not call any libascend_hal consumer API. On each task FIN the
 * scheduler thread reads the AICore-published (start_cycle, end_cycle) from
 * the per-core L0PerfAicoreRing slot, enriches it with the real PTO2 task_id
 * and the task's biu_perf (group, block, core_type), and appends the marker
 * into a per-thread L0MarkerBuffer. When the buffer fills it is enqueued into
 * the per-thread ready queue and a fresh one is popped from the free queue
 * (same buffer-pool model as L2/PMU). A partial buffer is flushed at thread
 * exit. Host L0PerfCollector recycles buffers, stores all marker windows, and
 * (offline, at finalize) matches each decoded biu_perf pipe stamp to the task
 * whose cycle window contains it.
 */

#include "aicpu/l0_perf_collector_aicpu.h"

#include "aicpu/platform_regs.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "common/unified_log.h"

// ---------------------------------------------------------------------------
// Module state (process-wide; each scheduler thread writes only its own
// cores_owned[] entries in the logical→group table, so no cross-thread locking).
// ---------------------------------------------------------------------------

static uint64_t g_platform_l0_perf_base = 0;
static bool g_enable_l0_swimlane = false;

static L0PerfDataHeader *s_l0_header = nullptr;

// Logical core id -> biu_perf group (0..5), or -1 if outside coverage. Each
// scheduler thread writes only the entries for the cores it owns (in init) and
// reads only those same entries (in complete_record) — see l0_perf_aicpu_init
// for why there is no shared cross-thread initialization.
static int s_l0_logical_to_group[PLATFORM_MAX_CORES];

// Logical core id -> biu_perf sub-core block (0=aic, 1=aiv0, 2=aiv1). Same
// per-thread write/read discipline as the group table.
static uint16_t s_l0_logical_to_block[PLATFORM_MAX_CORES];

// Logical core id -> raw sub-core phys id. Carried into the marker so host can
// verify the (group, block) → channel sub_core mapping (esp. aiv0/aiv1).
static uint32_t s_l0_logical_to_phys[PLATFORM_MAX_CORES];

// Logical core id -> AICore-side L0PerfAicoreRing pointer. Populated in init
// from L0PerfDataHeader::aicore_ring_ptrs[]; complete_record reads the slot
// at `expected_reg_task_id % PLATFORM_L0_AICORE_RING_SIZE`.
static L0PerfAicoreRing *s_l0_aicore_rings[PLATFORM_MAX_CORES];

// Per-thread buffer-pool state cache (mirrors L2's s_perf_buffer_states /
// s_perf_records_buffers). Each scheduler thread owns one entry.
static L0PerfBufferState *s_l0_thread_states[PLATFORM_MAX_AICPU_THREADS] = {};
static L0MarkerBuffer *s_l0_current_buf[PLATFORM_MAX_AICPU_THREADS] = {};

extern "C" void set_platform_l0_perf_base(uint64_t l0_perf_data_base) { g_platform_l0_perf_base = l0_perf_data_base; }
extern "C" uint64_t get_platform_l0_perf_base() { return g_platform_l0_perf_base; }
extern "C" void set_l0_swimlane_enabled(bool enable) { g_enable_l0_swimlane = enable; }
extern "C" bool is_l0_swimlane_enabled() { return g_enable_l0_swimlane; }

// ---------------------------------------------------------------------------
// Internal helpers — buffer-pool transport
// ---------------------------------------------------------------------------

// Push a full buffer pointer into this thread's ready queue. Returns -1 if
// the ready queue is full (host stalled). Publish protocol: write seq/index
// first, buffer_ptr (the torn-read publish flag) last, then bump tail.
static int enqueue_l0_ready_buffer(int thread_idx, uint64_t buffer_ptr, uint32_t buffer_seq) {
    uint32_t capacity = PLATFORM_L0_PERF_READYQUEUE_SIZE;
    uint32_t current_tail = s_l0_header->queue_tails[thread_idx];
    uint32_t current_head = s_l0_header->queue_heads[thread_idx];

    uint32_t next_tail = (current_tail + 1) % capacity;
    if (next_tail == current_head) {
        return -1;  // ready queue full
    }

    auto &slot = s_l0_header->queues[thread_idx][current_tail];
    slot.buffer_seq = buffer_seq;
    slot.thread_index = static_cast<uint32_t>(thread_idx);
    wmb();
    slot.buffer_ptr = buffer_ptr;  // publish last
    s_l0_header->queue_tails[thread_idx] = next_tail;
    return 0;
}

// Pop one free buffer pointer for this thread (SPSC consumer side). Returns 0
// if the free queue is empty. Mirrors L2's free_queue pop.
static uint64_t pop_l0_free_buffer(L0PerfBufferState *state) {
    rmb();
    uint32_t head = state->free_queue.head;
    uint32_t tail = state->free_queue.tail;
    if (head == tail) {
        return 0;
    }
    uint64_t buf_ptr = state->free_queue.buffer_ptrs[head % PLATFORM_L0_FREE_SLOT_COUNT];
    rmb();
    state->free_queue.head = head + 1;
    wmb();
    return buf_ptr;
}

// Buffer is full: enqueue it ready and switch to a fresh one from the free
// queue. If no free buffer is available, overwrite the current one (markers
// lost, counted) to keep the scheduler unblocked — same fallback as L2.
static void switch_l0_buffer(int thread_idx) {
    L0PerfBufferState *state = s_l0_thread_states[thread_idx];
    L0MarkerBuffer *full_buf = s_l0_current_buf[thread_idx];
    if (state == nullptr || full_buf == nullptr) {
        return;
    }

    uint32_t seq = state->current_buf_seq;
    uint64_t new_buf_ptr = pop_l0_free_buffer(state);
    if (new_buf_ptr == 0) {
        LOG_WARN("Thread %d: L0 no free buffer, overwriting current (%u markers lost)", thread_idx, full_buf->count);
        state->dropped_marker_count += full_buf->count;
        full_buf->count = 0;
        wmb();
        return;
    }

    if (enqueue_l0_ready_buffer(thread_idx, state->current_buf_ptr, seq) != 0) {
        LOG_ERROR("Thread %d: L0 ready queue full, %u markers lost", thread_idx, full_buf->count);
        state->dropped_marker_count += full_buf->count;
        full_buf->count = 0;
        // Return the just-popped buffer would complicate SPSC; simplest is to
        // adopt it and drop the full one's data (already accounted above).
    }

    L0MarkerBuffer *new_buf = reinterpret_cast<L0MarkerBuffer *>(new_buf_ptr);
    new_buf->count = 0;
    state->current_buf_ptr = new_buf_ptr;
    state->current_buf_seq = seq + 1;
    wmb();
    s_l0_current_buf[thread_idx] = new_buf;
}

// AIV sub-core phys layout (see l0_sub_core_phys_to_cluster_phys in
// l0_perf_profiling.h): two AIVs per cluster, the first (lower phys) is aiv0
// and the second is aiv1. We need the offset *within the die's AIV range* so
// the parity gives the block.
static uint16_t aiv_block_from_phys(uint32_t sub_core_phys) {
    if (sub_core_phys >= 18 && sub_core_phys < 54) {
        return static_cast<uint16_t>(((sub_core_phys - 18) % 2 == 0) ? 1 : 2);
    }
    if (sub_core_phys >= 72 && sub_core_phys < 108) {
        return static_cast<uint16_t>(((sub_core_phys - 72) % 2 == 0) ? 1 : 2);
    }
    return 0;  // Defensive — caller already validated `group >= 0` before using this.
}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

int l0_perf_aicpu_init(int thread_idx, const int *cores_owned, const uint32_t *phys_ids, int core_num) {
    void *base = reinterpret_cast<void *>(get_platform_l0_perf_base());
    if (base == nullptr) {
        LOG_ERROR("l0_perf_aicpu_init: l0_perf_data_base is NULL (thread %d)", thread_idx);
        return -1;
    }
    if (cores_owned == nullptr || phys_ids == nullptr) {
        LOG_ERROR("l0_perf_aicpu_init: null cores_owned/phys_ids (thread %d)", thread_idx);
        return -1;
    }
    s_l0_header = reinterpret_cast<L0PerfDataHeader *>(base);

    // No shared whole-array prefill: every scheduler thread runs this init
    // concurrently (right after the init_complete_ barrier), and a thread only
    // ever writes — and later, in complete_record, reads — the entries for the
    // cores it owns. Each owned core's entry is written unconditionally below
    // (group 0..5, or -1 when outside biu_perf coverage) before this thread's
    // dispatch loop starts, so reads always observe the owning thread's own
    // write in program order.
    for (int i = 0; i < core_num; ++i) {
        int logical = cores_owned[i];
        if (logical < 0 || logical >= PLATFORM_MAX_CORES) {
            LOG_ERROR("l0_perf_aicpu_init: thread %d invalid logical core %d", thread_idx, logical);
            continue;
        }
        // Derive AIC vs AIV from the sub-core's phys range. The historical
        // `logical < aic_count` heuristic broke once L0 retire leaves logical
        // IDs sparse (e.g. surviving AICs at logical {0,9,17,18,27,35}).
        bool is_aic = l0_sub_core_phys_is_aic(phys_ids[i]);
        int cluster_phys = l0_sub_core_phys_to_cluster_phys(phys_ids[i], is_aic);
        int group = (cluster_phys < 0) ? -1 : l0_perf_phys_to_group(static_cast<uint32_t>(cluster_phys));
        s_l0_logical_to_group[logical] = group;
        s_l0_logical_to_block[logical] = is_aic ? static_cast<uint16_t>(0) : aiv_block_from_phys(phys_ids[i]);
        s_l0_logical_to_phys[logical] = phys_ids[i];
        s_l0_aicore_rings[logical] = reinterpret_cast<L0PerfAicoreRing *>(s_l0_header->aicore_ring_ptrs[logical]);
        LOG_INFO_V1(
            "l0_perf_aicpu_init: thread %d core %d phys %u (%s) cluster %d group %d block %u ring 0x%lx",
            thread_idx, logical, phys_ids[i], is_aic ? "AIC" : "AIV", cluster_phys, group,
            s_l0_logical_to_block[logical],
            reinterpret_cast<unsigned long>(s_l0_aicore_rings[logical])
        );
    }

    // Bring up this thread's marker buffer pool: cache the per-thread state and
    // pop the first buffer from the (host-seeded) free queue.
    if (thread_idx >= 0 && thread_idx < PLATFORM_MAX_AICPU_THREADS) {
        L0PerfBufferState *state = &s_l0_header->thread_states[thread_idx];
        s_l0_thread_states[thread_idx] = state;
        uint64_t buf_ptr = pop_l0_free_buffer(state);
        if (buf_ptr != 0) {
            L0MarkerBuffer *buf = reinterpret_cast<L0MarkerBuffer *>(buf_ptr);
            buf->count = 0;
            state->current_buf_ptr = buf_ptr;
            state->current_buf_seq = 0;
            wmb();
            s_l0_current_buf[thread_idx] = buf;
        } else {
            LOG_ERROR("l0_perf_aicpu_init: thread %d free queue empty, no initial marker buffer", thread_idx);
            state->current_buf_ptr = 0;
            s_l0_current_buf[thread_idx] = nullptr;
        }
    }

    LOG_INFO_V1("l0_perf_aicpu_init: thread %d initialized %d cores", thread_idx, core_num);
    return 0;
}

int l0_perf_aicpu_complete_record(
    int core_id, int thread_idx, uint32_t expected_reg_task_id, uint64_t task_id, CoreType core_type
) {
    if (s_l0_header == nullptr || core_id < 0 || core_id >= PLATFORM_MAX_CORES) {
        return -1;
    }
    int group = s_l0_logical_to_group[core_id];
    if (group < 0) {
        return 0;  // core outside biu_perf coverage — host won't get a marker
    }
    L0PerfAicoreRing *ring = s_l0_aicore_rings[core_id];
    if (ring == nullptr) {
        LOG_ERROR(
            "l0_perf_aicpu_complete_record: core %d has no AICore ring (init missed it or L0 disabled)", core_id
        );
        return -1;
    }

    // Read AICore-published timing from the per-core staging ring.
    L0PerfRecordRaw *slot = &ring->slots[expected_reg_task_id % PLATFORM_L0_AICORE_RING_SIZE];
    cache_invalidate_range(slot, sizeof(L0PerfRecordRaw));
    if (static_cast<uint32_t>(slot->task_id) != expected_reg_task_id) {
        // Same invariant as L2: completion-before-dispatch guarantees AICore
        // has published the slot before AICPU sees FIN. A mismatch means
        // either in-flight depth > PLATFORM_L0_AICORE_RING_SIZE or AICore
        // failed to dcci before signaling.
        LOG_ERROR(
            "L0Perf invariant violated: core %d slot task_id=0x%x expected=0x%x "
            "(completion-before-dispatch broken or ring undersized)",
            core_id, static_cast<uint32_t>(slot->task_id), expected_reg_task_id
        );
        return -1;
    }

    L0PerfBufferState *state = s_l0_thread_states[thread_idx];
    L0MarkerBuffer *buf = s_l0_current_buf[thread_idx];
    if (state == nullptr || buf == nullptr) {
        // init failed to seed a buffer for this thread; nothing we can do.
        return -1;
    }
    state->total_marker_count += 1;

    buf->markers[buf->count] = L0TaskFinMarker{
        task_id, slot->start_cycle, slot->end_cycle, static_cast<uint32_t>(group),
        s_l0_logical_to_phys[core_id], static_cast<uint16_t>(core_type), s_l0_logical_to_block[core_id],
        /*reserved=*/0
    };
    buf->count += 1;

    // Full → publish + switch. Re-read the (possibly switched) current buffer
    // for the next caller via s_l0_current_buf.
    if (buf->count >= PLATFORM_L0_MARKERS_PER_BUFFER) {
        switch_l0_buffer(thread_idx);
    }
    return 0;
}

void l0_perf_aicpu_flush(int thread_idx) {
    if (s_l0_header == nullptr || thread_idx < 0 || thread_idx >= PLATFORM_MAX_AICPU_THREADS) {
        return;
    }
    L0PerfBufferState *state = s_l0_thread_states[thread_idx];
    L0MarkerBuffer *buf = s_l0_current_buf[thread_idx];
    if (state == nullptr || buf == nullptr || buf->count == 0) {
        return;  // nothing buffered
    }

    uint32_t seq = state->current_buf_seq;
    if (enqueue_l0_ready_buffer(thread_idx, state->current_buf_ptr, seq) != 0) {
        LOG_ERROR("Thread %d: L0 flush failed (ready queue full), %u markers lost", thread_idx, buf->count);
        state->dropped_marker_count += buf->count;
        buf->count = 0;
        wmb();
        return;
    }
    // Detach the flushed buffer; no new buffer needed (run is ending).
    state->current_buf_ptr = 0;
    s_l0_current_buf[thread_idx] = nullptr;
    wmb();
    LOG_INFO_V1("Thread %d: L0 flushed final buffer (%u markers)", thread_idx, buf->count);
}
