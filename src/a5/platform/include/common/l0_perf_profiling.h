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
 * @file l0_perf_profiling.h
 * @brief DAV_3510 (a5) core-swimlane (biu_perf / L0) profiling layout.
 *
 * Architecture:
 *
 *   AICore: mark_stamp -> HW DFX trace bus -> SoC trace ring (driver-managed).
 *           Also calls perf_aicore_l0_record_task on each task end to publish
 *           the task's (start, end) AICore cycle window into the per-core
 *           L0PerfAicoreRing slot.
 *   AICPU scheduler thread:
 *     - on task FIN, reads the L0PerfAicoreRing slot to lift (start, end),
 *       builds an enriched L0TaskFinMarker{real_task_id, group, start_cycle,
 *       end_cycle, core_type, block}, pushes it into the GM per-thread ready
 *       queue. No driver consumer API calls.
 *   Host L0PerfCollector:
 *     - initialize(): dlopen libascend_hal.so, prof_drv_start the 18
 *       biu_perf channels.
 *     - on_buffer_collected(marker): remembers the marker's cycle window for
 *       its group, calls prof_channel_read on the group's 3 sub-core channels,
 *       decodes chunks, and assigns each pipe stamp to the task whose
 *       (start_cycle, end_cycle) window contains the stamp's cycle. The
 *       marker that triggered the read no longer owns the resulting batch.
 *     - finalize(): prof_stop all 18 channels.
 *
 * AICPU cannot act as consumer: CANN 9.1.T500's device-stub
 * libascend_hal.so returns DRV_ERROR_NOT_SUPPORT for prof_drv_start, so the
 * consumer side has to live on host.
 *
 * a5 has no halHostRegister (DAV_3510), so host↔device SPSC fields go
 * through rtMemcpy against a host shadow buffer.
 */

#ifndef SRC_A5_PLATFORM_INCLUDE_COMMON_L0_PERF_PROFILING_H_
#define SRC_A5_PLATFORM_INCLUDE_COMMON_L0_PERF_PROFILING_H_

#include <cstddef>
#include <cstdint>

#include "common/platform_config.h"

// =============================================================================
// biu_perf channel / physical-core constants (CANN 9.1.T500, A5 onboard)
// =============================================================================
//
// biu_perf currently covers 6 AICore (18 sub-cores), fixed channel ids 11..28.
// The driver expanding coverage is a matter of editing this constant set only;
// business code reads through the helpers.

constexpr uint32_t kBiuPerfChanBase = 11;     // driver-assigned start id
constexpr uint32_t kBiuPerfNumGroups = 6;     // AICore covered today
constexpr uint32_t kBiuPerfSubPerGroup = 3;   // aic / aiv0 / aiv1
constexpr uint32_t kBiuPerfNumChans = kBiuPerfNumGroups * kBiuPerfSubPerGroup;  // 18

// CLUSTER phys ids (global, 0..35) biu_perf is wired to. This is a HARDWARE
// FIXED SET — the trace bus only carries DFX data for these 6 clusters
// (every 9th of the 36 across the SoC), and `prof_drv_start` with any other
// `aicore_phys_id` would never produce data.
//
// Dispatch is locked to these 6 clusters by AICPU
// SchedulerContext::retire_uncovered_cores_for_l0(): it enables every AICore
// at handshake so it can see which physical clusters it got, then sends an
// exit signal to all cores whose cluster is not in this set. ACL group
// affinity APIs are unsupported on a5 (ACL_ERROR_RT_FEATURE_NOT_SUPPORT)
// and aclrtSetDeviceResLimit only caps concurrency, not placement.
//
// Matches msprof's GenGroupVector(36); do not change without confirming the
// HW-fixed set has actually moved.
constexpr uint32_t kBiuPerfPhysAicore[kBiuPerfNumGroups] = {0, 9, 17, 18, 27, 35};

// After prof_channel_read returns 0 we re-read the channel up to this many
// extra times to catch chunks still in flight from the just-finished task.
// (prof_channel_poll is unusable on this hot path — it's a global readable-
// channel selector with a seconds-resolution timeout.) The primary flush
// guarantee is the kernel's trailing `bar.all`.
constexpr int kBiuPerfDrainRetries = 4;

// Optional debug sentinel region id a kernel may stamp at its tail to verify
// drain completeness. Not used by default.
constexpr uint16_t kRegionFinSentinel = 4095;

// Sentinel "no biu_perf coverage" group value carried in L0TaskFinMarker.group.
constexpr uint32_t kL0NoCoverageGroup = UINT32_MAX;

// channel_id = base + group * sub_per_group + sub  (sub: 0=aic,1=aiv0,2=aiv1)
inline uint32_t biu_perf_chan_id(uint32_t group, uint32_t sub) {
    return kBiuPerfChanBase + group * kBiuPerfSubPerGroup + sub;
}

// Map a global cluster phys id (0..35) to its biu_perf group index (0..5), or
// -1 if the cluster is outside the monitored set.
//
// IMPORTANT: input is *cluster phys*, NOT *sub-core phys*. AICPU's
// physical_core_ids_[] holds sub-core phys (per-AIC or per-AIV), which is a
// different ID space — convert first via l0_sub_core_phys_to_cluster_phys().
inline int l0_perf_phys_to_group(uint32_t cluster_phys) {
    for (uint32_t g = 0; g < kBiuPerfNumGroups; ++g) {
        if (kBiuPerfPhysAicore[g] == cluster_phys) {
            return static_cast<int>(g);
        }
    }
    return -1;
}

// Classify a sub-core phys id as AIC (cube) vs AIV (vec). Required because
// after L0 retire, logical core ids are no longer contiguous AIC-then-AIV,
// so callers must derive AIC/AIV from the phys layout rather than the
// historical `logical < aic_count` heuristic. Phys layout:
//   die 0 AIC: 0..17    die 0 AIV: 18..53
//   die 1 AIC: 54..71   die 1 AIV: 72..107
inline bool l0_sub_core_phys_is_aic(uint32_t sub_core_phys) {
    return sub_core_phys < 18 || (sub_core_phys >= 54 && sub_core_phys < 72);
}

// Convert sub-core phys id (per-AIC or per-AIV) to global cluster phys id
// (0..35). Reverse-engineered from dev=2 handshake logs (2026-05-25):
//
//   A5 SoC has 2 dies × 18 clusters each = 36 clusters total.
//   Die 0:  AIC phys 0..17    → cluster_phys = phys                (1:1)
//           AIV phys 18..53   → cluster_phys = (phys - 18) / 2     (2:1, two AIVs per cluster)
//   Die 1:  AIC phys 54..71   → cluster_phys = 18 + (phys - 54)
//           AIV phys 72..107  → cluster_phys = 18 + (phys - 72) / 2
//
// Returns -1 if the sub-core phys is outside the known ranges (defensive —
// shouldn't happen with handshake-reported phys ids).
inline int l0_sub_core_phys_to_cluster_phys(uint32_t sub_core_phys, bool is_aic) {
    if (is_aic) {
        if (sub_core_phys < 18) {
            return static_cast<int>(sub_core_phys);
        }
        if (sub_core_phys >= 54 && sub_core_phys < 72) {
            return 18 + static_cast<int>(sub_core_phys - 54);
        }
        return -1;
    }
    // AIV
    if (sub_core_phys >= 18 && sub_core_phys < 54) {
        return static_cast<int>((sub_core_phys - 18) / 2);
    }
    if (sub_core_phys >= 72 && sub_core_phys < 108) {
        return 18 + static_cast<int>((sub_core_phys - 72) / 2);
    }
    return -1;
}

/**
 * `prof_start_para.user_data` for a biu_perf channel — fixed 16-byte, all
 * little-endian uint32 (per CANN 9.1.T500's libascend_hal). One channel is
 * started at a time, so user_data points at a single sub-core.
 *
 * Built only on host (L0PerfCollector::initialize).
 */
struct BiuPerfStartUserData {
    uint32_t hdr_size;       // = sizeof(BiuPerfStartUserData) = 16, self-describing
    uint32_t biu_mode;       // 0 = perf monitor
    uint32_t sub_core_type;  // 0 = aic, 1 = aiv0, 2 = aiv1
    uint32_t aicore_phys_id;  // from kBiuPerfPhysAicore[]
};
static_assert(sizeof(BiuPerfStartUserData) == 16, "BiuPerfStartUserData must be 16 bytes");

// =============================================================================
// 4-byte chunk decode (authoritative layout from msprof biu_perf parser)
// =============================================================================
//
// Layout of one 4-byte LE chunk, taken verbatim from msprof's
// profiling_bean/biu_perf/biu_perf_bean.py::BiuPerfInstructionBean +
// msparser/biu_perf/biu_perf_chip6_parser.py::_get_ctrl_type:
//
//   bits[31:28] = ctrl_type (4 bit, pipe tag / sentinel kind)
//   bits[27:16] = region_id / events (12 bit)
//   bits[15:0]  = sys_cnt (16-bit delta from the last START_STAMP)
//
// The msprof parser is the ground truth for bit positions — do not "fix"
// them based on any other source diagram.
//
// ctrl_type semantics:
//   0..6  pipe stamps (SU/VEC/CUBE/MTE1/MTE2/MTE3/FIXP) — region_id is the
//         user mark_stamp id; emit an L0PerfRecord.
//   14    START_STAMP — this chunk + the next 3 chunks form a 64-bit absolute
//         base cycle + 16-bit block_id; rebases the running cycle.
//   15    STATE — periodic pipe activity bitmask (not a mark_stamp); accumulate
//         its delta to keep cycles aligned, but do NOT emit a record.

constexpr uint32_t kL0ChunkEndFlag = 0xEFFFFFFFu;  // INSTR_END_FLAG (block end)
constexpr uint32_t kL0ChunkFiller = 0xDDDDDDDDu;    // INSTR_END_FILLER (padding)
constexpr uint8_t kL0CtrlStartStamp = 14;
constexpr uint8_t kL0CtrlState = 15;
constexpr uint8_t kL0CtrlMaxPipe = 6;  // 0..6 are real pipe stamps

inline uint8_t l0_chunk_ctrl(uint32_t w) { return static_cast<uint8_t>((w >> 28) & 0xFu); }
inline uint16_t l0_chunk_region(uint32_t w) { return static_cast<uint16_t>((w >> 16) & 0xFFFu); }
inline uint16_t l0_chunk_delta(uint32_t w) { return static_cast<uint16_t>(w & 0xFFFFu); }

/**
 * Streaming decode state for one channel. base_cycle is the running absolute
 * cycle rebuilt from START_STAMP groups + accumulated deltas. carry[] holds a
 * partial tail (incomplete 4-byte chunk, or an incomplete START_STAMP group)
 * from a previous prof_channel_read so decoding survives read boundaries.
 *
 * Host-side state (one per sub-core, kept across on_buffer_collected calls).
 */
struct L0DecodeState {
    uint64_t base_cycle = 0;
    uint16_t block_id = 0;
    bool have_base = false;     // seen a START_STAMP yet
    uint8_t carry[16] = {0};    // up to 3 stamp chunks (12B) + a partial chunk
    uint8_t carry_len = 0;      // bytes valid in carry[]
};

inline uint32_t l0_read_chunk_le(const uint8_t *p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

/**
 * Assemble a START_STAMP group (4 consecutive ctrl_type==14 chunks) into a
 * 64-bit base cycle + 16-bit block_id. Mirrors
 * biu_perf_chip6_parser._init_timestamp_and_block: the 64-bit cycle is the four
 * chunks' low-16 bits, most-significant chunk first (reversed); block_id is
 * byte[2] of chunks 1 and 0.
 */
inline void l0_assemble_start_stamp(const uint8_t *g, uint64_t &base_cycle, uint16_t &block_id) {
    uint64_t syscnt = 0;
    for (int i = 3; i >= 0; --i) {
        uint16_t low16 = static_cast<uint16_t>(g[i * 4] | (g[i * 4 + 1] << 8));
        syscnt = (syscnt << 16) | low16;
    }
    base_cycle = syscnt;
    block_id = static_cast<uint16_t>((g[1 * 4 + 2] << 8) | g[0 * 4 + 2]);
}

/**
 * Decode a freshly-read trace buffer, emitting one pipe record per ctrl_type
 * 0..6 chunk via `emit(region_id, pipe, cycle, block_id)`.
 *
 * Stateless w.r.t. the caller's buffer storage: incomplete trailing bytes
 * (a sub-4-byte fragment, or a START_STAMP group split across a read boundary)
 * are stashed in `st.carry[]` and resumed on the next call.
 *
 * @return number of pipe records emitted.
 */
template <typename Emit>
inline int l0_decode_chunks(const uint8_t *buf, int bytes, L0DecodeState &st, Emit &&emit) {
    // Splice any carried-over bytes in front of the new buffer into a small
    // local window so chunk/group boundaries are contiguous. We process from a
    // logical stream = carry[] ++ buf[].
    int emitted = 0;
    uint8_t local[16];
    int local_len = st.carry_len;
    for (int i = 0; i < st.carry_len; ++i) {
        local[i] = st.carry[i];
    }
    st.carry_len = 0;

    int src = 0;  // index into buf
    auto next_chunk = [&](uint32_t &out) -> bool {
        // Prefer draining the carried bytes first (they precede buf logically).
        if (local_len >= 4) {
            out = l0_read_chunk_le(local);
            // shift remaining local bytes down
            for (int i = 4; i < local_len; ++i) {
                local[i - 4] = local[i];
            }
            local_len -= 4;
            return true;
        }
        // Need to refill local from buf to complete a chunk, or read directly.
        if (local_len == 0 && bytes - src >= 4) {
            out = l0_read_chunk_le(buf + src);
            src += 4;
            return true;
        }
        // Top up local with bytes from buf to reach 4.
        while (local_len < 4 && src < bytes) {
            local[local_len++] = buf[src++];
        }
        if (local_len >= 4) {
            out = l0_read_chunk_le(local);
            for (int i = 4; i < local_len; ++i) {
                local[i - 4] = local[i];
            }
            local_len -= 4;
            return true;
        }
        return false;
    };

    // peek/availability for START_STAMP grouping: count chunks left without
    // consuming. A chunk is available if local_len + (bytes - src) >= 4*n.
    auto chunks_available = [&]() -> int { return (local_len + (bytes - src)) / 4; };

    uint32_t w;
    while (next_chunk(w)) {
        if (w == kL0ChunkEndFlag) {
            // Block end — following fillers are skipped naturally as plain
            // chunks (0xDDDDDDDD decodes to ctrl_type 13, region 0xDDD; we just
            // drop them explicitly to avoid spurious records).
            continue;
        }
        if (w == kL0ChunkFiller) {
            continue;
        }

        uint8_t ctrl = l0_chunk_ctrl(w);
        if (ctrl == kL0CtrlStartStamp) {
            // Need this chunk + 3 more. If they aren't all here, stash the
            // whole partial group into carry[] and resume next call.
            if (chunks_available() < 3) {
                // Re-stash: w (already consumed) + whatever remains.
                st.carry[0] = static_cast<uint8_t>(w & 0xFF);
                st.carry[1] = static_cast<uint8_t>((w >> 8) & 0xFF);
                st.carry[2] = static_cast<uint8_t>((w >> 16) & 0xFF);
                st.carry[3] = static_cast<uint8_t>((w >> 24) & 0xFF);
                st.carry_len = 4;
                while (st.carry_len < 16 && local_len > 0) {
                    st.carry[st.carry_len++] = local[0];
                    for (int i = 1; i < local_len; ++i) local[i - 1] = local[i];
                    local_len--;
                }
                while (st.carry_len < 16 && src < bytes) {
                    st.carry[st.carry_len++] = buf[src++];
                }
                break;
            }
            uint8_t group[16];
            group[0] = static_cast<uint8_t>(w & 0xFF);
            group[1] = static_cast<uint8_t>((w >> 8) & 0xFF);
            group[2] = static_cast<uint8_t>((w >> 16) & 0xFF);
            group[3] = static_cast<uint8_t>((w >> 24) & 0xFF);
            for (int c = 1; c < 4; ++c) {
                uint32_t cw = 0;
                next_chunk(cw);  // guaranteed by chunks_available() check
                group[c * 4 + 0] = static_cast<uint8_t>(cw & 0xFF);
                group[c * 4 + 1] = static_cast<uint8_t>((cw >> 8) & 0xFF);
                group[c * 4 + 2] = static_cast<uint8_t>((cw >> 16) & 0xFF);
                group[c * 4 + 3] = static_cast<uint8_t>((cw >> 24) & 0xFF);
            }
            l0_assemble_start_stamp(group, st.base_cycle, st.block_id);
            st.have_base = true;
            continue;
        }

        // Normal chunk (pipe stamp 0..6, or STATE 15): accumulate the delta.
        st.base_cycle += l0_chunk_delta(w);
        if (ctrl <= kL0CtrlMaxPipe) {
            emit(l0_chunk_region(w), ctrl, st.base_cycle, st.block_id);
            ++emitted;
        }
    }

    // Any leftover sub-chunk bytes (bytes not a multiple of 4 and no pending
    // START_STAMP group) get carried forward.
    if (st.carry_len == 0 && local_len > 0) {
        for (int i = 0; i < local_len; ++i) st.carry[i] = local[i];
        st.carry_len = static_cast<uint8_t>(local_len);
    }
    return emitted;
}

// =============================================================================
// AICore -> AICPU staging ring (per-core stable address)
// =============================================================================

/**
 * One AICore-published task window. The host attributes each decoded pipe
 * stamp to the task whose [start_cycle, end_cycle] range contains the
 * stamp's cycle, so this struct carries the minimum needed for that match.
 *
 * task_id holds the register dispatch token (low 32 bits) and is written
 * last so AICPU can validate the slot before consuming it.
 */
struct L0PerfRecordRaw {
    uint64_t start_cycle;
    uint64_t end_cycle;
    uint64_t task_id;
} __attribute__((aligned(64)));

static_assert(sizeof(L0PerfRecordRaw) == 64, "L0PerfRecordRaw must be one cache line");

/**
 * Per-core staging ring. Lives at a fixed host-allocated address so AICore's
 * write target is stable across the whole run. AICPU reads
 * `slots[reg_task_id % PLATFORM_L0_AICORE_RING_SIZE]` in
 * l0_perf_aicpu_complete_record.
 */
struct L0PerfAicoreRing {
    L0PerfRecordRaw slots[PLATFORM_L0_AICORE_RING_SIZE];
} __attribute__((aligned(64)));

// =============================================================================
// L0 perf record + GM ReadyQueue (marker-based)
// =============================================================================

/**
 * One decoded pipe stamp. Written exclusively by the **host** L0PerfCollector
 * in on_buffer_collected after draining a marker's channels. task_id is set
 * by matching the stamp's cycle against the per-group window deque, not by
 * the marker that triggered the read (driver flushes are batched).
 */
struct L0PerfRecord {
    uint64_t cycle;        // absolute cycle (START_STAMP base + accumulated delta)
    uint64_t task_id;      // owner task resolved by cycle-window match
    uint16_t region_id;    // 12-bit mark_stamp id
    uint8_t pipe;          // ctrl_type 0..6
    uint8_t sub_core_id;   // group * 3 + sub, 0..17
    uint16_t core_type;    // CoreType (AIC/AIV) of the owning task, 0xFFFF if unmatched
    uint16_t block;        // logical block-in-cluster of the owning task, 0xFFFF if unmatched
} __attribute__((aligned(8)));

static_assert(sizeof(L0PerfRecord) == 24, "L0PerfRecord must be 24 bytes");

/**
 * AICPU → host signal. Pushed by AICPU scheduler thread on task FIN; consumed
 * by host mgmt thread which forwards into manager_.ready_queue, then by
 * collector_thread in on_buffer_collected. Carries the task's biu_perf group
 * (which 3 sub-core channels to drain) plus the AICore-published cycle window
 * and AICPU-side enrichment (real task_id, core_type, block) needed for
 * cycle-based stamp attribution.
 *
 * group == kL0NoCoverageGroup means "this task's AICore is not in the biu_perf
 * 6-core coverage" — AICPU never pushes such a marker in practice (it
 * short-circuits in l0_perf_aicpu_complete_record), but host treats it as a
 * no-op for safety.
 */
struct L0TaskFinMarker {
    uint64_t task_id;        // real PTO2 id (from scheduler), not the reg dispatch token
    uint64_t start_cycle;    // AICore HW cycle at task start (from L0PerfAicoreRing)
    uint64_t end_cycle;      // AICore HW cycle at task end   (from L0PerfAicoreRing)
    uint32_t group;          // 0..kBiuPerfNumGroups-1, or kL0NoCoverageGroup
    uint32_t sub_core_phys;  // raw sub-core phys id of the executing core (for aiv0/aiv1 verification)
    uint16_t core_type;      // CoreType (AIC=0, AIV=1)
    uint16_t block;          // logical block-in-cluster (0..2: aic, aiv0, aiv1)
    uint32_t reserved;       // pad to 40 bytes
} __attribute__((aligned(8)));

static_assert(sizeof(L0TaskFinMarker) == 40, "L0TaskFinMarker must be 40 bytes");

// =============================================================================
// L0 marker buffer-pool transport (mirrors L2/PMU)
// =============================================================================
//
// AICPU appends markers into a per-thread L0MarkerBuffer; when full it enqueues
// the buffer pointer into the per-thread ready queue and pops a fresh buffer
// from the free queue (host pre-seeds + recycles). One ready-queue entry = one
// full buffer, so the host's per-entry rtMemcpy cost is amortized over
// PLATFORM_L0_MARKERS_PER_BUFFER markers.

/**
 * Fixed-size marker buffer. Owned/written exclusively by one AICPU thread;
 * host reads after it's enqueued full. `count` is the committed marker count.
 */
struct L0MarkerBuffer {
    L0TaskFinMarker markers[PLATFORM_L0_MARKERS_PER_BUFFER];
    volatile uint32_t count;
} __attribute__((aligned(64)));

/**
 * SPSC free queue of available buffer pointers. Host (producer) pushes via
 * tail, AICPU (consumer) pops via head. Same shape as L2PerfFreeQueue.
 */
struct L0FreeQueue {
    volatile uint64_t buffer_ptrs[PLATFORM_L0_FREE_SLOT_COUNT];  // 4 * 8 = 32 B
    volatile uint32_t head;  // AICPU increments (pop)
    volatile uint32_t tail;  // host increments (push)
    uint32_t pad[6];         // pad to 64 bytes (32 + 8 + 24)
} __attribute__((aligned(64)));

static_assert(sizeof(L0FreeQueue) == 64, "L0FreeQueue must be 64 bytes (assumes PLATFORM_L0_FREE_SLOT_COUNT == 4)");

/**
 * Per-thread buffer state: free queue + current active buffer + tallies.
 * Writers: host owns free_queue.tail; AICPU owns free_queue.head,
 * current_buf_ptr, current_buf_seq, and the *_count fields.
 */
struct L0PerfBufferState {
    L0FreeQueue free_queue;                    // 64 B
    volatile uint64_t current_buf_ptr;        // active buffer (0 = none)
    volatile uint32_t current_buf_seq;        // monotonic ordering
    volatile uint32_t total_marker_count;     // markers AICPU attempted to commit
    volatile uint32_t dropped_marker_count;   // markers dropped (no free buffer / full)
    uint32_t pad[11];                         // pad to 128 bytes (64 + 20 + 44)
} __attribute__((aligned(64)));

static_assert(sizeof(L0PerfBufferState) == 128, "L0PerfBufferState must be 128 bytes");

/**
 * Ready queue entry: pointer to a full buffer. `buffer_ptr == 0` doubles as
 * the framework's torn-read publish flag (try_pop_aicpu_entry skips it until
 * AICPU writes the non-zero pointer last).
 */
struct L0ReadyQueueEntry {
    uint64_t buffer_ptr;    // device pointer to the full L0MarkerBuffer (0 = empty)
    uint32_t buffer_seq;    // ordering
    uint32_t thread_index;  // producing AICPU thread
} __attribute__((aligned(16)));

/**
 * L0 perf data fixed header at the start of L0 shared memory.
 *   - per-thread ready queues (AICPU pushes full-buffer pointers, host pops)
 *   - per-thread buffer states (free queue + current buffer)
 *   - per-core AICore L0 staging ring pointers
 * Actual L0MarkerBuffers are allocated dynamically by host and pushed into the
 * per-thread free queues.
 */
struct L0PerfDataHeader {
    L0ReadyQueueEntry queues[PLATFORM_MAX_AICPU_THREADS][PLATFORM_L0_PERF_READYQUEUE_SIZE];
    volatile uint32_t queue_heads[PLATFORM_MAX_AICPU_THREADS];  // Host reads (consumer)
    volatile uint32_t queue_tails[PLATFORM_MAX_AICPU_THREADS];  // AICPU writes (producer)
    L0PerfBufferState thread_states[PLATFORM_MAX_AICPU_THREADS];
    uint32_t num_chans;  // = kBiuPerfNumChans
    uint32_t num_threads;
    uint32_t pad[2];
    // Per-core L0PerfAicoreRing device pointers. Written once by the host
    // during init (one entry per AICore worker, indexed by logical core id);
    // AICPU's l0_perf_aicpu_init reads them so complete_record can pull each
    // task's (start_cycle, end_cycle) out of the staging ring slot.
    uint64_t aicore_ring_ptrs[PLATFORM_MAX_CORES];
} __attribute__((aligned(64)));

// =============================================================================
// Helper Functions
// =============================================================================

inline size_t calc_l0_perf_data_size() {
    return sizeof(L0PerfDataHeader);
}

inline L0PerfDataHeader *get_l0_perf_header(void *base_ptr) {
    return reinterpret_cast<L0PerfDataHeader *>(base_ptr);
}

#endif  // SRC_A5_PLATFORM_INCLUDE_COMMON_L0_PERF_PROFILING_H_
