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
 * @file l0_perf_collector.h
 * @brief Host-side core-swimlane (biu_perf / L0) consumer.
 *
 * Data plane: the 18 biu_perf channels (chan 11..28) deliver per-pipe
 * stamps via the host driver's DFX trace ring; host reads them with
 * prof_channel_read. AICPU only pushes an L0TaskFinMarker into a GM
 * per-thread queue on each task FIN.
 *
 * initialize() arms the driver in a fixed order (rollback on any failure):
 *   1. aclprofInit + aclprofCreateConfig(ACL_PROF_AICORE_METRICS,
 *      ACL_AICORE_PIPE_UTILIZATION) + aclprofStart
 *   2. rtProfSetProSwitch(PROF_INSTR, START)
 *      ↑ Steps 1+2 are mandatory — without them prof_drv_start succeeds
 *        but prof_channel_read returns 0 bytes forever; the driver only
 *        routes DFX trace data to biu_perf channels once this handshake
 *        is in place.
 *   3. dlopen libascend_hal.so + dlsym prof_drv_*
 *   4. allocate GM L0PerfDataHeader (per-thread marker queues only;
 *      no buffer pool because payload is in the driver ring, not GM)
 *   5. prof_drv_start the 18 biu_perf channels — synchronous, completes
 *      before return so the HW DFX trace ring is recording before the
 *      caller launches any kernel.
 *
 * on_buffer_collected(marker) runs on ProfilerBase's collector_thread:
 *   prof_channel_read the marker's 3 sub-core channels, decode 4-byte
 *   chunks, append L0PerfRecord tagged with marker.task_id.
 *
 * finalize() reverses initialize: ProfilerBase::stop joins mgmt + collector
 * → prof_stop x18 → dlclose → rtProfSetProSwitch(PROF_INSTR, STOP) →
 * aclprofStop / DestroyConfig / Finalize → release GM.
 *
 * export_swimlane_json() emits two files under <output_prefix>:
 *   l0_perf_records.json — records grouped by task_id
 *   l0_perf_trace.json   — Perfetto / Chrome traceEvents
 */

#ifndef SRC_A5_PLATFORM_INCLUDE_HOST_L0_PERF_COLLECTOR_H_
#define SRC_A5_PLATFORM_INCLUDE_HOST_L0_PERF_COLLECTOR_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "common/l0_perf_profiling.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "host/profiling_common/profiler_base.h"

// Forward declarations to avoid pulling acl_prof.h into a header.
struct aclprofConfig;

// ---------------------------------------------------------------------------
// L0 perf hand-off + Module (buffer-pool — one L0MarkerBuffer per ready entry)
// ---------------------------------------------------------------------------

struct L0PerfReadyBufferInfo {
    uint32_t thread_index;      // producing AICPU scheduler thread
    void *dev_buffer_ptr;       // device L0MarkerBuffer
    void *host_buffer_ptr;      // host shadow (filled by ProfilerAlgorithms)
    uint32_t buffer_seq;
};

struct L0PerfModule {
    using DataHeader = L0PerfDataHeader;
    using ReadyEntry = L0ReadyQueueEntry;
    using ReadyBufferInfo = ::L0PerfReadyBufferInfo;
    using FreeQueue = L0FreeQueue;

    static constexpr int kBufferKinds = 1;
    static constexpr uint32_t kReadyQueueSize = PLATFORM_L0_PERF_READYQUEUE_SIZE;
    static constexpr uint32_t kSlotCount = PLATFORM_L0_FREE_SLOT_COUNT;
    static constexpr const char *kSubsystemName = "L0PerfModule";

    // Refill a fully-drained recycled pool to PLATFORM_L0_BUFFERS_PER_THREAD
    // in one tick (clamped to >= 1).
    static constexpr int batch_size(int /*kind*/) {
        constexpr int b = PLATFORM_L0_BUFFERS_PER_THREAD - PLATFORM_L0_FREE_SLOT_COUNT;
        return b < 1 ? 1 : b;
    }

    static int kind_of(const ReadyBufferInfo & /*info*/) { return 0; }

    static DataHeader *header_from_shm(void *shm) { return get_l0_perf_header(shm); }

    static std::optional<profiling_common::EntrySite<L0PerfModule>>
    resolve_entry(void * /*shm*/, DataHeader *header, int /*q*/, const ReadyEntry &entry) {
        if (entry.thread_index >= static_cast<uint32_t>(PLATFORM_MAX_AICPU_THREADS)) {
            LOG_ERROR("L0PerfModule: invalid ready entry thread=%u", entry.thread_index);
            return std::nullopt;
        }
        profiling_common::EntrySite<L0PerfModule> site;
        site.kind = 0;
        site.free_queue = &header->thread_states[entry.thread_index].free_queue;
        site.buffer_size = sizeof(L0MarkerBuffer);
        site.info.thread_index = entry.thread_index;
        site.info.dev_buffer_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
        site.info.host_buffer_ptr = nullptr;  // filled by ProfilerAlgorithms
        site.info.buffer_seq = entry.buffer_seq;
        return site;
    }

    template <typename Cb>
    static void for_each_instance(void * /*shm*/, DataHeader *header, Cb &&cb) {
        int n = static_cast<int>(header->num_threads);
        if (n > PLATFORM_MAX_AICPU_THREADS) n = PLATFORM_MAX_AICPU_THREADS;
        for (int t = 0; t < n; t++) {
            cb(/*kind=*/0, &header->thread_states[t].free_queue, sizeof(L0MarkerBuffer));
        }
    }
};

// ---------------------------------------------------------------------------
// Memory operation callbacks (injected by DeviceRunner).
// ---------------------------------------------------------------------------

using L0PerfAllocCallback = profiling_common::ProfAllocCallback;
using L0PerfRegisterCallback = profiling_common::ProfRegisterCallback;
using L0PerfUnregisterCallback = profiling_common::ProfUnregisterCallback;
using L0PerfFreeCallback = profiling_common::ProfFreeCallback;

// ---------------------------------------------------------------------------
// L0PerfCollector
// ---------------------------------------------------------------------------

class L0PerfCollector : public profiling_common::ProfilerBase<L0PerfCollector, L0PerfModule> {
public:
    L0PerfCollector() = default;
    ~L0PerfCollector();

    L0PerfCollector(const L0PerfCollector &) = delete;
    L0PerfCollector &operator=(const L0PerfCollector &) = delete;

    // ProfilerBase contract
    static constexpr int kIdleTimeoutSec = PLATFORM_L0_PERF_TIMEOUT_SECONDS;
    static constexpr const char *kSubsystemName = "L0Perf";

    /**
     * Bring up the L0 driver session and start the 18 biu_perf channels.
     * Returns 0 on success; on failure all already-acquired resources are
     * released before returning. See class-level doc for the step order.
     *
     * Also allocates `num_aicore` L0PerfAicoreRing slots and writes the
     * per-core device pointers into the shared header's `aicore_ring_ptrs`
     * table. KernelArgs::aicore_l0_perf_ring_addrs points at that same table
     * (= shm_dev + offsetof(L0PerfDataHeader, aicore_ring_ptrs)), so both
     * AICore kernel entry and AICPU init read the same array.
     *
     * @param output_prefix  Output directory for l0_perf_records.json and
     *                       l0_perf_trace.json. The ACL prof session also
     *                       writes its own metadata into <prefix>/acl_prof_l0/.
     */
    int initialize(
        int num_aicore, int num_threads, int device_id, const L0PerfAllocCallback &alloc_cb,
        L0PerfRegisterCallback register_cb, const L0PerfFreeCallback &free_cb, const std::string &output_prefix
    );

    /// Wire into KernelArgs::l0_perf_data_base after initialize() succeeds.
    void *get_l0_perf_setup_device_ptr() const { return shm_dev_; }

    /// Device pointer to `L0PerfDataHeader::aicore_ring_ptrs[]`. Wire into
    /// `KernelArgs::aicore_l0_perf_ring_addrs` so the AICore kernel entry can
    /// resolve its per-core ring just like it does for L2/PMU.
    void *get_aicore_ring_addrs_device_ptr() const;

    /// One-shot diagnostic. After initialize() succeeds, enumerate all ACL
    /// compute groups + (via libascend_hal) read each group's aicore_mask,
    /// logging which cluster phys IDs are actually accessible to this
    /// process. Used to confirm whether `aclrtSetDeviceResLimit(CUBE=6,
    /// VECTOR=12)` actually constrained dispatch to the biu_perf-monitored
    /// `kBiuPerfPhysAicore` set or just capped concurrency.
    void diagnose_group_affinity();

    /// ProfilerBase callback (runs on collector_thread). Iterates the full
    /// L0MarkerBuffer and appends each marker's cycle window to all_windows_.
    /// No matching here — matching is deferred to finalize().
    void on_buffer_collected(const L0PerfReadyBufferInfo &info);

    /**
     * Write accumulated records to two files under <output_prefix>:
     *   l0_perf_records.json — grouped by task_id
     *   l0_perf_trace.json   — Perfetto / Chrome traceEvents format
     */
    int export_swimlane_json();

    /// Reverse of initialize. Idempotent.
    void finalize(L0PerfUnregisterCallback unregister_cb, const L0PerfFreeCallback &free_cb);

    bool is_initialized() const { return initialized_; }

private:
    bool initialized_ = false;
    void *shm_dev_ = nullptr;
    std::string output_prefix_;

    // ACL / runtime prof session — open in initialize(), close in finalize().
    aclprofConfig *acl_prof_cfg_ = nullptr;
    std::string acl_prof_result_dir_;
    bool rt_prof_switch_started_ = false;  // gates the paired STOP call

    // Final matched records, indexed by sub_core_index (0..kBiuPerfNumChans-1).
    // Populated only in finalize()'s offline matching pass.
    std::vector<std::vector<L0PerfRecord>> collected_records_;
    uint64_t total_collected_ = 0;
    uint64_t markers_received_ = 0;
    uint64_t stamps_unmatched_ = 0;

    // Per-core L0PerfAicoreRing slab (one allocation holding num_aicore_
    // contiguous rings; per-core dev pointers also published in
    // L0PerfDataHeader::aicore_ring_ptrs so AICPU init reads them).
    void *aicore_rings_dev_ = nullptr;
    int num_aicore_ = 0;
    int num_threads_ = 0;  // AICPU scheduler thread count (free-queue seeding)

    // Marker buffer pool: one slab of L0MarkerBuffers seeded into the per-thread
    // free queues at init; released in finalize.
    void *marker_buffers_dev_ = nullptr;
    size_t marker_buffer_count_ = 0;

    // ---- Deferred-matching accumulators (filled during the run, matched at
    // finalize). Two independent collection lines, joined before matching:
    //   - all_windows_: written by the collector thread (on_buffer_collected),
    //     one entry per task marker, grouped by biu_perf group.
    //   - all_stamps_:  written by the stamp drain thread (drain_all_channels),
    //     one entry per decoded pipe stamp, grouped by biu_perf group.
    // Because matching is offline (after both threads join), the two lines need
    // no cross-locking.
    struct PendingStamp {
        uint64_t cycle;
        uint16_t region_id;
        uint8_t pipe;
        uint8_t sub_core_idx;  // 0..kBiuPerfNumChans-1
    };
    std::vector<L0TaskFinMarker> all_windows_[kBiuPerfNumGroups];
    std::vector<PendingStamp> all_stamps_[kBiuPerfNumGroups];

    // Dedicated stamp-drain thread: periodically reads all 18 biu_perf channels
    // into all_stamps_ so the driver ring never overflows. Independent of
    // marker arrival. Started in initialize(), joined in finalize().
    std::thread stamp_drain_thread_;
    std::atomic<bool> stamp_drain_running_{false};
    bool collection_finished_ = false;  // guards the one-shot finish+match
    static constexpr int kStampDrainPeriodMs = 2;

    // biu_perf HW DFX cycle vs AICore syscnt: same rate, independent epoch per
    // run. Matching converts a stamp into the syscnt domain via a per-sub-core
    // offset (estimate_subcore_offset) and compares against window ranges with
    // kCycleMatchTolerance to absorb the mark_stamp ramp at task edges. Kept
    // small (< the inter-task dispatch gap) so a stamp can only slop into an
    // immediately-adjacent window, never skip across tightly-packed ones.
    static constexpr int64_t kCycleMatchTolerance = 2048;

    // Mirror of prof_start_para from driver/ascend_hal_base.h (CANN 9.1.T500).
    // Field order + alignment must match the driver's own struct.
    struct ProfStartParaMirror {
        int channel_type;
        unsigned int sample_period;
        unsigned int real_time;
        void *user_data;             // for biu_perf: &BiuPerfStartUserData
        unsigned int user_data_size;
    };
    using ProfDrvStartFn = int (*)(unsigned int dev, unsigned int chan, ProfStartParaMirror *para);
    using ProfChannelReadFn = int (*)(unsigned int dev, unsigned int chan, char *out_buf, unsigned int buf_size);
    using ProfStopFn = int (*)(unsigned int dev, unsigned int chan);

    void *hal_handle_ = nullptr;
    ProfDrvStartFn prof_drv_start_ = nullptr;
    ProfChannelReadFn prof_channel_read_ = nullptr;
    ProfStopFn prof_stop_ = nullptr;
    bool chan_started_[kBiuPerfNumChans] = {false};
    L0DecodeState decode_state_[kBiuPerfNumChans];

    // ---- internal helpers ----
    int open_acl_prof_session(int device_id);
    void close_acl_prof_session();
    int resolve_driver_symbols();
    int start_all_biu_perf_channels();
    void stop_all_biu_perf_channels();
    void *alloc_single_buffer(size_t size, void **host_ptr_out);
    int seed_marker_free_queues(L0PerfDataHeader *hdr_host);  // alloc L0MarkerBuffers into per-thread free queues
    void stamp_drain_loop();                // body of stamp_drain_thread_
    int drain_all_channels();               // one pass over 18 channels → all_stamps_; returns bytes read
    void run_offline_match();               // finalize: per-sub-core offset + match all_stamps_ × all_windows_
    // Estimate the DFX→syscnt offset for one sub-core: try a few leading
    // windows as the home of the first stamp, pick the offset matching the most
    // stamps (tie-broken toward the tightest-fitting windows).
    int64_t estimate_subcore_offset(
        const std::vector<L0TaskFinMarker> &windows, const std::vector<PendingStamp> &stamps
    );
};

#endif  // SRC_A5_PLATFORM_INCLUDE_HOST_L0_PERF_COLLECTOR_H_
