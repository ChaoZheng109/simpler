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
 * @file l0_perf_collector.cpp
 * @brief Host-side L0 (biu_perf) consumer. See l0_perf_collector.h.
 */

#include "host/l0_perf_collector.h"

#include <dlfcn.h>

#include <acl/acl_prof.h>
#include <acl/acl_rt.h>
#include <runtime/rt.h>

#include <algorithm>
#include <chrono>
#include <thread>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <utility>

#include "common/memory_barrier.h"
#include "common/unified_log.h"
#include "host/profiling_copy.h"

namespace {
constexpr int kReadBufSize = 8192;  // per-channel scratch (multiple of 4)

// ---------------------------------------------------------------------------
// Mirror of MsprofCommandHandle from runtime/src/dfx/msprof/inc/toolchain/
// aprof_pub.h (not installed in public CANN headers). Used by
// rtProfSetProSwitch — the runtime entry point for instr-profiling.
// Layout must match the upstream struct exactly.
// ---------------------------------------------------------------------------
constexpr uint64_t kProfInstrBit = 0x00800000ULL;  // PROF_INSTR
constexpr uint32_t kProfCmdStart = 1;              // PROF_COMMANDHANDLE_TYPE_START
constexpr uint32_t kProfCmdStop  = 2;              // PROF_COMMANDHANDLE_TYPE_STOP
constexpr uint32_t kMsprofMaxDev = 64;
constexpr uint32_t kProfPathMax  = 1024;           // PATH_LEN_MAX + 1
constexpr uint32_t kProfParamMax = 4096;           // PARAM_LEN_MAX + 1

struct MsprofCommandHandleParamsMirror {
    uint32_t pathLen;
    uint32_t storageLimit;
    uint32_t profDataLen;
    char path[kProfPathMax];
    char profData[kProfParamMax];
};

struct MsprofCommandHandleMirror {
    uint64_t profSwitch;
    uint64_t profSwitchHi;
    uint32_t devNums;
    uint32_t devIdList[kMsprofMaxDev];
    uint32_t modelId;
    uint32_t type;
    uint32_t cacheFlag;
    MsprofCommandHandleParamsMirror params;
};

int call_rt_prof_set_pro_switch(int device_id, uint32_t cmd_type) {
    MsprofCommandHandleMirror cmd{};
    cmd.profSwitch = kProfInstrBit;
    cmd.devNums = 1;
    cmd.devIdList[0] = static_cast<uint32_t>(device_id);
    cmd.modelId = 0xFFFFFFFFUL;
    cmd.type = cmd_type;
    return static_cast<int>(rtProfSetProSwitch(&cmd, static_cast<uint32_t>(sizeof(cmd))));
}

// ---------------------------------------------------------------------------
// Display helpers for JSON export.
// ---------------------------------------------------------------------------
// Map ctrl_type (= our `pipe` field, range 0..6) to a short human-readable
// name. Mirrors msprof's biu_perf parser:
//   tools/biu_perf_chip6_parser.py
const char *pipe_name(uint8_t pipe) {
    switch (pipe) {
        case 0: return "SU";    // scalar
        case 1: return "VEC";   // PIPE_V
        case 2: return "CUBE";  // PIPE_M
        case 3: return "MTE1";
        case 4: return "MTE2";
        case 5: return "MTE3";
        case 6: return "FIXP";  // PIPE_FIX
        default: return "UNK";
    }
}

struct SubCoreInfo {
    uint32_t group;
    uint32_t sub_in_group;
    const char *type;  // "aic" / "aiv0" / "aiv1"
};
SubCoreInfo decode_sub_core(uint8_t sub_core_id) {
    uint32_t g = sub_core_id / kBiuPerfSubPerGroup;
    uint32_t s = sub_core_id % kBiuPerfSubPerGroup;
    const char *t = (s == 0) ? "aic" : (s == 1) ? "aiv0" : "aiv1";
    return {g, s, t};
}

// kernel mark_stamp region_id is encoded as op_id*10 + phase, where
//   phase 0 = instruction begin, 1 = instruction end, other = instant point.
// op_info maps op_id to the display name + a Chrome/Perfetto `cname` color so
// l0_perf_trace.json renders like msprof's trace.json. To track a new
// instruction, pick an op_id, emit begin/end (or instant) mark_stamps in the
// kernel, and add a row here.
struct L0OpInfo {
    const char *name;
    const char *cname;
};
L0OpInfo op_info(uint32_t op_id) {
    switch (op_id) {
        case 1: return {"TLOAD", "thread_state_iowait"};   // MTE2 load — amber
        case 2: return {"SET_FLAG", "black"};
        case 3: return {"TMUL", "good"};                   // VEC compute — green
        case 4: return {"SET_FLAG", "black"};
        case 5: return {"TSTORE", "rail_response"};        // MTE3 store — blue
        case 9: return {"END", "grey"};
        default: return {"OP", "thread_state_runnable"};
    }
}

// msprof lane (tid) name per biu_perf ctrl_type (our `pipe` field). The
// scalar-issue duplicate of every mark_stamp (pipe 0) is routed to the
// USERMARK lane, mirroring tools/biu_perf_chip6_parser.py.
const char *pipe_lane_name(uint8_t pipe) {
    switch (pipe) {
        case 1: return "VECTOR";
        case 2: return "CUBE";
        case 3: return "MTE1";
        case 4: return "MTE2";
        case 5: return "MTE3";
        case 6: return "FIXP";
        default: return "USERMARK";  // pipe 0 (SU/scalar) + anything unknown
    }
}
// Vertical lane order in the swimlane, matching msprof's thread_sort_index.
int pipe_sort_index(uint8_t pipe) {
    switch (pipe) {
        case 1: return 7;   // VECTOR
        case 2: return 4;   // CUBE
        case 3: return 3;   // MTE1
        case 4: return 6;   // MTE2
        case 5: return 8;   // MTE3
        case 6: return 5;   // FIXP
        default: return 12;  // USERMARK
    }
}
}  // namespace

L0PerfCollector::~L0PerfCollector() {
    // Join the stamp drain thread before stop()/destruction; a still-joinable
    // std::thread destructor would call std::terminate.
    stamp_drain_running_.store(false, std::memory_order_release);
    if (stamp_drain_thread_.joinable()) {
        stamp_drain_thread_.join();
    }
    stop();
}

// ---------------------------------------------------------------------------
// ACL / runtime prof session — armed in initialize(), torn down in finalize()
// ---------------------------------------------------------------------------

int L0PerfCollector::open_acl_prof_session(int device_id) {
    acl_prof_result_dir_ = output_prefix_ + "/acl_prof_l0";
    std::error_code ec;
    std::filesystem::create_directories(acl_prof_result_dir_, ec);
    if (ec) {
        LOG_WARN(
            "L0PerfCollector: failed to create acl_prof result dir %s: %s",
            acl_prof_result_dir_.c_str(), ec.message().c_str()
        );
    }

    aclError prof_init_rc = aclprofInit(acl_prof_result_dir_.c_str(), acl_prof_result_dir_.size());
    LOG_INFO_V1("L0PerfCollector: aclprofInit(%s) -> %d", acl_prof_result_dir_.c_str(), (int)prof_init_rc);
    if (prof_init_rc != ACL_SUCCESS) {
        return -1;
    }

    // ACL_PROF_AICORE_METRICS + ACL_AICORE_PIPE_UTILIZATION is the bit set
    // that maps to biu_perf on dav-c310. ACL_PROF_TASK_TIME is required
    // alongside for the session to accept the AICORE_METRICS bit.
    uint32_t dev_list[1] = {static_cast<uint32_t>(device_id)};
    acl_prof_cfg_ = aclprofCreateConfig(
        dev_list, /*deviceNums=*/1, ACL_AICORE_PIPE_UTILIZATION, /*aicoreEvents=*/nullptr,
        ACL_PROF_AICORE_METRICS | ACL_PROF_TASK_TIME
    );
    LOG_INFO_V1(
        "L0PerfCollector: aclprofCreateConfig(AICORE_METRICS|TASK_TIME, PIPE_UTILIZATION) -> %p",
        (void *)acl_prof_cfg_
    );
    if (acl_prof_cfg_ == nullptr) {
        aclprofFinalize();
        return -1;
    }

    aclError prof_start_rc = aclprofStart(acl_prof_cfg_);
    LOG_INFO_V1("L0PerfCollector: aclprofStart -> %d", (int)prof_start_rc);
    if (prof_start_rc != ACL_SUCCESS) {
        aclprofDestroyConfig(acl_prof_cfg_);
        acl_prof_cfg_ = nullptr;
        aclprofFinalize();
        return -1;
    }

    // PROF_INSTR is the runtime-level switch that routes DFX trace data to
    // biu_perf channels. aclprofStart alone does not do this on dav-c310.
    int rt_rc = call_rt_prof_set_pro_switch(device_id, kProfCmdStart);
    LOG_INFO_V1("L0PerfCollector: rtProfSetProSwitch(PROF_INSTR, START) -> %d", rt_rc);
    if (rt_rc != 0) {
        aclprofStop(acl_prof_cfg_);
        aclprofDestroyConfig(acl_prof_cfg_);
        acl_prof_cfg_ = nullptr;
        aclprofFinalize();
        return -1;
    }
    rt_prof_switch_started_ = true;
    return 0;
}

void L0PerfCollector::close_acl_prof_session() {
    // Teardown is the reverse of open. Each step is best-effort — keep going
    // even if one returns an error so we don't leak the rest.
    if (rt_prof_switch_started_) {
        int rt_rc = call_rt_prof_set_pro_switch(device_id_, kProfCmdStop);
        LOG_INFO_V1("L0PerfCollector: rtProfSetProSwitch(PROF_INSTR, STOP) -> %d", rt_rc);
        rt_prof_switch_started_ = false;
    }
    if (acl_prof_cfg_ != nullptr) {
        aclError stop_rc = aclprofStop(acl_prof_cfg_);
        LOG_INFO_V1("L0PerfCollector: aclprofStop -> %d", (int)stop_rc);
        aclError destroy_rc = aclprofDestroyConfig(acl_prof_cfg_);
        LOG_INFO_V1("L0PerfCollector: aclprofDestroyConfig -> %d", (int)destroy_rc);
        acl_prof_cfg_ = nullptr;
        aclError fin_rc = aclprofFinalize();
        LOG_INFO_V1("L0PerfCollector: aclprofFinalize -> %d", (int)fin_rc);
    }
}

// ---------------------------------------------------------------------------
// dlopen + dlsym
// ---------------------------------------------------------------------------

int L0PerfCollector::resolve_driver_symbols() {
    if (hal_handle_ != nullptr) {
        return 0;
    }
    hal_handle_ = dlopen("libascend_hal.so", RTLD_NOW | RTLD_LOCAL);
    if (hal_handle_ == nullptr) {
        LOG_ERROR("L0PerfCollector: dlopen(libascend_hal.so) failed: %s", dlerror());
        return -1;
    }
    prof_drv_start_ = reinterpret_cast<ProfDrvStartFn>(dlsym(hal_handle_, "prof_drv_start"));
    prof_channel_read_ = reinterpret_cast<ProfChannelReadFn>(dlsym(hal_handle_, "prof_channel_read"));
    prof_stop_ = reinterpret_cast<ProfStopFn>(dlsym(hal_handle_, "prof_stop"));
    if (prof_drv_start_ == nullptr || prof_channel_read_ == nullptr || prof_stop_ == nullptr) {
        LOG_ERROR(
            "L0PerfCollector: dlsym failed for prof_drv_start/prof_channel_read/prof_stop: %s",
            dlerror() != nullptr ? dlerror() : "(null)"
        );
        dlclose(hal_handle_);
        hal_handle_ = nullptr;
        prof_drv_start_ = nullptr;
        prof_channel_read_ = nullptr;
        prof_stop_ = nullptr;
        return -1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Channel lifecycle (host driver consumer API)
// ---------------------------------------------------------------------------

int L0PerfCollector::start_all_biu_perf_channels() {
    auto t0 = std::chrono::steady_clock::now();

    // CCU/STARS/AICPU prereq channels (147/148/149/151, 44/50/52/53, 143)
    // are claimed by the ACL prof session opened earlier; trying to
    // prof_drv_start them here would return EBUSY. Only the 18 biu_perf
    // channels (chan 11..28) are our data-plane consumer.
    for (uint32_t g = 0; g < kBiuPerfNumGroups; ++g) {
        for (uint32_t sub = 0; sub < kBiuPerfSubPerGroup; ++sub) {
            BiuPerfStartUserData ud{
                static_cast<uint32_t>(sizeof(BiuPerfStartUserData)), /*biu_mode=*/0u, sub,
                kBiuPerfPhysAicore[g]
            };
            ProfStartParaMirror sp{};
            sp.channel_type = 0;     // PROF_TS_TYPE
            sp.sample_period = 0;
            sp.real_time = 1;
            sp.user_data = &ud;
            sp.user_data_size = static_cast<unsigned int>(sizeof(ud));

            unsigned int chan = biu_perf_chan_id(g, sub);
            int r = prof_drv_start_(static_cast<unsigned int>(device_id_), chan, &sp);
            if (r != 0) {
                LOG_ERROR(
                    "L0PerfCollector: prof_drv_start(dev=%d, chan=%u) -> %d; rolling back %u previously started "
                    "channels and aborting L0 init",
                    device_id_, chan, r, g * kBiuPerfSubPerGroup + sub
                );
                stop_all_biu_perf_channels();
                return r;
            }
            chan_started_[g * kBiuPerfSubPerGroup + sub] = true;
        }
    }
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    LOG_INFO_V1("L0PerfCollector: prof_drv_start x%u took %ld ms", kBiuPerfNumChans, (long)elapsed_ms);
    return 0;
}

void L0PerfCollector::stop_all_biu_perf_channels() {
    if (prof_stop_ == nullptr) {
        return;
    }
    uint32_t to_stop = 0;
    for (uint32_t c = 0; c < kBiuPerfNumChans; ++c) {
        if (chan_started_[c]) ++to_stop;
    }
    if (to_stop == 0) {
        return;
    }
    // Each biu_perf prof_stop takes ~3-4s in the driver (~60s total for
    // 18 channels). Log per-channel progress so the wait looks intentional
    // instead of a hang.
    LOG_INFO_V1(
        "L0PerfCollector: stopping %u biu_perf channels (driver tear-down ~3-4s each, total ~%us expected)",
        to_stop, to_stop * 4u
    );
    auto t0 = std::chrono::steady_clock::now();
    uint32_t stopped = 0;
    for (uint32_t c = 0; c < kBiuPerfNumChans; ++c) {
        if (!chan_started_[c]) {
            continue;
        }
        uint32_t g = c / kBiuPerfSubPerGroup;
        uint32_t sub = c % kBiuPerfSubPerGroup;
        unsigned int chan = biu_perf_chan_id(g, sub);
        auto ts0 = std::chrono::steady_clock::now();
        prof_stop_(static_cast<unsigned int>(device_id_), chan);
        auto ts1 = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(ts1 - ts0).count();
        ++stopped;
        LOG_INFO_V1(
            "L0PerfCollector: prof_stop %u/%u chan=%u took %ld ms", stopped, to_stop, chan, (long)ms
        );
        chan_started_[c] = false;
    }
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    LOG_INFO_V1("L0PerfCollector: prof_stop x%u total=%ld ms", stopped, (long)total_ms);
}

// ---------------------------------------------------------------------------
// GM allocation
// ---------------------------------------------------------------------------

void *L0PerfCollector::alloc_single_buffer(size_t size, void **host_ptr_out) {
    void *dev_ptr = alloc_cb_(size);
    if (dev_ptr == nullptr) {
        if (host_ptr_out) *host_ptr_out = nullptr;
        return nullptr;
    }

    void *host_ptr = nullptr;
    if (register_cb_ != nullptr) {
        int rc = register_cb_(dev_ptr, size, device_id_, &host_ptr);
        if (rc != 0 || host_ptr == nullptr) {
            LOG_ERROR("L0PerfCollector: register failed: %d", rc);
            free_cb_(dev_ptr);
            if (host_ptr_out) *host_ptr_out = nullptr;
            return nullptr;
        }
    } else {
        host_ptr = std::malloc(size);
        if (host_ptr == nullptr) {
            LOG_ERROR("L0PerfCollector: host shadow alloc failed for %zu bytes", size);
            free_cb_(dev_ptr);
            if (host_ptr_out) *host_ptr_out = nullptr;
            return nullptr;
        }
        std::memset(host_ptr, 0, size);
        profiling_copy_to_device(dev_ptr, host_ptr, size);
    }

    if (host_ptr_out) *host_ptr_out = host_ptr;
    manager_.register_mapping(dev_ptr, host_ptr);
    return dev_ptr;
}

// ---------------------------------------------------------------------------
// initialize — see header for the step order and the must-run-before-kernel
// invariant on prof_drv_start.
// ---------------------------------------------------------------------------

int L0PerfCollector::initialize(
    int num_aicore, int num_threads, int device_id, const L0PerfAllocCallback &alloc_cb,
    L0PerfRegisterCallback register_cb, const L0PerfFreeCallback &free_cb, const std::string &output_prefix
) {
    if (alloc_cb == nullptr || free_cb == nullptr || num_aicore <= 0 || num_aicore > PLATFORM_MAX_CORES) {
        LOG_ERROR("L0PerfCollector: initialize: invalid arguments (num_aicore=%d)", num_aicore);
        return -1;
    }
    if (num_threads <= 0 || num_threads > PLATFORM_MAX_AICPU_THREADS) {
        LOG_ERROR("L0PerfCollector: initialize: invalid num_threads=%d", num_threads);
        return -1;
    }
    if (initialized_) {
        LOG_ERROR("L0PerfCollector: already initialized");
        return -1;
    }

    num_aicore_ = num_aicore;
    num_threads_ = num_threads;
    output_prefix_ = output_prefix;
    total_collected_ = 0;
    markers_received_ = 0;
    stamps_unmatched_ = 0;
    collected_records_.assign(kBiuPerfNumChans, {});
    for (uint32_t i = 0; i < kBiuPerfNumChans; ++i) {
        decode_state_[i] = L0DecodeState{};
    }
    for (uint32_t g = 0; g < kBiuPerfNumGroups; ++g) {
        all_windows_[g].clear();
        all_stamps_[g].clear();
    }

    // device_id needs to be set before any helper that reads device_id_.
    // shm_dev_ stays null until the GM allocation in step 3.
    set_memory_context(
        alloc_cb, register_cb, free_cb, /*shm_dev=*/nullptr, /*shm_host=*/nullptr, /*shm_size=*/0, device_id
    );

    if (open_acl_prof_session(device_id) != 0) {
        return -1;
    }

    if (resolve_driver_symbols() != 0) {
        close_acl_prof_session();
        return -1;
    }

    size_t shm_size = calc_l0_perf_data_size();
    void *shm_host_local = nullptr;
    void *shm_dev_local = alloc_single_buffer(shm_size, &shm_host_local);
    if (shm_dev_local == nullptr) {
        LOG_ERROR("L0PerfCollector: failed to allocate L0 shared memory (%zu bytes)", shm_size);
        if (hal_handle_ != nullptr) {
            dlclose(hal_handle_);
            hal_handle_ = nullptr;
        }
        close_acl_prof_session();
        return -1;
    }
    std::memset(shm_host_local, 0, shm_size);
    L0PerfDataHeader *hdr = get_l0_perf_header(shm_host_local);
    hdr->num_chans = kBiuPerfNumChans;
    hdr->num_threads = static_cast<uint32_t>(num_threads_);

    // Allocate per-core L0PerfAicoreRing slab in one contiguous buffer and
    // publish each ring's device pointer into hdr->aicore_ring_ptrs[i] so
    // AICPU init can lift them; AICore reads the same array through
    // KernelArgs::aicore_l0_perf_ring_addrs (= shm_dev + offsetof).
    size_t ring_slab_size = static_cast<size_t>(num_aicore_) * sizeof(L0PerfAicoreRing);
    void *rings_host_local = nullptr;
    void *rings_dev_local = alloc_single_buffer(ring_slab_size, &rings_host_local);
    if (rings_dev_local == nullptr) {
        LOG_ERROR("L0PerfCollector: failed to allocate %zu bytes for %d AICore rings", ring_slab_size, num_aicore_);
        if (hal_handle_ != nullptr) {
            dlclose(hal_handle_);
            hal_handle_ = nullptr;
        }
        close_acl_prof_session();
        return -1;
    }
    std::memset(rings_host_local, 0, ring_slab_size);
    for (int i = 0; i < num_aicore_; ++i) {
        hdr->aicore_ring_ptrs[i] = reinterpret_cast<uint64_t>(rings_dev_local) + i * sizeof(L0PerfAicoreRing);
    }
    profiling_copy_to_device(rings_dev_local, rings_host_local, ring_slab_size);
    aicore_rings_dev_ = rings_dev_local;

    // Seed per-thread marker buffer pools into the host shadow's free queues
    // (1 buffer in free_queue, the rest into the framework recycled pool) BEFORE
    // pushing the shm to device, so AICPU's first pop in l0_perf_aicpu_init
    // finds a buffer.
    if (seed_marker_free_queues(hdr) != 0) {
        if (hal_handle_ != nullptr) {
            dlclose(hal_handle_);
            hal_handle_ = nullptr;
        }
        close_acl_prof_session();
        return -1;
    }

    profiling_copy_to_device(shm_dev_local, shm_host_local, shm_size);

    shm_dev_ = shm_dev_local;
    set_memory_context(alloc_cb, register_cb, free_cb, shm_dev_local, shm_host_local, shm_size, device_id);

    if (start_all_biu_perf_channels() != 0) {
        // Partial biu_perf starts were already rolled back inside
        // start_all_biu_perf_channels. shm_dev_ is left registered with
        // manager_; the caller's finalize() path releases it via
        // manager_.release_owned_buffers().
        if (hal_handle_ != nullptr) {
            dlclose(hal_handle_);
            hal_handle_ = nullptr;
        }
        close_acl_prof_session();
        return -1;
    }

    // Channels are recording — start the dedicated stamp-drain thread so the
    // driver biu_perf ring is read continuously (decoupled from markers).
    stamp_drain_running_.store(true, std::memory_order_release);
    stamp_drain_thread_ = std::thread(&L0PerfCollector::stamp_drain_loop, this);

    initialized_ = true;
    LOG_INFO_V1(
        "L0PerfCollector: initialized: dev=%d, SHM=0x%lx, 18 biu_perf channels started, output=%s",
        device_id, reinterpret_cast<unsigned long>(shm_dev_), output_prefix_.c_str()
    );
    return 0;
}

// Allocate PLATFORM_L0_BUFFERS_PER_THREAD L0MarkerBuffers for each of the
// num_threads_ AICPU threads. One goes into that thread's free_queue (in the
// host shadow); the rest seed the framework recycled pool so proactive_replenish
// can top the free_queue back up as the device drains. Must run before the
// shm host→device copy so AICPU sees the seeded queues.
int L0PerfCollector::seed_marker_free_queues(L0PerfDataHeader *hdr) {
    for (int t = 0; t < num_threads_; ++t) {
        L0PerfBufferState *state = &hdr->thread_states[t];
        std::memset(state, 0, sizeof(L0PerfBufferState));
        for (int s = 0; s < PLATFORM_L0_BUFFERS_PER_THREAD; ++s) {
            void *host_buf = nullptr;
            void *dev_buf = alloc_single_buffer(sizeof(L0MarkerBuffer), &host_buf);
            if (dev_buf == nullptr) {
                LOG_ERROR("L0PerfCollector: failed to alloc L0MarkerBuffer (thread %d, slot %d)", t, s);
                return -1;
            }
            ++marker_buffer_count_;
            if (s == 0) {
                state->free_queue.buffer_ptrs[0] = reinterpret_cast<uint64_t>(dev_buf);
            } else {
                manager_.push_recycled(/*kind=*/0, dev_buf);
            }
        }
        state->free_queue.tail = 1;
    }
    LOG_INFO_V1(
        "L0PerfCollector: seeded %d threads x %d buffers (%zu total marker buffers)", num_threads_,
        PLATFORM_L0_BUFFERS_PER_THREAD, marker_buffer_count_
    );
    return 0;
}

void *L0PerfCollector::get_aicore_ring_addrs_device_ptr() const {
    if (shm_dev_ == nullptr) {
        return nullptr;
    }
    return reinterpret_cast<void *>(
        reinterpret_cast<uintptr_t>(shm_dev_) + offsetof(L0PerfDataHeader, aicore_ring_ptrs)
    );
}

namespace {
// Mirror of capability_group_info from driver/ascend_hal_base.h (16 uint32 =
// 64 B). Field order + AICORE_MASK_NUM(=2) must match upstream exactly; if
// the driver header diverges, halGetCapabilityGroupInfo writes the wrong
// fields and the mask we read is garbage.
struct CapGroupInfoMirror {
    uint32_t group_id;
    uint32_t state;
    uint32_t extend_attribute;
    uint32_t aicore_number;
    uint32_t aivector_number;
    uint32_t sdma_number;
    uint32_t aicpu_number;
    uint32_t active_sq_number;
    uint32_t aicore_mask[2];
    uint32_t vfid;
    uint32_t poolid;
    uint32_t poolid_max;
    uint32_t res[3];
};
static_assert(sizeof(CapGroupInfoMirror) == 64, "CapGroupInfoMirror layout drift vs driver");

using HalGetCapGroupFn = int (*)(int, int, int, CapGroupInfoMirror *, int);

// kBiuPerfPhysAicore[] is a 6-element array of cluster phys IDs (0..35).
// Pack it into a 64-bit reference mask so we can XOR/AND against the
// driver-reported aicore_mask (low 32 bits in [0], high 32 in [1]).
uint64_t monitored_mask_ref() {
    uint64_t m = 0;
    for (uint32_t g = 0; g < kBiuPerfNumGroups; ++g) {
        m |= (uint64_t{1} << kBiuPerfPhysAicore[g]);
    }
    return m;
}
}  // namespace

void L0PerfCollector::diagnose_group_affinity() {
    uint32_t group_count = 0;
    aclError rc = aclrtGetGroupCount(&group_count);
    LOG_INFO_V1("L0PerfCollector: diag aclrtGetGroupCount -> rc=%d count=%u", (int)rc, group_count);
    if (rc != ACL_SUCCESS || group_count == 0) {
        return;
    }

    aclrtGroupInfo *info = aclrtCreateGroupInfo();
    if (info == nullptr) {
        LOG_WARN("L0PerfCollector: diag aclrtCreateGroupInfo returned null");
        return;
    }
    aclError gi_rc = aclrtGetAllGroupInfo(info);
    if (gi_rc != ACL_SUCCESS) {
        LOG_WARN("L0PerfCollector: diag aclrtGetAllGroupInfo -> %d", (int)gi_rc);
        aclrtDestroyGroupInfo(info);
        return;
    }

    // Resolve halGetCapabilityGroupInfo via the libascend_hal handle we
    // already dlopened for prof_drv_start (resolve_driver_symbols).
    HalGetCapGroupFn hal_get_cap = nullptr;
    if (hal_handle_ != nullptr) {
        hal_get_cap = reinterpret_cast<HalGetCapGroupFn>(dlsym(hal_handle_, "halGetCapabilityGroupInfo"));
        if (hal_get_cap == nullptr) {
            LOG_WARN(
                "L0PerfCollector: diag dlsym(halGetCapabilityGroupInfo) failed (%s); mask will be unavailable",
                dlerror() != nullptr ? dlerror() : "(null)"
            );
        }
    }

    const uint64_t want = monitored_mask_ref();
    LOG_INFO_V1(
        "L0PerfCollector: diag monitored kBiuPerfPhysAicore mask=0x%lx (%u clusters)",
        static_cast<unsigned long>(want), kBiuPerfNumGroups
    );

    for (uint32_t idx = 0; idx < group_count; ++idx) {
        int32_t group_id = -1, aicore_num = 0, aiv_num = 0, aic_num = 0;
        size_t ret_size = 0;
        aclrtGetGroupInfoDetail(info, idx, ACL_GROUP_GROUPID_INT, &group_id, sizeof(group_id), &ret_size);
        aclrtGetGroupInfoDetail(info, idx, ACL_GROUP_AICORE_INT, &aicore_num, sizeof(aicore_num), &ret_size);
        aclrtGetGroupInfoDetail(info, idx, ACL_GROUP_AIV_INT, &aiv_num, sizeof(aiv_num), &ret_size);
        aclrtGetGroupInfoDetail(info, idx, ACL_GROUP_AIC_INT, &aic_num, sizeof(aic_num), &ret_size);

        uint64_t mask = 0;
        bool mask_ok = false;
        if (hal_get_cap != nullptr) {
            CapGroupInfoMirror cgi{};
            // ts_id 0 = TS_AICORE. group_count argument is "how many entries
            // the caller's array can hold" — passing 1 reads only this group.
            int hal_rc = hal_get_cap(device_id_, 0, group_id, &cgi, 1);
            if (hal_rc == 0) {
                mask = static_cast<uint64_t>(cgi.aicore_mask[0]) |
                       (static_cast<uint64_t>(cgi.aicore_mask[1]) << 32);
                mask_ok = true;
            } else {
                LOG_WARN("L0PerfCollector: diag halGetCapabilityGroupInfo group=%d rc=%d", group_id, hal_rc);
            }
        }

        // Decode mask bits to a human-readable list of cluster IDs.
        char cluster_list[256];
        int cl_off = 0;
        cluster_list[0] = '\0';
        for (int b = 0; b < 64 && cl_off < (int)sizeof(cluster_list) - 6; ++b) {
            if (mask & (uint64_t{1} << b)) {
                cl_off += snprintf(cluster_list + cl_off, sizeof(cluster_list) - cl_off, "%s%d",
                                   cl_off == 0 ? "" : ",", b);
            }
        }

        const char *match_note;
        if (!mask_ok) {
            match_note = "(mask unavailable)";
        } else if (mask == want) {
            match_note = "MATCH monitored set";
        } else if ((mask & want) == want) {
            match_note = "SUPERSET of monitored set";
        } else if ((mask & want) == 0) {
            match_note = "DISJOINT from monitored set — NO data expected";
        } else {
            match_note = "PARTIAL overlap with monitored set";
        }

        LOG_INFO_V1(
            "L0PerfCollector: diag group[%u] id=%d aicore=%d aiv=%d aic=%d mask=0x%lx clusters={%s} %s",
            idx, group_id, aicore_num, aiv_num, aic_num, static_cast<unsigned long>(mask),
            cluster_list, match_note
        );
    }
    aclrtDestroyGroupInfo(info);
}

// ---------------------------------------------------------------------------
// on_buffer_collected runs on ProfilerBase's collector_thread. The driver
// allows prof_drv_start / prof_channel_read / prof_stop on different threads,
// so reading here (separate from the initialize thread that did the start)
// is fine.
// ---------------------------------------------------------------------------

void L0PerfCollector::on_buffer_collected(const L0PerfReadyBufferInfo &info) {
    if (!initialized_) {
        return;
    }
    // info.host_buffer_ptr points at the host shadow of a full L0MarkerBuffer
    // (the framework already copied it from device). Append each marker's
    // cycle window to all_windows_[group]. No matching here — deferred to
    // finalize's offline pass.
    const L0MarkerBuffer *buf = reinterpret_cast<const L0MarkerBuffer *>(info.host_buffer_ptr);
    if (buf == nullptr) {
        return;
    }
    uint32_t count = buf->count;
    if (count > PLATFORM_L0_MARKERS_PER_BUFFER) {
        count = PLATFORM_L0_MARKERS_PER_BUFFER;  // defensive against torn count
    }
    for (uint32_t i = 0; i < count; ++i) {
        const L0TaskFinMarker &m = buf->markers[i];
        ++markers_received_;
        if (m.group >= kBiuPerfNumGroups) {
            continue;  // outside biu_perf coverage (shouldn't happen)
        }
        all_windows_[m.group].push_back(m);
    }
}

// ---------------------------------------------------------------------------
// Dedicated stamp-drain thread: read all 18 biu_perf channels on a fixed
// period into all_stamps_, decoupled from marker arrival. Runs until
// stamp_drain_running_ is cleared in finalize(); a final pass after the loop
// catches the last driver flush.
// ---------------------------------------------------------------------------

int L0PerfCollector::drain_all_channels() {
    char read_buf[kReadBufSize];
    int total_bytes = 0;
    for (uint32_t g = 0; g < kBiuPerfNumGroups; ++g) {
        for (uint32_t sub = 0; sub < kBiuPerfSubPerGroup; ++sub) {
            const int sub_core_idx = static_cast<int>(g) * static_cast<int>(kBiuPerfSubPerGroup) +
                                     static_cast<int>(sub);
            unsigned int chan = biu_perf_chan_id(g, sub);
            L0DecodeState &ds = decode_state_[sub_core_idx];
            auto &stamps = all_stamps_[g];
            auto emit = [&](uint16_t region_id, uint8_t pipe, uint64_t cycle, uint16_t /*block_id*/) {
                stamps.push_back(PendingStamp{cycle, region_id, pipe, static_cast<uint8_t>(sub_core_idx)});
            };

            int retries_left = kBiuPerfDrainRetries;
            for (;;) {
                int bytes = prof_channel_read_(
                    static_cast<unsigned int>(device_id_), chan, read_buf, static_cast<unsigned int>(kReadBufSize)
                );
                if (bytes > 0) {
                    total_bytes += bytes;
                    l0_decode_chunks(reinterpret_cast<const uint8_t *>(read_buf), bytes, ds, emit);
                    retries_left = kBiuPerfDrainRetries;
                    continue;
                }
                if (bytes < 0) {
                    LOG_WARN("L0PerfCollector: prof_channel_read(dev=%d, chan=%u) -> %d", device_id_, chan, bytes);
                    break;
                }
                if (retries_left-- <= 0) break;
            }
        }
    }
    return total_bytes;
}

void L0PerfCollector::stamp_drain_loop() {
    while (stamp_drain_running_.load(std::memory_order_acquire)) {
        drain_all_channels();
        std::this_thread::sleep_for(std::chrono::milliseconds(kStampDrainPeriodMs));
    }
}

// ---------------------------------------------------------------------------
// Offline matching helpers.
// ---------------------------------------------------------------------------
namespace {
// Find the index of the window (sorted ascending by start_cycle, non-overlapping)
// that owns `adj`, or -1. Prefers STRICT containment [start, end]; only if no
// window strictly contains adj does it allow a `tol` slop at the immediate
// neighbour edges (adj just past one window's end, or just before the next
// window's start). This avoids the over-reach bug where a large tol let a
// stamp skip across several tightly-packed windows.
int find_window_idx(const std::vector<L0TaskFinMarker> &windows, int64_t adj, int64_t tol) {
    int lo = 0, hi = static_cast<int>(windows.size()) - 1, cand = -1;
    while (lo <= hi) {  // last window with start <= adj
        int mid = (lo + hi) / 2;
        if (static_cast<int64_t>(windows[mid].start_cycle) <= adj) {
            cand = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    // Strict containment in the candidate (largest start <= adj).
    if (cand >= 0 && adj <= static_cast<int64_t>(windows[cand].end_cycle)) {
        return cand;
    }
    if (tol <= 0) {
        return -1;
    }
    // Slop only at the two adjacent edges.
    if (cand >= 0 && adj <= static_cast<int64_t>(windows[cand].end_cycle) + tol) {
        return cand;
    }
    int nxt = cand + 1;
    if (nxt < static_cast<int>(windows.size()) &&
        adj >= static_cast<int64_t>(windows[nxt].start_cycle) - tol) {
        return nxt;
    }
    return -1;
}
}  // namespace

int64_t L0PerfCollector::estimate_subcore_offset(
    const std::vector<L0TaskFinMarker> &windows, const std::vector<PendingStamp> &stamps
) {
    if (windows.empty() || stamps.empty()) {
        return 0;
    }
    // The first stamp belongs to the first *instrumented* task on this sub-core
    // — under full instrumentation that's windows[0], but a few leading tasks
    // may be uninstrumented, so try the first K windows as its "home". For each
    // candidate offset, score = stamps that land in some window; tie-break
    // toward the offset whose matched windows are tightest (smallest summed
    // duration), which disambiguates a lone burst that could nest in either a
    // loose earlier window or its own tight one.
    const uint64_t first = stamps.front().cycle;  // stamps sorted ascending by caller
    const int K = std::min<int>(static_cast<int>(windows.size()), 8);
    int64_t best_off = static_cast<int64_t>(first) - static_cast<int64_t>(windows[0].start_cycle);
    uint64_t best_count = 0;
    uint64_t best_dur = UINT64_MAX;
    for (int k = 0; k < K; ++k) {
        int64_t cand = static_cast<int64_t>(first) - static_cast<int64_t>(windows[k].start_cycle);
        uint64_t count = 0;
        uint64_t dur = 0;
        // Score with STRICT containment (tol=0): a wrong offset that only
        // "fits" by spilling stamps across a window boundary must not earn the
        // same count as the true offset. Tie-break toward the offset whose
        // matched windows are tightest (the burst's real window is filled, not
        // loosely nested in an earlier/bigger one).
        for (const auto &s : stamps) {
            int idx = find_window_idx(windows, static_cast<int64_t>(s.cycle) - cand, /*tol=*/0);
            if (idx >= 0) {
                ++count;
                dur += windows[idx].end_cycle - windows[idx].start_cycle;
            }
        }
        if (count > best_count || (count == best_count && dur < best_dur)) {
            best_count = count;
            best_dur = dur;
            best_off = cand;
        }
    }
    return best_off;
}

// ---------------------------------------------------------------------------
// Offline matching (finalize): per-sub-core. Re-bucket windows by their exact
// sub-core (group*3 + block — verified to align with the channel's sub_core_id)
// and stamps by sub_core_id, so a stamp is only ever matched against tasks that
// ran on the SAME physical sub-core. Within a sub-core tasks are strictly
// sequential (non-overlapping windows), so after a robust per-sub-core offset
// each stamp maps to exactly one window. This removes the two failure modes of
// the old per-group match: cross-sub-core mis-attribution (parallel
// aic/aiv0/aiv1 windows overlap in cycle) and min-min offset skew.
// ---------------------------------------------------------------------------

void L0PerfCollector::run_offline_match() {
    std::vector<L0TaskFinMarker> win_sc[kBiuPerfNumChans];
    std::vector<PendingStamp> stamp_sc[kBiuPerfNumChans];
    for (uint32_t g = 0; g < kBiuPerfNumGroups; ++g) {
        for (const auto &w : all_windows_[g]) {
            uint32_t sc = g * kBiuPerfSubPerGroup + w.block;
            if (sc < kBiuPerfNumChans) {
                win_sc[sc].push_back(w);
            }
        }
        for (const auto &s : all_stamps_[g]) {
            if (s.sub_core_idx < kBiuPerfNumChans) {
                stamp_sc[s.sub_core_idx].push_back(s);
            }
        }
    }

    for (uint32_t sc = 0; sc < kBiuPerfNumChans; ++sc) {
        auto &windows = win_sc[sc];
        auto &stamps = stamp_sc[sc];
        if (stamps.empty()) {
            continue;
        }
        if (windows.empty()) {
            stamps_unmatched_ += stamps.size();
            continue;
        }

        std::sort(windows.begin(), windows.end(), [](const L0TaskFinMarker &a, const L0TaskFinMarker &b) {
            return a.start_cycle < b.start_cycle;
        });
        std::sort(stamps.begin(), stamps.end(), [](const PendingStamp &a, const PendingStamp &b) {
            return a.cycle < b.cycle;
        });

        const int64_t offset = estimate_subcore_offset(windows, stamps);

        // Clock-probe detail dump for tiny runs (e.g. vector_mul_stamp): print
        // every window + stamp so the DFX cycle can be compared by hand against
        // the get_sys_cnt_aicore() window. Gated so pa_unroll stays quiet.
        if (windows.size() <= 32 && stamps.size() <= 64) {
            for (const auto &w : windows) {
                LOG_INFO_V1(
                    "L0PerfCollector: sub_core=%u DETAIL window task=%lu start=%lu end=%lu block=%u phys=%u",
                    sc, static_cast<unsigned long>(w.task_id), static_cast<unsigned long>(w.start_cycle),
                    static_cast<unsigned long>(w.end_cycle), w.block, w.sub_core_phys
                );
            }
            for (const auto &s : stamps) {
                LOG_INFO_V1(
                    "L0PerfCollector: sub_core=%u DETAIL stamp cycle=%lu region=%u pipe=%u (adj=%ld)", sc,
                    static_cast<unsigned long>(s.cycle), s.region_id, s.pipe,
                    static_cast<long>(static_cast<int64_t>(s.cycle) - offset)
                );
            }
        }

        uint64_t matched = 0;
        for (const PendingStamp &s : stamps) {
            int idx = find_window_idx(windows, static_cast<int64_t>(s.cycle) - offset, kCycleMatchTolerance);
            if (idx < 0) {
                ++stamps_unmatched_;
                continue;
            }
            const L0TaskFinMarker &w = windows[idx];
            collected_records_[sc].push_back(L0PerfRecord{
                s.cycle, w.task_id, s.region_id, s.pipe, static_cast<uint8_t>(sc), w.core_type, w.block
            });
            ++total_collected_;
            ++matched;
        }
        LOG_INFO_V1(
            "L0PerfCollector: sub_core=%u match: %zu windows, %zu stamps, offset=%ld, matched=%lu",
            sc, windows.size(), stamps.size(), static_cast<long>(offset), static_cast<unsigned long>(matched)
        );
    }
}

// ---------------------------------------------------------------------------
// export_swimlane_json — two outputs:
//   l0_perf_records.json — v1 grouped by task_id, pretty-printed
//   l0_perf_trace.json   — Perfetto/Chrome traceEvents
// ---------------------------------------------------------------------------

int L0PerfCollector::export_swimlane_json() {
    // Finish collection exactly once: the collector thread (stop() in run())
    // has populated all_windows_; now stop the stamp drain thread, do a final
    // channel drain to catch the last driver flush, then run the offline match
    // that populates collected_records_. Channels are still up at this point
    // (finalize() tears them down later).
    if (!collection_finished_) {
        collection_finished_ = true;
        stamp_drain_running_.store(false, std::memory_order_release);
        if (stamp_drain_thread_.joinable()) {
            stamp_drain_thread_.join();
        }
        if (prof_channel_read_ != nullptr) {
            drain_all_channels();  // final catch of any straggler stamps
        }
        run_offline_match();
    }

    LOG_INFO_V1(
        "L0PerfCollector: export_swimlane_json: markers_received=%lu, total_records=%lu, unmatched=%lu",
        static_cast<unsigned long>(markers_received_), static_cast<unsigned long>(total_collected_),
        static_cast<unsigned long>(stamps_unmatched_)
    );
    if (output_prefix_.empty()) {
        return 0;
    }
    if (total_collected_ == 0) {
        return 0;  // nothing to write — avoid leaving an empty file
    }

    std::error_code ec;
    std::filesystem::create_directories(output_prefix_, ec);
    if (ec) {
        LOG_WARN("L0PerfCollector: failed to create output dir %s: %s", output_prefix_.c_str(), ec.message().c_str());
    }

    // Re-index by task_id since that's the user-meaningful unit for the
    // JSON consumer (swimlane_converter, MindStudio).
    struct TaggedRecord {
        uint8_t sub_core_id;
        L0PerfRecord r;
    };
    std::map<uint64_t, std::vector<TaggedRecord>> by_task;
    for (uint32_t sc = 0; sc < kBiuPerfNumChans; ++sc) {
        for (const L0PerfRecord &r : collected_records_[sc]) {
            by_task[r.task_id].push_back({static_cast<uint8_t>(sc), r});
        }
    }
    for (auto &kv : by_task) {
        std::sort(kv.second.begin(), kv.second.end(), [](const TaggedRecord &a, const TaggedRecord &b) {
            if (a.r.cycle != b.r.cycle) return a.r.cycle < b.r.cycle;
            if (a.sub_core_id != b.sub_core_id) return a.sub_core_id < b.sub_core_id;
            return a.r.pipe < b.r.pipe;
        });
    }

    // ---- (1) human-readable records.json (grouped by task_id) ----
    {
        std::string path = output_prefix_ + "/l0_perf_records.json";
        std::ofstream out(path, std::ios::out | std::ios::trunc);
        if (!out.is_open()) {
            LOG_ERROR("L0PerfCollector: failed to open %s", path.c_str());
            return -1;
        }
        out << "{\n"
            << "  \"version\": 1,\n"
            << "  \"markers_received\": " << markers_received_ << ",\n"
            << "  \"total_records\": " << total_collected_ << ",\n"
            << "  \"tasks\": [\n";
        bool first_task = true;
        for (const auto &kv : by_task) {
            if (!first_task) out << ",\n";
            first_task = false;
            out << "    {\n"
                << "      \"task_id\": " << kv.first << ",\n"
                << "      \"record_count\": " << kv.second.size() << ",\n"
                << "      \"records\": [\n";
            bool first_rec = true;
            for (const TaggedRecord &tr : kv.second) {
                if (!first_rec) out << ",\n";
                first_rec = false;
                SubCoreInfo s = decode_sub_core(tr.sub_core_id);
                out << "        {\"cycle\": " << tr.r.cycle
                    << ", \"sub_core_id\": " << static_cast<uint32_t>(tr.sub_core_id)
                    << ", \"group\": " << s.group << ", \"sub_type\": \"" << s.type << "\""
                    << ", \"pipe\": " << static_cast<uint32_t>(tr.r.pipe)
                    << ", \"pipe_name\": \"" << pipe_name(tr.r.pipe) << "\""
                    << ", \"region_id\": " << tr.r.region_id << "}";
            }
            out << "\n      ]\n    }";
        }
        out << "\n  ]\n}\n";
        out.flush();
        LOG_INFO_V1(
            "L0PerfCollector: wrote %lu records (grouped by %zu tasks) to %s",
            static_cast<unsigned long>(total_collected_), by_task.size(), path.c_str()
        );
    }

    // ---- (2) msprof-style trace.json (Chrome Trace Event Format) ----
    // Renders in https://ui.perfetto.dev, chrome://tracing, MindStudio. One
    // process per sub-core (pid "g{group}.{aic/aiv0/aiv1}"), per-pipe lanes
    // (tid), instruction duration bars built by pairing begin/end mark_stamp
    // regions, ns time base (1 cycle == 1 ns at PLATFORM_PROF_SYS_CNT_FREQ),
    // cname colors, and flow arrows for the cross-pipe set_flag/wait_flag chain.
    {
        std::string path = output_prefix_ + "/l0_perf_trace.json";
        std::ofstream out(path, std::ios::out | std::ios::trunc);
        if (!out.is_open()) {
            LOG_ERROR("L0PerfCollector: failed to open %s", path.c_str());
            return -1;
        }
        out << std::fixed << std::setprecision(3);

        // Time origin: shift to relative ns so the swimlane starts near 0.
        uint64_t t0 = UINT64_MAX;
        for (const auto &kv : by_task)
            for (const TaggedRecord &tr : kv.second) t0 = std::min(t0, tr.r.cycle);
        if (t0 == UINT64_MAX) t0 = 0;
        auto to_ns = [&](uint64_t c) {
            return static_cast<double>(c - t0) * 1e9 / static_cast<double>(PLATFORM_PROF_SYS_CNT_FREQ);
        };
        auto pid_of = [](uint8_t sub_core_id, char *buf, size_t n) {
            SubCoreInfo s = decode_sub_core(sub_core_id);
            snprintf(buf, n, "g%u.%s", s.group, s.type);
        };

        out << "{\n  \"displayTimeUnit\": \"ns\",\n  \"profilingType\": \"op\",\n"
            << "  \"schemaVersion\": 2,\n  \"traceEvents\": [\n";
        bool first = true;
        auto sep = [&]() {
            if (!first) out << ",\n";
            first = false;
        };

        // --- metadata: process order + per-lane order ---
        std::map<uint8_t, std::set<uint8_t>> lanes_by_subcore;  // sub_core_id -> pipes
        for (const auto &kv : by_task)
            for (const TaggedRecord &tr : kv.second)
                lanes_by_subcore[tr.sub_core_id].insert(tr.r.pipe);
        for (const auto &sk : lanes_by_subcore) {
            char pid[32];
            pid_of(sk.first, pid, sizeof pid);
            sep();
            out << "    {\"ph\":\"M\",\"name\":\"process_sort_index\",\"pid\":\"" << pid
                << "\",\"tid\":\"NA\",\"args\":{\"sort_index\":" << static_cast<int>(sk.first) << "}}";
            sep();
            out << "    {\"ph\":\"M\",\"name\":\"process_name\",\"pid\":\"" << pid
                << "\",\"tid\":\"NA\",\"args\":{\"name\":\"" << pid << "\"}}";
            for (uint8_t pipe : sk.second) {
                sep();
                out << "    {\"ph\":\"M\",\"name\":\"thread_sort_index\",\"pid\":\"" << pid
                    << "\",\"tid\":\"" << pipe_lane_name(pipe) << "\",\"args\":{\"sort_index\":"
                    << pipe_sort_index(pipe) << "}}";
            }
        }

        // --- events: duration bars + USERMARK/instant markers + flow arrows ---
        int flow_id = 0;
        for (const auto &kv : by_task) {
            uint64_t task_id = kv.first;
            // bucket records by (sub_core, pipe), preserving the by_task cycle order
            std::map<std::pair<uint8_t, uint8_t>, std::vector<const L0PerfRecord *>> lanes;
            for (const TaggedRecord &tr : kv.second) lanes[{tr.sub_core_id, tr.r.pipe}].push_back(&tr.r);

            // bar extents per (sub_core, op_id) for the flow chain below
            struct Span {
                double begin = 0, end = 0;
                uint8_t pipe = 0;
                bool ok = false;
            };
            std::map<std::pair<uint8_t, uint32_t>, Span> spans;

            for (const auto &lk : lanes) {
                uint8_t sub = lk.first.first;
                uint8_t pipe = lk.first.second;
                char pid[32];
                pid_of(sub, pid, sizeof pid);
                const char *lane = pipe_lane_name(pipe);
                std::map<uint32_t, double> open_begin;  // op_id -> begin ts

                for (const L0PerfRecord *r : lk.second) {
                    uint32_t op = r->region_id / 10;
                    uint32_t phase = r->region_id % 10;
                    L0OpInfo oi = op_info(op);
                    double ts = to_ns(r->cycle);

                    // pipe 0 (scalar-issue duplicate) -> USERMARK instant
                    if (pipe == 0) {
                        sep();
                        out << "    {\"ph\":\"X\",\"pid\":\"" << pid << "\",\"tid\":\"USERMARK\""
                            << ",\"ts\":" << ts << ",\"dur\":0,\"name\":\"" << oi.name
                            << "\",\"cname\":\"" << oi.cname << "\",\"args\":{\"task_id\":" << task_id
                            << ",\"region_id\":" << r->region_id << ",\"cycle\":" << r->cycle << "}}";
                        continue;
                    }
                    if (phase == 0) {
                        open_begin[op] = ts;
                    } else if (phase == 1) {
                        auto it = open_begin.find(op);
                        double b = (it != open_begin.end()) ? it->second : ts;
                        if (it != open_begin.end()) open_begin.erase(it);
                        double dur = ts - b;
                        if (dur < 0) dur = 0;
                        sep();
                        out << "    {\"ph\":\"X\",\"pid\":\"" << pid << "\",\"tid\":\"" << lane
                            << "\",\"ts\":" << b << ",\"dur\":" << dur << ",\"name\":\"" << oi.name
                            << "\",\"cname\":\"" << oi.cname << "\",\"args\":{\"task_id\":" << task_id
                            << ",\"op_id\":" << op << ",\"region_id\":" << r->region_id
                            << ",\"cycle\":" << r->cycle << "}}";
                        spans[{sub, op}] = {b, ts, pipe, true};
                    } else {
                        sep();
                        out << "    {\"ph\":\"X\",\"pid\":\"" << pid << "\",\"tid\":\"" << lane
                            << "\",\"ts\":" << ts << ",\"dur\":0,\"name\":\"" << oi.name
                            << "\",\"cname\":\"" << oi.cname << "\",\"args\":{\"task_id\":" << task_id
                            << ",\"op_id\":" << op << ",\"region_id\":" << r->region_id
                            << ",\"cycle\":" << r->cycle << "}}";
                        spans[{sub, op}] = {ts, ts, pipe, true};
                    }
                }
                // unmatched begins -> zero-dur markers so nothing is lost
                for (const auto &ob : open_begin) {
                    L0OpInfo oi = op_info(ob.first);
                    sep();
                    out << "    {\"ph\":\"X\",\"pid\":\"" << pid << "\",\"tid\":\"" << lane
                        << "\",\"ts\":" << ob.second << ",\"dur\":0,\"name\":\"" << oi.name
                        << "\",\"cname\":\"" << oi.cname << "\",\"args\":{\"task_id\":" << task_id << "}}";
                }
            }

            // flow arrows along the producer->consumer chain (TLOAD->TMUL->TSTORE)
            static const std::pair<uint32_t, uint32_t> kFlowChain[] = {{1, 3}, {3, 5}};
            std::set<uint8_t> subs;
            for (const auto &sp : spans) subs.insert(sp.first.first);
            for (uint8_t sub : subs) {
                char pid[32];
                pid_of(sub, pid, sizeof pid);
                for (const auto &fc : kFlowChain) {
                    auto its = spans.find({sub, fc.first});
                    auto itd = spans.find({sub, fc.second});
                    if (its == spans.end() || itd == spans.end()) continue;
                    if (!its->second.ok || !itd->second.ok) continue;
                    const char *src = pipe_lane_name(its->second.pipe);
                    const char *dst = pipe_lane_name(itd->second.pipe);
                    int fid = flow_id++;
                    sep();
                    out << "    {\"ph\":\"s\",\"id\":" << fid << ",\"name\":\"flow\",\"cat\":\"" << src
                        << "To" << dst << "\",\"pid\":\"" << pid << "\",\"tid\":\"" << src
                        << "\",\"ts\":" << its->second.end << "}";
                    sep();
                    out << "    {\"ph\":\"t\",\"id\":" << fid << ",\"name\":\"flow\",\"cat\":\"" << src
                        << "To" << dst << "\",\"pid\":\"" << pid << "\",\"tid\":\"" << dst
                        << "\",\"ts\":" << itd->second.begin << "}";
                }
            }
        }
        out << "\n  ]\n}\n";
        out.flush();
        LOG_INFO_V1(
            "L0PerfCollector: wrote msprof-style trace (%lu stamps) to %s",
            static_cast<unsigned long>(total_collected_), path.c_str()
        );
    }
    return 0;
}

// ---------------------------------------------------------------------------
// finalize — reverse of initialize. Idempotent.
// ---------------------------------------------------------------------------

void L0PerfCollector::finalize(L0PerfUnregisterCallback unregister_cb, const L0PerfFreeCallback &free_cb) {
    if (!initialized_) {
        // initialize() may have failed partway through, leaving a partial
        // ACL session or hal_handle_ behind. Best-effort cleanup of whatever
        // is set.
        close_acl_prof_session();
        if (hal_handle_ != nullptr) {
            dlclose(hal_handle_);
            hal_handle_ = nullptr;
            prof_drv_start_ = nullptr;
            prof_channel_read_ = nullptr;
            prof_stop_ = nullptr;
        }
        return;
    }

    // Defensive: if export_swimlane_json() was never called the stamp drain
    // thread is still running. Stop + join it before tearing channels down.
    stamp_drain_running_.store(false, std::memory_order_release);
    if (stamp_drain_thread_.joinable()) {
        stamp_drain_thread_.join();
    }

    // stop() must run before stop_all_biu_perf_channels so on_buffer_collected
    // (which runs on the collector thread) can't fire after the channels are
    // torn down.
    stop();
    stop_all_biu_perf_channels();

    // dlclose must come after the last prof_* call so the symbols stay
    // resolved.
    if (hal_handle_ != nullptr) {
        dlclose(hal_handle_);
        hal_handle_ = nullptr;
        prof_drv_start_ = nullptr;
        prof_channel_read_ = nullptr;
        prof_stop_ = nullptr;
    }

    close_acl_prof_session();

    auto release_dev = [&](void *p) { release_one_buffer(p, unregister_cb, free_cb); };

    // Release marker buffers still parked in per-thread free_queues + held as
    // current_buf_ptr (AICPU-owned, not tracked by release_owned_buffers).
    // Mirror the device shm first so we read the final queue state. Only the
    // device pointer is released here; the paired host shadow is freed by
    // clear_mappings() below.
    manager_.mirror_shm_from_device();
    if (shm_host_ != nullptr) {
        L0PerfDataHeader *hdr = get_l0_perf_header(shm_host_);
        for (int t = 0; t < num_threads_; ++t) {
            L0PerfBufferState *state = &hdr->thread_states[t];
            if (state->current_buf_ptr != 0) {
                release_dev(reinterpret_cast<void *>(state->current_buf_ptr));
                state->current_buf_ptr = 0;
            }
            uint32_t head = state->free_queue.head;
            uint32_t tail = state->free_queue.tail;
            uint32_t queued = tail - head;
            if (queued > PLATFORM_L0_FREE_SLOT_COUNT) {
                queued = PLATFORM_L0_FREE_SLOT_COUNT;
            }
            for (uint32_t k = 0; k < queued; ++k) {
                uint32_t slot = (head + k) % PLATFORM_L0_FREE_SLOT_COUNT;
                release_dev(reinterpret_cast<void *>(state->free_queue.buffer_ptrs[slot]));
                state->free_queue.buffer_ptrs[slot] = 0;
            }
            state->free_queue.head = tail;
        }
    }

    manager_.release_owned_buffers([&](void *p) { release_dev(p); });
    // shm_dev_ and aicore_rings_dev_ went through register_mapping only —
    // release_owned_buffers walks the recycle / done / ready queues and so
    // never sees them; release explicitly here.
    if (aicore_rings_dev_ != nullptr) {
        release_dev(aicore_rings_dev_);
        aicore_rings_dev_ = nullptr;
    }
    if (shm_dev_ != nullptr) {
        release_dev(shm_dev_);
        shm_dev_ = nullptr;
    }
    manager_.clear_mappings();

    collected_records_.clear();
    for (uint32_t g = 0; g < kBiuPerfNumGroups; ++g) {
        all_windows_[g].clear();
        all_stamps_[g].clear();
    }
    initialized_ = false;
    clear_memory_context();
    LOG_INFO_V1("L0PerfCollector: finalized");
}
