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
 * @file msprof_session.h
 * @brief Host-driven msprof daemon control via aclprof* C API (route 1).
 *
 * Triggered by env vars:
 *   SIMPLER_MSPROF=1                  Enable; otherwise everything is no-op.
 *   SIMPLER_MSPROF_OUT=<dir>          Output directory; default ./prof_output.
 *   SIMPLER_MSPROF_AICORE_METRICS=<n> aclprofAicoreMetrics enum value;
 *                                     default 1 (ACL_AICORE_PIPE_UTILIZATION).
 *   SIMPLER_MSPROF_MASK=<hex>         dataTypeConfig bitmask;
 *                                     default 0x4F (ACL_API|TASK_TIME|
 *                                     AICORE_METRICS|AICPU|MSPROFTX).
 *
 * libacl_prof.so is dlopen'd lazily on first use; if dlopen or any dlsym
 * fails the session stays disabled and every entry point becomes a no-op.
 *
 * Lifecycle:
 *   - Process: MsprofSession singleton dlopens + aclprofInit on first
 *              MsprofRunScope construction; aclprofFinalize + dlclose run
 *              from atexit().
 *   - Per run: MsprofRunScope wraps a single device launch with
 *              aclprofCreateConfig + Start (ctor) and Stop + DestroyConfig
 *              (dtor). Multiple consecutive runs each get their own slice.
 */

#ifndef SRC_A5_PLATFORM_INCLUDE_HOST_MSPROF_SESSION_H_
#define SRC_A5_PLATFORM_INCLUDE_HOST_MSPROF_SESSION_H_

#include <cstdint>
#include <string>

// Forward-declare CANN types so this header has no dependency on acl/acl_prof.h
// (the .cpp pulls the real header for typedefs).
struct aclprofConfig;

namespace msprof {

class MsprofSession {
public:
    static MsprofSession &Instance();

    // True iff SIMPLER_MSPROF=1, dlopen succeeded, and aclprofInit returned 0.
    bool enabled() const { return enabled_; }

    // Function pointers resolved via dlsym. Null when !enabled().
    using CreateConfigFn = aclprofConfig *(*)(uint32_t *, uint32_t, int, const void *, uint64_t);
    using DestroyConfigFn = int (*)(const aclprofConfig *);
    using StartFn = int (*)(const aclprofConfig *);
    using StopFn = int (*)(const aclprofConfig *);

    CreateConfigFn create_config = nullptr;
    DestroyConfigFn destroy_config = nullptr;
    StartFn start = nullptr;
    StopFn stop = nullptr;

    int aicore_metrics() const { return aicore_metrics_; }
    uint64_t data_type_config() const { return data_type_config_; }

private:
    MsprofSession();
    ~MsprofSession();
    MsprofSession(const MsprofSession &) = delete;
    MsprofSession &operator=(const MsprofSession &) = delete;

    void *handle_ = nullptr;
    bool enabled_ = false;
    int aicore_metrics_ = 1;  // ACL_AICORE_PIPE_UTILIZATION
    uint64_t data_type_config_ = 0x4Full;
    std::string output_dir_;

    using InitFn = int (*)(const char *, std::size_t);
    using FinalizeFn = int (*)();
    InitFn init_fn_ = nullptr;
    FinalizeFn finalize_fn_ = nullptr;

    static void AtExitFinalize();
};

class MsprofRunScope {
public:
    explicit MsprofRunScope(int device_id);
    ~MsprofRunScope();
    MsprofRunScope(const MsprofRunScope &) = delete;
    MsprofRunScope &operator=(const MsprofRunScope &) = delete;

private:
    aclprofConfig *cfg_ = nullptr;
};

}  // namespace msprof

#endif  // SRC_A5_PLATFORM_INCLUDE_HOST_MSPROF_SESSION_H_
