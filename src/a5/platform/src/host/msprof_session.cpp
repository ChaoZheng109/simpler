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

#include "host/msprof_session.h"

#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>

#include "common/unified_log.h"

namespace msprof {

namespace {

constexpr const char *kEnvEnable = "SIMPLER_MSPROF";
constexpr const char *kEnvOutDir = "SIMPLER_MSPROF_OUT";
constexpr const char *kEnvAicoreMetrics = "SIMPLER_MSPROF_AICORE_METRICS";
constexpr const char *kEnvMask = "SIMPLER_MSPROF_MASK";
constexpr const char *kDefaultOutDir = "./prof_output";
constexpr const char *kAclProfLib = "libacl_prof.so";

bool ParseEnabledFlag() {
    const char *v = std::getenv(kEnvEnable);
    if (v == nullptr) return false;
    return v[0] == '1' || v[0] == 't' || v[0] == 'T' || v[0] == 'y' || v[0] == 'Y';
}

std::string ParseOutDir() {
    const char *v = std::getenv(kEnvOutDir);
    return (v == nullptr || v[0] == '\0') ? std::string(kDefaultOutDir) : std::string(v);
}

int ParseAicoreMetrics(int fallback) {
    const char *v = std::getenv(kEnvAicoreMetrics);
    if (v == nullptr || v[0] == '\0') return fallback;
    char *end = nullptr;
    long parsed = std::strtol(v, &end, 0);
    if (end == v) return fallback;
    return static_cast<int>(parsed);
}

uint64_t ParseMask(uint64_t fallback) {
    const char *v = std::getenv(kEnvMask);
    if (v == nullptr || v[0] == '\0') return fallback;
    char *end = nullptr;
    unsigned long long parsed = std::strtoull(v, &end, 0);
    if (end == v) return fallback;
    return static_cast<uint64_t>(parsed);
}

}  // namespace

MsprofSession &MsprofSession::Instance() {
    static MsprofSession s;
    return s;
}

MsprofSession::MsprofSession() {
    if (!ParseEnabledFlag()) {
        return;
    }

    output_dir_ = ParseOutDir();
    aicore_metrics_ = ParseAicoreMetrics(aicore_metrics_);
    data_type_config_ = ParseMask(data_type_config_);

    handle_ = dlopen(kAclProfLib, RTLD_LAZY | RTLD_GLOBAL);
    if (handle_ == nullptr) {
        LOG_WARN("MsprofSession: dlopen(%s) failed: %s; profiling disabled", kAclProfLib, dlerror());
        return;
    }

    init_fn_ = reinterpret_cast<InitFn>(dlsym(handle_, "aclprofInit"));
    finalize_fn_ = reinterpret_cast<FinalizeFn>(dlsym(handle_, "aclprofFinalize"));
    create_config = reinterpret_cast<CreateConfigFn>(dlsym(handle_, "aclprofCreateConfig"));
    destroy_config = reinterpret_cast<DestroyConfigFn>(dlsym(handle_, "aclprofDestroyConfig"));
    start = reinterpret_cast<StartFn>(dlsym(handle_, "aclprofStart"));
    stop = reinterpret_cast<StopFn>(dlsym(handle_, "aclprofStop"));

    if (init_fn_ == nullptr || finalize_fn_ == nullptr || create_config == nullptr || destroy_config == nullptr ||
        start == nullptr || stop == nullptr) {
        LOG_WARN("MsprofSession: missing aclprof symbols in %s; profiling disabled", kAclProfLib);
        dlclose(handle_);
        handle_ = nullptr;
        init_fn_ = nullptr;
        finalize_fn_ = nullptr;
        create_config = nullptr;
        destroy_config = nullptr;
        start = nullptr;
        stop = nullptr;
        return;
    }

    int rc = init_fn_(output_dir_.c_str(), output_dir_.size());
    if (rc != 0) {
        LOG_WARN("MsprofSession: aclprofInit(%s) returned %d; profiling disabled", output_dir_.c_str(), rc);
        dlclose(handle_);
        handle_ = nullptr;
        return;
    }

    enabled_ = true;
    LOG_INFO_V5(
        "MsprofSession: enabled, out=%s aicore_metrics=%d mask=0x%llx", output_dir_.c_str(), aicore_metrics_,
        static_cast<unsigned long long>(data_type_config_)
    );
    std::atexit(&MsprofSession::AtExitFinalize);
}

MsprofSession::~MsprofSession() {
    // Finalization is owned by AtExitFinalize so it runs once even if the
    // singleton's static destructor races with other teardown paths.
}

void MsprofSession::AtExitFinalize() {
    auto &self = Instance();
    if (!self.enabled_) return;
    if (self.finalize_fn_ != nullptr) {
        int rc = self.finalize_fn_();
        if (rc != 0) {
            LOG_WARN("MsprofSession: aclprofFinalize returned %d", rc);
        }
    }
    if (self.handle_ != nullptr) {
        dlclose(self.handle_);
        self.handle_ = nullptr;
    }
    self.enabled_ = false;
}

MsprofRunScope::MsprofRunScope(int device_id) {
    auto &session = MsprofSession::Instance();
    if (!session.enabled()) return;

    uint32_t dev = static_cast<uint32_t>(device_id);
    cfg_ = session.create_config(&dev, 1, session.aicore_metrics(), nullptr, session.data_type_config());
    if (cfg_ == nullptr) {
        LOG_WARN("MsprofRunScope: aclprofCreateConfig returned null for device %d", device_id);
        return;
    }
    int rc = session.start(cfg_);
    if (rc != 0) {
        LOG_WARN("MsprofRunScope: aclprofStart returned %d", rc);
        session.destroy_config(cfg_);
        cfg_ = nullptr;
    }
}

MsprofRunScope::~MsprofRunScope() {
    if (cfg_ == nullptr) return;
    auto &session = MsprofSession::Instance();
    if (!session.enabled()) {
        cfg_ = nullptr;
        return;
    }
    int rc = session.stop(cfg_);
    if (rc != 0) {
        LOG_WARN("MsprofRunScope: aclprofStop returned %d", rc);
    }
    rc = session.destroy_config(cfg_);
    if (rc != 0) {
        LOG_WARN("MsprofRunScope: aclprofDestroyConfig returned %d", rc);
    }
    cfg_ = nullptr;
}

}  // namespace msprof
