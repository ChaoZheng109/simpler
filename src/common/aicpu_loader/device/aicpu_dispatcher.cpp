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
 * AICPU Dispatcher implementation — transient bootstrap-only upload helper.
 *
 * See aicpu_dispatcher.h for architecture. The dispatcher SO exists only
 * to provide a piece of code that runs with sched-thread (HwHiAiUser)
 * permissions for one purpose: write the bundled runtime SO bytes to
 * the main aicpu_scheduler's process directory under a content-fingerprint
 * filename. Once Init returns, this SO is no longer referenced — host's
 * subsequent rtsBinaryLoadFromFile calls target the runtime SO file directly.
 */

#include "aicpu_dispatcher.h"

#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <cerrno>
#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#include "aicpu/aicpu_schedule/aicpu_context.h"
#include "aicpu/cache_maintenance.h"
#include "utils/elf_build_id.h"

// dlog wrapper so error paths show up in device log without depending on
// our common/unified_log machinery (this SO is loaded standalone by CANN).
extern "C" void DlogRecord(int moduleId, int level, const char *fmt, ...);

namespace simpler_dispatcher {
constexpr int kDlogModuleCcecpu = 3;
constexpr int kDlogLevelError = 3;

void DispatcherLog(const char *fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    // DlogRecord is a non-weak extern: if it failed to resolve, this SO
    // would not have dlopen'd in the first place, so an address-vs-nullptr
    // guard here is dead code (and is folded to `true` by most compilers).
    DlogRecord(kDlogModuleCcecpu, kDlogLevelError, "[simpler-dispatcher] %s", buf);
}
}  // namespace simpler_dispatcher

// Bootstrap-time DeviceArgs view. Layout shared with host's BootstrapDispatcher.
// libaicpu_extend_kernels reads aicpu_so_bin/len/deviceId; we additionally read
// inner_so_bin/len and a result address in the remaining bootstrap fields.
struct KernelArgs {
    uint64_t unused[5] = {0};
    void *device_args{nullptr};
    void *runtime_args{nullptr};
    uint64_t regs{0};
};
static_assert(offsetof(KernelArgs, device_args) == 40, "KernelArgs::device_args offset drift");

namespace simpler_dispatcher {

void ReportFailure(BootstrapResult &result, BootstrapStage stage, int error_number, const std::string &path) {
    result.status = BootstrapStatus::Failed;
    result.stage = stage;
    result.error_number = error_number;
    snprintf(result.path, sizeof(result.path), "%s", path.c_str());
    if (error_number == ENOSPC || error_number == EDQUOT) {
        const std::string dir = path.substr(0, path.find_last_of('/'));
        struct statvfs capacity{};
        if (statvfs(dir.c_str(), &capacity) == 0) {
            result.capacity_valid = 1;
            result.block_bytes = capacity.f_frsize;
            result.free_blocks = capacity.f_bfree;
            result.available_blocks = capacity.f_bavail;
            result.free_inodes = capacity.f_ffree;
        } else {
            result.capacity_errno = errno;
        }
    }
    DispatcherLog(
        "SO cache failed: stage=%s errno=%d (%s) device=%d path=%s requested=%llu written=%llu context_rc=%d",
        bootstrap_stage_name(stage), error_number, error_number != 0 ? strerror(error_number) : "none",
        result.device_id, result.path, static_cast<unsigned long long>(result.requested_bytes),
        static_cast<unsigned long long>(result.written_bytes), result.context_error
    );
}

// ELF Build-ID-derived 64-bit fingerprint (linker SHA-1 truncated to 8
// bytes by `-Wl,--build-id`). Falls back to full-buffer FNV-1a if the SO
// was somehow linked without a build-id note. Host's
// load_aicpu_op.cpp::FingerprintBytes calls the same helper, so both sides
// produce identical fingerprints with no other channel of communication.
//
// The earlier "FNV-1a over the first 64 bytes XOR len" scheme collided in
// practice on same-toolchain runtime SOs whose ELF headers + size matched
// — wrong-code risk on the multi-runtime path. Build-IDs are strong by
// linker contract: identical Build-IDs imply byte-identical SOs.
uint64_t Fingerprint(const char *data, uint64_t len) {
    return simpler::common::utils::elf_build_id_64(data, static_cast<std::size_t>(len));
}

// CANN resolves and removes this process-owned directory at session teardown.
std::string MakeInnerSoPath(uint64_t fp, uint64_t device_id, BootstrapResult *result) {
    aicpu::aicpuContext_t ctx{};
    const int context_success = static_cast<int>(aicpu::AICPU_ERROR_NONE);
    int context_rc = context_success;
    if (aicpu::aicpuGetContext == nullptr || aicpu::GetUniqueVfId == nullptr) context_rc = -ENOSYS;
    else if ((context_rc = static_cast<int>(aicpu::aicpuGetContext(&ctx))) == context_success && ctx.hostPid <= 0)
        context_rc = -EINVAL;
    if (context_rc != context_success) {
        DispatcherLog("cannot determine CANN process SO directory");
        if (result != nullptr) {
            result->context_error = context_rc;
            ReportFailure(*result, BootstrapStage::Context, 0, {});
        }
        return {};
    }
    char buf[256];
    snprintf(
        buf, sizeof(buf), "/usr/lib64/aicpu_kernels/%u/aicpu_kernels_device/%d_%u/simpler_inner_%016llx_%llu.so",
        aicpu::GetUniqueVfId(), ctx.hostPid, ctx.deviceId, static_cast<unsigned long long>(fp),
        static_cast<unsigned long long>(device_id)
    );
    return buf;
}

// Per-upload temporary files and atomic rename publish complete SOs to concurrent
// readers. Existing dlopen handles remain bound to their original inode.
bool WriteBytes(const std::string &path, const char *data, uint64_t len, BootstrapResult &report) {
    report.requested_bytes = len;
    report.written_bytes = 0;
    char tmp_path[320];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp.XXXXXX", path.c_str());
    int fd = mkstemp(tmp_path);
    if (fd < 0) {
        const int error_number = errno;
        ReportFailure(report, BootstrapStage::Create, error_number, path);
        return false;
    }
    const auto fail = [&](BootstrapStage stage, int error_number) {
        ReportFailure(report, stage, error_number, path);
        if (fd >= 0) (void)close(fd);
        unlink(tmp_path);
        return false;
    };
    while (report.written_bytes < len) {
        const uint64_t remaining = len - report.written_bytes;
        const size_t chunk =
            static_cast<size_t>(std::min(remaining, static_cast<uint64_t>(std::numeric_limits<ssize_t>::max())));
        const ssize_t written = write(fd, data + report.written_bytes, chunk);
        if (written < 0) {
            const int error_number = errno;
            if (error_number == EINTR) continue;
            return fail(BootstrapStage::Write, error_number);
        }
        if (written == 0) return fail(BootstrapStage::Write, EIO);
        report.written_bytes += static_cast<uint64_t>(written);
    }
    const int close_rc = close(fd);
    const int close_error = errno;
    fd = -1;
    if (close_rc != 0) return fail(BootstrapStage::Close, close_error);
    if (chmod(tmp_path, 0755) != 0) return fail(BootstrapStage::Chmod, errno);
    if (rename(tmp_path, path.c_str()) != 0) {
        return fail(BootstrapStage::Rename, errno);
    }
    return true;
}

}  // namespace simpler_dispatcher

// =============================================================================
// C-style exported entry points dlsym'd by libaicpu_extend_kernels.
// =============================================================================

extern "C" {

// Stubs — libaicpu_extend_kernels::SetTileFwkKernelMap dlsym's all three at
// load time; absence makes the whole SO unmappable. We only reach Init in
// practice, but return 0 (success) here to mirror the happy-path return of
// the old AICPU kernel stubs we replaced. If a future CANN version begins
// invoking Static as a warm-up probe, returning failure would be a silent
// regression versus the prior behavior.
__attribute__((visibility("default"))) int StaticTileFwkBackendKernelServer(void *args) {
    (void)args;
    simpler_dispatcher::DispatcherLog("Static: stub (not expected to be called)");
    return 0;
}

__attribute__((visibility("default"))) uint32_t DynTileFwkBackendKernelServer(void *args) {
    (void)args;
    simpler_dispatcher::DispatcherLog("Server: stub (dispatcher is upload-only, not expected to be called)");
    return 0;
}

// Init publishes the bundled runtime SO into CANN's process-owned directory.
// Subsequent rtsBinaryLoadFromFile calls resolve it by basename.
__attribute__((visibility("default"))) uint32_t DynTileFwkBackendKernelServerInit(void *args) {
    if (args == nullptr) {
        simpler_dispatcher::DispatcherLog("Init: args==nullptr");
        return 1;
    }
    auto *k = reinterpret_cast<KernelArgs *>(args);
    auto *d = reinterpret_cast<simpler_dispatcher::BootstrapDeviceArgs *>(k->device_args);
    if (d == nullptr) {
        simpler_dispatcher::DispatcherLog("Init: device_args==nullptr");
        return 1;
    }
    simpler_dispatcher::BootstrapResult result;
    result.device_id = static_cast<int32_t>(d->device_id);
    result.requested_bytes = d->inner_so_len;
    const auto finish = [&]() -> uint32_t {
        if (d->result_addr != 0) {
            auto *output = reinterpret_cast<simpler_dispatcher::BootstrapResult *>(d->result_addr);
            *output = result;
            cache_flush_range(output, sizeof(*output));
            // Expected upload failures travel in the record without faulting the stream.
            return 0;
        }
        return result.status == simpler_dispatcher::BootstrapStatus::Success ? 0 : 1;
    };
    if (d->inner_so_bin == 0 || d->inner_so_len == 0) {
        simpler_dispatcher::DispatcherLog(
            "Init: empty inner SO bundle (bin=%lx len=%lu)", d->inner_so_bin, d->inner_so_len
        );
        simpler_dispatcher::ReportFailure(result, simpler_dispatcher::BootstrapStage::Write, EINVAL, {});
        return finish();
    }
    const char *inner_bytes = reinterpret_cast<const char *>(d->inner_so_bin);
    uint64_t fp = simpler_dispatcher::Fingerprint(inner_bytes, d->inner_so_len);
    const std::string path = simpler_dispatcher::MakeInnerSoPath(fp, d->device_id, &result);
    if (path.empty()) return finish();
    const std::string dir = path.substr(0, path.find_last_of('/'));
    if (mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
        const int error_number = errno;
        simpler_dispatcher::ReportFailure(result, simpler_dispatcher::BootstrapStage::Mkdir, error_number, dir);
        return finish();
    }
    if (!simpler_dispatcher::WriteBytes(path, inner_bytes, d->inner_so_len, result)) return finish();
    simpler_dispatcher::DispatcherLog("Init: wrote %s (%lu bytes)", path.c_str(), d->inner_so_len);
    result.status = simpler_dispatcher::BootstrapStatus::Success;
    result.stage = simpler_dispatcher::BootstrapStage::Complete;
    return finish();
}

}  // extern "C"
