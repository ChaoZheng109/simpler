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
#ifndef COMMON_CHIP_TIMING_H_
#define COMMON_CHIP_TIMING_H_

/**
 * CHIP_TIMING (issue #1012): the single source of truth for the opt-in
 * per-stage run_prepared() timing log grammar. Every emit across the host
 * c_api and the per-arch device kernels goes through these macros so the
 * grammar can never drift away from the parser
 * (simpler_setup/tools/chip_timing.py).
 *
 * Lines are emitted on the V4 log tier — hidden at the default V5 threshold
 * (4 < 5), enabled by lowering the log level to V4. The gate short-circuits
 * before formatting, so disabled is near-zero cost. The timestamp rides in the
 * message body (ns in the emitting clock's domain), never the dlog prefix, so
 * the parser is prefix-independent. Durations are only meaningful within a
 * clock domain: host events (steady_clock) reconcile against host_wall, device
 * events (sys_cnt) against device_wall.
 *
 * Grammar:
 *   [CHIP_TIMING] run=<n> clk=host name=<stage> ev=<B|E> t=<ns>
 *   [CHIP_TIMING]          clk=dev  name=<stage> ev=<B|E> t=<ns> tid=<t>
 *   [CHIP_TIMING] run=<n> clk=host name=host_wall   ev=WALL us=<us>
 *   [CHIP_TIMING] run=<n> clk=dev  name=device_wall ev=WALL us=<us>
 */

#include "common/unified_log.h"

/* ---------------------------------------------------------------------------
 * Format macros — no std dependencies, safe to expand in any TU (host, onboard
 * AICPU, sim). The caller supplies the timestamp already converted to ns in its
 * own clock domain.
 * ------------------------------------------------------------------------- */

#define CHIP_TIMING_HOST_EVENT(run, name, ev, t_ns)                                                           \
    LOG_INFO_V4(                                                                                              \
        "[CHIP_TIMING] run=%llu clk=host name=%s ev=%c t=%llu", static_cast<unsigned long long>(run), (name), \
        static_cast<char>(ev), static_cast<unsigned long long>(t_ns)                                          \
    )

#define CHIP_TIMING_HOST_WALL(run, name, us)                                                                     \
    LOG_INFO_V4(                                                                                                 \
        "[CHIP_TIMING] run=%llu clk=host name=%s ev=WALL us=%.3f", static_cast<unsigned long long>(run), (name), \
        static_cast<double>(us)                                                                                  \
    )

#define CHIP_TIMING_DEV_WALL(run, name, us)                                                                     \
    LOG_INFO_V4(                                                                                                \
        "[CHIP_TIMING] run=%llu clk=dev name=%s ev=WALL us=%.3f", static_cast<unsigned long long>(run), (name), \
        static_cast<double>(us)                                                                                 \
    )

#define CHIP_TIMING_DEV_EVENT(name, ev, t_ns, tid)                                                                   \
    LOG_INFO_V4(                                                                                                     \
        "[CHIP_TIMING] clk=dev name=%s ev=%c tid=%ld t=%llu", (name), static_cast<char>(ev), static_cast<long>(tid), \
        static_cast<unsigned long long>(t_ns)                                                                        \
    )

/* ---------------------------------------------------------------------------
 * Per-clock convenience wrappers. Each references symbols that exist only in
 * one TU flavor; because macros expand lazily, an unused wrapper costs nothing
 * in the other TUs.
 * ------------------------------------------------------------------------- */

// Onboard AICPU: raw sys-counter cycles → ns via cycles_to_us(), tagged GET_TID().
// Requires common/platform_config.h (cycles_to_us) and the device-log GET_TID().
#define CHIP_TIMING_DEV_CYCLES(name, ev, cycles) \
    CHIP_TIMING_DEV_EVENT((name), (ev), static_cast<uint64_t>(cycles_to_us(cycles) * 1000.0), GET_TID())

// Sim device: a steady_clock::time_point → ns in the steady_clock epoch.
// Requires <chrono>.
#define CHIP_TIMING_DEV_TP(name, ev, tp, tid)                                                                         \
    CHIP_TIMING_DEV_EVENT(                                                                                            \
        (name), (ev),                                                                                                 \
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>((tp).time_since_epoch()).count()), \
        (tid)                                                                                                         \
    )

/* ---------------------------------------------------------------------------
 * Host RAII span + run counter — opt-in because it pulls <atomic>/<chrono>.
 * A host TU defines CHIP_TIMING_HOST_SPANS before including this header; the
 * device kernels (which avoid the std deps) leave it undefined.
 * ------------------------------------------------------------------------- */

#ifdef CHIP_TIMING_HOST_SPANS
#include <atomic>
#include <chrono>

namespace simpler::chip_timing {

// One monotonic per-run index per process. Inline so both arches' host c_api
// TUs share one definition without an ODR clash.
inline std::atomic<uint64_t> g_run_counter{0};

inline uint64_t next_run() { return g_run_counter.fetch_add(1, std::memory_order_relaxed); }

inline uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count()
    );
}

// B on construction, E on destruction — exception-safe pairing that also fires
// on the early-return error paths inside run_prepared().
struct Span {
    uint64_t run;
    const char *name;
    Span(uint64_t run_, const char *name_) :
        run(run_),
        name(name_) {
        CHIP_TIMING_HOST_EVENT(run, name, 'B', now_ns());
    }
    ~Span() { CHIP_TIMING_HOST_EVENT(run, name, 'E', now_ns()); }
    Span(const Span &) = delete;
    Span &operator=(const Span &) = delete;
};

}  // namespace simpler::chip_timing

#define CHIP_TIMING_SCOPE(run, name) ::simpler::chip_timing::Span _chip_timing_span_(run, name)
#endif  // CHIP_TIMING_HOST_SPANS

#endif  // COMMON_CHIP_TIMING_H_
