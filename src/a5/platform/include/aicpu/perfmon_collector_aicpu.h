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
 * @file perfmon_collector_aicpu.h
 * @brief AICPU-side init/finalize for the perfmon writeback probe (a5 onboard).
 *
 * Probe scope (issue #905):
 *   AICPU programs the AICore perfmon HW registers (0xB000 block within the
 *   3MB per-core MMIO window) so that perfmon DMAs trace bytes into per-core
 *   GM buffers we own — bypassing the driver biu_perf channel + HDC pipeline.
 *
 *   This file is the probe's init/finalize half. There is no per-task drain
 *   on AICPU (perfmon writes autonomously), and there is no decoder here —
 *   host copies the raw buffer back at teardown and inspects bytes offline.
 *
 * Lifecycle (mirroring pmu_collector_aicpu):
 *   set_platform_perfmon_buf_addrs() / set_perfmon_enabled()
 *       — pushed once by simpler_aicpu_exec from KernelArgs.
 *   perfmon_aicpu_init(physical_core_ids, num_cores)
 *       — once per run, called from the scheduler one-time init alongside
 *         pmu_aicpu_init. Programs base_addr_l/h, clears counters, enables.
 *   perfmon_aicpu_finalize(cur_thread_cores, core_num)
 *       — once per thread shutdown, called alongside pmu_aicpu_finalize.
 *         Reads wptr_o / samp_wrt / samp_crt for diagnostic logging then
 *         disables global_en so HW stops writing.
 */

#ifndef PLATFORM_AICPU_PERFMON_COLLECTOR_AICPU_H_
#define PLATFORM_AICPU_PERFMON_COLLECTOR_AICPU_H_

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Push the device pointer to the per-core perfmon buffer address table.
 *
 * @param addrs  Device address of a uint64_t[num_aicore] table; 0 = probe off.
 */
void set_platform_perfmon_buf_addrs(uint64_t addrs);

/** Push the per-core buffer length (bytes); programmed into PERF_MON_BUF_LEN. */
void set_platform_perfmon_buf_len(uint32_t buf_len);

/** Enable/disable the perfmon probe. Mirrors set_pmu_enabled. */
void set_perfmon_enabled(bool enable);

/** Query enablement (consumed by scheduler dispatch + cold-path). */
bool is_perfmon_enabled();

#ifdef __cplusplus
}
#endif

/**
 * Blind-configure perfmon HW on cores regs[0..num_cores-1]: program
 * BASE_ADDR_L/H from the buf-addr table (same index), buf_len, reset wptr /
 * glitch / sample counters, open global_en then en.
 *
 * "Blind" = indexed directly by regs[] order, NOT by handshake's
 * logical->physical map — so it can run BEFORE the AICore kernel is launched
 * (the perfmon config must be ready at AICore kickstart). Cores that end up
 * running no task simply have an empty buffer.
 *
 * Single-threaded; call once from simpler_aicpu_init.
 *
 * @param num_cores  Number of regs[] entries to configure.
 */
void perfmon_aicpu_init(int num_cores);

/**
 * Read wptr_o / samp_wrt / samp_crt for diagnostic logging then clear en /
 * global_en on cores regs[0..num_cores-1] (the same blind set as init).
 * Idempotent — guarded to run its body once even if called per-thread.
 */
void perfmon_aicpu_finalize();

#endif  // PLATFORM_AICPU_PERFMON_COLLECTOR_AICPU_H_
