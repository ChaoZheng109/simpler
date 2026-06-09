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

/**
 * Select addr-only sub-mode: skip the pre-launch blind config and instead
 * write ONLY base_addr per core after the AICore handshake (see
 * perfmon_aicpu_set_addr_after_handshake).
 */
void set_perfmon_addr_only(bool addr_only);

/** Query addr-only sub-mode (consumed by the scheduler handshake). */
bool is_perfmon_addr_only();

/**
 * Select unify sub-mode (extends addr-only): the post-handshake write also
 * forces buf_len + resets wptr + sets global_en=1 on every core, so the cores
 * the driver never armed (gen=0) start writing into their self-managed buffer
 * too and all cores share one buf_len.
 */
void set_perfmon_unify(bool unify);

/** Query unify sub-mode (consumed by perfmon_aicpu_set_addr_after_handshake). */
bool is_perfmon_unify();

/**
 * Select rearm-addr mode: blind-config still runs pre-launch (gen armed before
 * kickstart), and after the handshake we re-write base on every core to take
 * the driver-repointed monitored cores back. Also suppresses the L0 retire.
 */
void set_perfmon_rearm_addr(bool rearm);

/** Query rearm-addr mode (consumed by the scheduler handshake + retire skip). */
bool is_perfmon_rearm_addr();

/**
 * Select gen-only mode: pre-launch blind config opens only global_en, leaving
 * perf_mon_en for the kickstart to set. Consumed inside perfmon_aicpu_init.
 */
void set_perfmon_gen_only(bool gen_only);

/** Query gen-only mode. */
bool is_perfmon_gen_only();

/** Select skip-retire: keep all 108 cores in dispatch (composable with any mode). */
void set_perfmon_skip_retire(bool skip);

/** Query skip-retire (consumed by the scheduler retire decision). */
bool is_perfmon_skip_retire();

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
 * Addr-only sub-mode (PROFILING_FLAG_PERFMON_ADDR_ONLY): called per core from
 * the scheduler handshake AFTER the AICore kernel has entered (so firmware
 * kickstart has already programmed perfmon with its defaults). For physical
 * core @p phys_core_id at MMIO base @p reg_addr it:
 *   1. reads + logs the firmware-default regs (buf_len / en / global_en /
 *      base_addr / wptr) — answers "what does the default buf_len become";
 *   2. writes ONLY BASE_ADDR_L/H from the buf-addr table (indexed by
 *      phys_core_id) — no buf_len, no en, no counters;
 *   3. reads base_addr back + logs, so we see whether the override stuck while
 *      HW already had en set.
 * Records reg_addr so perfmon_aicpu_finalize() can read back wptr / samp.
 */
void perfmon_aicpu_set_addr_after_handshake(int phys_core_id, uint64_t reg_addr);

/**
 * Read wptr_o / samp_wrt / samp_crt for diagnostic logging then clear en /
 * global_en on cores regs[0..num_cores-1] (the same blind set as init).
 * Idempotent — guarded to run its body once even if called per-thread.
 */
void perfmon_aicpu_finalize();

#endif  // PLATFORM_AICPU_PERFMON_COLLECTOR_AICPU_H_
