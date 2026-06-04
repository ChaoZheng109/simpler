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
 * @file perfmon_collector_aicpu.cpp
 * @brief AICPU-side perfmon writeback probe (a5 onboard) — see header for design.
 *
 * Programming sequence per core (init, after a disable to clear any TS-firmware
 * residue): BASE_ADDR_L → BASE_ADDR_H → SAMP_CRT_CLR=1 → SAMP_WRT=0 → wmb → GLOBAL_EN=1.
 * Finalize per core: read WPTR_O / SAMP_WRT / SAMP_CRT (diagnostic) → GLOBAL_EN=0.
 */

#include "aicpu/perfmon_collector_aicpu.h"

#include "aicpu/platform_regs.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "common/unified_log.h"

static uint64_t g_perfmon_buf_addrs_table = 0;
static uint32_t g_perfmon_buf_len = 0;
static bool g_enable_perfmon = false;

// Per-core resolved AICore MMIO base address, indexed by regs[] order (blind
// config — not by handshake's logical->physical map). 0 = no perfmon for it.
static uint64_t s_perfmon_reg_addrs[PLATFORM_MAX_CORES] = {0};
static int s_perfmon_num_cores = 0;
static bool s_perfmon_finalized = false;

extern "C" void set_platform_perfmon_buf_addrs(uint64_t addrs) { g_perfmon_buf_addrs_table = addrs; }

extern "C" void set_platform_perfmon_buf_len(uint32_t buf_len) { g_perfmon_buf_len = buf_len; }

extern "C" void set_perfmon_enabled(bool enable) { g_enable_perfmon = enable; }

extern "C" bool is_perfmon_enabled() { return g_enable_perfmon; }

void perfmon_aicpu_init(int num_cores) {
    if (g_perfmon_buf_addrs_table == 0) {
        LOG_ERROR("perfmon_aicpu_init: buf_addrs table is NULL (host did not allocate)");
        return;
    }
    if (num_cores > PLATFORM_MAX_CORES) {
        LOG_WARN("perfmon_aicpu_init: num_cores %d capped to PLATFORM_MAX_CORES %d", num_cores, PLATFORM_MAX_CORES);
        num_cores = PLATFORM_MAX_CORES;
    }
    s_perfmon_num_cores = num_cores;

    uint64_t *regs_array = reinterpret_cast<uint64_t *>(get_platform_regs());
    uint64_t *buf_addr_table = reinterpret_cast<uint64_t *>(g_perfmon_buf_addrs_table);
    if (regs_array == nullptr) {
        LOG_ERROR("perfmon_aicpu_init: regs_array is NULL");
        return;
    }

    for (int i = 0; i < num_cores; i++) {
        // Blind: index regs[] and the buf table directly by i (regs[] order),
        // no handshake / physical_core_ids needed.
        uint64_t reg_addr = regs_array[i];
        uint64_t buf_addr = buf_addr_table[i];
        s_perfmon_reg_addrs[i] = reg_addr;
        if (reg_addr == 0 || buf_addr == 0) {
            LOG_WARN(
                "perfmon_aicpu_init: core %d skipped (reg_addr=0x%lx, buf_addr=0x%lx)", i, reg_addr, buf_addr
            );
            continue;
        }

        // Sentinel: stamp a probe value at the very start of the buffer BEFORE
        // programming perfmon. If at finalize the buffer's first uint32_t is
        // still this value, HW DMA never wrote anything (and we can rule out
        // "AICPU can't reach this GM region"). If overwritten by non-zero,
        // HW wrote at least one chunk. Diagnostic — no functional purpose.
        *reinterpret_cast<volatile uint32_t *>(buf_addr) = 0xDEADBEEFu;

        // Disable both gates first in case TS firmware (or a prior run) left
        // them on. HW writes iff PERF_MON_EN (0x00C4) AND PERF_MON_GLOBAL_EN
        // (0xB000) are both 1.
        write_reg(reg_addr, RegId::PERF_MON_GLOBAL_EN, 0);
        write_reg(reg_addr, RegId::PERF_MON_EN, 0);

        // Program 48-bit writeback base: low 32 bits + high 16 bits.
        uint32_t lo = static_cast<uint32_t>(buf_addr & 0xFFFFFFFFu);
        uint32_t hi = static_cast<uint32_t>((buf_addr >> 32) & 0xFFFFu);
        write_reg(reg_addr, RegId::PERF_MON_BASE_ADDR_L, lo);
        write_reg(reg_addr, RegId::PERF_MON_BASE_ADDR_H, hi);

        // Tell HW how big each buffer is. HW default 0 → refuses to write.
        write_reg(reg_addr, RegId::PERF_MON_BUF_LEN, g_perfmon_buf_len);

        // Reset write pointer to buffer start. Per spec this is a RW register
        // that must be reset to 0 at setup (only writable while en = 0); a
        // stale non-zero value would make HW start mid-buffer or treat it full.
        write_reg(reg_addr, RegId::PERF_MON_WPTR_O, 0);

        // Disable glitch filtering (0 = no signal width is filtered out). A
        // non-zero default could cancel all trace output.
        write_reg(reg_addr, RegId::PERF_MON_GLITCH_FILTER, 0);

        // Reset samples-produced and samples-written counters (only writable
        // while en = 0).
        write_reg(reg_addr, RegId::PERF_MON_SAMP_CRT_CLR, 1);
        write_reg(reg_addr, RegId::PERF_MON_SAMP_WRT, 0);

        // perf_mon_global_en is the per-core gate; perf_mon_en is the last
        // enable write. Open the gate first so that when en goes 0->1, global_en
        // is already 1 and the AND condition holds at that instant.
        // NOTE: this is a plain AICPU MMIO write that sets the en value — it is
        // NOT the hardware-scheduler "kickstart" the spec describes (which sets
        // en during task dispatch). Whether a software value-set is equivalent
        // to that kickstart is the open question (see perfmon_register_check.md).
        write_reg(reg_addr, RegId::PERF_MON_GLOBAL_EN, 1);

        // Experiment: no barrier between the two enable writes — a dsb here may
        // stall and stretch the gap before en. MMIO writes to the same
        // Device-nGnRnE region are already ordered.
        write_reg(reg_addr, RegId::PERF_MON_EN, 1);

        // Immediate readback on core 0 only — confirms whether the values we
        // wrote actually landed in MMIO (rules out write_reg / RegId mapping
        // bugs) and whether the gates are still hot right after enable
        // (rules out a firmware autonomous-disable in the very same cycle).
        if (i == 0) {
            uint64_t rb_lo = read_reg(reg_addr, RegId::PERF_MON_BASE_ADDR_L);
            uint64_t rb_hi = read_reg(reg_addr, RegId::PERF_MON_BASE_ADDR_H);
            uint64_t rb_len = read_reg(reg_addr, RegId::PERF_MON_BUF_LEN);
            uint64_t rb_wptr = read_reg(reg_addr, RegId::PERF_MON_WPTR_O);
            uint64_t rb_glf = read_reg(reg_addr, RegId::PERF_MON_GLITCH_FILTER);
            uint64_t rb_en = read_reg(reg_addr, RegId::PERF_MON_EN);
            uint64_t rb_gen = read_reg(reg_addr, RegId::PERF_MON_GLOBAL_EN);
            LOG_INFO_V0(
                "Perfmon core 0 init readback: base_l=0x%lx (wrote 0x%x), base_h=0x%lx (wrote 0x%x), "
                "buf_len=0x%lx (wrote 0x%x), wptr=0x%lx glitch=0x%lx en=0x%lx global_en=0x%lx (wrote 1/1), "
                "buf_addr=0x%lx",
                rb_lo, lo, rb_hi, hi, rb_len, g_perfmon_buf_len, rb_wptr, rb_glf, rb_en, rb_gen, buf_addr
            );
        }
    }
    wmb();

    LOG_INFO_V0("Perfmon probe initialized on %d cores", num_cores);
}

void perfmon_aicpu_finalize() {
    // Idempotent: scheduler shutdown calls this per-thread, but the blind set
    // spans all cores, so run the body once.
    if (s_perfmon_finalized) {
        return;
    }
    s_perfmon_finalized = true;

    for (int core_id = 0; core_id < s_perfmon_num_cores; core_id++) {
        uint64_t reg_addr = s_perfmon_reg_addrs[core_id];
        if (reg_addr == 0) {
            continue;
        }

        // Diagnostic read before disabling — wptr / samp tell us if HW wrote
        // anything (probe success signal), and samp_wrt > buf_len signals a
        // wrap that overwrote earlier bytes (data loss). Also re-read
        // base_addr / global_en so we can compare against init-time
        // readback (catches firmware-clobbers-during-run scenarios).
        uint64_t wptr = read_reg(reg_addr, RegId::PERF_MON_WPTR_O);
        uint64_t samp_wrt = read_reg(reg_addr, RegId::PERF_MON_SAMP_WRT);
        uint64_t samp_crt = read_reg(reg_addr, RegId::PERF_MON_SAMP_CRT);
        uint64_t rb_lo = read_reg(reg_addr, RegId::PERF_MON_BASE_ADDR_L);
        uint64_t rb_hi = read_reg(reg_addr, RegId::PERF_MON_BASE_ADDR_H);
        uint64_t rb_en = read_reg(reg_addr, RegId::PERF_MON_EN);
        uint64_t rb_gen = read_reg(reg_addr, RegId::PERF_MON_GLOBAL_EN);
        LOG_INFO_V0(
            "Perfmon core %d: wptr=%lu samp_wrt=%lu samp_crt=%lu base_l=0x%lx base_h=0x%lx en=0x%lx gen=0x%lx",
            core_id, wptr, samp_wrt, samp_crt, rb_lo, rb_hi, rb_en, rb_gen
        );

        // Reverse of init: kill the trigger (en) first, then close the gate.
        write_reg(reg_addr, RegId::PERF_MON_EN, 0);
        write_reg(reg_addr, RegId::PERF_MON_GLOBAL_EN, 0);
    }
}
