/**
 * @file inner_kernel.h
 * @brief Platform-specific AICore definitions for real hardware (a2a3)
 *
 * This header provides platform-specific macro definitions for AICore kernels
 * running on real Ascend hardware with CANN compiler support.
 */

#ifndef PLATFORM_A2A3_AICORE_INNER_KERNEL_H_
#define PLATFORM_A2A3_AICORE_INNER_KERNEL_H_

#include <cstdint>
#include "aicpu/aicpu_regs.h"

// AICore function attribute for CANN compiler
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

// dcci (Data Cache Clean and Invalidate) is provided by CANN headers
// No need to define it here - it's a hardware instruction

/**
 * Read task_id from DATA_MAIN_BASE register
 *
 * Return values: 0=idle, AICORE_EXIT_SIGNAL=shutdown, other=task_id+1
 */
 __aicore__ inline uint32_t read_task_id_by_reg() {
    uint32_t task_id;
    // MOV reads from AICore SPR, bypasses cache
    __asm__ volatile("MOV %0, DATA_MAIN_BASE\n" : "=l"(task_id));
    return task_id;
}

/**
 * Set AICore status to BUSY via COND register
 */
 __aicore__ inline void set_aicore_busy() {
    set_cond(AICoreStatus::BUSY);
}

/**
 * Set AICore status to IDLE via COND register
 */
 __aicore__ inline void set_aicore_idle() {
    set_cond(AICoreStatus::IDLE);
}

#endif  // PLATFORM_A2A3_AICORE_INNER_KERNEL_H_
