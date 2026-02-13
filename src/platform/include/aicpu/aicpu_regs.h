/**
 * @file aicpu_regs.h
 * @brief AICPU-side AICore register access via MMIO
 *
 * Provides MMIO-based register writes/reads for task dispatch and status polling.
 * Memory barriers ensure cross-core visibility.
 */

#ifndef PLATFORM_A2A3_AICPU_AICPU_REGS_H_
#define PLATFORM_A2A3_AICPU_AICPU_REGS_H_

#include <cstdint>
#include "common/platform_config.h"

/**
 * AICore execution status (read from COND register)
 */
enum class AICoreStatus : uint32_t {
    IDLE = 0,
    BUSY = 1,
};

/**
 * Enable register-based communication (call during handshake)
 */
void enable_aicore_register(uint64_t reg_base_addr);

/**
 * Disable register-based communication (call during shutdown)
 */
void disable_aicore_register(uint64_t reg_base_addr);

/**
 * Write task_id to AICore's DATA_MAIN_BASE register
 *
 * Special values: 0=idle, AICORE_EXIT_SIGNAL=shutdown
 */
void write_task_id_to_aicore(uint64_t reg_base_addr, uint32_t task_id);

/**
 * Read AICore status from COND register
 */
AICoreStatus read_aicore_status(uint64_t reg_base_addr);

#endif  // PLATFORM_A2A3_AICPU_AICPU_REGS_H_
