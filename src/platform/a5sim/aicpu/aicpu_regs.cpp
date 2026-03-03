/**
 * @file aicpu_regs.cpp
 * @brief Platform-specific AICPU register initialization for a5sim
 */

#include "aicpu/platform_regs.h"
#include "inner_platform_config.h"

void platform_init_aicore_regs(uint64_t reg_addr) {
    // A5sim platform does NOT support fast path control
    // Only initialize task dispatch register to idle state
    write_reg(reg_addr, RegId::DATA_MAIN_BASE, 0);
}

void platform_deinit_aicore_regs(uint64_t reg_addr) {
    // Send exit signal to AICore
    write_reg(reg_addr, RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);

    // A5sim does not have fast path control - no need to close it
}

uint32_t platform_get_physical_cores_count() {
    return DAV_3510::PLATFORM_MAX_PHYSICAL_CORES;
}
