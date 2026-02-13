/**
 * @file aicpu_regs.cpp
 * @brief AICPU-side AICore register access implementation
 */

#include "aicpu/aicpu_regs.h"

void enable_aicore_register(uint64_t reg_base_addr) {
    volatile uint32_t* reg_enable =
        reinterpret_cast<volatile uint32_t*>(reg_base_addr + REG_SPR_FAST_PATH_ENABLE_OFFSET);
    *reg_enable = REG_SPR_FAST_PATH_OPEN;
    __sync_synchronize();

    // Clear stale task_id from previous kernel runs
    volatile uint32_t* reg_task =
        reinterpret_cast<volatile uint32_t*>(reg_base_addr + REG_SPR_DATA_MAIN_BASE_OFFSET);
    *reg_task = 0;
}

void disable_aicore_register(uint64_t reg_base_addr) {
    volatile uint32_t* reg_enable =
        reinterpret_cast<volatile uint32_t*>(reg_base_addr + REG_SPR_FAST_PATH_ENABLE_OFFSET);
    *reg_enable = REG_SPR_FAST_PATH_CLOSE;
    __sync_synchronize();
}

void write_task_id_to_aicore(uint64_t reg_base_addr, uint32_t task_id) {
    volatile uint32_t* reg_task =
        reinterpret_cast<volatile uint32_t*>(reg_base_addr + REG_SPR_DATA_MAIN_BASE_OFFSET);
    *reg_task = task_id;
    __sync_synchronize();
}

AICoreStatus read_aicore_status(uint64_t reg_base_addr) {
    volatile uint32_t* reg_cond =
        reinterpret_cast<volatile uint32_t*>(reg_base_addr + REG_SPR_COND_OFFSET);
    __sync_synchronize();
    uint32_t status_val = *reg_cond;
    return static_cast<AICoreStatus>(status_val);
}
