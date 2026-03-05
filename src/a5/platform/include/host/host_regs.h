/**
 * @file host_regs.h
 * @brief AICore register address retrieval via CANN HAL APIs
 *
 * Provides register base addresses for AICPU to perform MMIO-based
 * task dispatch to AICore cores.
 */

#ifndef PLATFORM_HOST_HOST_REGS_H_
#define PLATFORM_HOST_HOST_REGS_H_

#include "common/platform_config.h"
#include <cstdint>
#include <vector>

// Forward declaration
class MemoryAllocator;

// =============================================================================
// HAL API Structures (not provided by CANN ascend_hal.h)
// =============================================================================

// AddrMapInPara/AddrMapOutPara: Used by legacy halMemCtl API (DAV_2201)
// These are not defined in CANN headers, defined here for compatibility
struct AddrMapInPara {
    int64_t devid;
    int addr_type;
};

struct AddrMapOutPara {
    uint64_t ptr;
    uint64_t len;
};

constexpr int ADDR_MAP_TYPE_REG_AIC_CTRL = 2;

// tagProcType/res_map_type: Used by halResMap API (DAV_3510)
// Note: res_map_info struct itself is defined in CANN ascend_hal.h
enum tagProcType {
    PROCESS_CP1 = 0,
    PROCESS_CP2,
    PROCESS_DEV_ONLY,
    PROCESS_QS,
    PROCESS_HCCP,
    PROCESS_USER,
    PROCESS_CPTYPE_MAX
};

enum res_map_type {
    RES_AICORE = 0,
    RES_HSCB_AICORE,
    RES_L2BUFF,
    RES_C2C,
    RES_MAP_TYPE_MAX
};

struct res_map_info {
    tagProcType target_proc_type;
    res_map_type res_type;
    uint32_t res_id;
    uint32_t flag;
    uint32_t rsv[1];
};

// MODULE_TYPE_AICORE and INFO_TYPE_OCCUPY are defined in CANN ascend_hal.h
// but we provide fallback definitions in case the header is not available
#ifndef MODULE_TYPE_AICORE
constexpr int32_t MODULE_TYPE_AICORE = 4;
#endif

#ifndef INFO_TYPE_OCCUPY
constexpr int32_t INFO_TYPE_OCCUPY = 8;
#endif

// AICore bitmap length for get_pg_mask (DAV_3510-specific)
constexpr uint8_t PLATFORM_AICORE_MAP_BUFF_LEN = 2;

// =============================================================================
// DAV_3510 Register Mapping Parameters
// =============================================================================

/**
 * Register address stride for sub-cores within an AICore
 * Each sub-core (AIC, AIV_0, AIV_1) has a 1M register space
 */
constexpr uint64_t REG_SUB_CORE_STRIDE = 0x100000ULL;

/**
 * Mapping region size per AICore
 * Covers all 3 sub-cores (1 AIC + 2 AIV) = 3M total
 */
constexpr uint32_t REG_AICORE_MAP_SIZE = 0x300000;

/**
 * Register offsets for AIV sub-cores relative to AICore base
 */
constexpr uint64_t REG_AIV_FIRST_OFFSET = REG_SUB_CORE_STRIDE;        // 1M
constexpr uint64_t REG_AIV_SECOND_OFFSET = 2 * REG_SUB_CORE_STRIDE;   // 2M

/**
 * Initialize AICore register addresses for runtime
 *
 * Retrieves register addresses from HAL, allocates device memory,
 * copies addresses to device, and stores the device pointer in runtime.
 *
 * @param runtime_regs_ptr Pointer to the regs field (e.g., KernelArgs.regs)
 * @param device_id Device ID
 * @param allocator Memory allocator for device memory
 * @return 0 on success, negative on failure
 */
int init_aicore_register_addresses(
    uint64_t* runtime_regs_ptr,
    uint64_t device_id,
    MemoryAllocator& allocator);

#endif  // PLATFORM_HOST_HOST_REGS_H_
