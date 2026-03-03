/**
 * @file inner_platform_config.h
 * @brief Platform-specific configuration for a5sim (Simulation)
 */

#ifndef PLATFORM_INNER_CONFIG_H_
#define PLATFORM_INNER_CONFIG_H_

#include <cstdint>

// =============================================================================
// Platform Capacity Constraints
// =============================================================================

/**
 * Maximum block dimension supported by a5sim platform
 */
constexpr int PLATFORM_MAX_BLOCKDIM = 32;

/**
 * Maximum AICPU scheduling threads for a5sim platform
 */
constexpr int PLATFORM_MAX_AICPU_THREADS = 8;

// =============================================================================
// Register Offsets (a5sim architecture - DAV_3510 simulation)
// =============================================================================

/**
 * Task dispatch register offset (AICPU→AICore)
 */
constexpr uint32_t REG_SPR_DATA_MAIN_BASE_OFFSET = 0xD4;

/**
 * Status register offset (AICore→AICPU)
 */
constexpr uint32_t REG_SPR_COND_OFFSET = 0x5108;

/**
 * Fast path control register offset (not supported on a5sim)
 */
constexpr uint32_t REG_SPR_FAST_PATH_ENABLE_OFFSET = 0;

// =============================================================================
// Fast Path Control Values (not supported on a5sim)
// =============================================================================

constexpr uint32_t REG_SPR_FAST_PATH_OPEN = 0;
constexpr uint32_t REG_SPR_FAST_PATH_CLOSE = 0;

// =============================================================================
// Chip-Specific Configuration (DAV_3510)
// =============================================================================

namespace DAV_3510 {
    /**
     * Maximum physical AICore count for DAV_3510 chip
     */
    constexpr uint32_t PLATFORM_MAX_PHYSICAL_CORES = 36;
}

#endif  // PLATFORM_INNER_CONFIG_H_
