/**
 * @file inner_kernel.h
 * @brief Platform-specific AICore definitions for simulation (a2a3sim)
 *
 * This header provides platform-specific macro definitions for AICore kernels
 * running in host-based simulation environment.
 */

#ifndef PLATFORM_A2A3SIM_AICORE_INNER_KERNEL_H_
#define PLATFORM_A2A3SIM_AICORE_INNER_KERNEL_H_

#include <chrono>
#include <cstdint>

#include "common/platform_config.h"

// AICore function attribute - no-op in simulation
#ifndef __aicore__
#define __aicore__
#endif

// dcci (Data Cache Clean and Invalidate) - no-op in simulation
#define dcci(addr, mode, opt) ((void)0)

// Cache coherency constants (no-op in simulation)
#define ENTIRE_DATA_CACHE 0
#define CACHELINE_OUT 0

// =============================================================================
// System Counter Simulation
// =============================================================================

/**
 * Simulated system counter for performance profiling
 *
 * Mimics hardware counter behavior by returning a monotonic value
 * at 1850 MHz frequency (matching a2a3 hardware counter).
 *
 * Implementation:
 * - Uses std::chrono::high_resolution_clock as time source
 * - Converts elapsed nanoseconds to counter ticks
 * - Counter frequency: PLATFORM_PROF_SYS_CNT_FREQ (1850 MHz)
 *
 * This ensures performance data calculation uses the same formula
 * as the real a2a3 platform:
 *   duration_us = (counter_ticks * 1000000) / 1850000000
 *
 * Thread-safety: The static variable is initialized once globally,
 * ensuring all threads share the same time base for consistent
 * cross-thread time comparison.
 *
 * @return Simulated counter value (ticks since program start)
 */
inline uint64_t get_sys_cnt() {
    // Use a global start time to ensure consistency across all threads
    static auto program_start = std::chrono::high_resolution_clock::now();

    auto now = std::chrono::high_resolution_clock::now();
    uint64_t elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now - program_start
    ).count();

    // Convert nanoseconds to counter ticks at PLATFORM_PROF_SYS_CNT_FREQ (1850 MHz)
    // Formula: ticks = (ns * freq_hz) / 1e9
    //
    // To avoid overflow, break down the calculation:
    // 1. Split elapsed_ns into seconds and remainder nanoseconds
    // 2. Calculate ticks for each part separately
    uint64_t seconds = elapsed_ns / 1000000000ULL;
    uint64_t remaining_ns = elapsed_ns % 1000000000ULL;

    uint64_t ticks = seconds * PLATFORM_PROF_SYS_CNT_FREQ +
                     (remaining_ns * PLATFORM_PROF_SYS_CNT_FREQ) / 1000000000ULL;

    return ticks;
}

#endif  // PLATFORM_A2A3SIM_AICORE_INNER_KERNEL_H_
