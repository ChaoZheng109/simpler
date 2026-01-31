/**
 * @file aicore.h
 * @brief AICore Platform Abstraction Layer
 *
 * Provides unified AICore qualifiers and macros for both real hardware
 * and simulation environments. Uses conditional compilation to select
 * the appropriate implementation.
 *
 * Platform Support:
 * - a2a3 (__PLATFORM_A2A3__): Real Ascend hardware with CANN compiler
 * - a2a3sim (default): Host-based simulation using standard C++
 */

#ifndef PLATFORM_AICORE_H_
#define PLATFORM_AICORE_H_

// =============================================================================
// Common Memory Qualifiers (All Platforms)
// =============================================================================

#ifndef __gm__
#define __gm__
#endif

#ifndef __global__
#define __global__
#endif

#ifndef __in__
#define __in__
#endif

#ifndef __out__
#define __out__
#endif

// =============================================================================
// Platform-Specific Definitions
// =============================================================================

#if defined(__PLATFORM_A2A3__)
    // Real hardware: Use CANN compiler attributes
    #ifndef __aicore__
    #define __aicore__ [aicore]
    #endif

#else
    // Simulation: No-op macros for host execution
    #ifndef __aicore__
    #define __aicore__
    #endif

    // Cache coherency operations (no-op on unified memory)
    #define ENTIRE_DATA_CACHE 0
    #define CACHELINE_OUT 0
    #define dcci(addr, mode, opt) ((void)0)
#endif

#endif  // PLATFORM_AICORE_H_
