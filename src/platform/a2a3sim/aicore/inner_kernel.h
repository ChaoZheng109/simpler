/**
 * @file inner_kernel.h
 * @brief Platform-specific AICore definitions for simulation (a2a3sim)
 *
 * This header provides platform-specific macro definitions for AICore kernels
 * running in host-based simulation environment.
 */

#ifndef PLATFORM_A2A3SIM_AICORE_INNER_KERNEL_H_
#define PLATFORM_A2A3SIM_AICORE_INNER_KERNEL_H_

// AICore function attribute - no-op in simulation
#ifndef __aicore__
#define __aicore__
#endif

// dcci (Data Cache Clean and Invalidate) - no-op in simulation
#define dcci(addr, mode, opt) ((void)0)

// Cache coherency constants (no-op in simulation)
#define ENTIRE_DATA_CACHE 0
#define CACHELINE_OUT 0

#endif  // PLATFORM_A2A3SIM_AICORE_INNER_KERNEL_H_
