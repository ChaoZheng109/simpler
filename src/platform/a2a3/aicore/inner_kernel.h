/**
 * @file inner_kernel.h
 * @brief Platform-specific AICore definitions for real hardware (a2a3)
 *
 * This header provides platform-specific macro definitions for AICore kernels
 * running on real Ascend hardware with CANN compiler support.
 */

#ifndef PLATFORM_A2A3_AICORE_INNER_KERNEL_H_
#define PLATFORM_A2A3_AICORE_INNER_KERNEL_H_

// AICore function attribute for CANN compiler
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

// dcci (Data Cache Clean and Invalidate) is provided by CANN headers
// No need to define it here - it's a hardware instruction

#endif  // PLATFORM_A2A3_AICORE_INNER_KERNEL_H_
