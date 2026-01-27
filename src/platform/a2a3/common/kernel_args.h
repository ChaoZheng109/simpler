/**
 * KernelArgs Structure - Shared between Host, AICPU, and AICore
 *
 * This structure is used to pass arguments to both AICPU and AICore kernels.
 * It contains pointers to device memory for arguments and graph data.
 *
 * Memory Layout:
 * This structure's layout is hardcoded in libaicpu_extend_kernels.so, which
 * expects specific offsets for deviceArgs fields. The unused[5] array provides
 * the required offset alignment for compatibility with the CANN runtime.
 */

#ifndef RUNTIME_COMMON_KERNEL_ARGS_H
#define RUNTIME_COMMON_KERNEL_ARGS_H

#include <cstdint>

// Forward declaration
class Graph;

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__AIV__) || defined(__AIC__)
#define __may_used_by_aicore__ __gm__
#else
#define __may_used_by_aicore__
#endif

/**
 * Kernel arguments structure
 *
 * This structure is passed to AICPU kernels by the host.
 *
 * Field Access Patterns:
 * - unused[5]: Padding for alignment with CANN runtime expectations
 * - deviceArgs: Written by host, read by AICPU (contains aicpuSoBin/aicpuSoLen)
 * - graphArgs: Written by host, read by AICPU (task graph with execution parameters)
 *              Graph contains: workers[], worker_count, block_dim, scheCpuNum, tasks[]
 *
 * Note: AICore kernels receive Graph* directly, not KernelArgs
 *       - AICPU: accesses graphArgs->workers, graphArgs->block_dim, etc.
 *       - AICore: receives Graph* pointer with workers at offset 0
 */
struct KernelArgs {
    uint64_t unused[5] = {0};        // Alignment padding (required by CANN runtime offset)
    uint64_t *deviceArgs{nullptr};    // Device arguments (AICPU reads, contains SO info)
    uint64_t *runtime{nullptr};       // Runtime context (AICPU reads)
    Graph *graphArgs{nullptr};        // Task graph in device memory
};

#ifdef __cplusplus
}
#endif

#endif  // RUNTIME_COMMON_KERNEL_ARGS_H
