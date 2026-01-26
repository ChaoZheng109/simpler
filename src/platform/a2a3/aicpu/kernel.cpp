#include <cstdint>
#include <cstdio>
#include "device_log.h"
#include "graph.h"
#include "kernel_args.h"

// Forward declaration of AicpuExecute (implemented in graphexecutor.cpp)
extern "C" int AicpuExecute(void* arg);

extern "C" __attribute__((visibility("default"))) int StaticTileFwkBackendKernelServer(void *arg) {
    if (arg == nullptr) {
        DEV_ERROR("%s", "Invalid kernel arguments: null pointer");
        return -1;
    }

    return 0;
}

/**
 * AICPU kernel initialization entry point
 *
 * This function is called once during kernel initialization by the CANN runtime.
 * It initializes logging and validates kernel arguments.
 *
 * Note: Function name is hardcoded in libaicpu_extend_kernels.so
 *
 * @param arg Pointer to KernelArgs structure
 * @return 0 on success, -1 on error
 */
extern "C" __attribute__((visibility("default"))) int DynTileFwkBackendKernelServerInit(void *arg) {
    InitLogSwitch();
    if (arg == nullptr) {
        DEV_ERROR("%s", "Invalid kernel arguments: null pointer");
        return -1;
    }

    DEV_INFO("%s", "Graph Executor Init: Initializing AICPU kernel");
    return 0;
}

/**
 * AICPU kernel main execution entry point
 *
 * This is the main entry point for the AICPU graph executor kernel.
 * It delegates to AicpuExecute which is implemented in graphexecutor.cpp.
 *
 * Note: Function name is hardcoded in libaicpu_extend_kernels.so
 *
 * @param arg Pointer to KernelArgs structure containing:
 *            - deviceArgs: device-specific arguments
 *            - block_dim: number of blocks (1 block = 1 AIC + 2 AIV)
 *            - graphArgs: task graph to execute (includes handshake buffers in workers[])
 * @return 0 on success, non-zero on error
 */
extern "C" __attribute__((visibility("default"))) int DynTileFwkBackendKernelServer(void *arg) {
    if (arg == nullptr) {
        DEV_ERROR("%s", "Invalid kernel arguments: null pointer");
        return -1;
    }

    DEV_INFO("%s", "DynTileFwkBackendKernelServer: Calling AicpuExecute");
    int rc = AicpuExecute(arg);
    if (rc != 0) {
        DEV_ERROR("DynTileFwkBackendKernelServer: AicpuExecute failed with rc=%d", rc);
        return rc;
    }
    DEV_INFO("%s", "DynTileFwkBackendKernelServer: AicpuExecute completed successfully");
    
    return rc;
}
