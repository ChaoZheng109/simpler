/**
 * PTO Runtime C API - Implementation
 *
 * Wraps C++ classes as opaque pointers, providing C interface for ctypes
 * bindings. Simplified single-concept model: Runtime only.
 */

#include "pto_runtime_c_api.h"

#include <iostream>
#include <new>  // for placement new
#include <vector>

#include "device_runner.h"
#include "runtime.h"

extern "C" {

/* ===========================================================================
 */
/* Runtime Implementation Functions (defined in runtimemaker.cpp) */
/* ===========================================================================
 */
int init_runtime_impl(Runtime* runtime,
                    const uint8_t* orch_so_binary,
                    size_t orch_so_size,
                    const char* orch_func_name,
                    uint64_t* func_args,
                    int func_args_count);
int validate_runtime_impl(Runtime* runtime);

/* Forward declarations for device memory functions used in init_runtime */
void* device_malloc(size_t size);
void device_free(void* dev_ptr);
int copy_to_device(void* dev_ptr, const void* host_ptr, size_t size);
int copy_from_device(void* host_ptr, const void* dev_ptr, size_t size);

/* ===========================================================================
 */
/* Runtime API Implementation */
/* ===========================================================================
 */

size_t get_runtime_size(void) { return sizeof(Runtime); }

int init_runtime(RuntimeHandle runtime,
                const uint8_t* orch_so_binary,
                size_t orch_so_size,
                const char* orch_func_name,
                uint64_t* func_args,
                int func_args_count) {
    if (runtime == NULL) {
        return -1;
    }
    if (orch_so_binary == NULL || orch_so_size == 0 || orch_func_name == NULL) {
        std::cerr << "Error: Invalid orchestration parameters\n";
        return -1;
    }

    try {
        // Placement new to construct Runtime in user-allocated memory
        Runtime* r = new (runtime) Runtime();

        // Initialize host API function pointers (host-only, not available on device)
        r->host_api.device_malloc = device_malloc;
        r->host_api.device_free = device_free;
        r->host_api.copy_to_device = copy_to_device;
        r->host_api.copy_from_device = copy_from_device;

        // Delegate SO loading and orchestration to init_runtime_impl
        return init_runtime_impl(r, orch_so_binary, orch_so_size,
                               orch_func_name, func_args, func_args_count);
    } catch (...) {
        return -1;
    }
}

/* ===========================================================================
 */
/* Device Memory API Implementation */
/* ===========================================================================
 */

void* device_malloc(size_t size) {
    try {
        DeviceRunner& runner = DeviceRunner::get();
        return runner.allocate_tensor(size);
    } catch (...) {
        return NULL;
    }
}

void device_free(void* dev_ptr) {
    if (dev_ptr == NULL) {
        return;
    }
    try {
        DeviceRunner& runner = DeviceRunner::get();
        runner.free_tensor(dev_ptr);
    } catch (...) {
        // Ignore errors during free
    }
}

int copy_to_device(void* dev_ptr, const void* host_ptr, size_t size) {
    if (dev_ptr == NULL || host_ptr == NULL) {
        return -1;
    }
    try {
        DeviceRunner& runner = DeviceRunner::get();
        return runner.copy_to_device(dev_ptr, host_ptr, size);
    } catch (...) {
        return -1;
    }
}

int copy_from_device(void* host_ptr, const void* dev_ptr, size_t size) {
    if (host_ptr == NULL || dev_ptr == NULL) {
        return -1;
    }
    try {
        DeviceRunner& runner = DeviceRunner::get();
        return runner.copy_from_device(host_ptr, dev_ptr, size);
    } catch (...) {
        return -1;
    }
}

int launch_runtime(RuntimeHandle runtime,
    int aicpu_thread_num,
    int block_dim,
    int device_id,
    const uint8_t* aicpu_binary,
    size_t aicpu_size,
    const uint8_t* aicore_binary,
    size_t aicore_size) {
    if (runtime == NULL) {
        return -1;
    }
    if (aicpu_binary == NULL || aicpu_size == 0 || aicore_binary == NULL || aicore_size == 0) {
        return -1;
    }
    try {
        DeviceRunner& runner = DeviceRunner::get();

        // Convert to vectors for run()
        std::vector<uint8_t> aicpu_vec(aicpu_binary, aicpu_binary + aicpu_size);
        std::vector<uint8_t> aicore_vec(aicore_binary, aicore_binary + aicore_size);

        // Run the runtime (device initialization is handled internally)
        Runtime* r = static_cast<Runtime*>(runtime);
        return runner.run(*r, block_dim, device_id, aicpu_vec, aicore_vec, aicpu_thread_num);
    } catch (...) {
        return -1;
    }
}

int finalize_runtime(RuntimeHandle runtime) {
    if (runtime == NULL) {
        return -1;
    }
    try {
        Runtime* r = static_cast<Runtime*>(runtime);
        int rc = validate_runtime_impl(r);
        // Call destructor (user will call free())
        r->~Runtime();
        return rc;
    } catch (...) {
        return -1;
    }
}

int set_device(int device_id) {
    try {
        DeviceRunner& runner = DeviceRunner::get();
        return runner.ensure_device_set(device_id);
    } catch (...) {
        return -1;
    }
}

int register_kernel(int func_id, const uint8_t* bin_data, size_t bin_size) {
    if (bin_data == NULL || bin_size == 0) {
        return -1;
    }
    try {
        DeviceRunner& runner = DeviceRunner::get();
        return runner.register_kernel(func_id, bin_data, bin_size);
    } catch (...) {
        return -1;
    }
}

} /* extern "C" */
