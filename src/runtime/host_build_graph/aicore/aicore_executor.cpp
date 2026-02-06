#include "aicore/aicore.h"
#include "runtime.h"
#include "common/perf_profiling.h"
#include "common/memory_barrier.h"

/**
 * Unified function pointer type for kernel dispatch
 *
 * All kernels follow the same signature: void kernel(__gm__ int64_t* args)
 * This enables simple, switch-free dispatch.
 */
typedef void (*UnifiedKernelFunc)(__gm__ int64_t*);

/**
 * Task execution wrapper - dispatches tasks using function pointers
 *
 * This function demonstrates the runtime function pointer dispatch pattern.
 * Following the production system flow:
 * - function_bin_addr points to compiled kernel code in device GM memory
 * - The address is cast to a function pointer: UnifiedKernelFunc kernel =
 * (UnifiedKernelFunc)function_bin_addr
 * - The kernel is invoked: kernel(task->args)
 *
 * This is the KEY difference from compile-time linking:
 * - OLD: extern "C" declarations, resolved at link time
 * - NEW: function_bin_addr from GM memory, cast at runtime
 *
 * With unified kernel signature, no switch statement is needed.
 * All kernels unpack their own arguments from the args array.
 *
 * @param task Pointer to task in global memory (null during initialization)
 */
__aicore__ __attribute__((always_inline)) static void execute_task(__gm__ Task* task) {
    // Null task pointer indicates no work assigned (initialization state)
    if (task == nullptr) {
        return;
    }

    // Check for valid function_bin_addr
    if (task->function_bin_addr == 0) {
        // Invalid address - skip execution
        return;
    }

    // Cast function_bin_addr to unified function pointer and invoke
    // All kernels have signature: void kernel(__gm__ int64_t* args)
    UnifiedKernelFunc kernel = (UnifiedKernelFunc)task->function_bin_addr;
    kernel(reinterpret_cast<__gm__ int64_t*>(task->args));
}

__aicore__ __attribute__((weak)) void aicore_execute(__gm__ Runtime* runtime, int block_idx, CoreType core_type) {
    __gm__ Handshake* my_hank = (__gm__ Handshake*)(&runtime->workers[block_idx]);

    // Phase 1: Wait for AICPU initialization signal
    while (my_hank->aicpu_ready == 0) {
        dcci(my_hank, ENTIRE_DATA_CACHE, CACHELINE_OUT);
    }

    // Phase 2: Signal AICore is ready and report core type
    my_hank->core_type = core_type;        // Report core type to AICPU
    my_hank->aicore_done = block_idx + 1;  // Signal ready (use block_idx + 1 to avoid 0)

    // Check if profiling is enabled
    bool profiling_enabled = runtime->enable_profiling;

    // Phase 3: Main execution loop - poll for tasks until quit signal
    while (true) {
        dcci(my_hank, ENTIRE_DATA_CACHE, CACHELINE_OUT);

        // Check for quit command from AICPU
        if (my_hank->control == 1) {
            break;  // Exit kernel
        }

        // Execute task if assigned (task != 0 means valid Task* pointer)
        if (my_hank->task_status == 1 && my_hank->task != 0) {
            __gm__ Task* task_ptr = reinterpret_cast<__gm__ Task*>(my_hank->task);

            // Performance profiling: record start time
            uint64_t start_time = 0;
            if (profiling_enabled) {
                start_time = get_sys_cnt();
            }

            // Execute the task
            execute_task(task_ptr);

            // Performance profiling: record task execution
            if (profiling_enabled && my_hank->perf_buffer_status == 0) {
                uint64_t end_time = get_sys_cnt();

                // Get current performance buffer pointer
                __gm__ PerfBuffer* perf_buf = (__gm__ PerfBuffer*)my_hank->perf_records_addr;

                // Get current count (no atomic operation needed - single writer)
                uint32_t idx = perf_buf->count;

                // Check if buffer has space
                if (idx < PLATFORM_PROF_BUFFER_SIZE) {
                    // Get pointer to the record slot
                    __gm__ PerfRecord* record = (__gm__ PerfRecord*)&perf_buf->records[idx];

                    // Write record data
                    record->start_time = start_time;
                    record->end_time = end_time;
                    record->duration = end_time - start_time;
                    record->task_id = task_ptr->task_id;
                    record->func_id = task_ptr->func_id;
                    record->core_id = block_idx;
                    record->core_type = core_type;
                    record->fanout_count = task_ptr->fanout_count;

                    // Copy fanout array
                    for (int32_t i = 0; i < task_ptr->fanout_count && i < RUNTIME_MAX_FANOUT; i++) {
                        record->fanout[i] = task_ptr->fanout[i];
                    }

                    // Record first task time if this is the first record
                    if (idx == 0) {
                        perf_buf->first_task_time = start_time;
                    }

                    // Increment count after writing record
                    perf_buf->count = idx + 1;

                    // Write memory barrier: ensure performance data is visible to Host
                    wmb();

                    // Check if buffer is full after this write
                    if (perf_buf->count >= PLATFORM_PROF_BUFFER_SIZE) {
                        my_hank->perf_buffer_status = 1;  // Notify AICPU: buffer full
                    }
                } else {
                    // Buffer is already full
                    my_hank->perf_buffer_status = 1;
                }
            }

            // Mark task as complete (task_status: 0=idle, 1=busy)
            my_hank->task_status = 0;
        }
    }
}
