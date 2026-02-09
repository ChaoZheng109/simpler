/**
 * Device Runner Implementation - Thread-Based Simulation
 *
 * This file implements the simulated device execution using host threads.
 * It provides the same API as the real a2a3 implementation but uses
 * std::thread instead of CANN runtime APIs.
 *
 * aicpu_execute and aicore_execute_wrapper are loaded dynamically via dlopen from
 * the binaries passed to launch_runtime.
 *
 * Cross-platform notes:
 * - Linux: Uses MAP_ANONYMOUS for anonymous memory mapping
 * - macOS: Uses MAP_ANON (aliased) and MAP_JIT for executable memory on Apple Silicon
 *   which requires W^X (write xor execute) protection toggling via pthread_jit_write_protect_np
 */

#include "device_runner.h"

#include <optional>
#include <set>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <ctime>

// Function pointer types for dynamically loaded executors
typedef int (*aicpu_execute_func_t)(Runtime* runtime);
typedef void (*aicore_execute_func_t)(Runtime* runtime, int block_idx, CoreType core_type);

// =============================================================================
// DeviceRunner Implementation
// =============================================================================

DeviceRunner& DeviceRunner::get() {
    static DeviceRunner runner;
    return runner;
}

DeviceRunner::~DeviceRunner() {
    finalize();
}

int DeviceRunner::ensure_device_initialized(int device_id,
                                            const std::vector<uint8_t>& aicpu_so_binary,
                                            const std::vector<uint8_t>& aicore_kernel_binary) {
    device_id_ = device_id;
    return ensure_binaries_loaded(aicpu_so_binary, aicore_kernel_binary);
}

int DeviceRunner::ensure_binaries_loaded(const std::vector<uint8_t>& aicpu_so_binary,
                                         const std::vector<uint8_t>& aicore_kernel_binary) {
    // Skip if already loaded
    if (aicpu_execute_func_ != nullptr && aicore_execute_func_ != nullptr) {
        return 0;
    }

    // Write AICPU binary to temp file and dlopen
    if (!aicpu_so_binary.empty() && aicpu_execute_func_ == nullptr) {
        aicpu_so_path_ = "/tmp/aicpu_sim_" + std::to_string(getpid()) + ".so";
        std::ofstream ofs(aicpu_so_path_, std::ios::binary);
        if (!ofs) {
            LOG_ERROR("Failed to create temp file for AICPU SO: %s", aicpu_so_path_.c_str());
            return -1;
        }
        ofs.write(reinterpret_cast<const char*>(aicpu_so_binary.data()), aicpu_so_binary.size());
        ofs.close();

        aicpu_so_handle_ = dlopen(aicpu_so_path_.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (aicpu_so_handle_ == nullptr) {
            LOG_ERROR("dlopen failed for AICPU SO: %s", dlerror());
            return -1;
        }

        aicpu_execute_func_ = reinterpret_cast<int(*)(Runtime*)>(dlsym(aicpu_so_handle_, "aicpu_execute"));
        if (aicpu_execute_func_ == nullptr) {
            LOG_ERROR("dlsym failed for aicpu_execute: %s", dlerror());
            return -1;
        }
        LOG_INFO("DeviceRunner(sim): Loaded aicpu_execute from %s", aicpu_so_path_.c_str());
    }

    // Write AICore binary to temp file and dlopen
    if (!aicore_kernel_binary.empty() && aicore_execute_func_ == nullptr) {
        aicore_so_path_ = "/tmp/aicore_sim_" + std::to_string(getpid()) + ".so";
        std::ofstream ofs(aicore_so_path_, std::ios::binary);
        if (!ofs) {
            LOG_ERROR("Failed to create temp file for AICore SO: %s", aicore_so_path_.c_str());
            return -1;
        }
        ofs.write(reinterpret_cast<const char*>(aicore_kernel_binary.data()), aicore_kernel_binary.size());
        ofs.close();

        aicore_so_handle_ = dlopen(aicore_so_path_.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (aicore_so_handle_ == nullptr) {
            LOG_ERROR("dlopen failed for AICore SO: %s", dlerror());
            return -1;
        }

        aicore_execute_func_ = reinterpret_cast<void(*)(Runtime*, int, CoreType)>(dlsym(aicore_so_handle_, "aicore_execute_wrapper"));
        if (aicore_execute_func_ == nullptr) {
            LOG_ERROR("dlsym failed for aicore_execute_wrapper: %s", dlerror());
            return -1;
        }
        LOG_INFO("DeviceRunner(sim): Loaded aicore_execute_wrapper from %s", aicore_so_path_.c_str());
    }

    return 0;
}

void* DeviceRunner::allocate_tensor(size_t bytes) {
    return mem_alloc_.alloc(bytes);
}

void DeviceRunner::free_tensor(void* dev_ptr) {
    if (dev_ptr != nullptr) {
        mem_alloc_.free(dev_ptr);
    }
}

int DeviceRunner::copy_to_device(void* dev_ptr, const void* host_ptr, size_t bytes) {
    // In simulation, this is just a memcpy
    std::memcpy(dev_ptr, host_ptr, bytes);
    return 0;
}

int DeviceRunner::copy_from_device(void* host_ptr, const void* dev_ptr, size_t bytes) {
    // In simulation, this is just a memcpy
    std::memcpy(host_ptr, dev_ptr, bytes);
    return 0;
}

int DeviceRunner::run(Runtime& runtime,
                      int block_dim,
                      int device_id,
                      const std::vector<uint8_t>& aicpu_so_binary,
                      const std::vector<uint8_t>& aicore_kernel_binary,
                      int launch_aicpu_num) {

    // Validate launch_aicpu_num
    if (launch_aicpu_num < 1 || launch_aicpu_num > PLATFORM_MAX_AICPU_THREADS) {
        LOG_ERROR("launch_aicpu_num (%d) must be in range [1, %d]", 
                       launch_aicpu_num, PLATFORM_MAX_AICPU_THREADS);
        return -1;
    }

    // Validate block_dim
    if (block_dim < 1 || block_dim > PLATFORM_MAX_BLOCKDIM) {
        LOG_ERROR("block_dim (%d) must be in range [1, %d]", 
                       block_dim, PLATFORM_MAX_BLOCKDIM);
        return -1;
    }

    // Validate even distribution: block_dim must be divisible by scheduler thread count
    // When launch_aicpu_num == 4: 3 schedulers + 1 orchestrator (thread 3 has 0 cores)
    int scheduler_thread_num = (launch_aicpu_num == 4) ? 3 : launch_aicpu_num;
    if (block_dim % scheduler_thread_num != 0) {
        LOG_ERROR("block_dim (%d) must be evenly divisible by scheduler_thread_num (%d)",
                       block_dim, scheduler_thread_num);
        return -1;
    }

    // Ensure device is initialized
    int rc = ensure_device_initialized(device_id, aicpu_so_binary, aicore_kernel_binary);
    if (rc != 0) {
        LOG_ERROR("ensure_device_initialized failed: %d", rc);
        return rc;
    }

    // Calculate execution parameters
    block_dim_ = block_dim;
    int num_aicore = block_dim * cores_per_blockdim_;

    if (num_aicore > RUNTIME_MAX_WORKER) {
        LOG_ERROR("num_aicore (%d) exceeds RUNTIME_MAX_WORKER (%d)",
                       num_aicore, RUNTIME_MAX_WORKER);
        return -1;
    }

    // Initialize handshake buffers
    runtime.worker_count = num_aicore;
    worker_count_ = num_aicore;
    runtime.sche_cpu_num = launch_aicpu_num;

    // Calculate number of AIC cores
    int num_aic = block_dim;

    for (int i = 0; i < num_aicore; i++) {
        runtime.workers[i].aicpu_ready = 0;
        runtime.workers[i].aicore_done = 0;
        runtime.workers[i].control = 0;
        runtime.workers[i].task = 0;
        runtime.workers[i].task_status = 0;
        // First 1/3 are AIC, remaining 2/3 are AIV
        runtime.workers[i].core_type = (i < num_aic) ? CoreType::AIC : CoreType::AIV;
    }

    // Set function_bin_addr for each task from Runtime's func_id_to_addr_[] array
    // (addresses were stored there during init_runtime via upload_kernel_binary)
    LOG_DEBUG("Setting function_bin_addr for Tasks (Simulation)");
    for (int i = 0; i < runtime.get_task_count(); i++) {
        Task* task = runtime.get_task(i);
        if (task != nullptr) {
            uint64_t addr = runtime.get_function_bin_addr(task->func_id);
            task->function_bin_addr = addr;
            LOG_DEBUG("Task %d (func_id=%d) -> function_bin_addr=0x%lx",
                          i, task->func_id, addr);
        }
    }

    // Store runtime pointer for print_handshake_results
    last_runtime_ = &runtime;

    // Initialize performance profiling if enabled
    if (runtime.enable_profiling) {
        rc = init_performance_profiling(runtime, num_aicore, device_id);
        if (rc != 0) {
            LOG_ERROR("init_performance_profiling failed: %d", rc);
            return rc;
        }
    }

    // Check if executors are loaded
    if (aicpu_execute_func_ == nullptr || aicore_execute_func_ == nullptr) {
        LOG_ERROR("Executor functions not loaded. Call ensure_binaries_loaded first.");
        return -1;
    }

    // Launch AICPU threads
    LOG_INFO("Launching %d AICPU thread(s)", launch_aicpu_num);
    std::vector<std::thread> aicpu_threads;
    for (int i = 0; i < launch_aicpu_num; i++) {
        aicpu_threads.emplace_back([this, &runtime]() {
            aicpu_execute_func_(&runtime);
        });
    }

    // Launch AICore threads
    LOG_INFO("Launching %d AICore thread(s)", num_aicore);
    std::vector<std::thread> aicore_threads;
    for (int i = 0; i < num_aicore; i++) {
        CoreType core_type = runtime.workers[i].core_type;
        aicore_threads.emplace_back([this, &runtime, i, core_type]() {
            aicore_execute_func_(&runtime, i, core_type);
        });
    }

    // Poll and collect performance data during execution (if enabled)
    std::thread collector_thread;
    if (runtime.enable_profiling) {
        collector_thread = std::thread([this, &runtime, num_aicore]() {
            poll_and_collect_performance_data(num_aicore, runtime.get_task_count());
        });
    }

    // Wait for all threads to complete
    LOG_INFO("Waiting for threads to complete");
    for (auto& t : aicpu_threads) {
        t.join();
    }
    for (auto& t : aicore_threads) {
        t.join();
    }

    // Wait for collector thread if it was launched
    if (runtime.enable_profiling && collector_thread.joinable()) {
        collector_thread.join();
    }

    LOG_INFO("All threads completed");

    // Print performance data after execution completes
    if (runtime.enable_profiling) {
        print_performance_data();
        export_swimlane_json();
    }

    return 0;
}

void DeviceRunner::print_handshake_results() {
    if (worker_count_ == 0 || last_runtime_ == nullptr) {
        return;
    }

    LOG_DEBUG("Handshake results for %d cores:", worker_count_);
    for (int i = 0; i < worker_count_; i++) {
        LOG_DEBUG("  Core %d: aicore_done=%d aicpu_ready=%d control=%d task=%d",
                      i,
                      last_runtime_->workers[i].aicore_done,
                      last_runtime_->workers[i].aicpu_ready,
                      last_runtime_->workers[i].control,
                      last_runtime_->workers[i].task);
    }
}

int DeviceRunner::finalize() {
    // Skip if already finalized
    if (device_id_ == -1 && aicpu_so_handle_ == nullptr && aicore_so_handle_ == nullptr) {
        return 0;
    }

    // Print handshake results before cleanup
    print_handshake_results();

    // Cleanup performance profiling resources (inline, matching a2a3 style)
    if (perf_shared_mem_dev_ != nullptr) {
        free(perf_shared_mem_dev_);
        perf_shared_mem_dev_ = nullptr;
        perf_shared_mem_host_ = nullptr;
    }
    collected_perf_records_.clear();

    // Close all dlopen'd kernel libraries
    for (auto& pair : func_id_to_addr_) {
        MappedKernel& kernel = pair.second;
        if (kernel.dl_handle != nullptr) {
            dlclose(kernel.dl_handle);
            LOG_DEBUG("Closed dlopen kernel: func_id=%d", pair.first);
            kernel.dl_handle = nullptr;
            kernel.func_addr = 0;
        }
    }
    func_id_to_addr_.clear();

    // Close dynamically loaded libraries and remove temp files
    if (aicpu_so_handle_ != nullptr) {
        dlclose(aicpu_so_handle_);
        aicpu_so_handle_ = nullptr;
        aicpu_execute_func_ = nullptr;
    }
    if (!aicpu_so_path_.empty()) {
        std::remove(aicpu_so_path_.c_str());
        aicpu_so_path_.clear();
    }

    if (aicore_so_handle_ != nullptr) {
        dlclose(aicore_so_handle_);
        aicore_so_handle_ = nullptr;
        aicore_execute_func_ = nullptr;
    }
    if (!aicore_so_path_.empty()) {
        std::remove(aicore_so_path_.c_str());
        aicore_so_path_.clear();
    }

    // Free all remaining allocations
    mem_alloc_.finalize();

    device_id_ = -1;
    worker_count_ = 0;
    last_runtime_ = nullptr;

    LOG_INFO("DeviceRunner(sim) finalized");
    return 0;
}

// =============================================================================
// Kernel Binary Upload (returns function address for caller to store in Runtime)
// =============================================================================

uint64_t DeviceRunner::upload_kernel_binary(int func_id, const uint8_t* bin_data, size_t bin_size) {
    if (bin_data == nullptr || bin_size == 0) {
        LOG_ERROR("Invalid kernel data");
        return 0;
    }

    // Return cached address if already uploaded
    auto it = func_id_to_addr_.find(func_id);
    if (it != func_id_to_addr_.end()) {
        LOG_INFO("Kernel func_id=%d already uploaded, returning cached address", func_id);
        return it->second.func_addr;
    }

    // 1. Generate temp file path
    char tmpfile[256];
    snprintf(tmpfile, sizeof(tmpfile), "/tmp/kernel_%d_%d.so", func_id, getpid());

    // 2. Write to temp file
    std::ofstream ofs(tmpfile, std::ios::binary);
    if (!ofs) {
        LOG_ERROR("Failed to create temp file: %s", tmpfile);
        return 0;
    }
    ofs.write(reinterpret_cast<const char*>(bin_data), bin_size);
    ofs.close();

    LOG_DEBUG("Uploading kernel .so: %s (size=%zu bytes)", tmpfile, bin_size);

    // 3. dlopen to load .so (RTLD_NOW ensures all symbols resolved immediately)
    void* handle = dlopen(tmpfile, RTLD_NOW | RTLD_LOCAL);

    // 4. Remove temp file immediately (.so is already in memory)
    std::remove(tmpfile);

    if (!handle) {
        LOG_ERROR("dlopen failed: %s", dlerror());
        return 0;
    }

    // 5. dlsym to get kernel function address (unified entry point: "kernel_entry")
    void* func = dlsym(handle, "kernel_entry");
    if (!func) {
        LOG_ERROR("dlsym failed for 'kernel_entry': %s", dlerror());
        dlclose(handle);
        return 0;
    }

    // 6. Store mapping info for cleanup
    MappedKernel kernel;
    kernel.dl_handle = handle;
    kernel.func_addr = reinterpret_cast<uint64_t>(func);

    func_id_to_addr_[func_id] = kernel;

    LOG_DEBUG("Registered kernel (dlopen): func_id=%d -> addr=0x%lx, handle=%p",
                  func_id, kernel.func_addr, handle);

    return kernel.func_addr;
}

// =============================================================================
// Performance Profiling Implementation
// =============================================================================

int DeviceRunner::init_performance_profiling(Runtime& runtime, int num_aicore, int device_id) {
    (void)device_id;  // Unused in simulation

    LOG_INFO("Initializing Performance Profiling");

    // Calculate total memory size (header + all DoubleBuffers)
    size_t total_size = calc_perf_data_size(num_aicore);

    size_t header_size = sizeof(PerfDataHeader);
    size_t single_db_size = sizeof(DoubleBuffer);
    size_t buffers_size = num_aicore * single_db_size;

    LOG_DEBUG("Memory allocation plan:");
    LOG_DEBUG("Number of cores: %d", num_aicore);
    LOG_DEBUG("Header size: %zu bytes (includes ready queue: %d entries)", header_size, PLATFORM_PROF_READYQUEUE_SIZE);
    LOG_DEBUG("Single DoubleBuffer: %zu bytes", single_db_size);
    LOG_DEBUG("All DoubleBuffers: %zu bytes", buffers_size);
    LOG_DEBUG("Total size: %zu bytes (%zu KB, %zu MB)",
             total_size, total_size / 1024, total_size / (1024 * 1024));

    // Allocate device shared memory (simulation: use malloc for host memory)
    void* perf_dev_ptr = malloc(total_size);
    if (perf_dev_ptr == nullptr) {
        LOG_ERROR("Failed to allocate device memory for profiling (%zu bytes)", total_size);
        return -1;
    }
    LOG_DEBUG("Allocated device memory: %p", perf_dev_ptr);

    // Register to host mapping (simulation: both pointers point to same memory)
    void* perf_host_ptr = perf_dev_ptr;  // In simulation, both point to same memory
    LOG_DEBUG("Mapped to host memory: %p", perf_host_ptr);

    // Initialize fixed header (using host_ptr)
    PerfDataHeader* header = get_perf_header(perf_host_ptr);

    // Initialize queue
    memset(header->queue, 0, sizeof(header->queue));
    header->queue_head = 0;
    header->queue_tail = 0;

    // Initialize metadata
    header->num_cores = num_aicore;

    LOG_DEBUG("Initialized PerfDataHeader:");
    LOG_DEBUG("num_aicore: %d", header->num_cores);
    LOG_DEBUG("buffer_capacity: %d", PLATFORM_PROF_BUFFER_SIZE);
    LOG_DEBUG("queue capacity: %d", PLATFORM_PROF_READYQUEUE_SIZE);

    // Initialize all DoubleBuffers (all buffers start as 0=idle)
    DoubleBuffer* buffers = get_double_buffers(perf_host_ptr);

    for (int i = 0; i < num_aicore; i++) {
        DoubleBuffer* db = &buffers[i];

        // Initialize buffer1
        memset(&db->buffer1, 0, sizeof(PerfBuffer));
        db->buffer1.count = 0;
        db->buffer1_status = BufferStatus::IDLE;

        // Initialize buffer2
        memset(&db->buffer2, 0, sizeof(PerfBuffer));
        db->buffer2.count = 0;
        db->buffer2_status = BufferStatus::IDLE;
    }

    LOG_DEBUG("Initialized %d DoubleBuffers (all status=0, idle)", num_aicore);

    // Write memory barrier (ensure all initialization visible to workers)
    wmb();

    // Pass to Runtime (device base address)
    runtime.perf_data_base = (uint64_t)perf_dev_ptr;

    LOG_DEBUG("Set runtime.perf_data_base = 0x%lx", runtime.perf_data_base);

    // Save pointers to member variables
    perf_shared_mem_dev_ = perf_dev_ptr;
    perf_shared_mem_host_ = perf_host_ptr;

    LOG_INFO("Performance Profiling Initialized");

    return 0;
}

void DeviceRunner::poll_and_collect_performance_data(int num_aicore, int expected_tasks) {
    if (perf_shared_mem_host_ == nullptr) {
        return;  // Profiling not enabled
    }

    LOG_INFO("Collecting Performance Data (Before Stream Sync)");
    LOG_INFO("Expected tasks: %d", expected_tasks);

    PerfDataHeader* header = get_perf_header(perf_shared_mem_host_);
    DoubleBuffer* buffers = get_double_buffers(perf_shared_mem_host_);

    uint32_t capacity = PLATFORM_PROF_READYQUEUE_SIZE;
    int total_records_collected = 0;
    int buffers_processed = 0;

    // Clear previous collection
    collected_perf_records_.clear();

    // Timeout configuration: track continuous idle time
    const auto timeout_duration = std::chrono::seconds(PLATFORM_PROF_TIMEOUT_SECONDS);
    std::optional<std::chrono::steady_clock::time_point> idle_start;
    int empty_poll_count = 0;

    // Poll the ready queue until all expected tasks are collected
    while (total_records_collected < expected_tasks) {
        // Read queue status with memory barrier
        rmb();
        uint32_t head = header->queue_head;
        uint32_t tail = header->queue_tail;

        // Check if queue is empty
        if (head == tail) {
            // Queue is empty but we haven't collected all tasks yet
            // Record idle start time on first empty poll
            if (!idle_start.has_value()) {
                idle_start = std::chrono::steady_clock::now();
            }

            // Check for timeout periodically
            empty_poll_count++;
            if (empty_poll_count >= PLATFORM_PROF_EMPTY_POLLS_CHECK_NUM) {
                empty_poll_count = 0;
                auto elapsed = std::chrono::steady_clock::now() - idle_start.value();
                if (elapsed >= timeout_duration) {
                    LOG_WARN("Performance data collection idle timeout after %ld seconds",
                             std::chrono::duration_cast<std::chrono::seconds>(elapsed).count());
                    LOG_WARN("Collected %d / %d records before timeout",
                             total_records_collected, expected_tasks);
                    break;  // Exit with partial data
                }
            }
            // Continue polling (AICPU may still be producing data)
            continue;
        }

        // Reset idle tracking when we find data
        idle_start.reset();
        empty_poll_count = 0;

        // Dequeue entry
        ReadyQueueEntry entry = header->queue[head];
        uint32_t core_index = entry.core_index;
        uint32_t buffer_id = entry.buffer_id;

        // Validate core index
        if (core_index >= static_cast<uint32_t>(num_aicore)) {
            LOG_ERROR("Invalid core_index %u (max=%d)", core_index, num_aicore);
            break;
        }

        LOG_DEBUG("Processing: core=%u, buffer=%u", core_index, buffer_id);

        // Get the buffer and status pointer
        DoubleBuffer* db = &buffers[core_index];
        PerfBuffer* buf = nullptr;
        volatile BufferStatus* status = nullptr;
        get_buffer_and_status(db, buffer_id, &buf, &status);

        // Read buffer data with memory barrier
        rmb();
        uint32_t count = buf->count;

        LOG_DEBUG("Records in buffer: %u", count);

        // Collect records
        for (uint32_t i = 0; i < count && i < PLATFORM_PROF_BUFFER_SIZE; i++) {
            collected_perf_records_.push_back(buf->records[i]);
            total_records_collected++;
        }

        // Clear buffer
        buf->count = 0;

        // Set buffer status to IDLE
        *status = BufferStatus::IDLE;
        wmb();  // Ensure status is visible to AICPU

        // Update queue head
        header->queue_head = (head + 1) % capacity;
        wmb();  // Ensure head update is visible to AICPU

        buffers_processed++;
    }

    LOG_INFO("Total buffers processed: %d", buffers_processed);
    LOG_INFO("Total records collected: %d", total_records_collected);

    if (total_records_collected < expected_tasks) {
        LOG_WARN("Incomplete collection (%d / %d records)",
                 total_records_collected, expected_tasks);
    }

    LOG_INFO("Performance Data Collection Complete");
}

void DeviceRunner::print_performance_data() {
    if (collected_perf_records_.empty()) {
        LOG_INFO("No Performance Data to Print");
        return;
    }

    LOG_INFO("Performance Records Detail");

    // Use minimum kernel_ready_time as base time for normalization (same as export_swimlane_json)
    // Each AICore has different kernel_ready_time, use minimum for proper alignment
    uint64_t base_time = UINT64_MAX;
    for (const auto& record : collected_perf_records_) {
        if (record.kernel_ready_time < base_time) {
            base_time = record.kernel_ready_time;
        }
    }

    // Convert cycles to microseconds
    auto cycles_to_us = [base_time](uint64_t cycles) -> double {
        uint64_t normalized_cycles = cycles - base_time;
        return (static_cast<double>(normalized_cycles) / PLATFORM_PROF_SYS_CNT_FREQ) * 1000000.0;
    };

    // Print detailed records only in DEBUG mode
    LOG_DEBUG("Base time (kernel_ready_time): %lu cycles", base_time);
    LOG_DEBUG("Task execution records:");
    LOG_DEBUG("┌────────┬─────────┬─────────┬────────────┬──────────────────┬──────────────────┬──────────────┬──────────┐");
    LOG_DEBUG("│ Task ID│ Func ID │ Core ID │ Core Type  │    Start (us)    │     End (us)     │Duration (us) │  Fanout  │");
    LOG_DEBUG("├────────┼─────────┼─────────┼────────────┼──────────────────┼──────────────────┼──────────────┼──────────┤");

    for (size_t i = 0; i < collected_perf_records_.size() && i < 10; i++) {  // Limit to first 10 for display
        const PerfRecord& record = collected_perf_records_[i];

        // Convert times to microseconds
        double start_us = cycles_to_us(record.start_time);
        double end_us = cycles_to_us(record.end_time);
        // Calculate duration from start_time and end_time (not using record.duration)
        uint64_t duration_cycles = record.end_time - record.start_time;
        double duration_us = (static_cast<double>(duration_cycles) / PLATFORM_PROF_SYS_CNT_FREQ) * 1000000.0;

        char line_buf[256];
        snprintf(line_buf, sizeof(line_buf),
                 "│ %6u │ %7u │ %7u │ %10s │ %16.3f │ %16.3f │ %12.3f │ %8u │",
                 record.task_id, record.func_id, record.core_id,
                 (record.core_type == CoreType::AIC ? "AIC" : "AIV"),
                 start_us, end_us, duration_us, record.fanout_count);
        LOG_DEBUG("%s", line_buf);
    }

    LOG_DEBUG("└────────┴─────────┴─────────┴────────────┴──────────────────┴──────────────────┴──────────────┴──────────┘");

    if (collected_perf_records_.size() > 10) {
        LOG_DEBUG("... (%zu more records not shown)", collected_perf_records_.size() - 10);
    }

    // Calculate statistics (compute duration from start_time and end_time)
    uint64_t total_duration = 0;
    uint64_t max_duration = 0;
    uint64_t min_duration = UINT64_MAX;

    for (const auto& record : collected_perf_records_) {
        uint64_t duration = record.end_time - record.start_time;
        total_duration += duration;
        if (duration > max_duration) max_duration = duration;
        if (duration < min_duration) min_duration = duration;
    }

    double avg_duration = static_cast<double>(total_duration) / collected_perf_records_.size();

    // Convert durations to microseconds for statistics
    double avg_duration_us = (avg_duration / PLATFORM_PROF_SYS_CNT_FREQ) * 1000000.0;
    double min_duration_us = (static_cast<double>(min_duration) / PLATFORM_PROF_SYS_CNT_FREQ) * 1000000.0;
    double max_duration_us = (static_cast<double>(max_duration) / PLATFORM_PROF_SYS_CNT_FREQ) * 1000000.0;
    double total_duration_us = (static_cast<double>(total_duration) / PLATFORM_PROF_SYS_CNT_FREQ) * 1000000.0;

    LOG_INFO("Performance Statistics:");
    LOG_INFO("Total tasks: %zu", collected_perf_records_.size());
    LOG_INFO("Avg duration: %.3f us (%lu cycles)", avg_duration_us, static_cast<uint64_t>(avg_duration));
    LOG_INFO("Min duration: %.3f us (%lu cycles)", min_duration_us, min_duration);
    LOG_INFO("Max duration: %.3f us (%lu cycles)", max_duration_us, max_duration);
    LOG_INFO("Total duration: %.3f us (%lu cycles)", total_duration_us, total_duration);

    LOG_INFO("Performance Data Print Complete");
}

int DeviceRunner::export_swimlane_json(const std::string& output_path) {
    if (collected_perf_records_.empty()) {
        LOG_WARN("Warning: No performance data to export.");
        return -1;
    }

    // Step 1: Create output directory if it doesn't exist
    struct stat st;
    if (stat(output_path.c_str(), &st) == -1) {
        if (mkdir(output_path.c_str(), 0755) != 0) {
            LOG_ERROR("Error: Failed to create output directory. ");
            return -1;
        }
    }

    // Step 2: Generate timestamp for filename
    time_t now = time(nullptr);
    struct tm* timeinfo = localtime(&now);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", timeinfo);

    // Step 3: Open output file with timestamp
    std::string filepath = output_path + "/merged_swimlane_" + timestamp + ".json";
    std::ofstream outfile(filepath);
    if (!outfile.is_open()) {
        LOG_ERROR("Error: Failed to open file: %s", filepath.c_str());
        return -1;
    }

    // Step 3: Use minimum kernel_ready_time as base time for normalization
    // kernel_ready_time represents when AICore entered main loop (ready to execute)
    // Each AICore has different kernel_ready_time, use minimum for proper alignment
    uint64_t base_time_cycles = UINT64_MAX;
    for (const auto& record : collected_perf_records_) {
        if (record.kernel_ready_time < base_time_cycles) {
            base_time_cycles = record.kernel_ready_time;
        }
    }

    // Convert cycles to microseconds: timestamp_us = (cycles / freq) * 1e6
    // freq = PLATFORM_PROF_SYS_CNT_FREQ (1850 MHz = 1850000000 Hz)
    auto cycles_to_us = [base_time_cycles](uint64_t cycles) -> double {
        uint64_t normalized_cycles = cycles - base_time_cycles;
        return (static_cast<double>(normalized_cycles) / PLATFORM_PROF_SYS_CNT_FREQ) * 1000000.0;
    };

    // Step 4: Find all unique cores and build core index mapping
    std::map<uint32_t, int> core_to_tid;  // core_id -> thread_id
    std::set<uint32_t> unique_cores;
    for (const auto& record : collected_perf_records_) {
        unique_cores.insert(record.core_id);
    }

    int tid_counter = 1000;
    for (uint32_t core_id : unique_cores) {
        core_to_tid[core_id] = tid_counter++;
    }

    // Step 5: Write JSON header
    outfile << "{\n";
    outfile << "  \"traceEvents\": [\n";

    bool first_event = true;
    auto write_comma = [&]() {
        if (!first_event) {
            outfile << ",\n";
        }
        first_event = false;
    };

    // Step 6: Write metadata events - Process name
    write_comma();
    outfile << "    {\n";
    outfile << "      \"args\": {\"name\": \"Machine View\"},\n";
    outfile << "      \"cat\": \"__metadata\",\n";
    outfile << "      \"name\": \"process_name\",\n";
    outfile << "      \"ph\": \"M\",\n";
    outfile << "      \"pid\": 1\n";
    outfile << "    }";

    // Step 7: Write metadata events - Thread names (one per core)
    for (const auto& [core_id, tid] : core_to_tid) {
        write_comma();

        // Determine core type name
        std::string core_name;
        // Find first record with this core_id to get core_type
        CoreType core_type = CoreType::AIV;
        for (const auto& record : collected_perf_records_) {
            if (record.core_id == core_id) {
                core_type = record.core_type;
                break;
            }
        }

        if (core_type == CoreType::AIC) {
            core_name = "AIC_" + std::to_string(core_id);
        } else {
            core_name = "AIV_" + std::to_string(core_id);
        }

        outfile << "    {\n";
        outfile << "      \"args\": {\"name\": \"" << core_name << "\"},\n";
        outfile << "      \"cat\": \"__metadata\",\n";
        outfile << "      \"name\": \"thread_name\",\n";
        outfile << "      \"ph\": \"M\",\n";
        outfile << "      \"pid\": 1,\n";
        outfile << "      \"tid\": " << tid << "\n";
        outfile << "    }";
    }

    // Step 8: Write duration events (task execution records)
    // Build task_id -> event_id mapping for flow events
    std::map<uint32_t, int> task_to_event_id;
    int event_id = 0;
    for (const auto& record : collected_perf_records_) {
        write_comma();

        double start_us = cycles_to_us(record.start_time);
        double end_us = cycles_to_us(record.end_time);
        double duration_us = end_us - start_us;
        int tid = core_to_tid[record.core_id];

        // Build fanout hint string
        std::ostringstream fanout_str;
        fanout_str << "[";
        for (int i = 0; i < record.fanout_count && i < RUNTIME_MAX_FANOUT; i++) {
            if (i > 0) fanout_str << ", ";
            fanout_str << record.fanout[i];
        }
        fanout_str << "]";

        // Calculate duration_cycles from start_time and end_time
        uint64_t duration_cycles = record.end_time - record.start_time;

        outfile << "    {\n";
        outfile << "      \"args\": {\n";
        outfile << "        \"event-hint\": \"Task:" << record.task_id
                << ", FuncId:" << record.func_id
                << ", CoreId:" << record.core_id << "\",\n";
        outfile << "        \"fanout-hint\": \"" << fanout_str.str() << "\",\n";
        outfile << "        \"duration-cycles\": " << duration_cycles << ",\n";
        outfile << "        \"taskId\": " << record.task_id << "\n";
        outfile << "      },\n";
        outfile << "      \"cat\": \"event\",\n";
        outfile << "      \"id\": " << event_id << ",\n";
        outfile << "      \"name\": \"Task_" << record.task_id << "_Func_" << record.func_id << "\",\n";
        outfile << "      \"ph\": \"X\",\n";
        outfile << "      \"pid\": 1,\n";
        outfile << "      \"tid\": " << tid << ",\n";
        outfile << "      \"ts\": " << std::fixed << std::setprecision(5) << start_us << ",\n";
        outfile << "      \"dur\": " << std::fixed << std::setprecision(5) << duration_us << "\n";
        outfile << "    }";

        // Record mapping for flow events
        task_to_event_id[record.task_id] = event_id;
        event_id++;
    }

    // Step 9: Write flow events (dependencies)
    // Build task_id -> record mapping for dependency arrows
    std::map<uint32_t, const PerfRecord*> task_map;
    for (const auto& record : collected_perf_records_) {
        task_map[record.task_id] = &record;
    }

    int flow_id = 0;
    for (const auto& record : collected_perf_records_) {
        // For each fanout (successor), draw a flow arrow
        for (int i = 0; i < record.fanout_count && i < RUNTIME_MAX_FANOUT; i++) {
            int successor_task_id = record.fanout[i];
            auto it = task_map.find(successor_task_id);
            if (it == task_map.end()) {
                continue;  // Successor not found in records
            }
            const PerfRecord* succ_record = it->second;

            // Get event IDs for source and destination tasks
            int src_event_id = task_to_event_id[record.task_id];
            int dst_event_id = task_to_event_id[succ_record->task_id];

            // Calculate timestamps
            double src_end_us = cycles_to_us(record.end_time);
            double dst_start_us = cycles_to_us(succ_record->start_time);

            // Flow start event (at end of source task - half cycle)
            constexpr double half_cycle_us = (0.5 / PLATFORM_PROF_SYS_CNT_FREQ) * 1000000.0;
            double flow_start_us = src_end_us - half_cycle_us;
            int src_tid = core_to_tid[record.core_id];

            write_comma();
            outfile << "    {\n";
            outfile << "      \"cat\": \"flow\",\n";
            outfile << "      \"id\": " << flow_id << ",\n";
            outfile << "      \"name\": \"dependency\",\n";
            outfile << "      \"ph\": \"s\",\n";
            outfile << "      \"pid\": 1,\n";
            outfile << "      \"tid\": " << src_tid << ",\n";
            outfile << "      \"ts\": " << std::fixed << std::setprecision(5) << flow_start_us << ",\n";
            outfile << "      \"bind_id\": " << src_event_id << "\n";
            outfile << "    }";

            // Flow finish event (at start of destination task)
            write_comma();
            int dst_tid = core_to_tid[succ_record->core_id];

            outfile << "    {\n";
            outfile << "      \"cat\": \"flow\",\n";
            outfile << "      \"id\": " << flow_id << ",\n";
            outfile << "      \"name\": \"dependency\",\n";
            outfile << "      \"ph\": \"f\",\n";
            outfile << "      \"pid\": 1,\n";
            outfile << "      \"tid\": " << dst_tid << ",\n";
            outfile << "      \"ts\": " << std::fixed << std::setprecision(5) << dst_start_us << ",\n";
            outfile << "      \"bp\": \"e\",\n";
            outfile << "      \"bind_id\": " << dst_event_id << "\n";
            outfile << "    }";

            flow_id++;
        }
    }

    // Step 10: Close JSON
    outfile << "\n  ]\n";
    outfile << "}\n";

    outfile.close();

    LOG_INFO("=== Export %s Complete ===", filepath.c_str());

    return 0;
}
