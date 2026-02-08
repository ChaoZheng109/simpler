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
    int num_cores = block_dim * cores_per_blockdim_;

    if (num_cores > RUNTIME_MAX_WORKER) {
        LOG_ERROR("num_cores (%d) exceeds RUNTIME_MAX_WORKER (%d)", 
                       num_cores, RUNTIME_MAX_WORKER);
        return -1;
    }

    // Initialize handshake buffers
    runtime.worker_count = num_cores;
    worker_count_ = num_cores;
    runtime.sche_cpu_num = launch_aicpu_num;

    // Calculate number of AIC cores
    int num_aic = block_dim;

    for (int i = 0; i < num_cores; i++) {
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
    LOG_DEBUG("\n=== Setting function_bin_addr for Tasks (Simulation) ===");
    for (int i = 0; i < runtime.get_task_count(); i++) {
        Task* task = runtime.get_task(i);
        if (task != nullptr) {
            uint64_t addr = runtime.get_function_bin_addr(task->func_id);
            task->function_bin_addr = addr;
            LOG_DEBUG("  Task %d (func_id=%d) -> function_bin_addr=0x%lx",
                          i, task->func_id, addr);
        }
    }
    LOG_DEBUG("");

    // Store runtime pointer for print_handshake_results
    last_runtime_ = &runtime;

    // Initialize performance profiling if enabled
    if (runtime.enable_profiling) {
        rc = init_performance_profiling(runtime, num_cores, device_id);
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
    LOG_INFO("=== Launching %d AICPU thread(s) ===", launch_aicpu_num);
    std::vector<std::thread> aicpu_threads;
    for (int i = 0; i < launch_aicpu_num; i++) {
        aicpu_threads.emplace_back([this, &runtime]() {
            aicpu_execute_func_(&runtime);
        });
    }

    // Launch AICore threads
    LOG_INFO("=== Launching %d AICore thread(s) ===", num_cores);
    std::vector<std::thread> aicore_threads;
    for (int i = 0; i < num_cores; i++) {
        CoreType core_type = runtime.workers[i].core_type;
        aicore_threads.emplace_back([this, &runtime, i, core_type]() {
            aicore_execute_func_(&runtime, i, core_type);
        });
    }

    // Poll and collect performance data during execution (if enabled)
    std::thread collector_thread;
    if (runtime.enable_profiling) {
        collector_thread = std::thread([this, &runtime, num_cores]() {
            poll_and_collect_performance_data(num_cores, runtime.get_task_count());
        });
    }

    // Wait for all threads to complete
    LOG_INFO("=== Waiting for threads to complete ===");
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

    LOG_INFO("=== All threads completed ===");

    // Print performance data after execution completes
    if (runtime.enable_profiling) {
        print_performance_data();
    }

    return 0;
}

void DeviceRunner::print_handshake_results() {
    if (worker_count_ == 0 || last_runtime_ == nullptr) {
        return;
    }

    LOG_DEBUG("\nHandshake results for %d cores:", worker_count_);
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

int DeviceRunner::init_performance_profiling(Runtime& runtime, int num_cores, int device_id) {
    (void)device_id;  // Unused in simulation

    LOG_INFO("\n=== Initializing Performance Profiling ===");

    // Step 1: Calculate total memory size (header + all DoubleBuffers)
    size_t total_size = calc_perf_data_size(num_cores);

    size_t header_size = sizeof(PerfDataHeader);
    size_t single_db_size = sizeof(DoubleBuffer);
    size_t buffers_size = num_cores * single_db_size;

    LOG_INFO("  Memory allocation plan:");
    LOG_INFO("    - Number of cores:      %d", num_cores);
    LOG_INFO("    - Header size:          %zu bytes", header_size);
    LOG_INFO("      (includes ready queue: %d entries)", PLATFORM_MAX_CORES * 2);
    LOG_INFO("    - Single DoubleBuffer:  %zu bytes", single_db_size);
    LOG_INFO("    - All DoubleBuffers:    %zu bytes", buffers_size);
    LOG_INFO("    - Total size:           %zu bytes (%zu KB, %zu MB)",
             total_size, total_size / 1024, total_size / (1024 * 1024));

    // Step 2: Allocate device shared memory (simulation: use malloc for host memory)
    void* perf_dev_ptr = malloc(total_size);
    if (perf_dev_ptr == nullptr) {
        LOG_ERROR("Failed to allocate device memory for profiling (%zu bytes)", total_size);
        return -1;
    }
    LOG_INFO("  Allocated device memory: %p", perf_dev_ptr);

    // Step 3: Register to host mapping (simulation: both pointers point to same memory)
    void* perf_host_ptr = perf_dev_ptr;  // In simulation, both point to same memory
    LOG_INFO("  Mapped to host memory:   %p", perf_host_ptr);

    // Step 4: Initialize fixed header (using host_ptr)
    PerfDataHeader* header = get_perf_header(perf_host_ptr);

    // Initialize queue
    memset(header->queue, 0, sizeof(header->queue));
    header->queue_head = 0;
    header->queue_tail = 0;

    // Initialize metadata
    header->num_cores = num_cores;

    LOG_INFO("  Initialized PerfDataHeader:");
    LOG_INFO("    - num_cores:        %d", header->num_cores);
    LOG_INFO("    - buffer_capacity:  %d", PLATFORM_PROF_BUFFER_SIZE);
    LOG_INFO("    - queue capacity:   %d", PLATFORM_MAX_CORES * 2);

    // Step 5: Initialize all DoubleBuffers (all buffers start as 0=idle)
    DoubleBuffer* buffers = get_double_buffers(perf_host_ptr);

    for (int i = 0; i < num_cores; i++) {
        DoubleBuffer* db = &buffers[i];

        // Initialize buffer1
        memset(&db->buffer1, 0, sizeof(PerfBuffer));
        db->buffer1.count = 0;
        db->buffer1.first_task_time = 0;
        db->buffer1_status = BufferStatus::IDLE;

        // Initialize buffer2
        memset(&db->buffer2, 0, sizeof(PerfBuffer));
        db->buffer2.count = 0;
        db->buffer2.first_task_time = 0;
        db->buffer2_status = BufferStatus::IDLE;
    }

    LOG_INFO("  Initialized %d DoubleBuffers (all status=0, idle)", num_cores);

    // Step 6: Write memory barrier (ensure all initialization visible to workers)
    wmb();

    // Step 7: Pass to Runtime (device base address)
    runtime.perf_data_base = (uint64_t)perf_dev_ptr;

    LOG_INFO("  Set runtime.perf_data_base = 0x%lx", runtime.perf_data_base);

    // Step 8: Save pointers to member variables
    perf_shared_mem_dev_ = perf_dev_ptr;
    perf_shared_mem_host_ = perf_host_ptr;

    LOG_INFO("=== Performance Profiling Initialized ===");

    return 0;
}

void DeviceRunner::poll_and_collect_performance_data(int num_cores, int expected_tasks) {
    if (perf_shared_mem_host_ == nullptr) {
        return;  // Profiling not enabled
    }

    LOG_INFO("=== Collecting Performance Data (Before Stream Sync) ===");
    LOG_INFO("  Expected tasks: %d", expected_tasks);

    PerfDataHeader* header = get_perf_header(perf_shared_mem_host_);
    DoubleBuffer* buffers = get_double_buffers(perf_shared_mem_host_);

    uint32_t capacity = PLATFORM_MAX_CORES * 2;
    int total_records_collected = 0;
    int buffers_processed = 0;

    // Clear previous collection
    collected_perf_records_.clear();

    // Timeout configuration
    const auto timeout_duration = std::chrono::seconds(PLATFORM_PROF_TIMEOUT_SECONDS);  // 30 second timeout
    const auto start_time = std::chrono::steady_clock::now();
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
            // Check for timeout periodically
            empty_poll_count++;
            if (empty_poll_count >= PLATFORM_PROF_EMPTY_POLLS_CHECK_NUM) {
                empty_poll_count = 0;
                auto elapsed = std::chrono::steady_clock::now() - start_time;
                if (elapsed >= timeout_duration) {
                    LOG_WARN("  WARNING: Performance data collection timeout after %ld seconds",
                             std::chrono::duration_cast<std::chrono::seconds>(elapsed).count());
                    LOG_WARN("  Collected %d / %d records before timeout",
                             total_records_collected, expected_tasks);
                    break;  // Exit with partial data
                }
            }
            // Continue polling (AICPU may still be producing data)
            continue;
        }

        // Reset empty poll counter when we find data
        empty_poll_count = 0;

        // Dequeue entry
        ReadyQueueEntry entry = header->queue[head];
        uint32_t core_index = entry.core_index;
        uint32_t buffer_id = entry.buffer_id;

        // Validate core index
        if (core_index >= static_cast<uint32_t>(num_cores)) {
            LOG_ERROR("  ERROR: Invalid core_index %u (max=%d)", core_index, num_cores);
            break;
        }

        LOG_DEBUG("  Processing: core=%u, buffer=%u", core_index, buffer_id);

        // Get the buffer and status pointer
        DoubleBuffer* db = &buffers[core_index];
        PerfBuffer* buf = nullptr;
        volatile BufferStatus* status = nullptr;
        get_buffer_and_status(db, buffer_id, &buf, &status);

        // Read buffer data with memory barrier
        rmb();
        uint32_t count = buf->count;
        uint64_t first_task_time = buf->first_task_time;

        LOG_DEBUG("    Records in buffer: %u", count);
        LOG_DEBUG("    First task time: %lu", first_task_time);

        // Collect records
        for (uint32_t i = 0; i < count && i < PLATFORM_PROF_BUFFER_SIZE; i++) {
            collected_perf_records_.push_back(buf->records[i]);
            total_records_collected++;
        }

        // Clear buffer
        buf->count = 0;
        buf->first_task_time = 0;

        // Set buffer status to IDLE
        *status = BufferStatus::IDLE;
        wmb();  // Ensure status is visible to AICPU

        // Update queue head
        header->queue_head = (head + 1) % capacity;
        wmb();  // Ensure head update is visible to AICPU

        buffers_processed++;
    }

    LOG_INFO("  Total buffers processed: %d", buffers_processed);
    LOG_INFO("  Total records collected: %d", total_records_collected);

    if (total_records_collected < expected_tasks) {
        LOG_WARN("  WARNING: Incomplete collection (%d / %d records)",
                 total_records_collected, expected_tasks);
    }

    LOG_INFO("=== Performance Data Collection Complete ===\n");
}

void DeviceRunner::print_performance_data() {
    if (collected_perf_records_.empty()) {
        LOG_INFO("=== No Performance Data to Print ===");
        return;
    }

    LOG_INFO("=== Performance Records Detail ===");

    // Calculate min start time for normalization
    uint64_t min_time = UINT64_MAX;
    for (const auto& record : collected_perf_records_) {
        if (record.start_time < min_time) {
            min_time = record.start_time;
        }
    }

    // Print detailed records only in DEBUG mode
    LOG_DEBUG("  Base time (for normalization): %lu", min_time);
    LOG_DEBUG("");
    LOG_DEBUG("  Task execution records:");
    LOG_DEBUG("  ┌────────┬─────────┬─────────┬────────────┬──────────────────┬──────────────────┬──────────────┬──────────┐");
    LOG_DEBUG("  │ Task ID│ Func ID │ Core ID │ Core Type  │  Start (cycles)  │   End (cycles)   │Duration(cyc) │  Fanout  │");
    LOG_DEBUG("  ├────────┼─────────┼─────────┼────────────┼──────────────────┼──────────────────┼──────────────┼──────────┤");

    for (size_t i = 0; i < collected_perf_records_.size() && i < 50; i++) {  // Limit to first 50 for display
        const PerfRecord& record = collected_perf_records_[i];

        // Normalize times
        uint64_t norm_start = record.start_time - min_time;
        uint64_t norm_end = record.end_time - min_time;

        char line_buf[256];
        snprintf(line_buf, sizeof(line_buf),
                 "  │ %6u │ %7u │ %7u │ %10s │ %16lu │ %16lu │ %12lu │ %8u │",
                 record.task_id, record.func_id, record.core_id,
                 (record.core_type == CoreType::AIC ? "AIC" : "AIV"),
                 norm_start, norm_end, record.duration, record.fanout_count);
        LOG_DEBUG("%s", line_buf);
    }

    LOG_DEBUG("  └────────┴─────────┴─────────┴────────────┴──────────────────┴──────────────────┴──────────────┴──────────┘");

    if (collected_perf_records_.size() > 50) {
        LOG_DEBUG("  ... (%zu more records not shown)", collected_perf_records_.size() - 50);
    }

    // Calculate statistics
    uint64_t total_duration = 0;
    uint64_t max_duration = 0;
    uint64_t min_duration = UINT64_MAX;

    for (const auto& record : collected_perf_records_) {
        total_duration += record.duration;
        if (record.duration > max_duration) max_duration = record.duration;
        if (record.duration < min_duration) min_duration = record.duration;
    }

    double avg_duration = static_cast<double>(total_duration) / collected_perf_records_.size();

    LOG_INFO("");
    LOG_INFO("  Performance Statistics:");
    LOG_INFO("    Total tasks:     %zu", collected_perf_records_.size());
    LOG_INFO("    Avg duration:    %lu cycles", static_cast<uint64_t>(avg_duration));
    LOG_INFO("    Min duration:    %lu cycles", min_duration);
    LOG_INFO("    Max duration:    %lu cycles", max_duration);
    LOG_INFO("    Total duration:  %lu cycles", total_duration);

    LOG_INFO("=== Performance Data Print Complete ===");
}
