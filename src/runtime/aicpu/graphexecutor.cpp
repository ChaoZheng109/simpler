#include <cstdint>
#include <atomic>
#include <mutex>
#include "device_log.h"
#include "graph.h"
#include "kernel_args.h"

constexpr int MAX_AICPU_THREADS = 4;
constexpr int MAX_AIC_PER_THREAD = 24;
constexpr int MAX_AIV_PER_THREAD = 48;
constexpr int MAX_CORES_PER_THREAD = MAX_AIC_PER_THREAD + MAX_AIV_PER_THREAD;

struct GraphExecutor {
    // ===== Thread management state =====
    std::atomic<int> thread_idx_{0};
    std::atomic<bool> initialized_{false};
    std::atomic<bool> init_done_{false};
    std::atomic<bool> init_failed_{false};
    std::atomic<bool> finished_{false};

    int thread_num_{0};
    int total_cores_{0};
    int coresPerBlockdim_{3};
    int cores_per_thread_{0};
    int core_assignments_[MAX_AICPU_THREADS][MAX_CORES_PER_THREAD];

    // ===== Task queue state =====
    std::mutex ready_queue_aic_mutex_;
    int ready_queue_aic_[GRAPH_MAX_TASKS];
    std::atomic<int> ready_count_aic_{0};

    std::mutex ready_queue_aiv_mutex_;
    int ready_queue_aiv_[GRAPH_MAX_TASKS];
    std::atomic<int> ready_count_aiv_{0};

    // Task execution tracking
    std::atomic<int> completed_tasks_{0};
    std::atomic<int> total_tasks_{0};
    std::atomic<int> finished_count_{0};

    // ===== Methods =====
    int Init(KernelArgs* kargs);
    int HankAiCore(void* arg, int thread_idx, const int* cur_thread_cores);
    int Execute(Graph& g, Handshake* hank, int thread_idx,
                const int* cur_thread_cores, int core_num);
    int ShutdownAiCore(void* arg, int thread_idx, const int* cur_thread_cores);
    int Run(void* arg);
    void DeInit(KernelArgs* kargs);
};

static GraphExecutor g_executor;

// ===== GraphExecutor Method Implementations =====

int GraphExecutor::Init(KernelArgs* kargs) {
    bool expected = false;
    if (!initialized_.compare_exchange_strong(expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return 0;
    }

    DEV_INFO("GraphExecutor: Initializing");

    thread_num_ = kargs->scheCpuNum;
    if (thread_num_ == 0) thread_num_ = 1;

    total_cores_ = kargs->block_dim * coresPerBlockdim_;
    cores_per_thread_ = total_cores_ / thread_num_;

    DEV_INFO("Config: threads=%d, cores=%d, cores_per_thread=%d",
             thread_num_, total_cores_, cores_per_thread_);

    if (thread_num_ < 1 || thread_num_ > MAX_AICPU_THREADS) {
        DEV_ERROR("Invalid thread_num: %d", thread_num_);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    if (cores_per_thread_ > MAX_CORES_PER_THREAD) {
        DEV_ERROR("Cores per thread %d exceeds maximum %d", cores_per_thread_, MAX_CORES_PER_THREAD);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    // Pre-compute core assignments for each thread
    int num_aic = kargs->nrAic;
    for (int t = 0; t < thread_num_; t++) {
        // Each thread manages: 1 AIC + 2 AIV
        int aic_idx = t;
        int aiv_base = num_aic;
        int aiv_idx0 = aiv_base + t * 2;
        int aiv_idx1 = aiv_idx0 + 1;

        core_assignments_[t][0] = aic_idx;
        core_assignments_[t][1] = aiv_idx0;
        core_assignments_[t][2] = aiv_idx1;

        DEV_INFO("Thread %d: AIC[%d] AIV[%d,%d]", t, aic_idx, aiv_idx0, aiv_idx1);
    }

    // Initialize graph execution state
    if (kargs->graphArgs != nullptr) {
        Graph* g = kargs->graphArgs;

        total_tasks_.store(g->get_task_count(), std::memory_order_release);
        completed_tasks_.store(0, std::memory_order_release);

        int initial_ready[GRAPH_MAX_TASKS];
        int initial_count = g->get_initial_ready_tasks(initial_ready);

        DEV_INFO("Init: Found %d initially ready tasks", initial_count);

        int aic_count = 0;
        int aiv_count = 0;
        for (int i = 0; i < initial_count; i++) {
            Task* task = g->get_task(initial_ready[i]);
            if (task->core_type == 0) {  // AIC
                ready_queue_aic_[aic_count++] = initial_ready[i];
            } else {  // AIV
                ready_queue_aiv_[aiv_count++] = initial_ready[i];
            }
        }
        ready_count_aic_.store(aic_count, std::memory_order_release);
        ready_count_aiv_.store(aiv_count, std::memory_order_release);

        DEV_INFO("Init: Initial ready tasks: AIC=%d, AIV=%d", aic_count, aiv_count);

        finished_count_.store(0, std::memory_order_release);
    }

    init_done_.store(true, std::memory_order_release);
    DEV_INFO("GraphExecutor: Init complete");
    return 0;
}

/**
 * Handshake AICore - Initialize and synchronize with AICore kernels
 */
int GraphExecutor::HankAiCore(void *arg, int thread_idx, const int* cur_thread_cores) {
    auto kargs = (KernelArgs *)arg;
    Handshake* all_hanks = (Handshake*)kargs->graphArgs->workers;

    DEV_INFO("Thread %d: Handshaking with %d cores", thread_idx, cores_per_thread_);

    for (int i = 0; i < cores_per_thread_; i++) {
        int core_id = cur_thread_cores[i];
        Handshake* hank = &all_hanks[core_id];
        DEV_INFO("Thread %d: AICPU hank addr = 0x%lx", thread_idx, (uint64_t)hank);
        hank->aicpu_ready = 1;
    }

    for (int i = 0; i < cores_per_thread_; i++) {
        int core_id = cur_thread_cores[i];
        Handshake* hank = &all_hanks[core_id];
        while (hank->aicore_done == 0) { }
        DEV_INFO("Thread %d: success hank->aicore_done = %u", thread_idx, (uint64_t)hank->aicore_done);
    }
    return 0;
}

/**
 * Shutdown AICore - Send quit signal to all AICore kernels
 */
int GraphExecutor::ShutdownAiCore(void *arg, int thread_idx, const int* cur_thread_cores) {
    auto kargs = (KernelArgs *)arg;
    Handshake* all_hanks = (Handshake*)kargs->graphArgs->workers;

    DEV_INFO("Thread %d: Shutting down %d cores", thread_idx, cores_per_thread_);

    for (int i = 0; i < cores_per_thread_; i++) {
        int core_id = cur_thread_cores[i];
        Handshake* hank = &all_hanks[core_id];
        DEV_INFO("Thread %d: AICPU hank addr = 0x%lx", thread_idx, (uint64_t)hank);
        hank->control = 1;
    }
    DEV_INFO("Thread %d: Shutdown complete", thread_idx);
    return 0;
}

/**
 * Execute task graph using polling-based dispatch to AICore
 */
int GraphExecutor::Execute(Graph& g, Handshake* hank, int thread_idx,
                           const int* cur_thread_cores, int core_num) {

    DEV_INFO("Thread %d: Starting execution with %d cores", thread_idx, core_num);

    int cur_thread_completed = 0;
    int cur_thread_tasks_in_flight = 0;
    int task_count = total_tasks_.load(std::memory_order_acquire);

    // Execute tasks using polling-based dispatch
    while (completed_tasks_.load(std::memory_order_acquire) < task_count) {

        // Phase 1: Process completed tasks on my managed cores
        for (int i = 0; i < core_num; i++) {
            int core_id = cur_thread_cores[i];
            Handshake* h = &hank[core_id];

            // Core finished a task (idle + task not null)
            if (h->task_status == 0 && h->task != 0) {
                // Get completed task
                Task* task = reinterpret_cast<Task*>(h->task);
                int task_id = task->task_id;

                DEV_INFO("Thread %d: Core %d completed task %d", thread_idx, core_id, task_id);

                // Update fanin of successors atomically and add to appropriate shared ready queue
                for (int j = 0; j < task->fanout_count; j++) {
                    int dep_id = task->fanout[j];
                    Task* dep = g.get_task(dep_id);

                    // Atomic decrement fanin
                    int prev_fanin = dep->fanin.fetch_sub(1, std::memory_order_acq_rel);

                    // Dependency resolved, add to appropriate shared ready queue
                    if (prev_fanin == 1) {
                        if (dep->core_type == 0) {  // AIC task
                            std::lock_guard<std::mutex> lock(ready_queue_aic_mutex_);
                            int idx = ready_count_aic_.load(std::memory_order_relaxed);
                            ready_queue_aic_[idx] = dep_id;
                            ready_count_aic_.fetch_add(1, std::memory_order_release);
                            DEV_INFO("Thread %d: Task %d became ready -> AIC queue", thread_idx, dep_id);
                        } else {  // AIV task
                            std::lock_guard<std::mutex> lock(ready_queue_aiv_mutex_);
                            int idx = ready_count_aiv_.load(std::memory_order_relaxed);
                            ready_queue_aiv_[idx] = dep_id;
                            ready_count_aiv_.fetch_add(1, std::memory_order_release);
                            DEV_INFO("Thread %d: Task %d became ready -> AIV queue", thread_idx, dep_id);
                        }
                    }
                }

                // Clear task pointer and update counters
                h->task = 0;
                cur_thread_tasks_in_flight--;
                completed_tasks_.fetch_add(1, std::memory_order_release);
                cur_thread_completed++;
            }
        }

        // Load balancing: Skip dispatch if all my cores are busy
        if (cur_thread_tasks_in_flight >= core_num) {
            continue;
        }

        // Phase 2: Dispatch new tasks from matching ready queue to idle cores
        for (int i = 0; i < core_num; i++) {
            int core_id = cur_thread_cores[i];
            Handshake* h = &hank[core_id];

            // Core is idle and available (idle + task is null)
            if (h->task_status == 0 && h->task == 0) {
                // Dispatch from matching queue based on core type
                if (h->core_type == 0) {  // AIC core
                    if (ready_count_aic_.load(std::memory_order_acquire) > 0) {
                        std::lock_guard<std::mutex> lock(ready_queue_aic_mutex_);
                        int count = ready_count_aic_.load(std::memory_order_relaxed);
                        if (count > 0) {
                            ready_count_aic_.fetch_sub(1, std::memory_order_release);
                            int task_id = ready_queue_aic_[count - 1];
                            Task* task = g.get_task(task_id);

                            DEV_INFO("Thread %d: Dispatching AIC task %d to core %d",
                                    thread_idx, task_id, core_id);

                            h->task = reinterpret_cast<uint64_t>(task);
                            h->task_status = 1;  // Mark as busy
                            cur_thread_tasks_in_flight++;
                        }
                    }
                } else if (h->core_type == 1) {  // AIV core
                    if (ready_count_aiv_.load(std::memory_order_acquire) > 0) {
                        std::lock_guard<std::mutex> lock(ready_queue_aiv_mutex_);
                        int count = ready_count_aiv_.load(std::memory_order_relaxed);
                        if (count > 0) {
                            ready_count_aiv_.fetch_sub(1, std::memory_order_release);
                            int task_id = ready_queue_aiv_[count - 1];
                            Task* task = g.get_task(task_id);

                            DEV_INFO("Thread %d: Dispatching AIV task %d to core %d",
                                    thread_idx, task_id, core_id);

                            h->task = reinterpret_cast<uint64_t>(task);
                            h->task_status = 1;  // Mark as busy
                            cur_thread_tasks_in_flight++;
                        }
                    }
                }
            }
        }
    }

    DEV_INFO("Thread %d: Execution complete, completed %d tasks", thread_idx, cur_thread_completed);
    return cur_thread_completed;
}

int GraphExecutor::Run(void *arg) {
    int thread_idx = thread_idx_++;

    auto kargs = (KernelArgs *)arg;

    DEV_INFO("Thread %d: Start", thread_idx);

    const int* cur_thread_cores = core_assignments_[thread_idx];

    auto rc = HankAiCore(arg, thread_idx, cur_thread_cores);
    if (rc != 0) {
        return rc;
    }

    if (kargs->graphArgs != nullptr) {
        Graph* g = kargs->graphArgs;
        Handshake* hank = (Handshake*)kargs->graphArgs->workers;
        DEV_INFO("Thread %d: Graph has %d tasks", thread_idx, g->get_task_count());
        int completed = Execute(*g, hank, thread_idx, cur_thread_cores, cores_per_thread_);
        DEV_INFO("Thread %d: Executed %d tasks from graph", thread_idx, completed);
    }

    rc = ShutdownAiCore(arg, thread_idx, cur_thread_cores);
    if (rc != 0) {
        return rc;
    }

    DEV_INFO("Thread %d: Completed", thread_idx);

    // Check if this is the last thread to finish
    int prev_finished = finished_count_.fetch_add(1, std::memory_order_acq_rel);
    if (prev_finished + 1 == thread_num_) {
        finished_.store(true, std::memory_order_release);
        DEV_INFO("Thread %d: Last thread, marking executor finished", thread_idx);
    }

    return 0;
}

void GraphExecutor::DeInit(KernelArgs* kargs) {
    // Cleanup graph execution state
    if (kargs->graphArgs != nullptr) {
        ready_count_aic_.store(0, std::memory_order_release);
        ready_count_aiv_.store(0, std::memory_order_release);
        completed_tasks_.store(0, std::memory_order_release);
        total_tasks_.store(0, std::memory_order_release);
        finished_count_.store(0, std::memory_order_release);

        DEV_INFO("DeInit: Graph execution state reset");
    }

    initialized_.store(false, std::memory_order_release);
    init_done_.store(false, std::memory_order_release);
    init_failed_.store(false, std::memory_order_release);
    thread_idx_.store(0, std::memory_order_release);
    finished_.store(false, std::memory_order_release);

    DEV_INFO("DeInit: GraphExecutor reset complete");
}

// ===== Public Entry Point =====

/**
 * AicpuExecute - Main AICPU kernel execution entry point
 *
 * This is called by DynTileFwkBackendKernelServer in kernel.cpp.
 * Orchestrates the complete task graph execution:
 * 1. Initialize executor (thread-safe, first thread only)
 * 2. Wait for initialization to complete
 * 3. Execute tasks on managed cores
 * 4. Cleanup when last thread finishes
 *
 * @param arg Pointer to KernelArgs structure containing:
 *            - deviceArgs: device-specific arguments
 *            - hankArgs: handshake buffer array
 *            - core_num: number of cores
 *            - graphArgs: task graph to execute
 * @return 0 on success, non-zero on error
 */
extern "C" int AicpuExecute(void *arg) {
    if (arg == nullptr) {
        DEV_ERROR("%s", "Invalid kernel arguments: null pointer");
        return -1;
    }

    auto kargs = (KernelArgs *)arg;

    DEV_INFO("%s", "AicpuExecute: Starting AICPU kernel execution");

    g_executor.Init(kargs);

    while (!g_executor.init_done_.load(std::memory_order_acquire)) {
        if (g_executor.init_failed_.load(std::memory_order_acquire)) {
            DEV_ERROR("%s", "AicpuExecute: Initialization failed, aborting execution");
            return -1;
        }
    }

    int rc = g_executor.Run(arg);
    if (rc != 0) {
        DEV_ERROR("AicpuExecute: Thread execution failed with rc=%d", rc);
        return rc;
    }

    // Last thread cleans up
    if (g_executor.finished_.load(std::memory_order_acquire)) {
        DEV_INFO("AicpuExecute: Last thread finished, cleaning up");
        g_executor.DeInit(kargs);
    }

    DEV_INFO("%s", "AicpuExecute: Kernel execution completed successfully");
    return 0;
}
