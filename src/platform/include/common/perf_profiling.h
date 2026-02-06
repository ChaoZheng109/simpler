/**
 * @file perf_profiling.h
 * @brief Performance profiling data structures (Simplified Version)
 *
 * 架构设计：固定头部 + 动态尾部
 *
 * 内存布局：
 * ┌────────────────────────────────────────────────────────────┐
 * │ PerfDataHeader（固定头部）                                  │
 * │  - ReadyQueue（FIFO，容量=PLATFORM_MAX_CORES*2）           │
 * │  - Metadata（num_cores, buffer_capacity, flags）           │
 * ├────────────────────────────────────────────────────────────┤
 * │ DoubleBuffer[0] (Core 0)                                   │
 * │  - buffer1, buffer2 (PerfBuffer)                           │
 * │  - buffer1_status, buffer2_status (0/1/2)                  │
 * ├────────────────────────────────────────────────────────────┤
 * │ DoubleBuffer[1] (Core 1)                                   │
 * ├────────────────────────────────────────────────────────────┤
 * │ ...                                                        │
 * ├────────────────────────────────────────────────────────────┤
 * │ DoubleBuffer[num_cores-1]                                  │
 * └────────────────────────────────────────────────────────────┘
 *
 * 总大小 = sizeof(PerfDataHeader) + num_cores * sizeof(DoubleBuffer)
 */

#ifndef PLATFORM_COMMON_PERF_PROFILING_H_
#define PLATFORM_COMMON_PERF_PROFILING_H_

#include <cstdint>
#include <vector>

#include "platform_config.h"
#include "core_type.h"

// Maximum number of successor tasks per PerfRecord (matches Task::fanout)
#ifndef RUNTIME_MAX_FANOUT
#define RUNTIME_MAX_FANOUT 512
#endif

// Maximum cores that can be profiled simultaneously
#ifndef PLATFORM_MAX_CORES
#define PLATFORM_MAX_CORES 72  // 24 blocks × 3 cores/block
#endif

// =============================================================================
// Buffer Status Constants (Simple 3-state design)
// =============================================================================

/**
 * Buffer 状态常量（使用数字 0/1/2）
 *
 * 状态转换流程：
 * 0 (空闲) → 1 (写入中) → 2 (待读取) → 0 (空闲)
 *
 * - AICPU: 0→1 (分配时), 1→2 (检测满时)
 * - AICore: 只写入数据，不修改状态
 * - Host:   2→0 (读取完成后)
 */
constexpr uint32_t PERF_BUFFER_STATUS_IDLE    = 0;  ///< 空闲：可被 AICPU 分配
constexpr uint32_t PERF_BUFFER_STATUS_WRITING = 1;  ///< 写入中：AICore 正在使用
constexpr uint32_t PERF_BUFFER_STATUS_READY   = 2;  ///< 待读取：已满，等待 Host

// =============================================================================
// PerfRecord - Single Task Execution Record
// =============================================================================

/**
 * 单个任务执行记录
 */
struct PerfRecord {
    // 时间信息（设备时钟时间戳）
    uint64_t start_time;      ///< 任务开始时间戳（get_sys_cnt）
    uint64_t end_time;        ///< 任务结束时间戳
    uint64_t duration;        ///< 执行时长（end - start）

    // 任务标识
    uint32_t task_id;         ///< 任务唯一标识
    uint32_t func_id;         ///< 内核函数标识
    uint32_t core_id;         ///< 物理核心 ID（0-71）
    CoreType core_type;       ///< 核心类型（AIC/AIV）

    // 依赖关系（fanout only）
    int32_t fanout[RUNTIME_MAX_FANOUT];  ///< 后继任务 ID 数组
    int32_t fanout_count;                 ///< 后继任务数量

    uint8_t padding[4];       ///< 对齐到 64 字节
} __attribute__((aligned(64)));

static_assert(sizeof(PerfRecord) % 64 == 0,
              "PerfRecord must be 64-byte aligned for optimal cache performance");

// =============================================================================
// PerfBuffer - Fixed-Size Record Buffer (with time alignment)
// =============================================================================

/**
 * 固定大小的性能记录缓冲区
 *
 * 容量：PLATFORM_PROF_BUFFER_SIZE（定义在 platform_config.h）
 *
 * 新增功能：记录第一个任务的时间，用于跨核心时间对齐
 */
struct PerfBuffer {
    PerfRecord records[PLATFORM_PROF_BUFFER_SIZE];  ///< 记录数组
    volatile uint32_t count;                         ///< 当前记录数量

    // 时间对齐支持（每个 buffer 独立记录）
    volatile uint64_t first_task_time;               ///< 第一个任务的开始时间
    volatile uint32_t first_task_recorded;           ///< first_task_time 是否有效（0/1）

    uint32_t padding[13];                            ///< 对齐到 64 字节边界
} __attribute__((aligned(64)));

// =============================================================================
// DoubleBuffer - Per-Core Ping-Pong Buffers (with status)
// =============================================================================

/**
 * 每核心的双缓冲区（简化版，带状态管理）
 *
 * 设计要点：
 * 1. 两个独立的 PerfBuffer（buffer1, buffer2）
 * 2. 每个 buffer 有独立的状态字段（0/1/2）
 * 3. AICPU 通过检查状态决定是否可以分配给 AICore
 * 4. Host 通过队列获取就绪的 buffer 进行读取
 *
 * 状态转换：
 * - AICPU: 0→1 (分配), 1→2 (检测满后入队)
 * - AICore: 只写入 records，递增 count（不修改状态）
 * - Host:   2→0 (读取完成后清零)
 *
 * 优先级策略：
 * - 两个 buffer 都空闲时，AICPU 优先分配 buffer1
 */
struct DoubleBuffer {
    // Buffer 1（Ping）
    PerfBuffer buffer1;                              ///< 第一个缓冲区
    volatile uint32_t buffer1_status;                ///< Buffer1 状态（0/1/2）

    // Buffer 2（Pong）
    PerfBuffer buffer2;                              ///< 第二个缓冲区
    volatile uint32_t buffer2_status;                ///< Buffer2 状态（0/1/2）

    uint32_t padding[14];                            ///< 对齐到 64 字节边界
} __attribute__((aligned(64)));

// =============================================================================
// ReadyQueueEntry - Queue Entry for Ready Buffers
// =============================================================================

/**
 * 就绪队列条目
 *
 * 当某个核心的某个 buffer 满时，AICPU 将此条目加入队列。
 * Host 从队列中取出条目，定位到 (core_index, buffer_id) 进行读取。
 */
struct ReadyQueueEntry {
    uint32_t core_index;      ///< 核心索引（0 ~ num_cores-1）
    uint32_t buffer_id;       ///< Buffer ID（1=buffer1, 2=buffer2）
    uint32_t padding[2];      ///< 对齐到 16 字节
} __attribute__((aligned(16)));

// =============================================================================
// PerfDataHeader - Fixed Header (includes ready queue)
// =============================================================================

/**
 * 性能数据固定头部
 *
 * 位于共享内存起始位置，包含：
 * 1. 就绪队列（FIFO Circular Buffer）
 * 2. 元数据（核心数量、buffer 容量等）
 * 3. 控制标志（使能、完成标志）
 *
 * 就绪队列设计：
 * - 容量：PLATFORM_MAX_CORES * 2（每个核心最多2个buffer就绪）
 * - 实现：循环队列（Circular Buffer）
 * - 生产者：AICPU（添加满的 buffer）
 * - 消费者：Host（读取并清理 buffer）
 * - 队列空：head == tail
 * - 队列满：(tail + 1) % capacity == head
 */
struct PerfDataHeader {
    // =========================================================================
    // 就绪队列（FIFO Circular Buffer）
    // =========================================================================
    ReadyQueueEntry queue[PLATFORM_MAX_CORES * 2];  ///< 队列数组（容量=144）
    volatile uint32_t queue_head;                    ///< 消费者读取位置（Host 修改）
    volatile uint32_t queue_tail;                    ///< 生产者写入位置（AICPU 修改）

    // =========================================================================
    // 元数据（Host 初始化，Device 只读）
    // =========================================================================
    uint32_t num_cores;                              ///< 实际启动的核心数量
    uint32_t buffer_capacity;                        ///< 每个 buffer 的记录容量

    // =========================================================================
    // 控制标志
    // =========================================================================
    volatile uint32_t profiling_enabled;             ///< 性能分析使能（1=enabled）
    volatile uint32_t profiling_finished;            ///< 完成标志（1=finished）

    uint32_t padding[10];                            ///< 对齐到 64 字节边界
} __attribute__((aligned(64)));

// =============================================================================
// Helper Functions - Memory Layout
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 计算性能数据总内存大小
 *
 * 公式：总大小 = 固定头部 + 动态尾部
 *             = sizeof(PerfDataHeader) + num_cores × sizeof(DoubleBuffer)
 *
 * @param num_cores 核心数量（block_dim × PLATFORM_CORES_PER_BLOCKDIM）
 * @return 总字节数
 */
inline size_t calc_perf_data_size(int num_cores) {
    return sizeof(PerfDataHeader) + num_cores * sizeof(DoubleBuffer);
}

/**
 * 获取头部指针
 *
 * @param base_ptr 共享内存基地址（device_ptr 或 host_ptr）
 * @return PerfDataHeader 指针
 */
inline PerfDataHeader* get_perf_header(void* base_ptr) {
    return (PerfDataHeader*)base_ptr;
}

/**
 * 获取 DoubleBuffer 数组起始地址
 *
 * @param base_ptr 共享内存基地址
 * @return DoubleBuffer 数组指针
 */
inline DoubleBuffer* get_double_buffers(void* base_ptr) {
    return (DoubleBuffer*)((char*)base_ptr + sizeof(PerfDataHeader));
}

/**
 * 获取指定核心的 DoubleBuffer
 *
 * @param base_ptr 共享内存基地址
 * @param core_index 核心索引（0 ~ num_cores-1）
 * @return DoubleBuffer 指针
 */
inline DoubleBuffer* get_core_double_buffer(void* base_ptr, int core_index) {
    DoubleBuffer* buffers = get_double_buffers(base_ptr);
    return &buffers[core_index];
}

/**
 * 获取指定 buffer 的指针和状态指针
 *
 * @param db DoubleBuffer 指针
 * @param buffer_id Buffer ID（1=buffer1, 2=buffer2）
 * @param[out] buf PerfBuffer 指针
 * @param[out] status 状态指针
 */
inline void get_buffer_and_status(DoubleBuffer* db, uint32_t buffer_id,
                                  PerfBuffer** buf, volatile uint32_t** status) {
    if (buffer_id == 1) {
        *buf = &db->buffer1;
        *status = &db->buffer1_status;
    } else {
        *buf = &db->buffer2;
        *status = &db->buffer2_status;
    }
}

#ifdef __cplusplus
}
#endif

// =============================================================================
// HostPerfData - Host-Side Aggregated Data (C++ only)
// =============================================================================

#ifdef __cplusplus

/**
 * Host-side 性能数据聚合（简化版）
 *
 * 用途：
 * 1. 收集所有 PerfRecord 到 std::vector
 * 2. 时间归一化（基于各核心的 first_task_time）
 */
struct HostPerfData {
    std::vector<PerfRecord> records;          ///< 所有收集的记录

    // 元数据
    uint32_t total_cores;                     ///< 总核心数
    uint32_t total_tasks;                     ///< 总任务数

    // 时间对齐
    std::vector<uint64_t> core_first_times;   ///< 每个核心的第一个任务时间
    uint64_t global_first_time;               ///< 全局最早时间（t=0 锚点）

    HostPerfData()
        : total_cores(0),
          total_tasks(0),
          global_first_time(UINT64_MAX) {}

    /**
     * 归一化所有时间戳（相对于 global_first_time）
     */
    void normalize_timestamps() {
        if (global_first_time == UINT64_MAX) {
            return;
        }

        for (auto& record : records) {
            record.start_time -= global_first_time;
            record.end_time -= global_first_time;
        }
    }

    /**
     * 更新全局最早时间
     */
    void update_global_first_time(uint32_t core_id, uint64_t first_time) {
        if (core_first_times.size() <= core_id) {
            core_first_times.resize(core_id + 1, UINT64_MAX);
        }

        core_first_times[core_id] = first_time;

        if (first_time < global_first_time) {
            global_first_time = first_time;
        }
    }
};

#endif  // __cplusplus

#endif  // PLATFORM_COMMON_PERF_PROFILING_H_
