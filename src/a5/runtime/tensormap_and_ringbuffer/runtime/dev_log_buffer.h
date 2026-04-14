/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
/**
 * dev_log_buffer.h - Device-side diagnostic log buffer (temporary debug patch)
 *
 * Provides a host-allocated, device-written ring buffer for capturing AICPU
 * stall diagnostics on hardware where DEV_ALWAYS is broken.
 * The host allocates and zeros the buffer, copies it to device memory before
 * execution, and copies it back for printing after execution.
 */

#ifndef SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_DEV_LOG_BUFFER_H_
#define SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_DEV_LOG_BUFFER_H_

#include <atomic>
#include <cstdint>

#define DEV_LOG_MAX_ENTRIES 512
#define DEV_LOG_MSG_SIZE 128

struct DevLogEntry {
    int32_t thread_idx;
    char msg[DEV_LOG_MSG_SIZE];
};

struct alignas(64) DevLogBuffer {
    std::atomic<int32_t> write_pos;  // atomically claimed slot index
    int32_t capacity;                // DEV_LOG_MAX_ENTRIES
    char _pad[56];                   // pad header to 64 bytes
    DevLogEntry entries[DEV_LOG_MAX_ENTRIES];
};

#endif  // SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_DEV_LOG_BUFFER_H_
