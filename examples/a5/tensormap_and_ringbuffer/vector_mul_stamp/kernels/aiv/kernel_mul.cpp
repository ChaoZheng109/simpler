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
 * Element-wise Tensor Multiplication Kernel — INSTRUMENTED with mark_stamp.
 *
 * Implements: out[i] = src0[i] * src1[i]
 *
 * This is the only instrumented kernel in the vector_mul_stamp example, and
 * the orchestration DAG runs kernel_mul exactly once (t3: g = d * e). So every
 * biu_perf pipe stamp the run produces is attributable to this single task,
 * giving a clean dataset to (a) measure the offset between the mark_stamp DFX
 * trace clock and the AICore get_sys_cnt_aicore() task window, and (b) check
 * whether any non-mark_stamp pipe activity also shows up on the channels.
 *
 * Region-id scheme: region_id = op_id*10 + phase, where
 *   phase 0 = instruction begin, 1 = instruction end, 5 = instant point.
 * The host (l0_perf_collector op_info / pipe lanes) pairs begin/end into a
 * duration bar and renders instants as markers. To track a new instruction,
 * pick an op_id, emit stamps here, and add a row to op_info().
 *   10/11  TLOAD begin/end       (MTE2)  ← kernel start anchor
 *   25     SET_FLAG MTE2->V      (MTE2)
 *   30/31  TMUL  begin/end       (VEC)
 *   45     SET_FLAG V->MTE3      (VEC)
 *   50/51  TSTORE begin/end      (MTE3)
 *   95     END instant after bar.all (MTE3) ← kernel end anchor
 */

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "tensor.h"

using namespace pto;

#include "pipe_sync.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

/**
 * Element-wise multiplication kernel implementation
 *
 * Unified signature: all arguments passed via int64_t array
 * @param args  Argument array:
 *              args[0] = src0 pointer (first input tensor)
 *              args[1] = src1 pointer (second input tensor)
 *              args[2] = out pointer (output tensor)
 *              args[3] = size (number of elements)
 */
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Unpack arguments (Tensor* pointers from runtime)
    __gm__ Tensor *src0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *src1_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ float *src0 = reinterpret_cast<__gm__ float *>(src0_tensor->buffer.addr) + src0_tensor->start_offset;
    __gm__ float *src1 = reinterpret_cast<__gm__ float *>(src1_tensor->buffer.addr) + src1_tensor->start_offset;
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    // Configuration: float, 128, 128, 128, 128
    constexpr int kTRows_ = 128;
    constexpr int kTCols_ = 128;
    constexpr int vRows = 128;
    constexpr int vCols = 128;

    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kTCols_, 1>;
    using GlobalData = GlobalTensor<float, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, float, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData src0Tile(vRows, vCols);
    TileData src1Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x10000);
    TASSIGN(dstTile, 0x20000);

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);

    asm volatile("bar.MTE2");
    bisheng::cce::mark_stamp<PIPE_MTE2, 10>();

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);

    asm volatile("bar.MTE2");
    bisheng::cce::mark_stamp<PIPE_MTE2, 11>();

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    bisheng::cce::mark_stamp<PIPE_MTE2, 25>();
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    pipe_barrier(PIPE_ALL);
    bisheng::cce::mark_stamp<PIPE_V, 30>();

    TMUL(dstTile, src0Tile, src1Tile);

    pipe_barrier(PIPE_ALL);
    bisheng::cce::mark_stamp<PIPE_V, 31>();

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    bisheng::cce::mark_stamp<PIPE_V, 45>();
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    asm volatile("bar.MTE3");
    bisheng::cce::mark_stamp<PIPE_MTE3, 50>();

    TSTORE(dstGlobal, dstTile);

    asm volatile("bar.MTE3");
    bisheng::cce::mark_stamp<PIPE_MTE3, 51>();

    pipe_sync();

    // Tail anchor: bar.all guarantees the trace bus has drained the kernel's
    // stamps before this final one, so region 95 (END) brackets the kernel end.
    asm volatile("bar.all");
    bisheng::cce::mark_stamp<PIPE_MTE3, 95>();

    // Trailing flush — NOP burst interleaved with mark_stamps keeps the AICore
    // busy after the last functional stamp, both giving the perfmon HW
    // wall-clock time to drain/write its trace before the task ends (and
    // perf_mon_en is cleared) and actively emitting trace events for it to
    // capture. Mirrors the paged_attention_unroll aic_qk_matmul trailing flush.
    asm volatile("nop"); asm volatile("nop"); asm volatile("nop"); asm volatile("nop");
    asm volatile("nop"); asm volatile("nop"); asm volatile("nop"); asm volatile("nop");
    bisheng::cce::mark_stamp<PIPE_MTE3, 96>();
    asm volatile("nop");
    bisheng::cce::mark_stamp<PIPE_MTE3, 97>();
    asm volatile("nop");
    bisheng::cce::mark_stamp<PIPE_MTE3, 98>();
    asm volatile("nop");
    bisheng::cce::mark_stamp<PIPE_MTE3, 99>();
    asm volatile("nop");
    bisheng::cce::mark_stamp<PIPE_MTE3, 100>();
    asm volatile("nop");
    bisheng::cce::mark_stamp<PIPE_MTE3, 101>();
    asm volatile(".rept 3500\n\tNOP \n\t.endr");
    bisheng::cce::mark_stamp<PIPE_MTE3, 102>();
}
