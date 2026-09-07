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
 * Sliding-window mean kernel: out = (in[0] + in[1] + ... + in[NUM_INPUTS-1]) / NUM_INPUTS
 *
 * in[0] is the step's own base slice; in[1..WINDOW] are the previous WINDOW
 * step outputs. Averaging rather than summing keeps the recurrence a
 * contraction, so the value sequence stays bounded over thousands of steps.
 *
 * Only two tiles are resident at a time (accumulator + one source), so the
 * input count is not bounded by unified-buffer capacity.
 *
 * Args (Tensor*):
 *   args[0 .. NUM_INPUTS-1] = inputs, each ROWS x COLS per tile
 *   args[NUM_INPUTS]        = out (OUTPUT)
 */

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "tensor.h"

#include "pipe_sync.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

// WINDOW previous outputs plus the step's own base slice.
static constexpr int WINDOW = 16;
static constexpr int NUM_INPUTS = WINDOW + 1;
static constexpr uint64_t TILE_ELEMS = 128 * 128;

static __aicore__ inline int get_num_tiles(__gm__ Tensor *tensor, uint64_t tile_elems) {
    uint64_t total_elems = tensor->shapes[0];
    return static_cast<int>(total_elems / tile_elems);
}

template <int ROWS, int COLS>
static __aicore__ void window_mean_impl(__gm__ float *const *srcs, __gm__ float *out) {
    using DynShapeDim5 = pto::Shape<1, 1, 1, ROWS, COLS>;
    using DynStridDim5 = pto::Stride<1, 1, 1, COLS, 1>;
    using GlobalData = pto::GlobalTensor<float, DynShapeDim5, DynStridDim5>;
    using TileData = pto::Tile<pto::TileType::Vec, float, ROWS, COLS, pto::BLayout::RowMajor, -1, -1>;

    TileData accTile(ROWS, COLS);
    TileData srcTile(ROWS, COLS);
    TASSIGN(accTile, 0x0);
    TASSIGN(srcTile, 0x10000);

    GlobalData firstGlobal(srcs[0]);
    TLOAD(accTile, firstGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    for (int k = 1; k < NUM_INPUTS; k++) {
        GlobalData srcGlobal(srcs[k]);
        TLOAD(srcTile, srcGlobal);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TADD(accTile, accTile, srcTile);
        pipe_barrier(PIPE_V);
        // srcTile is a single buffer reloaded every iteration, so the next
        // TLOAD (MTE2) overwrites what this TADD (V) just read. pipe_barrier
        // orders the vector pipe against itself only; the write-after-read
        // across pipes needs its own flag, and without it MTE2 runs ahead and
        // corrupts the accumulation on hardware.
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    }

    TMULS(accTile, accTile, 1.0f / static_cast<float>(NUM_INPUTS));
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    GlobalData dstGlobal(out);
    TSTORE(dstGlobal, accTile);
    pipe_sync();
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *in_tensors[NUM_INPUTS];
    for (int k = 0; k < NUM_INPUTS; k++) {
        in_tensors[k] = reinterpret_cast<__gm__ Tensor *>(args[k]);
    }
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[NUM_INPUTS]);

    int num_tiles = get_num_tiles(out_tensor, TILE_ELEMS);

    __gm__ float *in_base[NUM_INPUTS];
    for (int k = 0; k < NUM_INPUTS; k++) {
        in_base[k] = reinterpret_cast<__gm__ float *>(in_tensors[k]->buffer.addr) + in_tensors[k]->start_offset;
    }
    __gm__ float *out_base = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        __gm__ float *srcs[NUM_INPUTS];
        for (int k = 0; k < NUM_INPUTS; k++) {
            srcs[k] = in_base[k] + (tile_idx * TILE_ELEMS);
        }
        window_mean_impl<128, 128>(srcs, out_base + (tile_idx * TILE_ELEMS));
    }
}
