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
 * Copy kernel: out = in
 *
 * Moves the recurrence's last runtime-allocated output into the caller's
 * buffer so the result is observable on the host.
 *
 * Args (Tensor*):
 *   args[0] = in  (INPUT)
 *   args[1] = out (OUTPUT_EXISTING)
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

static constexpr uint64_t TILE_ELEMS = 128 * 128;

static __aicore__ inline int get_num_tiles(__gm__ Tensor *tensor, uint64_t tile_elems) {
    uint64_t total_elems = tensor->shapes[0];
    return static_cast<int>(total_elems / tile_elems);
}

template <int ROWS, int COLS>
static __aicore__ void copy_impl(__gm__ float *src, __gm__ float *out) {
    using DynShapeDim5 = pto::Shape<1, 1, 1, ROWS, COLS>;
    using DynStridDim5 = pto::Stride<1, 1, 1, COLS, 1>;
    using GlobalData = pto::GlobalTensor<float, DynShapeDim5, DynStridDim5>;
    using TileData = pto::Tile<pto::TileType::Vec, float, ROWS, COLS, pto::BLayout::RowMajor, -1, -1>;

    TileData tile(ROWS, COLS);
    TASSIGN(tile, 0x0);

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    TLOAD(tile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, tile);
    pipe_sync();
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *in_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);

    int num_tiles = get_num_tiles(out_tensor, TILE_ELEMS);

    __gm__ float *in_base = reinterpret_cast<__gm__ float *>(in_tensor->buffer.addr) + in_tensor->start_offset;
    __gm__ float *out_base = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        copy_impl<128, 128>(in_base + (tile_idx * TILE_ELEMS), out_base + (tile_idx * TILE_ELEMS));
    }
}
