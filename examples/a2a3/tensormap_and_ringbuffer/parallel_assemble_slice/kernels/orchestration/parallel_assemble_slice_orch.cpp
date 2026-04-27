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
 * Reproducer orchestration for issue #663.
 *
 * Models the pypto-lib lowering of a `pl.parallel(0, batch, 1)` loop whose body
 * does an `assemble` (in-place write into a per-iteration slice of a shared
 * tensor X) followed by a `slice` (read from the same per-iteration slice).
 *
 * For each iteration b in [0, batch):
 *   step1 (assemble): kernel_add_scalar(src.view(b), X.view(b) INOUT, +0.0f)
 *                     → copies src region b into X region b (in-place on X).
 *   step2 (slice):    kernel_add_scalar(X.view(b),   out.view(b)     , +0.0f)
 *                     → copies X region b into out region b.
 *
 * Expected: out == src on every region. Real hardware satisfies this. The
 * a2a3 simulator (per issue #663) is reported to corrupt subsequent slice
 * reads of the shared tensor X for batch >= 2.
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_ADD_SCALAR 0

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 3,  // src, out, X
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    Tensor ext_src = from_tensor_arg(orch_args.tensor(0));
    Tensor ext_out = from_tensor_arg(orch_args.tensor(1));
    Tensor ext_X = from_tensor_arg(orch_args.tensor(2));

    constexpr uint32_t CHUNK = 128 * 128;  // kernel_add_scalar hard-coded tile
    uint32_t total = orch_args.tensor(0).shapes[0];
    uint32_t batch = total / CHUNK;

    LOG_INFO("issue663 repro: batch=%u CHUNK=%u total=%u", batch, CHUNK, total);

    uint32_t view_shapes[1] = {CHUNK};

    // Sequential lowering of `for b in pl.parallel(0, batch, 1): ...`.
    // The simulator bug surfaces when the same shared tensor X is updated and
    // re-read across iterations within a single orchestration entry.
    for (uint32_t b = 0; b < batch; ++b) {
        uint32_t off[1] = {b * CHUNK};
        Tensor src_b = ext_src.view(view_shapes, off);
        Tensor X_b = ext_X.view(view_shapes, off);
        Tensor out_b = ext_out.view(view_shapes, off);

        // step1 (assemble): X[b] := src[b] + 0
        {
            Arg p;
            p.add_input(src_b);
            p.add_inout(X_b);
            p.add_scalar(0.0f);
            p.add_scalar(static_cast<uint32_t>(CHUNK));
            pto2_rt_submit_aiv_task(FUNC_ADD_SCALAR, p);
        }

        // step2 (slice):    out[b] := X[b] + 0
        {
            Arg p;
            p.add_input(X_b);
            p.add_output(out_b);
            p.add_scalar(0.0f);
            p.add_scalar(static_cast<uint32_t>(CHUNK));
            pto2_rt_submit_aiv_task(FUNC_ADD_SCALAR, p);
        }
    }
}

}  // extern "C"
