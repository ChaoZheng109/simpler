#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reproducer for issue #663.

Models the pypto-lib lowering of:
    for b in pl.parallel(0, BATCH, 1):
        X_view_b = pl.slice(X, [CHUNK], [b*CHUNK])      # view of shared tensor X
        X_view_b = pl.assemble(X_view_b, src_view_b)    # write into X region b  (INOUT)
        out_view_b = pl.slice(X, [CHUNK], [b*CHUNK])    # read X region b again
        out_view_b -> out region b                      # consume

Lowered to a sequential simpler orchestration that, for each iteration b, does:
  step1 (assemble): kernel_add_scalar(src.view(b), X.view(b)_INOUT, scalar=0)
  step2 (slice):    kernel_add_scalar(X.view(b),  out.view(b),       scalar=0)

Expected: out == src for every region b. Real hardware passes; the a2a3 simulator
fails for BATCH >= 2 because it does not properly isolate SSA tensor versions
across the per-region writes to the shared tensor X.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestParallelAssembleSlice(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/parallel_assemble_slice_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.OUT, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aiv/kernel_add_scalar.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
        ],
    }

    CHUNK = 128 * 128  # kernel_add_scalar hard-codes a 128x128 tile

    CASES = [
        {
            "name": "Batch1_pass_expected",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 3},
            "params": {"batch": 5},
        },
        {
            "name": "Batch2_repro_issue663",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 3},
            "params": {"batch": 10},
        },
        {
            "name": "Batch4_repro_issue663",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 3},
            "manual": True,
            "params": {"batch": 4},
        },
    ]

    def generate_args(self, params):
        batch = params["batch"]
        size = batch * self.CHUNK
        # src: distinct value per region so any cross-iteration mix-up shows up
        src = torch.empty(size, dtype=torch.float32)
        for b in range(batch):
            src[b * self.CHUNK : (b + 1) * self.CHUNK] = float(b + 1)
        return TaskArgsBuilder(
            Tensor("src", src),
            Tensor("out", torch.zeros(size, dtype=torch.float32)),
            Tensor("X", torch.zeros(size, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        # out == src per-region; X ends up holding src too (assembled in-place).
        args.out[:] = args.src
        args.X[:] = args.src


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
