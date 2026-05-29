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
 * @file l0_perf_collector_aicpu.h
 * @brief AICPU-side core-swimlane (biu_perf / L0) marker producer (a5).
 *
 * Architecture: the consumer (prof_drv_start / prof_channel_read / prof_stop)
 * lives on the **host** L0PerfCollector. AICPU:
 *   - builds per-thread logical_core_id → (biu_perf group, sub-core block)
 *     tables at init, and caches per-core L0PerfAicoreRing pointers from the
 *     shared header
 *   - on task FIN, reads the AICore-published (start_cycle, end_cycle) from
 *     the per-core ring slot and pushes an enriched L0TaskFinMarker
 *     {real_task_id, group, start_cycle, end_cycle, core_type, block} into
 *     the per-thread GM ready queue
 *
 * Driver consumer calls on AICPU return DRV_ERROR_NOT_SUPPORT (CANN 9.1.T500
 * device-stub never implemented them), which is why the consumer side has
 * to live on host.
 *
 * Lifecycle (called from the scheduler):
 *   l0_perf_aicpu_init()              — per scheduler thread, AFTER it knows
 *                                       its cores_owned[]. Builds the
 *                                       logical→(group, block) tables and
 *                                       caches the per-core ring pointers.
 *                                       Pure in-memory; never fails.
 *   [task loop]
 *     l0_perf_aicpu_complete_record() — after each task FIN, reads the AICore
 *                                       ring slot, builds the enriched
 *                                       marker, pushes it into the
 *                                       per-thread ready queue.
 */

#ifndef PLATFORM_AICPU_L0_PERF_COLLECTOR_AICPU_H_
#define PLATFORM_AICPU_L0_PERF_COLLECTOR_AICPU_H_

#include <cstdint>

#include "common/core_type.h"
#include "common/l0_perf_profiling.h"

/**
 * L0 perf platform setters — called by the AICPU kernel entry (onboard) before
 * l0_perf_aicpu_init() so AICPU code can read perf state without reaching into
 * the generic Runtime struct. Profiling state lives in KernelArgs, never Handshake.
 */
extern "C" void set_platform_l0_perf_base(uint64_t l0_perf_data_base);
extern "C" uint64_t get_platform_l0_perf_base();
extern "C" void set_l0_swimlane_enabled(bool enable);
extern "C" bool is_l0_swimlane_enabled();

/**
 * True if the AICore at `sub_core_phys_id` belongs to a cluster monitored by
 * biu_perf (i.e. its cluster is in kBiuPerfPhysAicore[]). The scheduler uses
 * this to retire uncovered workers when L0 swimlane is on: without retiring
 * them, dispatch would still spread to uncovered clusters and most tasks
 * would produce no per-pipe data — `aclrtSetDeviceResLimit` only caps the
 * concurrency count, it cannot lock placement to specific clusters on a5.
 *
 * `is_aic` selects the AIC vs AIV sub-core phys layout (see
 * l0_sub_core_phys_to_cluster_phys for the dual-die mapping).
 */
inline bool l0_perf_cluster_is_covered(uint32_t sub_core_phys_id, bool is_aic) {
    int cluster_phys = l0_sub_core_phys_to_cluster_phys(sub_core_phys_id, is_aic);
    if (cluster_phys < 0) {
        return false;
    }
    return l0_perf_phys_to_group(static_cast<uint32_t>(cluster_phys)) >= 0;
}

/**
 * Per-scheduler-thread L0 init. Builds the per-logical-core lookups used by
 * complete_record:
 *   - group  (0..5, or -1 if outside biu_perf 6-cluster coverage)
 *   - block  (0=aic, 1=aiv0, 2=aiv1 within the cluster — directly indexes
 *             the matching biu_perf sub-core channel)
 *   - aicore ring pointer (from L0PerfDataHeader::aicore_ring_ptrs[])
 *
 * `phys_ids[i]` is the *sub-core* phys id of `cores_owned[i]`
 * (= physical_core_ids_[cores_owned[i]] in the scheduler). Sub-core phys is
 * converted to global cluster phys (via l0_sub_core_phys_to_cluster_phys, see
 * l0_perf_profiling.h for the dual-die formula) and the residue inside the
 * cluster gives the sub-core block: AIC → block 0; AIV → block 1 if the
 * sub-core's offset within its die's AIV range is even, else block 2.
 *
 * Core type (AIC vs AIV) is derived from the sub-core's phys range via
 * `l0_sub_core_phys_is_aic`. We can't use `logical < aic_count` here
 * because L0 retire leaves logical IDs sparse (e.g. surviving AICs at
 * logical {0,9,17,18,27,35}, not contiguous).
 *
 * Pure in-memory operation; **never fails**. No driver calls.
 *
 * @param thread_idx   AICPU scheduler thread index
 * @param cores_owned  logical core ids this thread schedules
 * @param phys_ids     sub-core phys id for each entry in cores_owned
 * @param core_num     entries in cores_owned / phys_ids
 * @return always 0
 */
int l0_perf_aicpu_init(int thread_idx, const int *cores_owned, const uint32_t *phys_ids, int core_num);

/**
 * Enrich and publish an L0TaskFinMarker for the just-finished task. Reads the
 * AICore-published (start_cycle, end_cycle) out of the per-core L0PerfAicoreRing
 * slot, validates the reg-dispatch token, then pushes the marker into the
 * per-thread GM ready queue. Host L0PerfCollector pops it, drains the
 * group's 3 sub-core trace channels, and attributes each decoded pipe stamp
 * to the task whose cycle window contains it.
 *
 * No-op if the core is outside the biu_perf 6-core coverage (group lookup is
 * -1). Returns -1 on hard error (ring missing, reg-token mismatch) so the
 * scheduler can log it; the host can still proceed since stamps without a
 * marker simply stay unattributed.
 *
 * @param core_id               logical core that just completed a task
 * @param thread_idx            AICPU scheduler thread index (selects ready_queue slot)
 * @param expected_reg_task_id  register dispatch token to validate the slot
 * @param task_id               full PTO2 task id to carry in the marker
 * @param core_type             CoreType of the owning task (from hank[core_id].core_type)
 */
int l0_perf_aicpu_complete_record(
    int core_id, int thread_idx, uint32_t expected_reg_task_id, uint64_t task_id, CoreType core_type
);

/**
 * Flush this thread's partial (non-full) marker buffer into the ready queue at
 * thread exit, so the tail markers that didn't fill a buffer still reach the
 * host. Mirrors l2_perf_aicpu_flush_buffers. Call once per scheduler thread
 * after its dispatch loop ends.
 *
 * @param thread_idx AICPU scheduler thread index
 */
void l0_perf_aicpu_flush(int thread_idx);

#endif  // PLATFORM_AICPU_L0_PERF_COLLECTOR_AICPU_H_
