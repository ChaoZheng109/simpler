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

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

#include "async_wait.h"
#include "backend/urma/urma_completion_scheduler.h"

namespace {

using pto::comm::urma::UrmaCqCtx;
using pto::comm::urma::UrmaInfo;
using pto::comm::urma::UrmaWQCtx;
using pto2::urma_backend::poll_urma_event_handle;

constexpr uint32_t kRows = 8;
constexpr uint32_t kDepth = 8;
constexpr uint32_t kCqeWords = 16;

struct alignas(64) TestCqe {
    std::array<uint32_t, kCqeWords> words{};
};

struct WorkspaceFixture {
    UrmaInfo info{};
    std::array<UrmaWQCtx, kRows> sq{};
    std::array<UrmaCqCtx, kRows> cq{};
    std::array<uint32_t, kRows> sq_head{};
    std::array<uint32_t, kRows> sq_tail{};
    std::array<uint32_t, kRows> sq_doorbell{};
    std::array<uint32_t, kRows> cq_tail{};
    std::array<uint32_t, kRows> cq_doorbell{};
    std::array<std::array<TestCqe, kDepth>, kRows> entries{};

    WorkspaceFixture() {
        info.rowsPerPeer = 4;
        info.rankCount = 2;
        info.sqPtr = reinterpret_cast<uint64_t>(sq.data());
        info.scqPtr = reinterpret_cast<uint64_t>(cq.data());
        info.layout = static_cast<decltype(info.layout)>(0);
        info.jettyCount = kRows;
        info.jettiesPerCore = 2;
        for (uint32_t row = 0; row < kRows; ++row) {
            sq[row].bufAddr = reinterpret_cast<uint64_t>(entries[row].data());
            sq[row].wqeShiftSize = 6;
            sq[row].depth = kDepth;
            sq[row].headAddr = reinterpret_cast<uint64_t>(&sq_head[row]);
            sq[row].tailAddr = reinterpret_cast<uint64_t>(&sq_tail[row]);
            sq[row].dbAddr = reinterpret_cast<uint64_t>(&sq_doorbell[row]);
            cq[row].bufAddr = reinterpret_cast<uint64_t>(entries[row].data());
            cq[row].cqeShiftSize = 6;
            cq[row].depth = kDepth;
            cq[row].tailAddr = reinterpret_cast<uint64_t>(&cq_tail[row]);
            cq[row].dbAddr = reinterpret_cast<uint64_t>(&cq_doorbell[row]);
        }
    }

    uint64_t addr() { return reinterpret_cast<uint64_t>(&info); }

    void set_cqe(uint32_t row, uint32_t sequence, bool ready, uint32_t status = 0, uint32_t substatus = 0) {
        const uint32_t lap_owner = (sequence / kDepth) & 1u;
        const uint32_t owner = ready ? lap_owner ^ 1u : lap_owner;
        entries[row][sequence & (kDepth - 1)].words[0] = (owner << 2) | (substatus << 16) | (status << 24);
    }

    CompletionPollResult poll(uint32_t row, uint32_t target_bb, uint32_t target_cqe) {
        return poll_urma_event_handle(((static_cast<uint64_t>(row) << 32) | target_bb), addr(), target_cqe);
    }
};

}  // namespace

TEST(A5UrmaCompletionScheduler, NullWorkspaceFailsEvenForZeroTarget) {
    const auto result = poll_urma_event_handle(0, 0, 0);
    EXPECT_EQ(result.state, CompletionPollState::FAILED);
    EXPECT_EQ(result.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
}

TEST(A5UrmaCompletionScheduler, ActualQueueRowIsIndependentOfPeerNumber) {
    WorkspaceFixture fixture;
    fixture.set_cqe(6, 0, true);

    EXPECT_EQ(fixture.poll(6, 1, 1).state, CompletionPollState::READY);
    EXPECT_EQ(fixture.cq_tail[6], 1u);
    EXPECT_EQ(fixture.sq_tail[6], 1u);
    EXPECT_EQ(fixture.cq_tail[1], 0u);
    EXPECT_EQ(fixture.sq_tail[1], 0u);
}

TEST(A5UrmaCompletionScheduler, QueueRowOutsideWorkspaceFails) {
    WorkspaceFixture fixture;
    EXPECT_EQ(fixture.poll(kRows, 1, 1).state, CompletionPollState::FAILED);
}

TEST(A5UrmaCompletionScheduler, EmptyWorkspaceMetadataFails) {
    WorkspaceFixture fixture;
    fixture.info.rankCount = 0;
    EXPECT_EQ(fixture.poll(0, 1, 1).state, CompletionPollState::FAILED);
}

TEST(A5UrmaCompletionScheduler, InvalidCqDepthFails) {
    WorkspaceFixture fixture;
    fixture.cq[0].depth = 3;
    EXPECT_EQ(fixture.poll(0, 1, 1).state, CompletionPollState::FAILED);
}

TEST(A5UrmaCompletionScheduler, MissingSqTailFails) {
    WorkspaceFixture fixture;
    fixture.sq[0].tailAddr = 0;
    EXPECT_EQ(fixture.poll(0, 1, 1).state, CompletionPollState::FAILED);
}

TEST(A5UrmaCompletionScheduler, IndependentBbAndCqeTargetsArePublished) {
    WorkspaceFixture fixture;
    fixture.set_cqe(0, 0, true);
    fixture.set_cqe(0, 1, true);

    EXPECT_EQ(fixture.poll(0, 5, 2).state, CompletionPollState::READY);
    EXPECT_EQ(fixture.sq_tail[0], 5u);
    EXPECT_EQ(fixture.cq_tail[0], 2u);
    EXPECT_EQ(fixture.cq_doorbell[0], 2u);
}

TEST(A5UrmaCompletionScheduler, GenericCompletionConditionForwardsCqeTarget) {
    WorkspaceFixture fixture;
    fixture.set_cqe(6, 0, true);
    fixture.set_cqe(6, 1, true);
    CompletionCondition condition{};
    condition.addr = ((static_cast<uint64_t>(6) << 32) | 5);
    condition.backend_cookie = fixture.addr();
    condition.expected_value = 2;
    condition.completion_type = COMPLETION_TYPE_URMA_EVENT_HANDLE;

    EXPECT_EQ(condition.test().state, CompletionPollState::READY);
    EXPECT_EQ(fixture.sq_tail[6], 5u);
    EXPECT_EQ(fixture.cq_tail[6], 2u);
}

TEST(A5UrmaCompletionScheduler, CqeWatermarkAloneDoesNotCompleteBbTarget) {
    WorkspaceFixture fixture;
    fixture.cq_tail[0] = 2;
    fixture.sq_tail[0] = 4;

    EXPECT_EQ(fixture.poll(0, 5, 2).state, CompletionPollState::PENDING);
    EXPECT_EQ(fixture.cq_tail[0], 2u);
    EXPECT_EQ(fixture.sq_tail[0], 4u);
}

TEST(A5UrmaCompletionScheduler, PendingPrefixPublishesNoPartialTargets) {
    WorkspaceFixture fixture;
    fixture.set_cqe(0, 0, true);
    fixture.set_cqe(0, 1, false);

    EXPECT_EQ(fixture.poll(0, 3, 2).state, CompletionPollState::PENDING);
    EXPECT_EQ(fixture.cq_tail[0], 0u);
    EXPECT_EQ(fixture.sq_tail[0], 0u);
    EXPECT_EQ(fixture.cq_doorbell[0], 0u);

    fixture.set_cqe(0, 1, true);
    EXPECT_EQ(fixture.poll(0, 3, 2).state, CompletionPollState::READY);
    EXPECT_EQ(fixture.cq_tail[0], 2u);
    EXPECT_EQ(fixture.sq_tail[0], 3u);
}

TEST(A5UrmaCompletionScheduler, ErrorPrefixNeverBecomesSuccessfulWatermark) {
    WorkspaceFixture fixture;
    fixture.set_cqe(0, 0, true);
    fixture.set_cqe(0, 1, true, 7);
    fixture.set_cqe(0, 2, true);

    EXPECT_EQ(fixture.poll(0, 2, 2).state, CompletionPollState::FAILED);
    EXPECT_EQ(fixture.cq_tail[0], 0u);
    EXPECT_EQ(fixture.sq_tail[0], 0u);
    EXPECT_EQ(fixture.cq_doorbell[0], 0u);
    EXPECT_EQ(fixture.poll(0, 2, 2).state, CompletionPollState::FAILED);
    EXPECT_EQ(fixture.poll(0, 3, 3).state, CompletionPollState::FAILED);
}

TEST(A5UrmaCompletionScheduler, SubstatusInPrefixFails) {
    WorkspaceFixture fixture;
    fixture.set_cqe(0, 0, true, 0, 3);

    const auto result = fixture.poll(0, 1, 1);
    EXPECT_EQ(result.state, CompletionPollState::FAILED);
    EXPECT_EQ(result.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(fixture.cq_tail[0], 0u);
    EXPECT_EQ(fixture.sq_tail[0], 0u);
}

TEST(A5UrmaCompletionScheduler, ZeroTargetRemainsPendingAcrossCounterWrap) {
    WorkspaceFixture fixture;
    fixture.cq_tail[0] = UINT32_MAX;
    fixture.sq_tail[0] = UINT32_MAX;
    fixture.set_cqe(0, UINT32_MAX, false);

    EXPECT_EQ(fixture.poll(0, 0, 0).state, CompletionPollState::PENDING);
    EXPECT_EQ(fixture.cq_tail[0], UINT32_MAX);
    EXPECT_EQ(fixture.sq_tail[0], UINT32_MAX);
}

TEST(A5UrmaCompletionScheduler, ZeroTargetCanCompleteAcrossCounterWrap) {
    WorkspaceFixture fixture;
    fixture.cq_tail[0] = UINT32_MAX;
    fixture.sq_tail[0] = UINT32_MAX;
    fixture.set_cqe(0, UINT32_MAX, true);

    EXPECT_EQ(fixture.poll(0, 0, 0).state, CompletionPollState::READY);
    EXPECT_EQ(fixture.cq_tail[0], 0u);
    EXPECT_EQ(fixture.sq_tail[0], 0u);
    EXPECT_EQ(fixture.cq_doorbell[0], 0u);
}

TEST(A5UrmaCompletionScheduler, IndependentTargetsCrossCounterWrap) {
    WorkspaceFixture fixture;
    fixture.cq_tail[0] = UINT32_MAX;
    fixture.sq_tail[0] = UINT32_MAX;
    fixture.set_cqe(0, UINT32_MAX, true);

    EXPECT_EQ(fixture.poll(0, 1, 0).state, CompletionPollState::READY);
    EXPECT_EQ(fixture.cq_tail[0], 0u);
    EXPECT_EQ(fixture.sq_tail[0], 1u);
}

TEST(A5UrmaCompletionScheduler, OwnerPhaseChangesAcrossRingWrap) {
    WorkspaceFixture fixture;
    fixture.cq_tail[0] = kDepth - 1;
    fixture.sq_tail[0] = kDepth - 1;
    fixture.set_cqe(0, kDepth - 1, true);
    fixture.set_cqe(0, kDepth, false);

    EXPECT_EQ(fixture.poll(0, kDepth + 1, kDepth + 1).state, CompletionPollState::PENDING);
    fixture.set_cqe(0, kDepth, true);
    EXPECT_EQ(fixture.poll(0, kDepth + 1, kDepth + 1).state, CompletionPollState::READY);
    EXPECT_EQ(fixture.cq_tail[0], kDepth + 1);
    EXPECT_EQ(fixture.sq_tail[0], kDepth + 1);
}

TEST(A5UrmaCompletionScheduler, DelayedObservationSurvivesCqeSlotReuse) {
    WorkspaceFixture fixture;
    fixture.cq_tail[0] = kDepth + 1;
    fixture.sq_tail[0] = kDepth + 1;
    fixture.cq_doorbell[0] = kDepth + 1;
    fixture.set_cqe(0, kDepth, true);

    EXPECT_EQ(fixture.poll(0, 1, 1).state, CompletionPollState::READY);
    EXPECT_EQ(fixture.cq_tail[0], kDepth + 1);
    EXPECT_EQ(fixture.sq_tail[0], kDepth + 1);
    EXPECT_EQ(fixture.cq_doorbell[0], kDepth + 1);
}

TEST(A5UrmaCompletionScheduler, NewerEventCanRetireBeforeOlderEventIsObserved) {
    WorkspaceFixture fixture;
    for (uint32_t sequence = 0; sequence < 4; ++sequence)
        fixture.set_cqe(0, sequence, true);

    EXPECT_EQ(fixture.poll(0, 6, 4).state, CompletionPollState::READY);
    EXPECT_EQ(fixture.poll(0, 2, 1).state, CompletionPollState::READY);
    EXPECT_EQ(fixture.sq_tail[0], 6u);
    EXPECT_EQ(fixture.cq_tail[0], 4u);
}

TEST(A5UrmaCompletionScheduler, MultipleRowsMakeIndependentProgress) {
    WorkspaceFixture fixture;
    fixture.set_cqe(2, 0, true);
    fixture.set_cqe(3, 0, false);

    EXPECT_EQ(fixture.poll(2, 1, 1).state, CompletionPollState::READY);
    EXPECT_EQ(fixture.poll(3, 1, 1).state, CompletionPollState::PENDING);
    fixture.set_cqe(3, 0, true);
    EXPECT_EQ(fixture.poll(3, 1, 1).state, CompletionPollState::READY);
    EXPECT_EQ(fixture.cq_tail[2], 1u);
    EXPECT_EQ(fixture.cq_tail[3], 1u);
}

TEST(A5UrmaCompletionScheduler, TargetOutsideOutstandingWindowFails) {
    WorkspaceFixture fixture;
    EXPECT_EQ(fixture.poll(0, kDepth + 1, kDepth + 1).state, CompletionPollState::FAILED);
    EXPECT_EQ(fixture.cq_tail[0], 0u);
    EXPECT_EQ(fixture.sq_tail[0], 0u);
}

static_assert(sizeof(TestCqe) == 64);
