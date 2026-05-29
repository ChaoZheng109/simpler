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
 * Unit tests for l0_decode_chunks() in common/l0_perf_profiling.h.
 *
 * The 4-byte biu_perf chunk layout decoded here is the authoritative one from
 * msprof's BiuPerfInstructionBean + biu_perf_chip6_parser (see the header):
 *   bits[31:28]=ctrl_type, bits[27:16]=region_id, bits[15:0]=sys_cnt delta.
 * START_STAMP (ctrl 14) rebases an absolute 64-bit cycle from 4 chunks' low-16
 * bits (most-significant chunk first); normal chunks accumulate the delta.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "common/l0_perf_profiling.h"

namespace {

struct Stamp {
    uint16_t region;
    uint8_t pipe;
    uint64_t cycle;
};

// Encode one 4-byte chunk (host is little-endian, matching the device).
uint32_t make_chunk(uint8_t ctrl, uint16_t region, uint16_t delta) {
    return (static_cast<uint32_t>(ctrl & 0xF) << 28) | (static_cast<uint32_t>(region & 0xFFF) << 16) | delta;
}

void append_word(std::vector<uint8_t> &buf, uint32_t w) {
    buf.push_back(static_cast<uint8_t>(w & 0xFF));
    buf.push_back(static_cast<uint8_t>((w >> 8) & 0xFF));
    buf.push_back(static_cast<uint8_t>((w >> 16) & 0xFF));
    buf.push_back(static_cast<uint8_t>((w >> 24) & 0xFF));
}

// A 4-chunk START_STAMP group encoding the 16-bit low word `base_low` into the
// least-significant slot (chunk0) and zero elsewhere → base_cycle == base_low.
void append_start_stamp(std::vector<uint8_t> &buf, uint16_t base_low) {
    append_word(buf, make_chunk(kL0CtrlStartStamp, /*region=*/0, /*delta=*/base_low));
    append_word(buf, make_chunk(kL0CtrlStartStamp, 0, 0));
    append_word(buf, make_chunk(kL0CtrlStartStamp, 0, 0));
    append_word(buf, make_chunk(kL0CtrlStartStamp, 0, 0));
}

std::vector<Stamp> decode_all(const std::vector<uint8_t> &buf, L0DecodeState &st) {
    std::vector<Stamp> out;
    l0_decode_chunks(buf.data(), static_cast<int>(buf.size()), st, [&](uint16_t r, uint8_t p, uint64_t c, uint16_t) {
        out.push_back(Stamp{r, p, c});
    });
    return out;
}

}  // namespace

// A START_STAMP base followed by two pipe stamps accumulates deltas correctly.
TEST(L0Decode, BaseAndDeltaAccumulation) {
    std::vector<uint8_t> buf;
    append_start_stamp(buf, 0x1234);
    append_word(buf, make_chunk(/*ctrl=*/4, /*region=*/11, /*delta=*/100));   // MTE2
    append_word(buf, make_chunk(/*ctrl=*/1, /*region=*/12, /*delta=*/50));    // VEC

    L0DecodeState st;
    auto stamps = decode_all(buf, st);

    ASSERT_EQ(stamps.size(), 2u);
    EXPECT_EQ(stamps[0].region, 11);
    EXPECT_EQ(stamps[0].pipe, 4);
    EXPECT_EQ(stamps[0].cycle, 0x1234u + 100u);
    EXPECT_EQ(stamps[1].region, 12);
    EXPECT_EQ(stamps[1].pipe, 1);
    EXPECT_EQ(stamps[1].cycle, 0x1234u + 100u + 50u);
}

// STATE chunks (ctrl 15) accumulate their delta but emit no record; sentinels
// are skipped entirely.
TEST(L0Decode, StateAndSentinelsSkipped) {
    std::vector<uint8_t> buf;
    append_start_stamp(buf, 0);
    append_word(buf, make_chunk(/*ctrl=*/4, /*region=*/11, /*delta=*/10));      // MTE2 -> cycle 10
    append_word(buf, make_chunk(kL0CtrlState, /*bitmask=*/0x7F, /*delta=*/5));  // STATE: +5, no emit
    append_word(buf, kL0ChunkEndFlag);                                         // block end sentinel
    append_word(buf, kL0ChunkFiller);                                          // padding
    append_word(buf, make_chunk(/*ctrl=*/2, /*region=*/12, /*delta=*/3));      // CUBE -> cycle 10+5+3

    L0DecodeState st;
    auto stamps = decode_all(buf, st);

    ASSERT_EQ(stamps.size(), 2u);
    EXPECT_EQ(stamps[0].pipe, 4);
    EXPECT_EQ(stamps[0].cycle, 10u);
    EXPECT_EQ(stamps[1].pipe, 2);
    EXPECT_EQ(stamps[1].region, 12);
    EXPECT_EQ(stamps[1].cycle, 10u + 5u + 3u);  // STATE delta kept the base aligned
}

// A START_STAMP group split across two prof_channel_read buffers is carried
// over and decoded correctly on the second call.
TEST(L0Decode, StartStampSplitAcrossReads) {
    std::vector<uint8_t> full;
    append_start_stamp(full, 0x0042);
    append_word(full, make_chunk(/*ctrl=*/5, /*region=*/7, /*delta=*/9));  // MTE3

    // Split mid-START_STAMP group (after 2 of the 4 stamp chunks = 8 bytes).
    std::vector<uint8_t> part1(full.begin(), full.begin() + 8);
    std::vector<uint8_t> part2(full.begin() + 8, full.end());

    L0DecodeState st;
    auto s1 = decode_all(part1, st);
    EXPECT_TRUE(s1.empty());  // group incomplete → nothing emitted, carried over
    auto s2 = decode_all(part2, st);

    ASSERT_EQ(s2.size(), 1u);
    EXPECT_EQ(s2[0].region, 7);
    EXPECT_EQ(s2[0].pipe, 5);
    EXPECT_EQ(s2[0].cycle, 0x42u + 9u);
}

// A sub-4-byte trailing fragment is carried to the next call and completed.
TEST(L0Decode, PartialChunkCarry) {
    std::vector<uint8_t> full;
    append_start_stamp(full, 0);
    append_word(full, make_chunk(/*ctrl=*/1, /*region=*/3, /*delta=*/8));  // VEC

    // Cut 2 bytes off the last chunk.
    std::vector<uint8_t> part1(full.begin(), full.end() - 2);
    std::vector<uint8_t> part2(full.end() - 2, full.end());

    L0DecodeState st;
    auto s1 = decode_all(part1, st);
    EXPECT_TRUE(s1.empty());
    auto s2 = decode_all(part2, st);

    ASSERT_EQ(s2.size(), 1u);
    EXPECT_EQ(s2[0].region, 3);
    EXPECT_EQ(s2[0].pipe, 1);
    EXPECT_EQ(s2[0].cycle, 8u);
}
