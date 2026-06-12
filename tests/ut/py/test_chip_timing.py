# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for simpler_setup.tools.chip_timing — the [CHIP_TIMING] log parser (issue #1012)."""

from simpler_setup.tools.chip_timing import assemble, parse_lines, render

# A host log: one run_prepared envelope wrapping 5 sequential stages, plus the
# two authoritative WALL baselines. The log prefix is decorative — only the
# [CHIP_TIMING] substring matters. Timestamps in ns (steady_clock).
HOST_LINES = [
    "[V4] f: [CHIP_TIMING] run=0 clk=host name=run_prepared ev=B t=1000000000",
    "[V4] f: [CHIP_TIMING] run=0 clk=host name=attach ev=B t=1000100000",
    "[V4] f: [CHIP_TIMING] run=0 clk=host name=attach ev=E t=1000200000",
    "[V4] f: [CHIP_TIMING] run=0 clk=host name=bind_callable ev=B t=1000200000",
    "[V4] f: [CHIP_TIMING] run=0 clk=host name=bind_callable ev=E t=1001200000",
    "[V4] f: [CHIP_TIMING] run=0 clk=host name=bind_impl ev=B t=1001200000",
    "[V4] f: [CHIP_TIMING] run=0 clk=host name=bind_impl ev=E t=1101200000",
    "[V4] f: [CHIP_TIMING] run=0 clk=host name=runner_run ev=B t=1101200000",
    "[V4] f: [CHIP_TIMING] run=0 clk=host name=runner_run ev=E t=1121200000",
    "[V4] f: [CHIP_TIMING] run=0 clk=host name=validate ev=B t=1121200000",
    "[V4] f: [CHIP_TIMING] run=0 clk=host name=validate ev=E t=1121250000",
    "[V4] f: [CHIP_TIMING] run=0 clk=host name=run_prepared ev=E t=1121300000",
    "[V4] f: [CHIP_TIMING] run=0 clk=host name=host_wall ev=WALL us=121300.000",
    "[V4] f: [CHIP_TIMING] run=0 clk=dev name=device_wall ev=WALL us=19850.000",
]

# A device log line as the CANN dlog backend renders it: the message is wrapped
# in quotes (note the trailing "), exercising the value-parsing regression.
# aicpu_wall spans init-start..last-thread-exit; aicpu_exec is the inner window.
# Two exec threads (tids) exercise the min-B/max-E reduction.
DEV_LINES = [
    '[INFO] AICPU: 31125 init [V4] "[CHIP_TIMING] clk=dev name=aicpu_wall ev=B tid=31125 t=500000000"',
    '[INFO] AICPU: 31126 exec [V4] "[CHIP_TIMING] clk=dev name=aicpu_exec ev=B tid=31126 t=500100000"',
    '[INFO] AICPU: 31127 exec [V4] "[CHIP_TIMING] clk=dev name=aicpu_exec ev=B tid=31127 t=500150000"',
    '[INFO] AICPU: 31126 exec [V4] "[CHIP_TIMING] clk=dev name=aicpu_exec ev=E tid=31126 t=519600000"',
    '[INFO] AICPU: 31127 exec [V4] "[CHIP_TIMING] clk=dev name=aicpu_exec ev=E tid=31127 t=519700000"',
    '[INFO] AICPU: 31126 exec [V4] "[CHIP_TIMING] clk=dev name=aicpu_wall ev=E tid=31126 t=519800000"',
    '[INFO] AICPU: 31127 exec [V4] "[CHIP_TIMING] clk=dev name=aicpu_wall ev=E tid=31127 t=519850000"',
]


def _host_spans():
    reports = assemble(parse_lines(HOST_LINES))
    assert len(reports) == 1
    return {s.name: s for s in reports[0].host_spans}, reports[0]


def test_host_stage_durations_and_nesting():
    spans, _ = _host_spans()
    # ns deltas -> ms.
    assert spans["attach"].dur_ns == 100_000
    assert spans["bind_callable"].dur_ns == 1_000_000
    assert spans["bind_impl"].dur_ns == 100_000_000
    assert spans["runner_run"].dur_ns == 20_000_000
    assert spans["validate"].dur_ns == 50_000
    # run_prepared is the root (depth 0); the 5 stages are depth 1.
    assert spans["run_prepared"].depth == 0
    for name in ("attach", "bind_callable", "bind_impl", "runner_run", "validate"):
        assert spans[name].depth == 1


def test_host_wall_reconciliation():
    _, rep = _host_spans()
    root = next(s for s in rep.host_spans if s.name == "run_prepared")
    # run_prepared envelope == host_wall baseline exactly (same clock).
    assert root.dur_ns == 121_300_000
    assert rep.host_wall_ns == 121_300_000
    assert rep.device_wall_ns == 19_850_000


def test_device_reduction_min_b_max_e():
    reports = assemble(parse_lines(DEV_LINES))
    assert len(reports) == 1
    spans = {s.name: s for s in reports[0].dev_spans}
    # aicpu_wall: min B (500000000) .. max E (519850000).
    assert spans["aicpu_wall"].dur_ns == 19_850_000
    # aicpu_exec: min B over tids (500100000) .. max E over tids (519700000).
    assert spans["aicpu_exec"].dur_ns == 19_600_000
    # aicpu_exec nested under aicpu_wall.
    assert spans["aicpu_wall"].depth == 0
    assert spans["aicpu_exec"].depth == 1


def test_device_rounds_segmented_by_tid_reappearance():
    # Two back-to-back runs: tid 31125's second 'B' opens a new round.
    two_runs = DEV_LINES + DEV_LINES
    reports = assemble(parse_lines(two_runs))
    assert len(reports) == 2
    for rep in reports:
        names = {s.name for s in rep.dev_spans}
        assert names == {"aicpu_wall", "aicpu_exec"}


def test_combined_host_and_device_single_run():
    reports = assemble(parse_lines(HOST_LINES + DEV_LINES))
    assert len(reports) == 1
    rep = reports[0]
    assert rep.run == 0
    assert any(s.name == "run_prepared" for s in rep.host_spans)
    assert any(s.name == "aicpu_wall" for s in rep.dev_spans)


def test_missing_end_degrades_with_note():
    lines = [
        "[..][V4] f: [CHIP_TIMING] run=0 clk=host name=run_prepared ev=B t=1000000000",
        "[..][V4] f: [CHIP_TIMING] run=0 clk=host name=attach ev=B t=1000100000",
        # no attach E, no run_prepared E
    ]
    reports = assemble(parse_lines(lines))
    assert reports[0].notes  # at least one "missing end" note
    assert any("missing end" in n for n in reports[0].notes)


def test_quoted_device_value_parses_trailing_quote():
    # Regression: CANN dlog wraps the message in quotes; the last field must not
    # swallow the closing quote.
    events = parse_lines(DEV_LINES[:1])
    assert len(events) == 1
    assert events[0].t_ns == 500_000_000
    assert events[0].tid == 31125


def test_render_contains_tree_and_reconciliation():
    text = render(assemble(parse_lines(HOST_LINES + DEV_LINES)))
    assert "run_prepared" in text
    assert "host_wall" in text  # reconciliation suffix
    assert "device_wall" in text
    assert "aicpu_wall" in text
    assert "(unattributed)" in text


def test_no_chip_timing_lines_yields_no_events():
    assert parse_lines(["just some unrelated log line", "[INFO] hello world"]) == []
