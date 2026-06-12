#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
CHIP_TIMING log parser — per-stage run_prepared() wall-clock breakdown (issue #1012).

Reads the opt-in ``[CHIP_TIMING]`` log lines emitted by the runtime when the log
level is lowered to V4 and prints one indented per-run tree, reconciled against
the authoritative ``host_wall`` / ``device_wall`` numbers the runtime already
reports.

The grammar (one line per event, fields are space-separated ``key=value``; the
surrounding log prefix is ignored):

    [CHIP_TIMING] run=<n> clk=host name=<stage> ev=<B|E> t=<ns>
    [CHIP_TIMING]          clk=dev  name=<stage> ev=<B|E> t=<ns> tid=<t>
    [CHIP_TIMING] run=<n> clk=host name=host_wall   ev=WALL us=<us>
    [CHIP_TIMING] run=<n> clk=dev  name=device_wall ev=WALL us=<us>

``t`` is nanoseconds in the emitting clock domain (host steady_clock, device
get_sys_cnt). Durations are only meaningful *within* a domain, so host stages
reconcile against host_wall and device stages against device_wall. The value
rides in the message body (never the log prefix), matching the device-log
convention in tools/benchmark_rounds.sh.

host events (single-threaded run_prepared) are paired as a nested B/E stack.
device events (multi-threaded) are reduced per name to min(B)..max(E), and
device rounds are segmented by tid reappearance like benchmark_rounds.sh — the
host run index is not available on-device.

To extend the breakdown, add another ``[CHIP_TIMING] ... ev=B`` / ``ev=E`` pair
anywhere in the runtime using the same grammar; no change to this tool is needed.

Usage:
    # sim (host + device logs both land on stderr)
    python -m simpler_setup.tools.chip_timing --log run.log

    # onboard (host stderr + CANN device log dir)
    python -m simpler_setup.tools.chip_timing --host-log host.log \
        --device-log-dir ~/ascend/log/debug/device-5

    # default device-log dir resolution mirrors benchmark_rounds.sh
    python -m simpler_setup.tools.chip_timing --host-log host.log --device-id 5
"""

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

# The device backend (CANN dlog) wraps the message in quotes, so a value must
# stop at the closing quote as well as whitespace.
_LINE_RE = re.compile(r"\[CHIP_TIMING\]\s+(.*)$")
_KV_RE = re.compile(r'(\w+)=([^\s"]+)')


@dataclass
class Event:
    clk: str  # "host" | "dev"
    name: str
    ev: str  # "B" | "E" | "WALL"
    run: Optional[int] = None
    t_ns: Optional[int] = None
    tid: Optional[int] = None
    us: Optional[float] = None


@dataclass
class Span:
    name: str
    depth: int
    dur_ns: int
    start_ns: int = 0


@dataclass
class RunReport:
    run: Optional[int]
    host_spans: list = field(default_factory=list)  # list[Span]
    dev_spans: list = field(default_factory=list)  # list[Span]
    host_wall_ns: Optional[int] = None
    device_wall_ns: Optional[int] = None
    notes: list = field(default_factory=list)  # list[str]


def parse_lines(lines):
    """Extract Event objects from an iterable of log lines (order preserved)."""
    events = []
    for line in lines:
        m = _LINE_RE.search(line)
        if not m:
            continue
        fields = dict(_KV_RE.findall(m.group(1)))
        if "clk" not in fields or "name" not in fields or "ev" not in fields:
            continue
        ev = Event(clk=fields["clk"], name=fields["name"], ev=fields["ev"])
        if "run" in fields:
            try:
                ev.run = int(fields["run"])
            except ValueError:
                pass
        if "t" in fields:
            try:
                ev.t_ns = int(fields["t"])
            except ValueError:
                pass
        if "tid" in fields:
            try:
                ev.tid = int(fields["tid"])
            except ValueError:
                pass
        if "us" in fields:
            try:
                ev.us = float(fields["us"])
            except ValueError:
                pass
        events.append(ev)
    return events


def _pair_host_stack(events):
    """Pair host B/E events into nested spans. Returns (spans, notes).

    Sorted by timestamp (B before E on ties) so interleaved stderr still nests
    correctly; LIFO pop matches the scoped-span emission order.
    """
    # On equal timestamps, close (E) before open (B) so a stage that ends exactly
    # when the next sibling begins still nests correctly under a coarse clock.
    ordered = sorted(
        [e for e in events if e.ev in ("B", "E") and e.t_ns is not None],
        key=lambda e: (e.t_ns, 0 if e.ev == "E" else 1),
    )
    spans = []
    notes = []
    stack = []  # list[Event] of open B's
    for e in ordered:
        if e.ev == "B":
            stack.append(e)
        else:  # E
            if not stack:
                notes.append(f"host: unmatched end for '{e.name}'")
                continue
            opener = stack.pop()
            if opener.name != e.name:
                notes.append(f"host: nesting mismatch ('{opener.name}' B / '{e.name}' E)")
            spans.append(Span(name=e.name, depth=len(stack), dur_ns=max(0, e.t_ns - opener.t_ns), start_ns=opener.t_ns))
    for leftover in stack:
        notes.append(f"host: missing end for '{leftover.name}'")
    # Pre-order for display: parent (earliest B) before its children.
    spans.sort(key=lambda s: s.start_ns)
    return spans, notes


def _reduce_dev(events):
    """Reduce device B/E events per name to min(B)..max(E). Returns (spans, notes).

    Nesting is inferred by interval containment so an outer ``aicpu_wall`` shows
    its inner phases indented.
    """
    by_name = {}  # name -> {"minB": int|None, "maxE": int|None}
    for e in events:
        if e.ev not in ("B", "E") or e.t_ns is None:
            continue
        slot = by_name.setdefault(e.name, {"minB": None, "maxE": None})
        if e.ev == "B":
            slot["minB"] = e.t_ns if slot["minB"] is None else min(slot["minB"], e.t_ns)
        else:
            slot["maxE"] = e.t_ns if slot["maxE"] is None else max(slot["maxE"], e.t_ns)

    intervals = []  # (name, start, end)
    notes = []
    for name, slot in by_name.items():
        if slot["minB"] is None or slot["maxE"] is None:
            notes.append(f"device: incomplete span for '{name}' (missing {'B' if slot['minB'] is None else 'E'})")
            continue
        intervals.append((name, slot["minB"], max(slot["minB"], slot["maxE"])))

    intervals.sort(key=lambda iv: (iv[1], -(iv[2] - iv[1])))  # by start, widest first
    spans = []
    for name, start, end in intervals:
        depth = sum(1 for _, s2, e2 in intervals if s2 <= start and e2 >= end and (s2, e2) != (start, end))
        spans.append(Span(name=name, depth=depth, dur_ns=end - start, start_ns=start))
    return spans, notes


def _segment_dev_rounds(dev_events):
    """Split device events into rounds by tid reappearance (benchmark_rounds.sh).

    A 'B' event whose tid already produced a 'B' in the current round opens a new
    round. Events keep file order. Single run with no repeats → one round.
    """
    rounds = []
    cur = []
    seen_b_tids = set()
    for e in dev_events:
        if e.ev == "B" and e.tid is not None and e.tid in seen_b_tids:
            rounds.append(cur)
            cur = []
            seen_b_tids = set()
        cur.append(e)
        if e.ev == "B" and e.tid is not None:
            seen_b_tids.add(e.tid)
    if cur:
        rounds.append(cur)
    return rounds


def assemble(events):
    """Build per-run reports from a flat event list."""
    # WALL baselines are emitted host-side with a run index for both host_wall
    # (clk=host) and device_wall (clk=dev), so route them by run, not by clk.
    wall_by_run = {}
    for e in events:
        if e.ev == "WALL":
            wall_by_run.setdefault(e.run, []).append(e)

    dev_events = [e for e in events if e.clk == "dev" and e.ev in ("B", "E")]

    # host runs keyed by run index (None bucketed together as a single run).
    host_by_run = {}
    for e in events:
        if e.clk == "host" and e.ev in ("B", "E"):
            host_by_run.setdefault(e.run, []).append(e)
    run_ids = sorted(host_by_run.keys(), key=lambda r: (r is None, r))

    # device rounds aligned to host runs by position (host run index absent on
    # device). WALL device_wall comes from the host side, keyed by run.
    dev_rounds = _segment_dev_rounds(dev_events)

    reports = []
    for idx, run_id in enumerate(run_ids):
        evs = host_by_run[run_id]
        rep = RunReport(run=run_id)
        spans, notes = _pair_host_stack(evs)
        rep.host_spans = spans
        rep.notes.extend(notes)
        for e in wall_by_run.get(run_id, []):
            if e.us is None:
                continue
            if e.name == "host_wall":
                rep.host_wall_ns = int(e.us * 1000)
            elif e.name == "device_wall":
                rep.device_wall_ns = int(e.us * 1000)
        if idx < len(dev_rounds):
            dspans, dnotes = _reduce_dev(dev_rounds[idx])
            rep.dev_spans = dspans
            rep.notes.extend(dnotes)
        reports.append(rep)

    # Device rounds with no matching host run (e.g. device-only log).
    for idx in range(len(run_ids), len(dev_rounds)):
        rep = RunReport(run=None)
        dspans, dnotes = _reduce_dev(dev_rounds[idx])
        rep.dev_spans = dspans
        rep.notes.extend(dnotes)
        rep.notes.append("device round without matching host run")
        reports.append(rep)

    return reports


def _ms(ns):
    return ns / 1e6


def _fmt_delta(measured_ns, baseline_ns, label):
    if baseline_ns is None:
        return ""
    delta = _ms(measured_ns - baseline_ns)
    return f"  ({label} {_ms(baseline_ns):.3f} ms, Δ {delta:+.3f} ms)"


def render(reports):
    """Render reports to a human-readable indented tree string."""
    out = []
    for rep in reports:
        label = f"run={rep.run}" if rep.run is not None else "run=?"
        out.append(f"[CHIP_TIMING] {label}")

        if rep.host_spans:
            out.append("  host (steady_clock):")
            for s in rep.host_spans:
                indent = "    " + "  " * s.depth
                suffix = ""
                if s.name == "run_prepared":
                    suffix = _fmt_delta(s.dur_ns, rep.host_wall_ns, "host_wall")
                out.append(f"{indent}{s.name:<20}{_ms(s.dur_ns):>10.3f} ms{suffix}")
            # Unattributed = run_prepared - sum(direct children).
            root = next((s for s in rep.host_spans if s.name == "run_prepared"), None)
            if root is not None:
                child_sum = sum(s.dur_ns for s in rep.host_spans if s.depth == 1)
                rem = root.dur_ns - child_sum
                if rem > 0:
                    out.append(f"      {'(unattributed)':<20}{_ms(rem):>10.3f} ms")

        if rep.dev_spans:
            out.append("  device (sys_cnt):")
            for s in rep.dev_spans:
                indent = "    " + "  " * s.depth
                suffix = ""
                if s.name == "aicpu_wall":
                    suffix = _fmt_delta(s.dur_ns, rep.device_wall_ns, "device_wall")
                out.append(f"{indent}{s.name:<20}{_ms(s.dur_ns):>10.3f} ms{suffix}")

        if not rep.host_spans and not rep.dev_spans:
            out.append("  (no spans parsed)")
        for note in rep.notes:
            out.append(f"  ! {note}")
        out.append("")
    return "\n".join(out)


def _read_files(paths):
    lines = []
    for p in paths:
        try:
            with open(p, errors="replace") as f:
                lines.extend(f.readlines())
        except OSError as exc:
            print(f"warning: cannot read {p}: {exc}", file=sys.stderr)
    return lines


def _resolve_device_log_dir(device_id):
    work = os.environ.get("ASCEND_WORK_PATH")
    candidates = []
    if work:
        candidates.append(os.path.join(work, "log", "debug"))
    candidates.append(os.path.join(os.path.expanduser("~"), "ascend", "log", "debug"))
    for root in candidates:
        d = os.path.join(root, f"device-{device_id}")
        if os.path.isdir(d):
            return d
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Parse [CHIP_TIMING] logs into a per-run wall-clock breakdown.")
    parser.add_argument("--log", action="append", default=[], help="combined log file (sim: host+device on stderr)")
    parser.add_argument("--host-log", action="append", default=[], help="host stderr log file")
    parser.add_argument("--device-log-dir", help="CANN device log dir (parses device-*.log inside)")
    parser.add_argument("--device-id", type=int, help="resolve device log dir like benchmark_rounds.sh")
    args = parser.parse_args(argv)

    paths = list(args.log) + list(args.host_log)
    dev_dir = args.device_log_dir
    if dev_dir is None and args.device_id is not None:
        dev_dir = _resolve_device_log_dir(args.device_id)
        if dev_dir is None:
            print(f"warning: no device log dir found for device-{args.device_id}", file=sys.stderr)
    if dev_dir:
        paths.extend(sorted(glob.glob(os.path.join(dev_dir, "device-*.log"))))

    if not paths:
        parser.error("no input: pass --log / --host-log / --device-log-dir / --device-id")

    events = parse_lines(_read_files(paths))
    if not events:
        print("no [CHIP_TIMING] lines found — was the log level lowered to V4?", file=sys.stderr)
        return 1
    reports = assemble(events)
    print(render(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
