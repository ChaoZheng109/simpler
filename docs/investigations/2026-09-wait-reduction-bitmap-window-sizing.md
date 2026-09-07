# Widening the bounded WAIT-reduction bitmap past BL=64

**Date**: 2026-09-05
**Verdict**: BL=64 shipped (#2009, issue #1376); **BL=128 and BL=256
deferred**. On the two production workloads measured, the wider windows buy
either nothing at all (Qwen3) or twice the edge removals against a coverage
gain that the same measurement shows is worth no end-to-end time (DeepSeek-V4).
This entry also records the second, larger result: **BL=64 is end-to-end
neutral on both production workloads** — bounded transitive reduction
preserves WAIT reachability exactly, so no task's earliest start time moves,
and the bookkeeping it does make cheaper turns out to be ~5% of
dependency-pool pressure that is not on the critical path there.

**Amended 2026-09-07** — "neutral" is a property of those two workloads, not
of the mechanism. Both are near-chains: their per-task fanin degree is at most
1.5, so there is almost no wiring or fanout-walk work for reduction to remove.
Given a workload whose fanin degree is 16, the same BL=64 code is **−20.47%
`Orch` and −2.59% `Effective`**. See "Where the mechanism does pay" below.

## Question

Issue #1376 specifies a bounded transitive reduction of the WAIT dependency
graph: each submitted task publishes a fixed-width bitmap of recently
reachable WAIT ancestors, and a direct WAIT candidate inside that window is
removed when the bitmap proves another direct predecessor already provides a
transitive path. `BL` — the window width in submissions — is the one free
parameter, and acceptance #9 requires it be chosen from measurement rather
than intuition.

The intuition that makes a future contributor reach for a wider window: BL=64
is one machine word, which looks like an implementation convenience rather
than a data-driven choice. If real graphs have producer→consumer distances
well past 64 submissions, a 2- or 4-word bitmap would remove strictly more
edges for a bounded, fixed per-task cost.

## What was tried

**Offline coverage.** `simpler_setup/tools/wait_reduction_sim.py` replays a
`deps.json` capture (see `docs/dfx/dep-gen.md`) under two models: the exact
full-DAG transitive reduction as an upper bound, and a faithful mirror of the
runtime's `reduce_wait_edges` at each requested window. Captures taken on
a2a3 for Qwen3-14B decode and DeepSeek-V4 FLASH decode:

```bash
python -m simpler_setup.tools.wait_reduction_sim outputs/<case>_<ts>/deps.json --bl 64,128,256
```

**Onboard A/B.** 10 rounds per arm against merge-base `146370fe`, each arm
pinned to the same even die (Qwen3 on die 4, DeepSeek-V4 on dies 4+6) so no
row compares across dies — see the `onboard-measurement-pinning` habit.

**Runtime instrumentation.** Counters inside `reduce_wait_edges` and at
`wire_fanin_task`, plus an independent `scope_stats` run for dependency-pool
occupancy, on one DeepSeek-V4 decode step.

## Result

### Coverage: the wider window only moves one of the two workloads

| workload | full-DAG upper bound | BL64 | BL128 | BL256 |
| -------- | -------------------: | ---: | ----: | ----: |
| Qwen3-14B decode | 40 | 1 (2.50%) | 1 (2.50%) | 1 (2.50%) |
| DeepSeek-V4 FLASH decode | 21,698 | 10,065 (46.39%) | 13,678 (63.04%) | 20,214 (93.16%) |

Qwen3 gains **nothing** from a wider window: 39 of its 40 redundant edges are
cross-ring long edges that sit outside BL=256 as well. DeepSeek-V4 is the
window-sensitive case — BL=256 would remove **+10,149 edges, 2.0x** what BL=64
removes.

### Cost of widening

`WaitReachEntry` grows **16 → 24 → 40 B per slot** (1 → 1.5 → 2.5 MiB at the
default 65,536 slots), and the single-word shift-merge in `reduce_wait_edges`
becomes a 2- or 4-word one on the AICPU submit path — a per-submit cost paid
by **every** task, against a coverage gain only one of the two measured
workloads sees.

### End-to-end: BL=64 is neutral, and that bounds what widening could buy

| workload | metric | merge-base | BL64 | change |
| -------- | ------ | ---------: | ---: | -----: |
| Qwen3-14B decode | Effective | 35,093.0 µs | 35,098.0 µs | +0.01% |
| Qwen3-14B decode | Orch | 8,598.2 µs | 8,508.3 µs | −1.05% |
| DeepSeek-V4 FLASH decode | max-rank Effective | 29,858.0 µs | 30,182.9 µs | +1.09% |
| DeepSeek-V4 FLASH decode | max-rank Orch | 16,551.7 µs | 16,639.4 µs | +0.53% |

Every row sits inside the ±1.1% run-to-run band. **That is the expected
result, not a disappointment**: bounded transitive reduction preserves WAIT
reachability exactly, so no task's earliest start time moves. It can only make
the same schedule cheaper to account for.

### Why so little is left to make cheaper

Counting on one DeepSeek-V4 step:

| quantity | value |
| -------- | ----: |
| edges whose `DEP_WAIT` was cleared | 9,495 (`demote` 8,377 / `drop` 1,118) |
| of those, producer still live when the consumer was wired | ~965 |
| dep-pool entries actually saved (independent `scope_stats` run) | 990 of 19,114 (−5.2%) |
| dep-pool peak occupancy | 12,384 → 11,868 (−4.2%) |

About **90% of the reduced edges point at producers that had already reached
`CHIP_TASK_COMPLETED` by wiring time**. Those take the `completed_fanin`
branch and never call `dep_pool.prepend`, so removing them frees no dep-pool
entry and no completion-time traversal — only the readiness accounting and one
`fanout_lock` round trip each. The ~5% of dependency-pool pressure that
reduction does free is real and independently measured, but it is not on the
critical path of a step whose ~30 ms is AICore compute.

### Where the mechanism does pay (added 2026-09-07)

The paragraph above is a statement about *these graphs*, and the corpus that
produced it is uniformly sparse. Per-task fanin degree, from `--enable-dep-gen`
captures of the benchmark suite:

| case | tasks | WAIT edges | edges/task | full-DAG redundant |
| ---- | ----: | ---------: | ---------: | -----------------: |
| alternating_matmul_add (C1) | 1000 | 0 | 0.00 | 0 |
| benchmark_bgemm (C0) | 1000 | 750 | 0.75 | 0 |
| paged_attention_unroll (C1) | 1280 | 1280 | 1.00 | 256 |
| paged_attention_unroll (C2) | 576 | 704 | 1.22 | 192 |
| batch_paged_attention (C1) | 4112 | 6128 | 1.49 | 2032 |

Nothing in that corpus exceeds 1.5 edges per task, so there is almost no
wiring or completion-fanout work for reduction to remove — which is what
"neutral" was measuring.

`sliding_window_deps` (`out[i] = mean(base[i], out[i-1..i-16])`) puts the same
BL=64 code under a degree-16 graph built entirely from Step-A creator edges, no
explicit dependency. Measured at `steps=1000` on a2a3, same die, both arms
serial, 100 rounds, trimmed to 80:

| metric | pre-#2009 (`fab1a41e`) | #2009 | change |
| ------ | ---------------------: | ----: | -----: |
| Orch | 1879.8 ± 12.8 µs | 1495.0 ± 8.8 µs | **−20.47%** |
| Sched | 17133.5 ± 29.8 µs | 16689.7 ± 35.2 µs | **−2.59%** |
| Effective | 17135.3 ± 29.8 µs | 16691.4 ± 35.3 µs | **−2.59%** |
| Device wall | 17157.8 ± 29.7 µs | 16713.4 ± 35.2 µs | −2.59% |

dep_gen reports 1001 tasks / 15,865 WAIT edges / 14,865 redundant, and BL=64
removes **100%** of them — every redundant edge sits at submission distance
≤ 16.

The case exists for both architectures, since #2009 shipped the reduction to
a2a3 and a5 alike, and dep_gen on a5 produces the identical graph. The timing
above is a2a3 only: the host these numbers came from is a2a3 silicon, so the
a5 arm of the same comparison has not been run.

Two things this corrects:

- **`Orch` falls, it does not rise.** `wire_fanin_task` runs on the
  orchestrator thread and is charged to the `Orch` span, so dropping 15 of 16
  edges per task removes 15 `lock_fanout` round trips and 15
  `dep_pool.prepend` calls from *inside* that window. The per-submit bitmap
  cost is real but an order of magnitude smaller: 1878 → 1493 ns per task,
  **~26 ns saved per removed edge**. `Orch` only rises where reduction removes
  nothing.
- **The fixed baseline-first ordering is a confound and is not controlled
  for.** Both arms ran in that order inside one allocation, and the direction
  of any ordering effect on the device-side spans was never measured, so
  nothing here rules it out. What makes the gap hard to attribute to ordering
  alone is its size against the spread: `Orch` moves 385 µs where the
  per-round standard deviation is 9–13 µs. A swapped-order repeat is what
  would settle it.

### The simulator's resource columns overstate the runtime saving ~10x

`estimated_dep_pool_entries_removed` and
`estimated_readiness_fanout_nodes_removed` both assume one removed edge frees
one dependency-pool entry. The 90%-already-completed result above is exactly
why that does not hold: **they are edge-count upper bounds, not runtime
savings.** The tool's docstring and `simpler_setup/tools/README.md` say so.

Its *edge* counts, by contrast, hold up well — the measured `drop` count
matches its prediction exactly (1,118) and the total is within 6% (9,495
measured vs 10,065 predicted).

### Two DeepSeek-V4 aggregations that produce wrong answers

Worth recording because both look reasonable:

1. **Keeping decode step 0 in a run-total aggregation reports −37.8%**, which
   is one startup sample and not a steady-state effect. Step 0's two ranks
   differ **7x** (29.3 ms vs 214.9 ms) purely from launch skew: the slow
   rank's `Orch` window is normal at 17.1 ms while its `Sched` window holds
   all 214.7 ms of the spike, and that arm's host-side preamble is 218 ms
   longer. The shipped rows above take the maximum rank per decode step and
   drop step 0.

2. **Comparing the slow rank of the first step does not separate the arms.**
   Repeating the first step alone gives slow-rank spans of 37–215 ms on
   merge-base (n=4) against 29–55 ms at BL64 (n=5): the two sets interleave,
   and a rank-sum test does not separate them (U=5 against a critical value of
   1 at n=4/5). Which rank is the slow one flips between runs on both arms.

Isolating the **fast** rank — the one not waiting on its peer — is the only
clean first-step signal:

| first step, fast rank | merge-base | BL64 | change |
| --------------------- | ---------: | ---: | -----: |
| Effective | 29,262.4 ± 51 µs (n=4) | 29,218.1 ± 107 µs (n=5) | −0.15% |
| Orch | 17,611.8 ± 362 µs | 17,896.0 ± 310 µs | +1.61% |

At 0.2–0.4% measurement noise the Effective gap is nothing, so there is no
cold-start effect to claim. The `Orch` row leans the other way by less than
one standard deviation, which is the shape to expect: reduction pays
per-submit bitmap work up front and, on this workload, recovers too little
bookkeeping to earn it back.

## Why not (now)

BL=64 is kept because it is one native word: the shift-merge is a single
instruction, `d == BL` is a natural no-shift boundary case, and the side
storage stays at 1 MiB. Widening would double DeepSeek-V4's edge removals —
but the same measurement campaign shows that on **this** workload the removals
themselves buy no end-to-end time, so paying multiword submit work for more of
them has no measured return. Qwen3 would pay the cost for zero additional
coverage.

The reason BL=64 shipped at all, given a neutral A/B, is not latency: it is
that the mechanism is correct and exact inside its window, it takes ~5% off
dependency-pool pressure and 4.2% off peak occupancy (headroom that matters
for dense graphs against `Dependency Pool Deadlock`), and the two-pass
structure generalizes to N words unchanged so the parameter stays open.

## When to reconsider

Revisit when a capture shows **both**:

- `BL=64 removed` well under the full-DAG upper bound, **and**
- `pct_pairs_within_window` — not cross-ring distance — as the binding
  constraint.

Qwen3 fails the second test today (its misses are cross-ring long edges that
BL=256 does not reach either), which is why "BL=64 only removes 1 of 40" is
*not* on its own a reason to widen.

Widening is a constant-size change once an onboard A/B shows the extra
removals repay the multiword submit work. Re-measure with:

```bash
python -m simpler_setup.tools.wait_reduction_sim <deps.json> --bl 64,128,256
```

## References

- Issue #1376 (bounded reachability bitmap), blocked by and unblocked by
  #1375 (independent WAIT/RETAIN representation).
- PR #2009 — the shipped BL=64 implementation.
- Mechanism (shipped, not this doc's subject): step 5.5 in
  `src/{a2a3,a5}/runtime/tensormap_and_ringbuffer/docs/RUNTIME_LOGIC.md`.
- Tool: `simpler_setup/tools/README.md#wait_reduction_sim`;
  capture format `docs/dfx/dep-gen.md`.
- `.claude/rules/discipline.md` §4 — why this entry exists rather than
  living only in the PR description.
