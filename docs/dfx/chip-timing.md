# Chip Timing — per-stage `run_prepared()` wall-clock breakdown

`CHIP_TIMING` is an opt-in DFX facility that breaks a single `run_prepared()`
into per-stage durations on both the host and the device, reconciled against the
two wall numbers the runtime already reports (`host_wall_ns` / `device_wall_ns`,
see [profiling-framework.md](../profiling-framework.md)). It answers "where did
the time go" — host binding vs device execution — without intrusive
instrumentation: the runtime emits a handful of `[CHIP_TIMING]` log lines, and
the offline parser [`simpler_setup/tools/chip_timing.py`](../../simpler_setup/tools/chip_timing.py)
assembles them into an indented tree.

It is deliberately **minimal and extensible**: the runtime seeds only the
outermost boundaries (enough to reproduce `host_wall` / `device_wall`). To drill
into any stage, add one more `ev=B` / `ev=E` pair in the same grammar — the tool
needs no change.

## Enabling

The lines are emitted on the **V4** log tier, which is hidden at the default V5
threshold (V4 < V5). Lower the log level to V4 to collect them:

```bash
python <example>/test_*.py -p a5 --log-level v4 2>&1 | tee run.log
```

V4 (rather than V0/V1) keeps the collateral chatter low — only V4..V9 lines are
unmuted, not the V0 flood. When disabled, the emit path short-circuits before
formatting, so there is no measurable overhead.

## Grammar

One event per line. The surrounding log prefix (host timestamp or CANN dlog
header) is ignored — the timestamp rides in the message body as nanoseconds in
the **emitting clock's** domain, mirroring the embedded-cycle convention in
[`tools/benchmark_rounds.sh`](../../tools/benchmark_rounds.sh):

```text
[CHIP_TIMING] run=<n> clk=host name=<stage> ev=<B|E> t=<ns>
[CHIP_TIMING]          clk=dev  name=<stage> ev=<B|E> t=<ns> tid=<t>
[CHIP_TIMING] run=<n> clk=host name=host_wall   ev=WALL us=<us>
[CHIP_TIMING] run=<n> clk=dev  name=device_wall ev=WALL us=<us>
```

- `clk=host` — `t` is host `steady_clock` ns (the clock backing `host_wall_ns`).
- `clk=dev` — `t` is ns derived from `get_sys_cnt_aicpu()` (the counter backing
  `device_wall_ns`); `tid` distinguishes the AICPU threads.
- `ev=WALL` — the authoritative baseline, emitted host-side after the run with
  the value in the body. Both `host_wall` and `device_wall` carry the host
  `run` index.

Host and device are **separate clock domains**: durations are only meaningful
*within* a domain. Host stages reconcile against `host_wall`; device stages
against `device_wall`. There is no cross-domain absolute alignment.

## Seeded stages

| clk | name | bounds |
| --- | ---- | ------ |
| host | `run_prepared` | the whole envelope (== `host_wall`) |
| host | `attach` / `bind_callable` / `bind_impl` / `runner_run` / `validate` | the five `run_prepared` stages (onboard; sim has no `attach`) |
| dev | `aicpu_wall` | `simpler_aicpu_init` start .. last `simpler_aicpu_exec` exit (== `device_wall`) |
| dev | `aicpu_exec` | the `aicpu_execute()` window inside each AICPU thread |

Source: host in [`c_api_shared.cpp`](../../src/common/platform/onboard/host/c_api_shared.cpp)
(onboard) and its sim counterpart; device in
[`kernel.cpp`](../../src/a5/platform/onboard/aicpu/kernel.cpp) (onboard a5/a2a3)
and the sim `device_runner.cpp` execution loop.

## Tool usage

```bash
# sim — host + device logs both land on stderr
python -m simpler_setup.tools.chip_timing --log run.log

# onboard — host stderr + CANN device log dir
python -m simpler_setup.tools.chip_timing --host-log host.log \
    --device-log-dir ~/ascend/log/debug/device-5
#   ... or let it resolve the dir like benchmark_rounds.sh:
python -m simpler_setup.tools.chip_timing --host-log host.log --device-id 5
```

How it assembles events:

- **host** (single-threaded) — B/E paired as a nested stack → `run_prepared`
  with its children; `(unattributed)` is the remainder vs the child sum.
- **device** (multi-threaded) — reduced per name to `min(B) .. max(E)`; rounds
  are segmented by tid reappearance (the host `run` index is not available
  on-device), so point `--device-log-dir` at a fresh log. Stale rounds with no
  matching host run print under `run=?`.

Example output:

```text
[CHIP_TIMING] run=0
  host (steady_clock):
    run_prepared           121.300 ms  (host_wall 121.300 ms, Δ +0.000 ms)
      attach                   0.100 ms
      bind_callable            1.000 ms
      bind_impl              100.000 ms
      runner_run              20.000 ms
      validate                 0.050 ms
      (unattributed)           0.150 ms
  device (sys_cnt):
    aicpu_wall              19.850 ms  (device_wall 19.850 ms, Δ +0.000 ms)
      aicpu_exec              19.600 ms
```

The `Δ` against `host_wall` / `device_wall` is the built-in sanity check: a
large delta means a boundary is missing or mis-paired.

## Extending

To attribute time inside a stage (e.g. split `bind_impl` into
`args_malloc_copy` / `prebuilt_arena`, or `aicpu_exec` into `so_load` /
`graph_build`), emit a nested `ev=B` / `ev=E` pair with the same grammar at the
new boundary. Host uses `steady_clock` ns; device uses
`cycles_to_us(get_sys_cnt_aicpu()) * 1000`. No parser change is required —
nesting is inferred from the timestamps.

Tests: [`tests/ut/py/test_chip_timing.py`](../../tests/ut/py/test_chip_timing.py).
