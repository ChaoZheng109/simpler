# Qwen3-14B HBG Decode

This directory contains a fixed-shape Torch-NPU eager adapter for the Qwen3-14B
HBG release. It uses BF16 weights and KV on A3, the stage-0 25-argument ABI,
one active request, the existing borrowed-context Worker API, caller-stream
ordering, RunHandle retirement, and explicit close.

The callable has no inactive-row mask. Row 0 carries the request. Rows 1–15
execute the same token and position with private KV pages, and only row 0 logits
are returned. Padding rows avoid every page reserved in the request's block
table, including pages needed by later decode steps. A 32-column table is
accepted only when columns beyond the 28-column ABI contain `-1`.

## Artifact

`build_hbg_artifact.py` accepts a generated Qwen decode artifact and produces a
`host_build_graph` release. The reader verifies the distributed metadata,
callable source files, registration order, tensor directions, binary paths,
binary sizes, and SHA256 values before compiling the callable for the current
Simpler runtime.

```bash
python build_hbg_artifact.py \
  --artifact-source /path/to/generated/artifact \
  --output-dir /path/to/qwen3-14b-hbg-release \
  --platform a2a3
```

The supported release has 25 external tensors, 32 registered in-core kernels,
40 Qwen layers, 128-token pages, BF16 KV in BSND storage, and FP32 logits with
the valid vocabulary range `[0, 151936)`.

The stage-0 release directory and its producer must be trusted: its
`kernel_config.py` executes during inspection and its generated C++ sources
are compiled. Binary manifests verify the declared binaries. Load-time hashes
of the config and all compiled sources are recorded and rechecked before
compilation; these snapshots do not authenticate initially supplied source.

## Torch-NPU smoke

`run_torch_eager.py` loads the stage-0 checkpoint and release, allocates the
fixed batch-16 KV backing, and submits two decode steps with one active request.
It checks finite FP32 logits, every layer's active KV slot, completion, close, and caller
stream use after close. The zero-prefix KV input validates execution and
lifecycle wiring; it is not a prefill numerical reference.

Config, weight-index and consumed shard digests are checked against fixed pins
and recorded in the receipt. Success and failure receipts include completed
steps, the current step/position, and separate execution and cleanup errors.
Smoke checks remain active under `python -O`.

When a completed run reports a terminal execution error, `close()` preserves
that primary error while still attempting Worker and Buffer cleanup. Successful
cleanup makes a later close a no-op; an uncertain timeout retains resources for
a retry.

Run it through the shared device queue:

```bash
task-submit --device auto --device-num 1 --max-time 900 --run \
  'eval "$(pypto-setup --export)" && source "$ASCEND_HOME_PATH/set_env.sh" && \
   export TASK_QUEUE_ENABLE=1 && \
   python /path/to/qwen3_14b_serving_effective/run_torch_eager.py \
     --artifact /path/to/qwen3-14b-hbg-release \
     --model /path/to/Qwen3-14B \
     --prompt-ids /path/to/prompt-3338-token-ids.json \
     --device "$TASK_DEVICE" --output /path/to/result.json'
```

The caller owns tensor backing, context, and stream. The adapter retains tensor
backing through completion and closes its Buffer wrappers after retirement.
Returned logits use reusable storage; consume or copy them before the next
decode call.

RunHandle waits use a 120-second timeout, including the wait before close-time
stream synchronization. A timeout latches failure and retains backing and
Buffer wrappers until a later drain succeeds. `Worker.close()` uses its public
cleanup budget; Torch-NPU stream synchronization has no timeout argument, so
the RunHandle timeout is not an overall process deadline.

## Other qualification tools

The directory also contains longer fixture-driven qualification entry points:

- `run_standalone_0p1.sh` measures a sealed KV fixture through the standalone
  Worker adapter.
- `reference_worker_submit.py` compares a sealed logical-KV reference with
  appended decode KV, logits, and greedy tokens.
- `clone_worker_submit.py` exercises two request-private states.
- `device_feedback_worker_submit.py` exercises a bounded DEVICE feedback chain.
- `continuous_worker_submit.py` exercises bounded refill, retirement, and
  failure recovery.

Those tools use their own fixture manifests and acceptance contracts. Their
reports identify the source, artifact, model, and runtime identities used for
each run.

The vLLM Engine integration, native prefill, vLLM page metadata, sampling, and
Engine shutdown are tracked by [Simpler #2715](https://github.com/hw-native-sys/simpler/issues/2715).
