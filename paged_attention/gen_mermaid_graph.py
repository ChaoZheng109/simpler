#!/usr/bin/env python3
"""
Run Paged Attention simulation and generate Mermaid dependency graph.

Usage:
    python3 gen_mermaid_graph.py                  # Run + generate full graph
    python3 gen_mermaid_graph.py --batch 0        # Single batch graph
    python3 gen_mermaid_graph.py -o graph.md      # Custom output file
"""

import subprocess
import re
import argparse
import sys
from pathlib import Path

# Kernel name mapping (func_id -> short name)
FUNC_NAMES = {0: "QK", 1: "SF", 2: "PV", 3: "UP"}

# Core type for styling
CORE_TYPES = {0: "AIC", 1: "AIV", 2: "AIC", 3: "AIV"}


def run_simulation() -> str:
    """Run paged attention simulation via subprocess, return captured output."""
    script_dir = Path(__file__).parent
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; from pathlib import Path; "
         "sys.path.insert(0, str(Path('" + str(script_dir.parent.parent) + "') / 'python')); "
         "sys.path.insert(0, str(Path('" + str(script_dir.parent.parent) + "') / 'examples' / 'scripts')); "
         "from code_runner import CodeRunner; "
         "CodeRunner("
         "kernels_dir='" + str(script_dir / "kernels") + "', "
         "golden_path='" + str(script_dir / "golden.py") + "', "
         "platform='a2a3sim').run()"],
        capture_output=True,
        text=True,
    )
    return result.stdout + result.stderr


def parse_task_table(output: str) -> tuple:
    """Parse Task Table from simulation output."""
    tasks = []
    params = {"batch": 2, "num_heads": 16, "block_num": 4}

    for line in output.split("\n"):
        if "batch=" in line and "num_heads=" in line:
            m = re.search(r"batch=(\d+)", line)
            if m:
                params["batch"] = int(m.group(1))
            m = re.search(r"num_heads=(\d+)", line)
            if m:
                params["num_heads"] = int(m.group(1))
        if "block_num=" in line:
            m = re.search(r"block_num=(\d+)", line)
            if m:
                params["block_num"] = int(m.group(1))

    in_task_table = False
    for line in output.split("\n"):
        if "Task Table:" in line:
            in_task_table = True
            continue
        if in_task_table and line.startswith("==="):
            break
        if in_task_table and "Task " in line and "func_id=" in line:
            m = re.match(
                r"\s*Task (\d+): func_id=(\d+), fanin=(\d+), fanout=(\d+), args=\d+ \[(.*?)\]",
                line,
            )
            if m:
                fanout_str = m.group(5).strip()
                fanout = [int(x.strip()) for x in fanout_str.split(",") if x.strip()]
                tasks.append({
                    "task_id": int(m.group(1)),
                    "func_id": int(m.group(2)),
                    "fanin": int(m.group(3)),
                    "fanout": fanout,
                })

    return tasks, params


def infer_batch_block(task_id: int, func_id: int, params: dict) -> tuple:
    """Infer batch and block indices from task_id."""
    block_num = params["block_num"]
    tasks_per_batch = block_num * 4

    batch_idx = task_id // tasks_per_batch
    remainder = task_id % tasks_per_batch

    qk_sf_pv_count = 3 * block_num
    if remainder < qk_sf_pv_count:
        block_idx = remainder // 3
    else:
        block_idx = remainder - qk_sf_pv_count

    return batch_idx, block_idx


def generate_mermaid(tasks: list, params: dict, filter_batch: int = None) -> str:
    """Generate Mermaid flowchart from tasks."""
    lines = [
        "flowchart TD",
        "",
        "    %% Style definitions - high contrast colors",
        "    classDef aic fill:#4fc3f7,stroke:#0277bd,stroke-width:2px,color:#01579b",
        "    classDef aiv fill:#ffb74d,stroke:#ef6c00,stroke-width:2px,color:#e65100",
        "",
    ]

    included_tasks = set()
    for task in tasks:
        batch_idx, _ = infer_batch_block(task["task_id"], task["func_id"], params)
        if filter_batch is not None and batch_idx != filter_batch:
            continue
        included_tasks.add(task["task_id"])

    groups = {}
    for task in tasks:
        if task["task_id"] not in included_tasks:
            continue
        batch_idx, _ = infer_batch_block(task["task_id"], task["func_id"], params)
        groups.setdefault(batch_idx, []).append(task)

    for batch_idx in sorted(groups):
        lines.append(f"    subgraph B{batch_idx}[\"Batch {batch_idx}\"]")
        for task in sorted(groups[batch_idx], key=lambda t: t["task_id"]):
            tid = task["task_id"]
            fname = FUNC_NAMES.get(task["func_id"], f"F{task['func_id']}")
            _, blk = infer_batch_block(tid, task["func_id"], params)
            lines.append(f"        T{tid}[\"{fname}[{blk}]\"]")
        lines.append("    end")
        lines.append("")

    lines.append("    %% Dependencies")
    for task in tasks:
        if task["task_id"] not in included_tasks:
            continue
        for succ in task["fanout"]:
            if succ in included_tasks:
                lines.append(f"    T{task['task_id']} --> T{succ}")
    lines.append("")

    lines.append("    %% Apply styles (AIC=blue, AIV=orange)")
    aic = sorted(t["task_id"] for t in tasks if t["task_id"] in included_tasks and CORE_TYPES.get(t["func_id"]) == "AIC")
    aiv = sorted(t["task_id"] for t in tasks if t["task_id"] in included_tasks and CORE_TYPES.get(t["func_id"]) == "AIV")
    if aic:
        lines.append(f"    class {','.join(f'T{t}' for t in aic)} aic")
    if aiv:
        lines.append(f"    class {','.join(f'T{t}' for t in aiv)} aiv")

    return '```mermaid\n' + '\n'.join(lines) + '\n```'


def main():
    parser = argparse.ArgumentParser(description="Run Paged Attention and generate Mermaid task graph")
    parser.add_argument("--output", "-o", default="task_graph.md", help="Output file (default: task_graph.md)")
    parser.add_argument("--batch", "-b", type=int, default=None, help="Filter by batch index")
    args = parser.parse_args()

    print("Running simulation...")
    output = run_simulation()
    print("Simulation completed.")

    tasks, params = parse_task_table(output)
    if not tasks:
        print("ERROR: No tasks found in output!", file=sys.stderr)
        sys.exit(1)

    mermaid = generate_mermaid(tasks, params, filter_batch=args.batch)
    with open(args.output, "w") as f:
        f.write(mermaid)

    print(f"Generated: {args.output} ({len(tasks)} tasks)")


if __name__ == "__main__":
    main()
