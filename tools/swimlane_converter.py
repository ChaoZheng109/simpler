#!/usr/bin/env python3
"""
Swimlane JSON to Perfetto JSON Converter

Converts performance data JSON (.json) to Chrome Trace Event Format JSON
for visualization in Perfetto (https://ui.perfetto.dev/).

Usage:
    python3 swimlane_converter.py  # Uses latest perf_swimlane_*.json in outputs/
    python3 swimlane_converter.py perf_swimlane_20260210_143526.json
    python3 swimlane_converter.py perf_swimlane_20260210_143526.json -o custom_output.json
    python3 swimlane_converter.py perf_swimlane_20260210_143526.json --freq 100000000
    python3 swimlane_converter.py perf_swimlane_20260210_143526.json -k kernel_config.py
    python3 swimlane_converter.py perf_swimlane_20260210_143526.json -v
"""

import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
import importlib.util

# Core type constants (from CoreType enum)
CORE_TYPE_AIC = 0
CORE_TYPE_AIV = 1


def read_perf_data(filepath):
    """Read performance data from JSON file.

    Args:
        filepath: Path to input JSON file

    Returns:
        dict: Parsed performance data with keys:
            - version
            - frequency
            - base_time
            - timestamp
            - function_names (dict)
            - records (list)

    Raises:
        ValueError: If JSON format is invalid
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Validate required fields
    required_fields = ['version', 'frequency', 'base_time', 'timestamp', 'records']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    # Validate version
    if data['version'] != 1:
        raise ValueError(f"Unsupported version: {data['version']} (expected 1)")

    return data


def cycles_to_us(cycles, base_time, frequency):
    """Convert cycles to microseconds.

    Args:
        cycles: Time in cycles
        base_time: Base time offset in cycles
        frequency: System counter frequency in Hz

    Returns:
        float: Time in microseconds
    """
    normalized_cycles = cycles - base_time
    return (normalized_cycles / frequency) * 1e6


def load_kernel_config(config_path):
    """Load kernel configuration from kernel_config.py file.

    Args:
        config_path: Path to kernel_config.py file

    Returns:
        dict: Mapping from func_id (as string) to function name
              Example: {"0": "QK", "1": "SF", "2": "PV", "3": "UP"}
              Entries without 'func_id' or 'name' are skipped with a warning

    Raises:
        ValueError: If file cannot be loaded or KERNELS definition is missing
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise ValueError(f"Kernel config file not found: {config_path}")

    # Load the Python module dynamically
    spec = importlib.util.spec_from_file_location("kernel_config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load module from: {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Extract func_id to name mapping from KERNELS list
    if not hasattr(module, 'KERNELS'):
        raise ValueError(f"kernel_config.py missing KERNELS definition")

    func_id_to_name = {}
    for kernel in module.KERNELS:
        # Skip entries without func_id
        if 'func_id' not in kernel:
            print(f"Warning: Kernel entry missing 'func_id', skipping: {kernel}", file=sys.stderr)
            continue

        func_id = kernel['func_id']

        # If name is missing, we'll fall back to default naming (Func_{func_id})
        if 'name' not in kernel:
            print(f"Warning: Kernel entry for func_id={func_id} missing 'name', will use default naming", file=sys.stderr)
            continue

        # Store as string to match JSON format
        func_id_to_name[str(func_id)] = kernel['name']

    return func_id_to_name


def generate_chrome_trace_json(records, base_time, frequency, output_path, func_id_to_name=None, verbose=False):
    """Generate Chrome Trace Event Format JSON from PerfRecord data.

    This function replicates the logic from device_runner.cpp::export_swimlane_json()

    Args:
        records: List of performance record dicts
        base_time: Base timestamp in cycles
        frequency: System counter frequency in Hz
        output_path: Path to output JSON file
        func_id_to_name: Optional dict mapping func_id to function name
        verbose: Print progress information
    """
    if verbose:
        print(f"Generating Chrome Trace JSON...")
        print(f"  Records: {len(records)}")
        print(f"  Base time: {base_time} cycles")
        print(f"  Frequency: {frequency} Hz ({frequency/1e6:.1f} MHz)")
        if func_id_to_name:
            print(f"  Function names: {len(func_id_to_name)} entries")

    # Step 1: Build core_to_tid mapping (using only core_id, not core_type)
    unique_cores = set()
    for record in records:
        unique_cores.add(record['core_id'])

    core_to_tid = {}
    tid_counter = 1000
    for core_id in sorted(unique_cores):
        core_to_tid[core_id] = tid_counter
        tid_counter += 1

    if verbose:
        print(f"  Unique cores: {len(unique_cores)}")

    # Step 2: Generate JSON events
    events = []

    # Metadata event: Process name
    events.append({
        "args": {"name": "Machine View"},
        "cat": "__metadata",
        "name": "process_name",
        "ph": "M",
        "pid": 1
    })

    # Metadata events: Thread names (one per core)
    for core_id, tid in core_to_tid.items():
        # Find first record with this core_id to get core_type (matches C++ logic)
        core_type = None
        for record in records:
            if record['core_id'] == core_id:
                core_type = record['core_type']
                break

        core_type_str = "AIC" if core_type == CORE_TYPE_AIC else "AIV"
        thread_name = f"{core_type_str}_{core_id}"
        events.append({
            "args": {"name": thread_name},
            "cat": "__metadata",
            "name": "thread_name",
            "ph": "M",
            "pid": 1,
            "tid": tid
        })

    # Duration events (Complete events "X")
    # Build task_id -> event_id mapping for flow events
    task_to_event_id = {}
    event_id = 0

    for record in records:
        tid = core_to_tid[record['core_id']]
        ts = cycles_to_us(record['start_time'], base_time, frequency)
        dur = cycles_to_us(record['end_time'], base_time, frequency) - ts

        # Build fanout hint string
        fanout_str = "[" + ", ".join(str(x) for x in record['fanout']) + "]"

        # Calculate duration_cycles from start_time and end_time
        duration_cycles = record['end_time'] - record['start_time']

        # Get function name if available
        func_id = record['func_id']
        if func_id_to_name and str(func_id) in func_id_to_name:
            func_name = func_id_to_name[str(func_id)]
            # New format: FuncName(task_id)
            task_name = f"{func_name}({record['task_id']})"
        else:
            # Fallback format: Func_{func_id}(task_id)
            task_name = f"Func_{func_id}({record['task_id']})"

        events.append({
            "args": {
                "event-hint": f"Task:{record['task_id']}, FuncId:{func_id}, CoreId:{record['core_id']}",
                "fanout-hint": fanout_str,
                "duration-cycles": duration_cycles,
                "taskId": record['task_id']
            },
            "cat": "event",
            "id": event_id,
            "name": task_name,
            "ph": "X",
            "pid": 1,
            "tid": tid,
            "ts": ts,
            "dur": dur
        })

        # Record mapping for flow events
        task_to_event_id[record['task_id']] = event_id
        event_id += 1

    # Flow events (Flow events "s" and "f" for dependencies)
    task_map = {r['task_id']: r for r in records}
    flow_id = 0

    for record in records:
        src_tid = core_to_tid[record['core_id']]
        src_ts_end = cycles_to_us(record['end_time'], base_time, frequency)

        for succ_task_id in record['fanout']:
            if succ_task_id not in task_map:
                if verbose:
                    print(f"Warning: Task {record['task_id']} references non-existent successor {succ_task_id}")
                continue

            succ_record = task_map[succ_task_id]
            dst_tid = core_to_tid[succ_record['core_id']]
            dst_ts_start = cycles_to_us(succ_record['start_time'], base_time, frequency)

            # Get event IDs for source and destination tasks
            src_event_id = task_to_event_id[record['task_id']]
            dst_event_id = task_to_event_id[succ_record['task_id']]

            # Calculate flow start timestamp (at end of source task - half cycle)
            half_cycle_us = (0.5 / frequency) * 1e6
            flow_start_us = src_ts_end - half_cycle_us

            # Flow start event (at end of source task)
            events.append({
                "cat": "flow",
                "id": flow_id,
                "name": "dependency",
                "ph": "s",
                "pid": 1,
                "tid": src_tid,
                "ts": flow_start_us,
                "bind_id": src_event_id
            })

            # Flow finish event (at start of destination task)
            events.append({
                "cat": "flow",
                "id": flow_id,
                "name": "dependency",
                "ph": "f",
                "pid": 1,
                "tid": dst_tid,
                "ts": dst_ts_start,
                "bp": "e",
                "bind_id": dst_event_id
            })

            flow_id += 1

    if verbose:
        print(f"  Total events: {len(events)}")
        print(f"  Flow events: {flow_id}")

    # Step 3: Write JSON file (with traceEvents wrapper to match C++ output)
    with open(output_path, 'w') as f:
        json.dump({"traceEvents": events}, f, indent=2)

    if verbose:
        print(f"JSON written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert swimlane performance JSON to Chrome Trace Event JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                       # Use latest .json in outputs/, output to outputs/
  %(prog)s perf_swimlane_20260210_143526.json   # Output: outputs/merged_swimlane_20260210_143526.json
  %(prog)s perf_swimlane_20260210_143526.json -o custom_output.json
  %(prog)s perf_swimlane_20260210_143526.json --freq 100000000
  %(prog)s perf_swimlane_20260210_143526.json -k examples/host_build_graph/paged_attention/kernels/kernel_config.py
  %(prog)s perf_swimlane_20260210_143526.json -v
        """
    )

    parser.add_argument('input', nargs='?', help='Input JSON file (.json). If not specified, uses the latest perf_swimlane_*.json file in outputs/ directory')
    parser.add_argument('-o', '--output', help='Output JSON file (default: outputs/merged_swimlane_<timestamp>.json)')
    parser.add_argument('--freq', type=int, help='Override frequency from data file (Hz)')
    parser.add_argument('--base-time', type=int, help='Override base time from data file (cycles)')
    parser.add_argument('-k', '--kernel-config', help='Path to kernel_config.py file for func_id to function name mapping')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # If no input file specified, find the latest .json file in outputs/ directory
    if args.input is None:
        outputs_dir = Path(__file__).parent.parent / "outputs"
        json_files = list(outputs_dir.glob("perf_swimlane_*.json"))

        if not json_files:
            print(f"Error: No perf_swimlane_*.json files found in {outputs_dir}", file=sys.stderr)
            print("Please specify an input file or ensure .json files exist in outputs/", file=sys.stderr)
            return 1

        # Get the most recently modified file
        input_path = max(json_files, key=lambda p: p.stat().st_mtime)

        if args.verbose:
            print(f"Auto-selected latest file: {input_path.name}")
    else:
        input_path = Path(args.input)

    # Check input file exists
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        # Read performance data JSON
        if args.verbose:
            print(f"Reading performance data from: {input_path}")

        data = read_perf_data(input_path)

        if args.verbose:
            print("\n=== Performance Data ===")
            print(f"  Version: {data['version']}")
            print(f"  Record Count: {len(data['records'])}")
            print(f"  Frequency: {data['frequency']} Hz ({data['frequency']/1e6:.1f} MHz)")
            print(f"  Base Time: {data['base_time']} cycles")
            timestamp_str = datetime.fromtimestamp(data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  Timestamp: {data['timestamp']} ({timestamp_str})")

            func_names = data.get('function_names', {})
            if func_names:
                print(f"  Function Names: {len(func_names)} entries")
                for func_id, name in sorted(func_names.items(), key=lambda x: int(x[0])):
                    print(f"    func_id={func_id}: {name}")
            print()

        # Apply overrides if provided
        frequency = args.freq if args.freq is not None else data['frequency']
        base_time = args.base_time if args.base_time is not None else data['base_time']

        if args.freq is not None and args.verbose:
            print(f"Frequency override: {frequency} Hz ({frequency/1e6:.1f} MHz)")
        if args.base_time is not None and args.verbose:
            print(f"Base time override: {base_time} cycles")

        # Load function name mapping
        # Priority: kernel_config.py > data['function_names']
        func_names = {}
        if args.kernel_config:
            # Load from kernel_config.py
            if args.verbose:
                print(f"\nLoading kernel config from: {args.kernel_config}")
            func_names = load_kernel_config(args.kernel_config)
            if args.verbose:
                print(f"  Loaded {len(func_names)} function name mappings from kernel_config.py:")
                for func_id, name in sorted(func_names.items(), key=lambda x: int(x[0])):
                    print(f"    func_id={func_id}: {name}")
        else:
            # Fall back to data['function_names'] if available
            func_names = data.get('function_names', {})
            if args.verbose and func_names:
                print(f"\nUsing function names from input JSON: {len(func_names)} entries")

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            # Extract timestamp from input filename
            # Expected format: perf_swimlane_YYYYMMDD_HHMMSS.json
            input_stem = input_path.stem  # filename without extension

            # Try to extract timestamp from filename
            if input_stem.startswith("perf_swimlane_"):
                timestamp_part = input_stem[len("perf_swimlane_"):]  # e.g., "20260210_143526"
            else:
                # Fallback: use data timestamp if filename format is unexpected
                timestamp_part = str(data['timestamp'])

            # Generate output filename and save to outputs/ directory
            outputs_dir = Path(__file__).parent.parent / "outputs"
            outputs_dir.mkdir(exist_ok=True)  # Ensure outputs directory exists
            output_path = outputs_dir / f"merged_swimlane_{timestamp_part}.json"

        # Generate Perfetto JSON
        generate_chrome_trace_json(data['records'], base_time, frequency, str(output_path), func_names, args.verbose)

        print(f"\n✓ Conversion complete")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
        print(f"\nTo visualize: Open https://ui.perfetto.dev/ and drag in {output_path}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
