"""
Swimlane JSON Generator - Chrome Trace Event Format (Perfetto)

Generates merged_swimlane.json for performance visualization in Chrome Tracing.
The output follows the Chrome Trace Event Format specification.
"""

import json
from pathlib import Path
from typing import List, Dict, Any


class SwimlaneGenerator:
    """
    Generate Chrome Trace Event Format JSON for swimlane visualization.

    This class takes trace events from a Runtime instance and generates
    a JSON file compatible with Chrome Tracing (chrome://tracing).
    """

    def __init__(self, runtime, block_dim: int):
        """
        Initialize the swimlane generator.

        Args:
            runtime: Runtime instance with trace events
            block_dim: Number of blocks (used to determine core mapping)
        """
        self.runtime = runtime
        self.block_dim = block_dim
        self.num_aic = block_dim
        self.num_aiv = block_dim * 2

    def generate(self, output_path: str):
        """
        Generate merged_swimlane.json file.

        Args:
            output_path: Path to output JSON file
        """
        events = []

        # Generate all event types
        events.extend(self._generate_metadata())
        events.extend(self._generate_task_events())
        events.extend(self._generate_flow_events())
        events.extend(self._generate_queue_events())
        events.extend(self._generate_memory_events())

        # Write output
        output = {"traceEvents": events}
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

    def _generate_metadata(self) -> List[Dict]:
        """
        Generate process and thread metadata.

        Creates:
        - Process name: "Machine View"
        - Thread names for each AIC core: "AIC_0", "AIC_1", ...
        - Thread names for each AIV core: "AIV_0", "AIV_1", ...
        """
        events = []

        # Process name
        events.append({
            "args": {"name": "Machine View"},
            "cat": "__metadata",
            "name": "process_name",
            "ph": "M",
            "pid": 1
        })

        # Thread names for AIC cores
        for i in range(self.num_aic):
            events.append({
                "args": {"name": f"AIC_{i}"},
                "cat": "__metadata",
                "name": "thread_name",
                "ph": "M",
                "pid": 1,
                "tid": 1000 + i
            })

        # Thread names for AIV cores
        for i in range(self.num_aiv):
            events.append({
                "args": {"name": f"AIV_{i}"},
                "cat": "__metadata",
                "name": "thread_name",
                "ph": "M",
                "pid": 1,
                "tid": 2000 + i
            })

        return events

    def _generate_task_events(self) -> List[Dict]:
        """
        Generate task execution events (Duration Events).

        Each task creates a duration event (ph="X") showing:
        - When the task started (ts)
        - How long it ran (dur)
        - Which core executed it (tid)
        - Task name if available
        """
        events = []
        trace_events = self.runtime.get_trace_events()

        # Debug: print trace event count
        print(f"[DEBUG] Total trace events: {len(trace_events)}")
        task_events = [e for e in trace_events if e['type'] == 'task_exec']
        print(f"[DEBUG] Task execution events: {len(task_events)}")

        for evt in trace_events:
            if evt['type'] == 'task_exec':
                task_id = evt['task_id']
                core_id = evt['core_id']
                timestamp = evt['timestamp_us']
                duration = evt['duration_us']
                task_name = evt.get('name', '')  # Get task name from trace event

                # Map core_id to thread_id
                # AIC cores: 0 to num_aic-1 -> tid 1000 to 1000+num_aic-1
                # AIV cores: num_aic to num_aic+num_aiv-1 -> tid 2000 to 2000+num_aiv-1
                if core_id < self.num_aic:
                    tid = 1000 + core_id
                    core_name = f"AIC_{core_id}"
                else:
                    aiv_index = core_id - self.num_aic
                    tid = 2000 + aiv_index
                    core_name = f"AIV_{aiv_index}"

                # Generate display name: use custom name if available, otherwise Task_id
                if task_name:
                    display_name = f"{task_name}_{task_id}"
                    event_hint = f"{task_name} (Task {task_id}) on {core_name}"
                else:
                    display_name = f"Task_{task_id}"
                    event_hint = f"Task {task_id} on {core_name}"

                events.append({
                    "args": {
                        "taskId": task_id,
                        "event-hint": event_hint,
                    },
                    "cat": "event",
                    "id": task_id,
                    "name": display_name,
                    "ph": "X",  # Duration event
                    "pid": 1,
                    "tid": tid,
                    "ts": timestamp,
                    "dur": duration
                })

        return events

    def _generate_queue_events(self) -> List[Dict]:
        """
        Generate ready queue count events (Counter Events).

        Creates counter events (ph="C") showing:
        - AIC ready queue size over time
        - AIV ready queue size over time
        """
        events = []
        trace_events = self.runtime.get_trace_events()

        for evt in trace_events:
            if evt['type'] == 'queue_count':
                queue_type = evt['queue_type']
                timestamp = evt['timestamp_us']
                count = evt['count']

                # Queue type: 0=AIC, 1=AIV
                name = "ReadyCount_AIC" if queue_type == 0 else "ReadyCount_AIV"

                events.append({
                    "name": name,
                    "pid": 1,
                    "tid": 1,
                    "ph": "C",  # Counter event
                    "ts": timestamp,
                    "args": {"size": count}
                })

        return events

    def _generate_memory_events(self) -> List[Dict]:
        """
        Generate memory usage events (Counter Events).

        Creates counter events (ph="C") showing:
        - Memory usage over time
        """
        events = []
        trace_events = self.runtime.get_trace_events()

        for evt in trace_events:
            if evt['type'] == 'memory':
                events.append({
                    "name": "Ideal_Mem_Usage(Task)",
                    "pid": 1,
                    "tid": 1,
                    "ph": "C",  # Counter event
                    "ts": evt['timestamp_us'],
                    "args": {"/byte": evt['bytes']}
                })

        return events

    def _generate_flow_events(self) -> List[Dict]:
        """
        Generate flow events to visualize task dependencies.

        Creates flow events (ph="s" and ph="f") showing:
        - Task dependency arrows from predecessor to successor

        Flow event format:
        - ph="s" (flow start): Placed at the end of the source task
        - ph="f" (flow finish): Placed at the start of the target task
        - Shared "id" links the start and finish events
        - Shared "cat" for filtering
        """
        events = []

        # Get all task execution events to find timestamps
        trace_events = self.runtime.get_trace_events()
        task_times = {}  # task_id -> (start_ts, end_ts, tid)

        for evt in trace_events:
            if evt['type'] == 'task_exec':
                task_id = evt['task_id']
                core_id = evt['core_id']
                start_ts = evt['timestamp_us']
                end_ts = start_ts + evt['duration_us']

                # Map core_id to thread_id (same logic as _generate_task_events)
                if core_id < self.num_aic:
                    tid = 1000 + core_id
                else:
                    aiv_index = core_id - self.num_aic
                    tid = 2000 + aiv_index

                task_times[task_id] = (start_ts, end_ts, tid)

        # Generate flow events for each dependency
        task_count = self.runtime.get_task_count()
        flow_id = 0

        for task_id in range(task_count):
            if task_id not in task_times:
                continue  # Task didn't execute (shouldn't happen in normal cases)

            fanout = self.runtime.get_task_fanout(task_id)
            src_start, src_end, src_tid = task_times[task_id]

            for successor_id in fanout:
                if successor_id not in task_times:
                    continue  # Successor didn't execute

                dst_start, dst_end, dst_tid = task_times[successor_id]

                # Flow start: at the end of the source task
                events.append({
                    "name": f"dep_{task_id}→{successor_id}",
                    "cat": "dependency",
                    "ph": "s",  # Flow start
                    "id": flow_id,
                    "pid": 1,
                    "tid": src_tid,
                    "ts": src_end,
                    "bp": "e",  # Binding point: end of task
                })

                # Flow finish: at the start of the target task
                events.append({
                    "name": f"dep_{task_id}→{successor_id}",
                    "cat": "dependency",
                    "ph": "f",  # Flow finish
                    "id": flow_id,
                    "pid": 1,
                    "tid": dst_tid,
                    "ts": dst_start,
                    "bp": "e",  # Binding point: end of previous task
                })

                flow_id += 1

        return events
