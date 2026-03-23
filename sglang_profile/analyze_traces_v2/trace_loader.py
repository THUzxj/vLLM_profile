"""
Trace Loader - Load kernel events from torch profiler trace files.

Reuses patterns from analyze_traces/kernel_name_stats.py
"""

import gzip
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class KernelEvent:
    """Represents a GPU kernel execution event."""
    name: str
    ts: float  # Start timestamp in microseconds
    dur: float  # Duration in microseconds
    pid: int
    tid: int
    cat: str
    args: Dict


def list_trace_files(trace_dir: str) -> List[str]:
    """
    Recursively find all trace files in directory.

    Supports .trace.json and .trace.json.gz files.
    """
    matches: List[str] = []
    for dirpath, _, filenames in os.walk(trace_dir):
        for fname in filenames:
            if fname.endswith(".trace.json") or fname.endswith(".trace.json.gz"):
                matches.append(os.path.join(dirpath, fname))
    matches.sort()
    return matches


def _load_trace_json_streaming(path: str):
    """
    Stream parse trace file to extract kernel events using ijson.
    Yields kernel events one at a time to save memory.
    """
    import ijson

    # Open file (gzipped or plain)
    if path.endswith(".gz"):
        f = gzip.open(path, "rb")
    else:
        f = open(path, "rb")

    try:
        # Parse traceEvents array using ijson
        parser = ijson.parse(f)

        current_event = {}
        in_event = False
        in_args = False

        for prefix, event_type, value in parser:
            # We're looking for items in traceEvents array
            if not prefix.startswith('traceEvents.item'):
                continue

            # Parse the field path
            parts = prefix.split('.')

            # Start of a new event
            if event_type == 'start_map' and len(parts) == 2:  # traceEvents.item
                current_event = {}
                in_event = True
                in_args = False
                continue

            # End of an event
            if event_type == 'end_map' and len(parts) == 2 and in_event:
                # Check if this is a kernel event
                if current_event.get("ph") == "X":
                    cat = current_event.get("cat") or ""
                    if cat in {"kernel", "gpu_memcpy", "gpu_memset"}:
                        yield current_event
                in_event = False
                current_event = {}
                continue

            if not in_event:
                continue

            # Handle args sub-map
            if len(parts) == 3 and parts[2] == 'args' and event_type == 'start_map':
                in_args = True
                current_event['args'] = {}
                continue

            if len(parts) == 3 and parts[2] == 'args' and event_type == 'end_map':
                in_args = False
                continue

            # Extract field values
            if event_type in ('string', 'number'):
                if len(parts) == 3:  # direct field like traceEvents.item.name
                    field = parts[2]
                    current_event[field] = value
                elif len(parts) == 4 and parts[2] == 'args':
                    field = parts[3]
                    if 'args' not in current_event:
                        current_event['args'] = {}
                    current_event['args'][field] = value

    finally:
        f.close()


def load_kernel_events_from_file(path: str) -> List[KernelEvent]:
    """Load kernel events from trace file using streaming to save memory."""
    kernels: List[KernelEvent] = []

    for ev in _load_trace_json_streaming(path):
        dur = ev.get("dur")
        ts = ev.get("ts")
        if dur is None or ts is None:
            continue

        kernels.append(
            KernelEvent(
                name=str(ev.get("name", "")),
                ts=float(ts),
                dur=float(dur),
                pid=int(ev.get("pid", -1)),
                tid=int(ev.get("tid", -1)),
                cat=ev.get("cat") or "",
                args=ev.get("args") or {},
            )
        )

    return kernels


def extract_tp_rank_from_filename(filename: str) -> int:
    """Extract TP rank from trace filename."""
    import re
    match = re.search(r'TP-(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1
