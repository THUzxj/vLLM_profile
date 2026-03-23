"""
Step Segmenter - Segment trace kernels into steps using combine kernel intervals.
"""

from typing import List, Tuple

from trace_loader import KernelEvent


# Default threshold for step boundary detection (10ms)
DEFAULT_STEP_GAP_THRESHOLD_US = 10000


def find_combine_kernel_pairs(kernels: List[KernelEvent]) -> List[Tuple[KernelEvent, KernelEvent]]:
    """
    Find consecutive deep_ep::internode_ll::combine kernel pairs.

    Each combine stage consists of 2 consecutive kernels with the same name.
    Returns list of (first_combine, second_combine) tuples.

    Args:
        kernels: List of kernel events

    Returns:
        List of (first_combine, second_combine) tuples
    """
    # Filter for combine kernels only
    combine_kernels = [
        k for k in kernels
        if "deep_ep::internode_ll::combine" in k.name
        and "notify" not in k.name.lower()
    ]

    # Sort by timestamp
    combine_kernels.sort(key=lambda k: k.ts)

    # Find consecutive pairs (same name)
    pairs = []
    i = 0
    while i < len(combine_kernels) - 1:
        first = combine_kernels[i]
        second = combine_kernels[i + 1]

        # Check if they have the same name (consecutive pair)
        if first.name == second.name:
            pairs.append((first, second))
            i += 2  # Skip both
        else:
            i += 1

    return pairs


def segment_steps_by_combine_intervals(
    kernels: List[KernelEvent],
    gap_threshold_us: float = DEFAULT_STEP_GAP_THRESHOLD_US
) -> List[Tuple[float, float]]:
    """
    Segment trace into steps based on combine kernel intervals.

    Step boundaries are identified by large gaps (> threshold) between
    consecutive combine kernel pairs.

    Args:
        kernels: List of kernel events
        gap_threshold_us: Gap threshold in microseconds (default: 10ms)

    Returns:
        List of (step_start_ts, step_end_ts) tuples
    """
    # Get combine pairs
    combine_pairs = find_combine_kernel_pairs(kernels)

    if not combine_pairs:
        return []

    # Calculate intervals between consecutive pairs
    # Use the end time of second combine in each pair
    step_boundaries = []

    for i in range(1, len(combine_pairs)):
        prev_pair = combine_pairs[i - 1]
        curr_pair = combine_pairs[i]

        # End of previous pair (end of second combine)
        prev_end = prev_pair[1].ts + prev_pair[1].dur

        # Start of current pair (start of first combine)
        curr_start = curr_pair[0].ts

        # Calculate gap
        gap = curr_start - prev_end

        if gap >= gap_threshold_us:
            # This is a step boundary
            step_boundaries.append(prev_end)

    # Build step ranges
    if not step_boundaries:
        # Single step - from first combine to last combine
        first_ts = min(k.ts for k in kernels)
        last_ts = max(k.ts + k.dur for k in kernels)
        return [(first_ts, last_ts)]

    # Multiple steps
    steps = []
    first_ts = min(k.ts for k in kernels)

    # First step: from start to first boundary
    steps.append((first_ts, step_boundaries[0]))

    # Middle steps
    for i in range(len(step_boundaries) - 1):
        steps.append((step_boundaries[i], step_boundaries[i + 1]))

    # Last step: from last boundary to end
    last_ts = max(k.ts + k.dur for k in kernels)
    steps.append((step_boundaries[-1], last_ts))

    return steps


def get_step_kernel_count(
    kernels: List[KernelEvent],
    step_boundaries: Tuple[float, float]
) -> int:
    """Get number of kernels within a step."""
    step_start, step_end = step_boundaries
    return sum(1 for k in kernels if step_start <= k.ts < step_end)
