"""
Layer Segmenter - Segment trace kernels into layers and classify MOE stages.
"""

from typing import Dict, List, Tuple

from trace_loader import KernelEvent
from stage_classifier import classify_kernel
from step_segmenter import find_combine_kernel_pairs


# Model configuration
DENSE_LAYER_COUNT = 3  # First 3 layers are dense
MOE_LAYER_COUNT = 58   # Layers 3-60 are MoE (58 layers)
TOTAL_LAYER_COUNT = 61  # Total layers


def find_dispatch_kernel_pairs(kernels: List[KernelEvent]) -> List[Tuple[KernelEvent, KernelEvent]]:
    """
    Find consecutive deep_ep::internode_ll::dispatch kernel pairs.

    Each dispatch stage consists of 2 consecutive kernels with the same name.
    Returns list of (first_dispatch, second_dispatch) tuples.

    Args:
        kernels: List of kernel events

    Returns:
        List of (first_dispatch, second_dispatch) tuples
    """
    # Filter for dispatch kernels only
    dispatch_kernels = [
        k for k in kernels
        if "deep_ep::internode_ll::dispatch" in k.name
        and "notify" not in k.name.lower()
    ]

    # Sort by timestamp
    dispatch_kernels.sort(key=lambda k: k.ts)

    # Find consecutive pairs (same name)
    pairs = []
    i = 0
    while i < len(dispatch_kernels) - 1:
        first = dispatch_kernels[i]
        second = dispatch_kernels[i + 1]

        # Check if they have the same name (consecutive pair)
        if first.name == second.name:
            pairs.append((first, second))
            i += 2
        else:
            i += 1

    return pairs


def segment_layers_in_step(
    kernels: List[KernelEvent],
    step_start_ts: float,
    step_end_ts: float
) -> Dict[int, Dict[str, List[KernelEvent]]]:
    """
    Segment kernels in a step into layers.

    For MoE model:
    - Layers 0-2: Dense layers (no MoE stages)
    - Layers 3-60: MoE layers with 4 stages each

    Uses combine kernel pairs as layer boundaries.

    Args:
        kernels: All kernel events
        step_start_ts: Step start timestamp
        step_end_ts: Step end timestamp

    Returns:
        Dict mapping layer_idx -> {stage: [kernels]}
        For dense layers, stage is "dense"
    """
    # Filter kernels in this step
    step_kernels = [
        k for k in kernels
        if step_start_ts <= k.ts < step_end_ts
    ]

    # Find combine pairs in this step (layer boundaries)
    combine_pairs = find_combine_kernel_pairs(step_kernels)

    # Find dispatch pairs in this step
    dispatch_pairs = find_dispatch_kernel_pairs(step_kernels)

    result: Dict[int, Dict[str, List[KernelEvent]]] = {}

    # We expect 58 combine pairs (one per MoE layer)
    # Layer 3 ends at combine[0], Layer 4 at combine[1], etc.
    if len(combine_pairs) < MOE_LAYER_COUNT:
        # Not enough combines for all MoE layers
        return result

    # Limit to first MOE_LAYER_COUNT combines
    moe_combines = combine_pairs[:MOE_LAYER_COUNT]

    # Build layer boundaries
    # Layer 3: from step_start to combine[0] (end of first pair)
    # Layer 4: from combine[0] to combine[1]
    # ...
    # Layer 60: from combine[56] to combine[57]

    layer_boundaries = []

    # Dense layers (0-2) - from step_start to first MoE activity
    # Dense layers are processed together before MoE layers
    dense_end_ts = moe_combines[0][0].ts  # Start of first dispatch (before first combine)

    # Find where dense layers end (find first attention kernel in MoE section)
    # Actually, dense layers come first, then MoE layers
    # Dense layers don't have dispatch/combine pattern

    # Assign dense layers
    dense_kernels = [
        k for k in step_kernels
        if k.ts < moe_combines[0][0].ts  # Before first combine starts
        and classify_kernel(k.name) != "other"
    ]

    for layer_idx in range(DENSE_LAYER_COUNT):
        result[layer_idx] = {"dense": []}

    # Distribute dense kernels among 3 dense layers (roughly equal)
    if dense_kernels:
        kernels_per_layer = len(dense_kernels) // DENSE_LAYER_COUNT
        for i, layer_idx in enumerate(range(DENSE_LAYER_COUNT)):
            start_idx = i * kernels_per_layer
            end_idx = (i + 1) * kernels_per_layer if i < DENSE_LAYER_COUNT - 1 else len(dense_kernels)
            result[layer_idx] = {"dense": dense_kernels[start_idx:end_idx]}

    # Process MoE layers (3-60)
    for layer_idx in range(DENSE_LAYER_COUNT, TOTAL_LAYER_COUNT):
        moe_layer_idx = layer_idx - DENSE_LAYER_COUNT  # 0-57

        if moe_layer_idx >= len(moe_combines):
            break

        # Get combine pair for this layer
        combine_pair = moe_combines[moe_layer_idx]

        # Layer boundary
        if moe_layer_idx == 0:
            layer_start = moe_combines[0][0].ts  # First combine starts
        else:
            layer_start = moe_combines[moe_layer_idx - 1][1].ts + moe_combines[moe_layer_idx - 1][1].dur

        layer_end = combine_pair[1].ts + combine_pair[1].dur  # End of second combine

        # Get kernels in this layer
        layer_kernels = [
            k for k in step_kernels
            if layer_start <= k.ts < layer_end
            and classify_kernel(k.name) != "other"
        ]

        # Find dispatch pair for this layer
        layer_dispatch = None
        for dp in dispatch_pairs:
            if layer_start <= dp[0].ts < layer_end:
                layer_dispatch = dp
                break

        # Classify into 4 stages
        stages = {"attention": [], "dispatch": [], "expert": [], "combine": []}

        for k in layer_kernels:
            stage = classify_kernel(k.name)

            if stage == "dispatch":
                stages["dispatch"].append(k)
            elif stage == "combine":
                stages["combine"].append(k)
            elif stage == "attention":
                stages["attention"].append(k)
            elif stage == "expert":
                stages["expert"].append(k)

        result[layer_idx] = stages

    return result


def compute_stage_duration(kernels: List[KernelEvent]) -> Tuple[float, float, float]:
    """
    Compute wall-clock duration for a group of kernels.

    Args:
        kernels: List of kernel events

    Returns:
        Tuple of (start_ts, end_ts, duration_us)
    """
    if not kernels:
        return 0.0, 0.0, 0.0

    start_ts = min(k.ts for k in kernels)
    end_ts = max(k.ts + k.dur for k in kernels)
    duration = end_ts - start_ts

    return start_ts, end_ts, duration


def segment_trace(
    kernels: List[KernelEvent],
    step_boundaries: List[Tuple[float, float]]
) -> Dict[int, Dict[int, Dict[str, List[KernelEvent]]]]:
    """
    Segment entire trace into steps and layers.

    Args:
        kernels: All kernel events
        step_boundaries: List of (step_start, step_end) tuples

    Returns:
        Dict mapping {step_idx: {layer_idx: {stage: [kernels]}}}
    """
    result = {}

    for step_idx, (step_start, step_end) in enumerate(step_boundaries):
        layers = segment_layers_in_step(kernels, step_start, step_end)
        if layers:
            result[step_idx] = layers

    return result
