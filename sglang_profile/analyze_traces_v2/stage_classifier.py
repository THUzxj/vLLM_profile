"""
Stage Classifier - Classify GPU kernels into MOE stages.

Reuses patterns from analyze_traces/moe_stage_classifier.py
"""

from typing import List


# Stage classification patterns
ATTENTION_PATTERNS: List[str] = [
    "flashattnfwd",           # FlashAttention forward
    "flash::flashattnfwd",    # FlashAttention namespace
    "flash::flashattnfwdcombine",
    "flash::prepare_varlen",
    "set_mla_kv_buffer",      # MLA KV buffer setup
    "batchqkapplyrotary",     # Rotary position embedding
    "flashinfer::norm::rmsnorm",
    "flashinfer::norm::fusedaddrmsnorm",
    "flashinfer::batchqkapplyrotary",
    "cutlass::device_kernel<flash::",
]

DISPATCH_PATTERNS: List[str] = [
    "deep_ep::internode_ll::dispatch",
]

COMBINE_PATTERNS: List[str] = [
    "deep_ep::internode_ll::combine",
]

EXPERT_PATTERNS: List[str] = [
    "deep_gemm::sm90_fp8_gemm",
    "router_gemm_kernel",
    "deepseek_v3_topk",
    "_silu_and_mul",
    "act_and_mul_kernel",
    "per_token_group_quant",
    "nvjet_tst_",
    "transpose_fp32",
    "tensorrt_llm::kernels::",
    "void flashinfer::activation::",
]


def classify_kernel(kernel_name: str) -> str:
    """
    Classify a kernel name into one of the MOE stages.

    Args:
        kernel_name: Full kernel name from trace

    Returns:
        One of: "attention", "dispatch", "expert", "combine", "other"
    """
    name_lower = kernel_name.lower()

    # Check dispatch first (most specific)
    for pattern in DISPATCH_PATTERNS:
        if pattern.lower() in name_lower:
            return "dispatch"

    # Check combine
    for pattern in COMBINE_PATTERNS:
        if pattern.lower() in name_lower:
            return "combine"

    # Check attention
    for pattern in ATTENTION_PATTERNS:
        if pattern.lower() in name_lower:
            return "attention"

    # Check expert
    for pattern in EXPERT_PATTERNS:
        if pattern.lower() in name_lower:
            return "expert"

    # Default to other
    return "other"


def get_all_stage_names() -> List[str]:
    """Return list of all stage names."""
    return ["attention", "dispatch", "expert", "combine", "other"]


def get_main_stage_names() -> List[str]:
    """Return list of main stages (excluding other)."""
    return ["attention", "dispatch", "expert", "combine"]
