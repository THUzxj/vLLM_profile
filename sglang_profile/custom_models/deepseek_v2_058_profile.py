import json
import os
import time
import types

import torch
from sglang.srt.batch_overlap.single_batch_overlap import SboFlags, compute_overlap_args
from sglang.srt.batch_overlap.two_batch_overlap import MaybeTboDeepEPDispatcher
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher, CombineInput, DispatchOutput

from . import deepseek_v2_058 as base

# Re-export entry classes so that this module can be used as a drop-in replacement.
DeepseekV2ForCausalLM = base.DeepseekV2ForCausalLM
DeepseekV3ForCausalLM = base.DeepseekV3ForCausalLM
DeepseekV32ForCausalLM = base.DeepseekV32ForCausalLM


def _patched_moe_get_expert_statistics(self, router_logits, times):
    """Calculate activated expert statistics and optionally save them.

    This is adapted from `deepseek_v2.DeepseekV2MoE.get_expert_statistics`,
    but made self-contained and robust for the 0.5.8 model.
    """

    # router_logits: (num_tokens, n_experts)
    # Use the configured top-k for this MoE block.
    top_k = getattr(self, "top_k", None)
    if top_k is None:
        # Fallback: read from config if available.
        top_k = getattr(getattr(self, "config", None),
                        "num_experts_per_tok", None)
    if top_k is None:
        # As a last resort, just pick k=1 to avoid crashing.
        top_k = 1

    topk_values, topk_indices = torch.topk(router_logits, k=top_k, dim=-1)

    # Flatten to get all selected expert indices
    selected_expert_indices = topk_indices.flatten()  # (num_tokens * top_k,)

    # Count unique activated experts
    unique_experts = torch.unique(selected_expert_indices)
    num_activated_experts = unique_experts.numel()
    times["num_activated_experts"] = int(num_activated_experts)

    # Count token distribution for each activated expert
    expert_token_counts = torch.bincount(
        selected_expert_indices,
        minlength=getattr(getattr(self, "config", None),
                          "n_routed_experts", 0),
    )
    activated_expert_token_counts = expert_token_counts[unique_experts]

    expert_token_distribution = {
        int(expert_id.item()): int(count.item())
        for expert_id, count in zip(unique_experts, activated_expert_token_counts)
    }
    times["expert_token_distribution"] = expert_token_distribution
    times["router_logits_shape"] = tuple(router_logits.shape)
    times["num_expert_requests"] = int(
        topk_indices.shape[0] * topk_indices.shape[1])

    # Optionally save raw topk indices for further offline analysis.
    output_dir = os.getenv("PROFILE_COMPONENT_OUTPUT_DIR", None)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        count = getattr(self, "_expert_profile_count", 0)
        layer_id = getattr(self, "layer_id", -1)
        topk_path = os.path.join(
            output_dir, f"topk_indices_layer{layer_id}_count_{count}.pt"
        )
        try:
            torch.save(topk_indices.cpu(), topk_path)
        except Exception:
            # Make sure profiling never affects model correctness.
            pass
        setattr(self, "_expert_profile_count", count + 1)

    return topk_indices


def _patched_moe_forward_normal(
    self,
    hidden_states: torch.Tensor,
    should_allreduce_fusion: bool = False,
    use_reduce_scatter: bool = False,
    gemm_output_zero_allocator: base.BumpAllocator = None,
) -> torch.Tensor:
    """Patched `forward_normal` that also gathers MoE expert statistics."""
    if hasattr(self, "shared_experts") and base.use_intel_amx_backend(
        self.shared_experts.gate_up_proj
    ):
        return self.forward_cpu(hidden_states, should_allreduce_fusion)

    if hidden_states.shape[0] > 0:
        if not self._fuse_shared_experts_inside_sbo:  # TODO: check if it supports mtp
            shared_output = self._forward_shared_experts(
                hidden_states, gemm_output_zero_allocator
            )
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states, gemm_output_zero_allocator)

        # Collect expert statistics for this MoE block.
        times = {}
        _patched_moe_get_expert_statistics(self, router_logits, times)

        topk_output = self.topk(hidden_states, router_logits)
    else:
        shared_output = None
        topk_output = self.topk.empty_topk_output(hidden_states.device)

    if self._fuse_shared_experts_inside_sbo:
        shared_output = None

        def _pre_combine_hook(
            dispatcher: base.BaseDispatcher, combine_input: base.CombineInput
        ):
            nonlocal shared_output
            self.alt_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.alt_stream):
                shared_output = self._forward_shared_experts(
                    hidden_states, gemm_output_zero_allocator
                )
            pre_combine_hook_handle.remove()

        def _post_combine_hook(
            dispatcher: base.BaseDispatcher, hidden_states: torch.Tensor
        ):
            nonlocal shared_output
            torch.cuda.current_stream().wait_stream(self.alt_stream)
            post_combine_hook_handle.remove()

        pre_combine_hook_handle = self.experts.dispatcher.register_pre_combine_hook(
            _pre_combine_hook
        )
        post_combine_hook_handle = (
            self.experts.dispatcher.register_post_combine_hook(
                _post_combine_hook)
        )

    final_hidden_states = self.experts(
        hidden_states,
        topk_output,
    )
    if (
        (not base._is_cuda and not base._use_aiter)
        or isinstance(self.experts.quant_method, base.KTEPWrapperMethod)
    ):
        # fused in biased_grouped_topk so we can skip here
        final_hidden_states *= self.routed_scaling_factor
    if shared_output is not None:
        final_hidden_states += shared_output
    if (
        self.tp_size > 1
        and not should_allreduce_fusion
        and not use_reduce_scatter
        and not base.should_use_flashinfer_cutlass_moe_fp4_allgather()
    ):
        final_hidden_states = base.tensor_model_parallel_all_reduce(
            final_hidden_states)
    return final_hidden_states


_ORIG_DEEPEP_FORWARD = base.DeepseekV2MoE.forward_deepep

def _patched_moe_forward_deepep(
    self: base.DeepseekV2MoE,
    hidden_states: torch.Tensor,
    forward_batch: base.ForwardBatch,
) -> torch.Tensor:
    # Whether to enable detailed timing.
    profiling_enabled = (
        os.getenv("PROFILE_COMPONENT_OUTPUT_DIR") is not None
        or os.getenv("PROFILE_COMPONENT_BS") is not None
        or os.getenv("PROFILE_COMPONENT_IN") is not None
    )

    dispatch_time = 0.0
    moe_core_time = 0.0
    combine_time = 0.0

    timing_fn = time.perf_counter

    dispatcher = getattr(self.experts, "dispatcher", None)
    orig_dispatch_a = getattr(dispatcher, "dispatch_a", None) if dispatcher is not None else None
    orig_dispatch_b = getattr(dispatcher, "dispatch_b", None) if dispatcher is not None else None
    orig_combine_a = getattr(dispatcher, "combine_a", None) if dispatcher is not None else None
    orig_combine_b = getattr(dispatcher, "combine_b", None) if dispatcher is not None else None
    orig_run_moe_core = getattr(self.experts, "run_moe_core", None)

    def _time_dispatch(fn):
        if fn is None or not profiling_enabled:
            return fn

        def wrapped(this_dispatcher, *args, **kwargs):
            nonlocal dispatch_time
            start = timing_fn()
            out = fn(*args, **kwargs)
            end = timing_fn()
            dispatch_time += end - start
            return out

        return wrapped

    def _time_combine(fn):
        if fn is None or not profiling_enabled:
            return fn

        def wrapped(this_dispatcher, *args, **kwargs):
            nonlocal combine_time
            start = timing_fn()
            out = fn(*args, **kwargs)
            end = timing_fn()
            combine_time += end - start
            return out

        return wrapped

    def run_moe_core_timed(this_experts, *args, **kwargs):
        nonlocal moe_core_time
        if not profiling_enabled or orig_run_moe_core is None:
            return orig_run_moe_core(*args, **kwargs)
        start = timing_fn()
        out = orig_run_moe_core(*args, **kwargs)
        end = timing_fn()
        moe_core_time += end - start
        return out

    # Patch dispatcher / experts for timing, if available.
    if dispatcher is not None and orig_run_moe_core is not None and profiling_enabled:
        dispatcher.dispatch_a = types.MethodType(_time_dispatch(orig_dispatch_a), dispatcher)
        dispatcher.dispatch_b = types.MethodType(_time_dispatch(orig_dispatch_b), dispatcher)
        dispatcher.combine_a = types.MethodType(_time_combine(orig_combine_a), dispatcher)
        dispatcher.combine_b = types.MethodType(_time_combine(orig_combine_b), dispatcher)
        self.experts.run_moe_core = types.MethodType(run_moe_core_timed, self.experts)

    try:
        # Original forward_deepep logic from base implementation.
        shared_output = None
        sbo_enabled_flag = self._fuse_shared_experts_inside_sbo and not self.is_nextn
        sbo_overlap_dispatch_flag = (
            sbo_enabled_flag and SboFlags.enable_dispatch_shared_one_stream_overlap()
        )
        sbo_overlap_combine_flag = (
            sbo_enabled_flag and SboFlags.enable_combine_shared_two_stream_overlap()
        )

        if hidden_states.shape[0] > 0:
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states, forward_batch=forward_batch)
            if not sbo_enabled_flag:
                if self.alt_stream is not None:
                    self.alt_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(self.alt_stream):
                        shared_output = self._forward_shared_experts(hidden_states)
                        shared_output.record_stream(self.alt_stream)
                        shared_event = self.alt_stream.record_event()
                else:
                    shared_output = self._forward_shared_experts(hidden_states)
            topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
            )
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)

        if sbo_overlap_dispatch_flag:
            shared_output = None

            def _deepep_dispatch_hook(dispatcher: BaseDispatcher):
                nonlocal shared_output
                shared_output = self._forward_shared_experts(hidden_states)
                for handle in deepep_dispatch_hook_handle:
                    handle.remove()

            def _post_dispatch_hook(
                dispatcher: BaseDispatcher, dispatch_output: DispatchOutput
            ):
                combine_overlap_args, down_gemm_overlap_args, meta_overlap_args = (
                    compute_overlap_args(dispatch_output, self.alt_stream)
                )
                dispatcher.set_overlap_args(
                    combine_overlap_args=combine_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )
                self.experts.set_overlap_args(
                    down_gemm_overlap_args=down_gemm_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )
                post_dispatch_hook_handle.remove()

            def _post_combine_hook(
                dispatcher: BaseDispatcher, hidden_states: torch.Tensor
            ):
                dispatcher.clear_overlap_args()
                self.experts.clear_overlap_args()
                post_combine_hook_handle.remove()

            assert isinstance(self.experts.dispatcher, MaybeTboDeepEPDispatcher)
            deepep_dispatch_hook_handle = (
                self.experts.dispatcher.register_deepep_dispatch_hook(
                    _deepep_dispatch_hook
                )
            )
            post_dispatch_hook_handle = (
                self.experts.dispatcher.register_post_dispatch_hook(_post_dispatch_hook)
            )
            post_combine_hook_handle = (
                self.experts.dispatcher.register_post_combine_hook(_post_combine_hook)
            )

        elif sbo_overlap_combine_flag:
            shared_output = None

            def _post_dispatch_hook(
                dispatcher: BaseDispatcher, dispatch_output: DispatchOutput
            ):

                combine_overlap_args, down_gemm_overlap_args, meta_overlap_args = (
                    compute_overlap_args(dispatch_output, self.alt_stream)
                )
                dispatcher.set_overlap_args(
                    combine_overlap_args=combine_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )
                self.experts.set_overlap_args(
                    down_gemm_overlap_args=down_gemm_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )

                post_dispatch_hook_handle.remove()

            def _pre_combine_hook(
                dispatcher: BaseDispatcher, combine_input: CombineInput
            ):

                nonlocal shared_output

                if (
                    e := dispatcher.meta_overlap_args.get("record_event_after_down")
                ) is not None:
                    e.record()

                # TODO reduce sm for non-deepgemm
                with deep_gemm_wrapper.configure_deep_gemm_num_sms(
                    dispatcher.meta_overlap_args["compute_num_sms"]
                ):
                    shared_output = self._forward_shared_experts(hidden_states)

                pre_combine_hook_handle.remove()

            def _post_combine_hook(
                dispatcher: BaseDispatcher, hidden_states: torch.Tensor
            ):
                dispatcher.clear_overlap_args()
                self.experts.clear_overlap_args()
                post_combine_hook_handle.remove()

            post_dispatch_hook_handle = (
                self.experts.dispatcher.register_post_dispatch_hook(_post_dispatch_hook)
            )
            pre_combine_hook_handle = self.experts.dispatcher.register_pre_combine_hook(
                _pre_combine_hook
            )
            post_combine_hook_handle = (
                self.experts.dispatcher.register_post_combine_hook(_post_combine_hook)
            )
        elif envs.SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO.get():
            # On GB200: Shared experts overlapped on alt_stream, down gemm overlapped with DeepEP Combine

            def _post_dispatch_hook(
                dispatcher: BaseDispatcher, dispatch_output: DispatchOutput
            ):

                combine_overlap_args, down_gemm_overlap_args, meta_overlap_args = (
                    compute_overlap_args(dispatch_output, self.alt_stream)
                )
                dispatcher.set_overlap_args(
                    combine_overlap_args=combine_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )
                self.experts.set_overlap_args(
                    down_gemm_overlap_args=down_gemm_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )

                post_dispatch_hook_handle.remove()

            def _pre_combine_hook(
                dispatcher: BaseDispatcher, combine_input: CombineInput
            ):
                if (
                    e := dispatcher.meta_overlap_args.get("record_event_after_down")
                ) is not None:
                    e.record()
                pre_combine_hook_handle.remove()

            def _post_combine_hook(
                dispatcher: BaseDispatcher, hidden_states: torch.Tensor
            ):
                dispatcher.clear_overlap_args()
                self.experts.clear_overlap_args()
                post_combine_hook_handle.remove()

            post_dispatch_hook_handle = (
                self.experts.dispatcher.register_post_dispatch_hook(_post_dispatch_hook)
            )
            pre_combine_hook_handle = self.experts.dispatcher.register_pre_combine_hook(
                _pre_combine_hook
            )
            post_combine_hook_handle = (
                self.experts.dispatcher.register_post_combine_hook(_post_combine_hook)
            )

        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

        if (
            hidden_states.shape[0] > 0
            and not sbo_enabled_flag
            and self.alt_stream is not None
        ):
            torch.cuda.current_stream().wait_event(shared_event)
        if shared_output is not None:
            x = shared_output
            if self.experts.should_fuse_routed_scaling_factor_in_topk:
                x.add_(final_hidden_states)
            else:
                x.add_(final_hidden_states, alpha=self.routed_scaling_factor)
            final_hidden_states = x
        else:
            if not self.experts.should_fuse_routed_scaling_factor_in_topk:
                final_hidden_states *= self.routed_scaling_factor
    finally:
        # Restore dispatcher / experts methods after timing.
        if dispatcher is not None:
            if orig_dispatch_a is not None:
                dispatcher.dispatch_a = orig_dispatch_a
            if orig_dispatch_b is not None:
                dispatcher.dispatch_b = orig_dispatch_b
            if orig_combine_a is not None:
                dispatcher.combine_a = orig_combine_a
            if orig_combine_b is not None:
                dispatcher.combine_b = orig_combine_b
        if orig_run_moe_core is not None:
            self.experts.run_moe_core = orig_run_moe_core

    # Expose timing info for model-level profiler (written in the same JSON file).
    if profiling_enabled:
        self._last_deepep_times = {
            "dispatch_time": dispatch_time,
            "moe_core_time": moe_core_time,
            "combine_time": combine_time,
        }

    return final_hidden_states


_ORIG_DECODER_LAYER_FORWARD = base.DeepseekV2DecoderLayer.forward


def _patched_decoder_layer_forward(
    self,
    positions,
    hidden_states,
    forward_batch,
    residual,
    zero_allocator,
    gemm_output_zero_allocator=None,
    llama_4_scaling=None,
):
    """Patched `DeepseekV2DecoderLayer.forward` with attention / MLP timing."""

    profiling_enabled = (
        os.getenv("PROFILE_COMPONENT_OUTPUT_DIR") is not None
        or os.getenv("PROFILE_COMPONENT_BS") is not None
        or os.getenv("PROFILE_COMPONENT_IN") is not None
    )

    timing_fn = time.perf_counter
    attn_time = 0.0
    mlp_time = 0.0
    attn_time_cuda = None
    mlp_time_cuda = None

    attn_start_event = attn_end_event = None
    mlp_start_event = mlp_end_event = None
    if profiling_enabled and forward_batch.forward_mode == base.ForwardMode.DECODE and base._is_cuda:
        attn_start_event = torch.cuda.Event(enable_timing=True)
        attn_end_event = torch.cuda.Event(enable_timing=True)
        mlp_start_event = torch.cuda.Event(enable_timing=True)
        mlp_end_event = torch.cuda.Event(enable_timing=True)

    quant_format = (
        "mxfp4"
        if (
            base._is_gfx95_supported
            and getattr(self.self_attn, "fused_qkv_a_proj_with_mqa", None) is not None
            and getattr(self.self_attn.fused_qkv_a_proj_with_mqa, "weight", None) is not None
            and self.self_attn.fused_qkv_a_proj_with_mqa.weight.dtype == torch.uint8
        )
        else (
            "fp8"
            if (
                base._is_gfx95_supported
                and getattr(self.self_attn, "fused_qkv_a_proj_with_mqa", None) is not None
                and getattr(self.self_attn.fused_qkv_a_proj_with_mqa, "weight", None) is not None
                and self.self_attn.fused_qkv_a_proj_with_mqa.weight.dtype
                == getattr(torch, "float8_e4m3fn", None)
            )
            else ""
        )
    )

    hidden_states, residual = self.layer_communicator.prepare_attn(
        hidden_states,
        residual,
        forward_batch,
        quant_format,
    )

    if profiling_enabled and forward_batch.forward_mode == base.ForwardMode.DECODE:
        if attn_start_event is not None:
            attn_start_event.record()
        attn_start = timing_fn()
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
            llama_4_scaling=llama_4_scaling,
        )
        if base._is_cuda:
            torch.cuda.synchronize()
            if attn_end_event is not None:
                attn_end_event.record()
                torch.cuda.synchronize()
                attn_time_cuda = (
                    attn_start_event.elapsed_time(attn_end_event) / 1000.0
                )
        attn_end = timing_fn()
        attn_time = attn_end - attn_start
    else:
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
            llama_4_scaling=llama_4_scaling,
        )

    hidden_states, residual = self.layer_communicator.prepare_mlp(
        hidden_states, residual, forward_batch
    )

    should_allreduce_fusion = (
        self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
            forward_batch
        )
    )

    # For DP with padding, reduce scatter can be used instead of all-reduce.
    use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
        forward_batch
    )

    if isinstance(self.mlp, base.DeepseekV2MLP):
        gemm_output_zero_allocator = None

    if profiling_enabled and forward_batch.forward_mode == base.ForwardMode.DECODE:
        if mlp_start_event is not None:
            mlp_start_event.record()
        mlp_start = timing_fn()
        hidden_states = self.mlp(
            hidden_states,
            forward_batch,
            should_allreduce_fusion,
            use_reduce_scatter,
            gemm_output_zero_allocator,
        )
        if base._is_cuda:
            torch.cuda.synchronize()
            if mlp_end_event is not None:
                mlp_end_event.record()
                torch.cuda.synchronize()
                mlp_time_cuda = (
                    mlp_start_event.elapsed_time(mlp_end_event) / 1000.0
                )
        mlp_end = timing_fn()
        mlp_time = mlp_end - mlp_start
    else:
        hidden_states = self.mlp(
            hidden_states,
            forward_batch,
            should_allreduce_fusion,
            use_reduce_scatter,
            gemm_output_zero_allocator,
        )

    if not self.nsa_enable_prefill_cp and should_allreduce_fusion:
        hidden_states._sglang_needs_allreduce_fusion = True

    if not should_allreduce_fusion:
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )

    if profiling_enabled and forward_batch.forward_mode == base.ForwardMode.DECODE:
        # Store component times on layer instance for model-level profiler.
        self._last_layer_component_times = {
            "attn": attn_time,
            "mlp": mlp_time,
        }
        self._last_layer_component_times_cuda = {
            "attn": attn_time_cuda,
            "mlp": mlp_time_cuda,
        }

    return hidden_states, residual


def _patched_model_forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: base.ForwardBatch,
    input_embeds: torch.Tensor = None,
    pp_proxy_tensors: base.PPProxyTensors | None = None,
):
    """Patched `DeepseekV2Model.forward` with per-layer time profiling."""

    # Lazily initialize profiling output directory and counter.
    if not hasattr(self, "output_dir"):
        self.count = 0
        profile_component_output_dir = os.getenv(
            "PROFILE_COMPONENT_OUTPUT_DIR", None)
        if profile_component_output_dir is not None:
            self.output_dir = profile_component_output_dir
        else:
            profile_component_bs = os.getenv("PROFILE_COMPONENT_BS", None)
            if profile_component_bs is not None:
                profile_component_bs = int(profile_component_bs)

            profile_component_in = os.getenv("PROFILE_COMPONENT_IN", None)
            if profile_component_in is not None:
                profile_component_in = int(profile_component_in)

            profile_component_model = os.getenv(
                "PROFILE_COMPONENT_MODEL", "deepseek-v2-0.5.8"
            )
            self.output_dir = (
                f"component_times_{profile_component_model}/"
                f"{profile_component_model}_in{profile_component_in}_"
                f"bs{profile_component_bs}/"
            )
        os.makedirs(self.output_dir, exist_ok=True)

    profiling_enabled = (
        os.getenv("PROFILE_COMPONENT_OUTPUT_DIR") is not None
        or os.getenv("PROFILE_COMPONENT_BS") is not None
        or os.getenv("PROFILE_COMPONENT_IN") is not None
    )
    timing_function = time.perf_counter
    model_start = timing_function() if profiling_enabled else 0.0
    model_start_event = None
    model_end_event = None
    if profiling_enabled and base._is_cuda:
        model_start_event = torch.cuda.Event(enable_timing=True)
        model_end_event = torch.cuda.Event(enable_timing=True)

    total_num_layers = self.end_layer - self.start_layer
    device = input_embeds.device if input_embeds is not None else input_ids.device
    zero_allocator = base.BumpAllocator(
        buffer_size=total_num_layers * 2 *
        (2 if forward_batch.can_run_tbo else 1),
        dtype=torch.float32,
        device=device,
    )

    has_gemm_output_zero_allocator = hasattr(
        self, "gemm_output_zero_allocator_size"
    )

    gemm_output_zero_allocator = (
        base.BumpAllocator(
            buffer_size=self.gemm_output_zero_allocator_size,
            dtype=torch.float32,
            device=device,
        )
        if has_gemm_output_zero_allocator
        and self.gemm_output_zero_allocator_size > 0
        else None
    )

    if self.pp_group.is_first_rank:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        residual = None
    else:
        assert pp_proxy_tensors is not None
        hidden_states = pp_proxy_tensors["hidden_states"]
        residual = pp_proxy_tensors["residual"]

    if base.nsa_use_prefill_cp(forward_batch):
        if self.pp_group.is_first_rank:
            hidden_states = base.cp_split_and_rebuild_data(
                forward_batch, hidden_states)
        positions = base.cp_split_and_rebuild_position(
            forward_batch, positions)

    # llama_4_scaling: for supporting Mistral-Large-3 model
    llama_4_scaling: torch.Tensor | None = None
    if self.llama_4_scaling_config is not None:
        llama_4_scaling = base._get_llama_4_scaling(
            original_max_position_embeddings=self.llama_4_scaling_config[
                "original_max_position_embeddings"
            ],
            scaling_beta=self.llama_4_scaling_config["beta"],
            positions=positions,
        )

    normal_start_layer = self.start_layer
    normal_end_layer = self.end_layer
    if forward_batch.can_run_tbo:
        if (
            self.first_k_dense_replace > normal_start_layer
            and self.first_k_dense_replace < normal_end_layer
        ):
            normal_end_layer = self.first_k_dense_replace
        elif self.first_k_dense_replace < normal_start_layer:
            normal_end_layer = normal_start_layer = 0

    aux_hidden_states: list[torch.Tensor] = []
    all_layer_times: list[dict] = []
    all_layer_times_cuda: list[dict] = []

    for i in range(normal_start_layer, normal_end_layer):
        ctx = (
            base.nullcontext()
            if base.get_global_server_args().enable_piecewise_cuda_graph
            else base.get_global_expert_distribution_recorder().with_current_layer(i)
        )
        with ctx:
            if i in self.layers_to_capture:
                if self.enable_a2a_moe and i > self.first_k_dense_replace:
                    aux_hidden_state = base.tensor_model_parallel_all_gather(
                        hidden_states + residual, dim=0
                    )
                    aux_hidden_states.append(aux_hidden_state)
                else:
                    aux_hidden_states.append(hidden_states + residual)
            layer = self.layers[i]

            if profiling_enabled and forward_batch.forward_mode == base.ForwardMode.DECODE:
                layer_start_event = None
                layer_end_event = None
                if base._is_cuda:
                    layer_start_event = torch.cuda.Event(enable_timing=True)
                    layer_end_event = torch.cuda.Event(enable_timing=True)
                    layer_start_event.record()

                start = timing_function()
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    forward_batch,
                    residual,
                    zero_allocator,
                    gemm_output_zero_allocator,
                    llama_4_scaling,
                )
                if base._is_cuda:
                    torch.cuda.synchronize()
                end = timing_function()

                layer_total_time_cuda = None
                if (
                    base._is_cuda
                    and layer_start_event is not None
                    and layer_end_event is not None
                ):
                    # elapsed_time returns milliseconds
                    layer_end_event.record()
                    torch.cuda.synchronize()
                    layer_total_time_cuda = (
                        layer_start_event.elapsed_time(
                            layer_end_event) / 1000.0
                    )

                # Build per-layer detail dicts, aligned with Qwen3-MoE format.
                layer_details_cpu: dict = {}
                layer_details_cuda: dict = {}

                if isinstance(getattr(layer, "mlp", None), base.DeepseekV2MoE) and hasattr(
                    layer.mlp, "_last_deepep_times"
                ):
                    deepep_times = dict(layer.mlp._last_deepep_times)
                    for key_src, key_dst in [
                        ("dispatch_time", "moe_dispatch"),
                        ("moe_core_time", "moe_core"),
                        ("combine_time", "moe_combine"),
                    ]:
                        v = deepep_times.get(key_src, 0.0)
                        layer_details_cpu[key_dst] = v
                        layer_details_cuda[key_dst] = v

                # Add attention / MLP component times if available.
                if hasattr(layer, "_last_layer_component_times"):
                    comp = getattr(layer, "_last_layer_component_times", {}) or {}
                    layer_details_cpu["attn"] = float(comp.get("attn", 0.0))
                    layer_details_cpu["mlp"] = float(comp.get("mlp", 0.0))

                if hasattr(layer, "_last_layer_component_times_cuda"):
                    comp_cuda = getattr(
                        layer, "_last_layer_component_times_cuda", {}
                    ) or {}
                    # If CUDA-specific measurements are missing, fall back to CPU ones.
                    layer_details_cuda["attn"] = float(
                        comp_cuda.get(
                            "attn", layer_details_cpu.get("attn", 0.0)
                        )
                    )
                    layer_details_cuda["mlp"] = float(
                        comp_cuda.get(
                            "mlp", layer_details_cpu.get("mlp", 0.0)
                        )
                    )

                layer_time_entry = {
                    "layer_idx": i,
                    "total_layer_time": end - start,
                    "layer_details": layer_details_cpu,
                }
                all_layer_times.append(layer_time_entry)

                if layer_total_time_cuda is not None:
                    all_layer_times_cuda.append(
                        {
                            "layer_idx": i,
                            "total_layer_time": layer_total_time_cuda,
                            # Same schema as CPU JSON, but using CUDA-specific details when available.
                            "layer_details": layer_details_cuda or layer_details_cpu,
                        }
                    )
            else:
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    forward_batch,
                    residual,
                    zero_allocator,
                    gemm_output_zero_allocator,
                    llama_4_scaling,
                )

    if normal_end_layer != self.end_layer:
        hidden_states, residual = base.model_forward_maybe_tbo(
            layers=self.layers[normal_end_layer: self.end_layer],
            enable_tbo=True,
            positions=positions,
            forward_batch=forward_batch,
            hidden_states=hidden_states,
            residual=residual,
            input_data_scatter_mode=self.layers[
                normal_end_layer - 1
            ].layer_scatter_modes.layer_output_mode,
            zero_allocator=zero_allocator,
        )

    if not self.pp_group.is_last_rank:
        return base.PPProxyTensors(
            {
                "hidden_states": hidden_states,
                "residual": residual,
            }
        )
    else:
        if not forward_batch.forward_mode.is_idle():
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

    if base.nsa_use_prefill_cp(forward_batch) and self.pp_group.is_last_rank:
        hidden_states = base.cp_all_gather_rerange_output(
            hidden_states,
            self.cp_size,
            forward_batch,
            torch.cuda.current_stream(),
        )

    if profiling_enabled and forward_batch.forward_mode == base.ForwardMode.DECODE:
        if base._is_cuda:
            if model_start_event is not None:
                # Start CUDA timing for the whole model after inputs are prepared.
                model_start_event.record()
            torch.cuda.synchronize()
        model_end = timing_function()
        model_time = model_end - model_start
        model_time_cuda = None
        if (
            base._is_cuda
            and model_start_event is not None
            and model_end_event is not None
        ):
            model_end_event.record()
            torch.cuda.synchronize()
            # elapsed_time returns milliseconds
            model_time_cuda = model_start_event.elapsed_time(
                model_end_event) / 1000.0
        log_file = (
            f"{self.output_dir}/cputime/count_{self.count}_promptlenshape_"
            f"{str(input_ids.shape)}_time{timing_function()}.json"
        )
        log_file_cuda = (
            f"{self.output_dir}/cuda/count_{self.count}_promptlenshape_"
            f"{str(input_ids.shape)}_time{timing_function()}.json"
        )
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, "w") as f:
                json.dump(
                    {
                        "model_time": model_time,
                        # Same schema as Qwen3-MoE: per-layer entries with optional layer_details.
                        "layer_times": all_layer_times,
                    },
                    f,
                    indent=4,
                )
            if model_time_cuda is not None and base._is_cuda:
                os.makedirs(os.path.dirname(log_file_cuda), exist_ok=True)
                with open(log_file_cuda, "w") as f:
                    json.dump(
                        {
                            "model_time": model_time_cuda,
                            # CUDA side uses the same structure as CPU side.
                            "layer_times": all_layer_times_cuda,
                        },
                        f,
                        indent=4,
                    )
        except Exception:
            # Profiling must not break normal execution.
            pass
        self.count += 1

    if len(aux_hidden_states) == 0:
        return hidden_states
    return hidden_states, aux_hidden_states


def _apply_patches():
    # Patch MoE to add expert statistics and saving.
    base.DeepseekV2MoE.get_expert_statistics = _patched_moe_get_expert_statistics
    base.DeepseekV2MoE.forward_normal = _patched_moe_forward_normal
    base.DeepseekV2MoE.forward_deepep = _patched_moe_forward_deepep

    # Patch decoder layer to record attention / MLP timings.
    base.DeepseekV2DecoderLayer.forward = _patched_decoder_layer_forward

    # Patch model forward for time profiling.
    base.DeepseekV2Model.forward = _patched_model_forward


_apply_patches()


EntryClass = [DeepseekV2ForCausalLM,
              DeepseekV3ForCausalLM, DeepseekV32ForCausalLM]
