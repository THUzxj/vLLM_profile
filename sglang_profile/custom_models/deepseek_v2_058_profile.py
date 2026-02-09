import json
import os
import time

import torch

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
        top_k = getattr(getattr(self, "config", None), "num_experts_per_tok", None)
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
        minlength=getattr(getattr(self, "config", None), "n_routed_experts", 0),
    )
    activated_expert_token_counts = expert_token_counts[unique_experts]

    expert_token_distribution = {
        int(expert_id.item()): int(count.item())
        for expert_id, count in zip(unique_experts, activated_expert_token_counts)
    }
    times["expert_token_distribution"] = expert_token_distribution
    times["router_logits_shape"] = tuple(router_logits.shape)
    times["num_expert_requests"] = int(topk_indices.shape[0] * topk_indices.shape[1])

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
            self.experts.dispatcher.register_post_combine_hook(_post_combine_hook)
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
        final_hidden_states = base.tensor_model_parallel_all_reduce(final_hidden_states)
    return final_hidden_states


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
        profile_component_output_dir = os.getenv("PROFILE_COMPONENT_OUTPUT_DIR", None)
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

    total_num_layers = self.end_layer - self.start_layer
    device = input_embeds.device if input_embeds is not None else input_ids.device
    zero_allocator = base.BumpAllocator(
        buffer_size=total_num_layers * 2 * (2 if forward_batch.can_run_tbo else 1),
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
            hidden_states = base.cp_split_and_rebuild_data(forward_batch, hidden_states)
        positions = base.cp_split_and_rebuild_position(forward_batch, positions)

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

            if profiling_enabled:
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
                all_layer_times.append(
                    {
                        "layer_idx": i,
                        "total_layer_time": end - start,
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
            layers=self.layers[normal_end_layer : self.end_layer],
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

    if profiling_enabled:
        if base._is_cuda:
            torch.cuda.synchronize()
        model_end = timing_function()
        model_time = model_end - model_start
        log_file = (
            f"{self.output_dir}/count_{self.count}_promptlenshape_"
            f"{str(input_ids.shape)}_time{timing_function()}.json"
        )
        try:
            with open(log_file, "w") as f:
                json.dump(
                    {"model_time": model_time, "layer_times": all_layer_times},
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

    # Patch model forward for time profiling.
    base.DeepseekV2Model.forward = _patched_model_forward


_apply_patches()


EntryClass = [DeepseekV2ForCausalLM, DeepseekV3ForCausalLM, DeepseekV32ForCausalLM]



