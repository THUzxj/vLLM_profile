"""Launch the inference server."""

import asyncio
import os
import sys

from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
from sglang.srt.models.registry import ModelRegistry

def run_server(server_args):
    """Run the server based on server_args.grpc_mode and server_args.encoder_only."""
    if server_args.grpc_mode:
        from sglang.srt.entrypoints.grpc_server import serve_grpc

        asyncio.run(serve_grpc(server_args))
    elif server_args.encoder_only:
        from sglang.srt.disaggregation.encode_server import launch_server

        launch_server(server_args)
    else:
        # Default mode: HTTP mode.
        from sglang.srt.entrypoints.http_server import launch_server

        launch_server(server_args)


def import_custom_models(custom_models_path: str = "custom_models", mode: str = "torchprofile"):
    """
    Import custom models from the specified path and register them to ModelRegistry.

    Args:
        custom_models_path: Path to the directory containing custom model files.
        mode: Mode for importing custom models: "torchprofile" or "nvtx".
    """
    import sys
    from pathlib import Path

    custom_path = Path(custom_models_path)
    if not custom_path.exists():
        print(f"Warning: Custom models path '{custom_models_path}' does not exist.")
        return

    # Add the custom models path to sys.path for imports
    parent_dir = str(custom_path.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    package_name = custom_path.name

    # Import DeepseekV2ForCausalLM
    try:
        module = __import__(f"{package_name}.deepseek_v2_058_{mode}", fromlist=["DeepseekV2ForCausalLM"])
        DeepseekV2ForCausalLM = module.DeepseekV2ForCausalLM
        ModelRegistry.models["DeepSeekV2ForCausalLM"] = DeepseekV2ForCausalLM
        print(f"Successfully imported DeepseekV2ForCausalLM from {package_name}.deepseek_v2_058_{mode}")
    except ImportError as e:
        print(f"Failed to import DeepseekV2ForCausalLM: {e}")

    # Import Qwen3MoeForCausalLM
    try:
        module = __import__(f"{package_name}.qwen3_moe_058_{mode}", fromlist=["Qwen3MoeForCausalLM"])
        Qwen3MoeForCausalLM = module.Qwen3MoeForCausalLM
        ModelRegistry.models["Qwen3MoeForCausalLM"] = Qwen3MoeForCausalLM
        print(f"Successfully imported Qwen3MoeForCausalLM from {package_name}.qwen3_moe_058_{mode}")
    except ImportError as e:
        print(f"Failed to import Qwen3MoeForCausalLM: {e}")


if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    import_custom_models()

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
