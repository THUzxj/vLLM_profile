import argparse
import os
import time
from random import randint
from typing import Dict

import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt
from vllm.sampling_params import RequestOutputKind


def decode_bench_mixed_lengths(args, length_config: Dict[int, int], summary_path: str = None):
    """
    Run decoding benchmark with mixed prompt lengths in the same batch.
    
    Args:
        args: Arguments namespace similar to decode_bench
        length_config: Dict mapping prompt lengths to their counts in batch
            e.g., {128: 2, 1024: 3} means 2 prompts of length 128 and 3 prompts of length 1024
        summary_path: Optional path to a summary CSV file where all results will be aggregated
    """
    np.random.seed(42)
    
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Calculate total batch size from length_config
    total_batch_size = sum(length_config.values())
    
    # Create LLM instance
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        enforce_eager=False,
        max_num_seqs=1024,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=0.9,
    )
    llm_engine = llm.llm_engine

    data_path = os.path.join(args.log_dir, args.log_path)
    
    # Initialize CSV file with headers
    df_header = pd.DataFrame(columns=["context", "mixed_name", "bs", "repeat_idx", "tpot", 
                                    "decode_time", "total_length", "batched_tokens", "ttft", "requests"])
    df_header.to_csv(data_path, index=False)

    # Warm up
    print("Warming up...")
    output_len = args.__dict__.get("output_len", 100)
    sampling_params = SamplingParams(
        temperature=1,
        ignore_eos=True,
        max_tokens=output_len,
        output_kind=RequestOutputKind.DELTA,
    )

    # Simple warmup with smallest length
    min_length = int(min(length_config.keys()))
    print(f"Warmup with prompt length {min_length}")
    warmup_prompt = TokensPrompt(prompt_token_ids=[randint(0, 8192) for _ in range(min_length)])
    for _ in range(2):  # warmup iterations
        llm_engine.add_request(
            request_id="warmup",
            prompt=warmup_prompt,
            params=sampling_params,
        )
        while llm_engine.has_unfinished_requests():
            llm_engine.step()


    # Main benchmark loop
    for repeat_idx in range(args.num_repeat):

        # generate a plan for mixed lengths

        lengths = [length for length, v in length_config.items() for _ in range(v)]

        prefilled_tokens = sum(lengths)

        # Shuffle the lengths to create a mixed batch
        np.random.shuffle(lengths)

        print("plan:", lengths)

        print(f"\n=== Running mixed length batch, repeat {repeat_idx + 1}/{args.num_repeat} ===")
        
        # Clear any previous requests
        while llm_engine.has_unfinished_requests():
            llm_engine.step()

        # Create prompts for each length and add requests
        request_lengths = {}  # Track prompt length for each request
        rid = 0
        for length in lengths:
            prompt_tokens = [randint(0, 8192) for _ in range(length)]
            token_prompt = TokensPrompt(prompt_token_ids=prompt_tokens)
            request_id = f"mixed_{repeat_idx}_{rid}"
            
            llm_engine.add_request(
                request_id=request_id,
                prompt=token_prompt,
                params=sampling_params,
            )
            request_lengths[request_id] = length
            rid += 1

        window = args.window_size
        total_steps = 0
        ttft = None

        # Prefill and get TTFT
        while llm_engine.has_unfinished_requests():
            start = time.perf_counter()
            step_outputs = llm_engine.step()
            end = time.perf_counter()
            
            for output in step_outputs:
                if hasattr(output, 'outputs') and output.outputs:
                    if len(output.outputs[0].token_ids) > 0:
                        ttft = (end - start) * 1000  # in ms
                        break
            
            if ttft is not None:
                print(f"TTFT for mixed batch, repeat {repeat_idx + 1}: {ttft:.2f} ms")
                break
            else:
                print(f"Prefill step completed but no tokens generated yet")

        # Main generation loop
        while llm_engine.has_unfinished_requests():
            start = time.perf_counter()
            num_steps = 0
            
            for _ in range(window):
                if llm_engine.has_unfinished_requests():
                    step_outputs = llm_engine.step()
                    num_steps += 1
                else:
                    break
                    
            current_time = time.perf_counter()
            total_steps += num_steps
            tpot_dur_window = (current_time - start) / num_steps if num_steps > 0 else 0
            
            print(f"{total_steps=}, total_batch_size={total_batch_size}, "
                  f"repeat={repeat_idx+1}, tpot={tpot_dur_window * 1000:.2f}ms, "
                  f"unfinished={llm_engine.get_num_unfinished_requests()}")

            # Write results
            mixed_name = "_".join([f"{k}x{v}" for k, v in length_config.items()])
            result_row_data = {
                "context": total_steps,
                "mixed_name": mixed_name,
                "bs": total_batch_size,
                "repeat_idx": repeat_idx,
                "tpot": tpot_dur_window * 1000,
                "decode_time": (current_time - start),
                "total_length": length + total_steps,
                "batched_tokens": prefilled_tokens + total_steps * len(lengths),
                "ttft": ttft,
                "requests": lengths
            }
            
            # Save to individual result file
            result_row = pd.DataFrame([result_row_data])
            result_row.to_csv(data_path, mode='a', header=False, index=False)
            
            # Save to summary file if provided
            if summary_path:
                result_row_data["config_name"] = os.path.splitext(os.path.basename(args.log_path))[0]
                summary_row = pd.DataFrame([result_row_data])
                summary_row.to_csv(summary_path, mode='a', header=False, index=False)

        print(f"Mixed batch repeat {repeat_idx + 1} completed")


def load_length_configs(config_path: str):
    """Load length configurations from a JSON file."""
    import json
    
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Convert string keys to integers for each config
        processed_configs = {}
        for config in config_data['configs']:
            lengths = {int(k): int(v) for k, v in config['lengths'].items()}
            processed_configs[config['name']] = lengths
            
        return processed_configs
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return None
    except json.JSONDecodeError:
        print(f"Invalid JSON in config file: {config_path}")
        return None
    except KeyError as e:
        print(f"Missing required key in config file: {e}")
        return None

def test_mixed_lengths(config_file: str):
    import datetime
    import os
    
    # Load configurations from JSON
    configs = load_length_configs(config_file)
    
    if not configs:
        print("Failed to load configurations. Using default config.")
        configs = {
            "default": {
                1024: 4,
                8192: 4
            }
        }
    
    model_name = "/nfs/xjzhang/Qwen/Qwen3-4B"
    max_num_batched_tokens = 4096*32
    max_model_len = 40960
    tp_size = 1
    
    # Create a summary CSV file
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_path = os.path.join("profile_data_mixed", f"summary_mixed_lengths_{timestamp}.csv")
    os.makedirs("profile_data_mixed", exist_ok=True)
    
    # Initialize summary CSV with headers
    df_header = pd.DataFrame(columns=["context", "mixed_name", "bs", "repeat_idx", "tpot", 
                                    "decode_time", "total_length", "batched_tokens", "ttft",
                                    "config_name"])
    df_header.to_csv(summary_path, index=False)
    
    # Run benchmark for each configuration
    for config_name, length_config in configs.items():
        print(f"\nRunning benchmark for configuration: {config_name}")
        args = argparse.Namespace(
            model=model_name,
            tp_size=tp_size,
            log_path=f"{model_name.split('/')[-1]}_tp{tp_size}_{config_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            n_verify=1,
            num_repeat=3,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            output_len=100,
            window_size=128,
            log_dir="profile_data_mixed"
        )
        
        decode_bench_mixed_lengths(args, length_config, summary_path)


if __name__ == "__main__":
    import sys

    config_file = sys.argv[1] if len(sys.argv) > 1 else "length_config.json"

    test_mixed_lengths(config_file)