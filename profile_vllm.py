import argparse
import os
import time
from random import randint

import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt
from vllm.sampling_params import RequestOutputKind


# from specrl.patches.llm_engine import (draft_step, verify_step, rollback_draft_v1 as rollback_draft, cover_step_v1 as cover_step)
def decode_bench(args):
    np.random.seed(42)


    # only 1 token per prompt to show decode performance
    prompt_token_length = args.prompt_length
    prompt_token_ids = [randint(0, 8192) for _ in range(prompt_token_length)]
    token_prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
    
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        enforce_eager=False,
        max_num_seqs=1024,
        # max_num_batched_tokens=40960,
        max_num_batched_tokens=args.batched_tokens,
        # gpu_memory_utilization=0.9,
    )
    llm_engine = llm.llm_engine
    # batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    # batch_sizes = [64]

    batch_sizes = args.batch_sizes

    # batch_sizes = [8,8]
    # batch_sizes = [16,16,32]

    data_path = os.path.join("profile_data", args.log_path)
    
    # Initialize CSV file with headers
    df_header = pd.DataFrame(columns=["context", "bs", "repeat_idx", "tpot", "decode_time", "total_length", "batched_tokens"])
    df_header.to_csv(data_path, index=False)
    
    draft_tokens = {}
    all_outputs = []  # Store all final outputs
    
    for bs in batch_sizes:

        output_len = args.batched_tokens // bs - prompt_token_length

        if output_len <= 1:
            print(f"Skipping batch size {bs} due to insufficient output length, output_len={output_len}")
            continue

        sampling_params = SamplingParams(
            temperature=1,
            ignore_eos=True,
            max_tokens=output_len,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )


        for repeat_idx in range(args.num_repeat):
            print(f"\n=== Running batch size {bs}, repeat {repeat_idx + 1}/{args.num_repeat} output_len {output_len} ===")
            
            # Clear previous requests if any
            while llm_engine.has_unfinished_requests():
                llm_engine.step()
            
            # Reset for each repeat
            rid = bs * repeat_idx  # Unique request IDs across repeats
            batch_request_ids = []  # Track request IDs for this batch
            for i in range(bs):
                request_id = f"{bs}_{repeat_idx}_{rid}"  # More descriptive ID
                llm_engine.add_request(
                    request_id=request_id,
                    prompt=token_prompt,
                    params=sampling_params,
                )
                draft_tokens[request_id] = [randint(0, 8192) for _ in range(args.n_verify)]
                batch_request_ids.append(request_id)
                rid += 1
            window = args.window_size
            total_steps = 0
            batch_outputs = []  # Store outputs for this batch
            
            while llm_engine.has_unfinished_requests():
                start = time.perf_counter()
                num_steps = 0
                for _ in range(window):
                    if llm_engine.has_unfinished_requests():
                        # Get step outputs
                        step_outputs = llm_engine.step()
                        
                        # Process step outputs to collect finished requests
                        # for output in step_outputs:
                        #     if output.finished:
                        #         batch_outputs.append({
                        #             'request_id': output.request_id,
                        #             'outputs': output.outputs,
                        #             'finished': output.finished,
                        #             'batch_size': bs,
                        #             'repeat_idx': repeat_idx
                        #         })
                        #         print(f"Finished request {output.request_id} with {len(output.outputs)} outputs, outputs length: {[len(o.token_ids) if hasattr(o, 'token_ids') else 'N/A' for o in output.outputs]}")
                        
                        # llm_engine.cover_step(draft_tokens=draft_tokens)
                        num_steps += 1
                    else:
                        break
                total_steps += num_steps
                current_time = time.perf_counter()
                tpot_dur_window = (current_time - start) / num_steps
                print(f"{total_steps=}, {bs=}, repeat={repeat_idx+1}, {tpot_dur_window * 1000:.2f}, {llm_engine.get_num_unfinished_requests()=}")

                total_length = prompt_token_length + total_steps
                # Write result immediately to CSV with repeat information
                result_row = pd.DataFrame([{
                    "context": total_steps,
                    "bs": bs,
                    "repeat_idx": repeat_idx,
                    "tpot": tpot_dur_window * 1000,
                    "decode_time": (current_time - start),
                    "total_length": total_length,
                    "batched_tokens": total_length * bs,
                }])
                result_row.to_csv(data_path, mode='a', header=False, index=False)
            
            # Add batch outputs to all outputs
            all_outputs.extend(batch_outputs)
            print(f"Batch {bs}, repeat {repeat_idx + 1} completed with {len(batch_outputs)} finished requests")
        
        # Write result immediately to CSV
        
    
    # # Save all outputs to a separate file
    # print(f"total iterations: {total_steps}")
    # print(f"Total collected outputs: {len(all_outputs)}")
    
    # # Save detailed outputs information
    # outputs_log_path = args.log_path.replace('.csv', '_outputs.txt')
    # with open(os.path.join("profile_output", outputs_log_path), 'w') as f:
    #     f.write("=== Final Outputs Summary ===\n")
    #     f.write(f"Total requests processed: {len(all_outputs)}\n\n")
        
    #     for i, output_info in enumerate(all_outputs):
    #         f.write(f"Request {i+1}: {output_info['request_id']}\n")
    #         f.write(f"Batch Size: {output_info['batch_size']}\n")
    #         f.write(f"Repeat Index: {output_info['repeat_idx']}\n")
    #         f.write(f"Finished: {output_info['finished']}\n")
    #         f.write(f"Number of outputs: {len(output_info['outputs'])}\n")
            
    #         # Write first few tokens of each output
    #         for j, output in enumerate(output_info['outputs']):
    #             f.write(f"  Output {j+1}: ")
    #             if hasattr(output, 'text'):
    #                 text_preview = output.text[:100] + "..." if len(output.text) > 100 else output.text
    #                 f.write(f"'{text_preview}'\n")
    #             elif hasattr(output, 'token_ids'):
    #                 token_preview = str(output.token_ids[:10]) + "..." if len(output.token_ids) > 10 else str(output.token_ids)
    #                 f.write(f"Token IDs: {token_preview}\n")
    #             else:
    #                 f.write(f"Output object: {str(output)[:100]}...\n")
    #         f.write("\n")
    
    # print(f"finish log path: {data_path}")
    # print(f"outputs saved to: {outputs_log_path}")


def test():
    import datetime
    # model_name = "Qwen/Qwen2.5-7B"
    for model_name in ["/nfs/xjzhang/Qwen/Qwen3-4B"]:
        # seq_length = 2048
        # batched_tokens = 65536
        batched_tokens = 4096*16+1
        max_model_len = 40960
        if model_name == "/nfs/xjzhang/Qwen/Qwen3-8B":
            # tp_sizes = [1, 2, 4]
            tp_sizes = [1]
            # tp_sizes = [2, 4]
            # tp_sizes = [4]
        elif model_name == "/opt/tiger/open_verl/QWen2.5-32B":
            tp_sizes = [1, 2, 4]
        elif model_name == "/nfs/xjzhang/Qwen/Qwen3-1.7B":
            # tp_sizes = [1, 2, 4]
            tp_sizes = [1]
        elif model_name == "/nfs/xjzhang/Qwen/Qwen3-4B":
            # tp_sizes = [1, 2, 4]
            tp_sizes = [1]
        elif model_name == "/nfs/xjzhang/Qwen/Qwen3-14B":
            # tp_sizes = [1, 2, 4]
            # batched_tokens = 16384
            # max_model_len = 16384

            tp_sizes = [2, 4]
            # batched_tokens = 16384
            # max_model_len = 16384
        for tp_size in tp_sizes:
            # seq_length = 16384
        
            args = argparse.Namespace(
                model=model_name,
                tp_size=tp_size,
                # seq_len=seq_length,
                log_path=f"{model_name.split('/')[-1]}_tp{tp_size}_batched_tokens{batched_tokens}_decode_bench_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                n_verify=1,
                num_repeat=1,  # Repeat each batch size 3 times
                batched_tokens=batched_tokens,
                max_model_len=max_model_len,
                prompt_length=1,
                window_size=128,
                batch_sizes=[1, 4, 8, 16]
            )
            decode_bench(args)


if __name__ == "__main__":
    test()
