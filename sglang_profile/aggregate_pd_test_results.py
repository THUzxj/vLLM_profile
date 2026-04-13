import json
import csv
import glob
import re
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python aggregate_pd_test_results.py <results_dir> [output_file]")
    sys.exit(1)

results_dir = sys.argv[1]
output_file_arg = sys.argv[2] if len(sys.argv) >= 3 else None
jsonl_files = sorted(glob.glob(f"{results_dir}/results_cached*.jsonl"))

data = []
for file in jsonl_files:
    match = re.search(r'cached(\d+)', file)
    cached_tokens = int(match.group(1))

    with open(file, 'r') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                data.append({
                    "batch_size": record['batch_size'],
                    'cached_tokens': cached_tokens,
                    "extend_len": str(int(record['input_len']) - int(cached_tokens)),
                    "output_len": record['output_len'],
                    'ttft': record['last_ttft'],
                    'decode_latency': record['decode_latency'],
                    'tpot': record['tpot'],
                })

data.sort(key=lambda x: x['cached_tokens'])

dir_name = os.path.basename(results_dir.rstrip('/'))
output_file = (
    output_file_arg
    if output_file_arg is not None and output_file_arg.strip()
    else f"aggregated_results_{dir_name}.csv"
)
with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['batch_size', 'cached_tokens', 'extend_len', 'output_len', 'ttft', 'decode_latency', 'tpot'])
    writer.writeheader()
    writer.writerows(data)

print(f"Results saved to {output_file}")
for row in data:
    print(f"Batch Size: {row['batch_size']}, Cached: {row['cached_tokens']}, TTFT: {row['ttft']:.4f}s, TPOT: {row['tpot']:.4f}s, Decode Latency: {row['decode_latency']:.4f}s")
