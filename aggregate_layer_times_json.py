import argparse
import json
import os
import re
import csv

def parse_config_from_name(name):
    # pattern A: layer_times_stream{sid}_sm{sm}_bs{bs}_rep{rep}_pl{pl}_ol{ol}.json
    m = re.search(r"stream(\d+)_sm(\d+)_bs(\d+)_rep(\d+)_pl(\d+)_ol(\d+)", name)
    if m:
        return {
            'stream_idx': int(m.group(1)),
            'sm_count': int(m.group(2)),
            'batch_size': int(m.group(3)),
            'repeat_idx': int(m.group(4)),
            'prompt_length': int(m.group(5)),
            'output_length': int(m.group(6)),
        }
    # pattern B: layer_times_sm{sm}_bs{bs}_rep{rep}_pl{pl}_ol{ol}.json (no stream index)
    m2 = re.search(r"sm(\d+)_bs(\d+)_rep(\d+)_pl(\d+)_ol(\d+)", name)
    if m2:
        return {
            'stream_idx': 0,
            'sm_count': int(m2.group(1)),
            'batch_size': int(m2.group(2)),
            'repeat_idx': int(m2.group(3)),
            'prompt_length': int(m2.group(4)),
            'output_length': int(m2.group(5)),
        }
    return {}

def aggregate_file(path):
    with open(path, 'r') as fh:
        data = json.load(fh)
    # layer_records is a list over tokens; each token entry is a list over layers; each layer is a dict {'attn': t, 'ffn': t}
    layer_records = data.get('layer_records', [])
    if not layer_records:
        return None
    # transpose across tokens to get per-layer list
    # assume all token entries have same number of layers
    num_layers = len(layer_records[0])
    attn_means = []
    ffn_means = []
    for layer_idx in range(num_layers):
        attn_vals = []
        ffn_vals = []
        for token_entry in layer_records:
            layer_entry = token_entry[layer_idx]
            attn = layer_entry.get('attn_cuda')
            ffn = layer_entry.get('ffn_cuda')
            if attn is not None:
                attn_vals.append(attn)
            if ffn is not None:
                ffn_vals.append(ffn)
        attn_mean = sum(attn_vals) / len(attn_vals) if attn_vals else 0.0
        ffn_mean = sum(ffn_vals) / len(ffn_vals) if ffn_vals else 0.0
        attn_means.append(attn_mean)
        ffn_means.append(ffn_mean)
    # overall averages across layers
    overall_attn = sum(attn_means) / len(attn_means) if attn_means else 0.0
    overall_ffn = sum(ffn_means) / len(ffn_means) if ffn_means else 0.0
    cfg = parse_config_from_name(os.path.basename(path))
    return {
        **cfg,
        'num_layers': num_layers,
        'attn_mean_ms': overall_attn * 1000.0,
        'ffn_mean_ms': overall_ffn * 1000.0,
        'attn_std_ms': sum((x - overall_attn) ** 2 for x in attn_means) / len(attn_means) * 1000.0 if attn_means else 0.0,
        'ffn_std_ms': sum((x - overall_ffn) ** 2 for x in ffn_means) / len(ffn_means) * 1000.0 if ffn_means else 0.0,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing layer_times_*.json files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV path')
    args = parser.parse_args()

    rows = []
    for fname in os.listdir(args.input_dir):
        if not fname.startswith('layer_times_') or not fname.endswith('.json'):
            continue
        path = os.path.join(args.input_dir, fname)
        rec = aggregate_file(path)
        if rec is not None:
            rows.append(rec)

    if not rows:
        print('No valid JSON files found in', args.input_dir)
        return

    # sort by sm_count, repeat_idx, stream_idx ascending
    rows.sort(key=lambda r: (
        r.get('batch_size', 0),
        r.get('repeat_idx', 0),
        r.get('stream_idx', 0),
    ))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', newline='') as fh:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print('Wrote', len(rows), 'rows to', args.output)

if __name__ == '__main__':
    main()