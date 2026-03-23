# MOE Trace Analysis v2

Torch profiler trace analysis tool for DeepSeek MOE model.

## Features

- **Step Segmentation**: Uses combine kernel intervals to detect step boundaries
- **Layer Segmentation**: Identifies 61 layers (3 dense + 58 MoE)
- **MOE Stage Classification**: Categorizes MoE layers into 4 stages:
  - `attention`: Between previous combine and dispatch
  - `dispatch`: `deep_ep::internode_ll::dispatch` kernel pair
  - `expert`: Between dispatch and combine
  - `combine`: `deep_ep::internode_ll::combine` kernel pair
- **Per-File Output**: Each trace file generates its own CSV output

## Requirements

```bash
pip install ijson
```

## Usage

### Full Analysis

```bash
python run_analysis.py \
    --trace-dir "d:\repos\vLLM_profile\sglang_profile\data\ali\data_0322\torch_profile_16_2048_1k" \
    --output-dir "d:\repos\vLLM_profile\sglang_profile\data\ali\data_0322\analysis_v2" \
    --gap-threshold-us 10000
```

### Aggregate Analysis Results (Multi-Rank)

Aggregate analysis results from multiple ranks, taking max across ranks and computing statistics:

```bash
python aggregate_analysis.py \
    --input-dir "d:\repos\vLLM_profile\sglang_profile\data\ali\data_0322\analysis_v2" \
    --output-dir "d:\repos\vLLM_profile\sglang_profile\data\ali\data_0322\aggregated" \
    --last-n-steps 20
```

**Aggregation Logic:**
1. Read analysis CSV files from all ranks
2. For each rank, take the last N steps (default: 20)
3. For each (step, layer, stage), take the MAX value across all ranks
4. Compute mean and stddev across steps for each (layer, stage)

**Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input-dir` | Yes | - | Directory containing analysis CSV files |
| `--output-dir` | Yes | - | Output directory for aggregated results |
| `--last-n-steps` | No | 20 | Number of last steps to use from each rank |

**Output Files:**
- `aggregated_stats.csv` - Aggregated statistics per (layer, stage)
- `aggregation_summary.txt` - Human-readable summary

### Export Layer Kernels (for Verification)

Export all kernels for a specific MoE layer to CSV and JSON for boundary verification:

```bash
python export_layer_kernels.py \
    --trace-file "d:\repos\vLLM_profile\sglang_profile\data\ali\data_0322\torch_profile_16_2048_1k\torch_profile\...\bs-2048-il-1000-...-TP-0.trace.json.gz" \
    --step-idx 0 \
    --layer-idx 4 \
    --output-dir "d:\repos\vLLM_profile\sglang_profile\data\ali\data_0322\layer_export"
```

**Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--trace-file` | Yes | - | Path to single trace file |
| `--step-idx` | No | 0 | Step index to export |
| `--layer-idx` | No | 4 | Layer index to export (4 = first MoE layer) |
| `--output-dir` | Yes | - | Output directory for CSV/JSON files |
| `--gap-threshold-us` | No | 10000 | Step detection threshold |

**Output Files:**
- `{trace_file}_step{N}_layer{M}.kernels.csv` - All kernels with stage classification
- `{trace_file}_step{N}_layer{M}.kernels.json` - Structured JSON with metadata and boundaries

## Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--trace-dir` | Yes | - | Directory containing trace files |
| `--output-dir` | Yes | - | Output directory for CSV files |
| `--gap-threshold-us` | No | 10000 | Step detection threshold (microseconds) |
| `--max-traces` | No | 0 | Max files to process (0 = all) |

## Output Format

Each trace file generates `{trace_filename}.analysis.csv`:

| Column | Description |
|--------|-------------|
| trace_file | Source filename |
| tp_rank | Tensor parallel rank |
| step_idx | Step index |
| layer_idx | Layer index (0-60) |
| layer_type | "dense" or "moe" |
| stage | attention/dispatch/expert/combine/dense |
| start_us | Start timestamp |
| end_us | End timestamp |
| dur_us | Duration (microseconds) |
| dur_ms | Duration (milliseconds) |
| kernel_count | Number of kernels |

## Implementation Details

### Step Segmentation
- Extracts `deep_ep::internode_ll::combine` kernel pairs
- Calculates intervals between consecutive pairs
- Gap > threshold indicates step boundary

### Layer Segmentation
- Dense layers (0-2): Kernels before first MoE combine
- MoE layers (3-60): Each ends with a combine kernel pair
- 58 MoE layers total

### Kernel Pair Handling
- Each dispatch and combine consists of 2 consecutive kernels with the same name
- Pairs are treated as single units for layer boundaries

## Full Analysis Pipeline

One-click script to run the complete analysis pipeline:

### Linux/Mac (Bash)

```bash
./run_full_analysis.sh /path/to/traces \
    --gap-threshold-us 10000 \
    --last-n-steps 20 \
    --max-traces 0
```

### Windows (Batch)

```cmd
run_full_analysis.bat D:\path\to\traces ^
    --gap-threshold-us 10000 ^
    --last-n-steps 20 ^
    --max-traces 0
```

### Pipeline Steps

1. **Individual Analysis**: Analyze each trace file (step/layer segmentation)
2. **Aggregation**: Aggregate results across all ranks

### Output Structure

```
input_trace_folder/
├── analysis_results/               # Individual analysis output
│   ├── TP-0.trace.json.gz.analysis.csv
│   ├── TP-1.trace.json.gz.analysis.csv
│   └── ...
└── aggregated_results/             # Aggregated output
    ├── aggregated_stats_per_layer.csv   # Per-layer statistics
    ├── aggregated_stats_averaged.csv    # Averaged across layers
    └── aggregation_summary.txt          # Human-readable report
```

```
analyze_traces_v2/
├── __init__.py
├── trace_loader.py           # Load trace files
├── stage_classifier.py       # Classify kernels into stages
├── step_segmenter.py         # Step segmentation
├── layer_segmenter.py        # Layer and MOE stage segmentation
├── analyzer.py               # Main analysis logic
├── run_analysis.py           # CLI entry point (full analysis)
├── export_layer_kernels.py   # CLI entry point (export layer kernels)
├── aggregate_analysis.py     # CLI entry point (aggregate multi-rank results)
└── README.md
```
