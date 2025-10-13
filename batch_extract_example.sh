#!/bin/bash
# Batch KV Cache Log Processing Example
# This script demonstrates how to use the batch processor

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Batch KV Cache Log Processing Example${NC}"
echo "======================================"
echo

# Check if directories exist
LOG_DIR="${1:-benchmark_results/logs}"
OUTPUT_DIR="${2:-benchmark_results/kv_cache_analysis}"

if [ ! -d "$LOG_DIR" ]; then
    echo -e "${YELLOW}Warning: Log directory not found: $LOG_DIR${NC}"
    echo "Creating example directory structure..."
    mkdir -p "$LOG_DIR"
    echo "Please place your .log files in: $LOG_DIR"
    echo "Then run: $0 $LOG_DIR $OUTPUT_DIR"
    exit 0
fi

echo -e "${GREEN}Processing logs from: $LOG_DIR${NC}"
echo -e "${GREEN}Output directory: $OUTPUT_DIR${NC}"
echo

# Run the batch processor
python3 batch_extract_kv_cache.py \
    "$LOG_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --pattern "*.log" \
    --max-workers 4 \
    --extractor-script extract_kv_cache_logs.py

echo
echo -e "${GREEN}Batch processing completed!${NC}"
echo
echo "Generated files:"
echo "  📁 $OUTPUT_DIR/"
echo "  ├── batch_processing_summary.json  # Overall processing summary"
echo "  ├── *_kv_cache.json              # Individual extracted data"
echo "  ├── *_kv_cache.csv               # Individual CSV data"
echo "  └── *_kv_cache_summary.json      # Individual summaries"
echo
echo "To view the batch summary:"
echo "  cat $OUTPUT_DIR/batch_processing_summary.json | jq"
echo
echo "To merge all extracted data:"
echo "  python3 -c \""
echo "import json, glob"
echo "files = glob.glob('$OUTPUT_DIR/*_kv_cache.json')"
echo "all_data = []"
echo "for f in files:"
echo "    with open(f) as fp:"
echo "        data = json.load(fp)"
echo "        all_data.extend(data)"
echo "with open('$OUTPUT_DIR/merged_kv_cache_data.json', 'w') as fp:"
echo "    json.dump(all_data, fp, indent=2)"
echo "print(f'Merged {len(all_data)} KV cache entries')"
echo "\""