#!/bin/bash
# =============================================================================
# MOE Trace Analysis Pipeline
# =============================================================================
# Complete analysis pipeline for DeepSeek MOE model traces:
# 1. Analyze individual trace files (step/layer segmentation)
# 2. Aggregate results across all ranks
#
# Usage:
#   ./run_full_analysis.sh <trace_directory> [options]
#
# Arguments:
#   trace_directory    Directory containing *.trace.json or *.trace.json.gz files
#
# Options:
#   --gap-threshold-us  Step detection threshold in microseconds (default: 10000)
#   --last-n-steps      Number of last steps to use for aggregation (default: 20)
#   --max-traces        Max trace files to process (0=all, default: 0)
#
# Example:
#   ./run_full_analysis.sh /path/to/traces --gap-threshold-us 10000 --last-n-steps 20
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
GAP_THRESHOLD_US=10000
LAST_N_STEPS=20
MAX_TRACES=0

# Parse arguments
TRACE_DIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --gap-threshold-us)
            GAP_THRESHOLD_US="$2"
            shift 2
            ;;
        --last-n-steps)
            LAST_N_STEPS="$2"
            shift 2
            ;;
        --max-traces)
            MAX_TRACES="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 <trace_directory> [options]"
            echo ""
            echo "Arguments:"
            echo "  trace_directory    Directory containing trace files"
            echo ""
            echo "Options:"
            echo "  --gap-threshold-us  Step detection threshold (default: 10000)"
            echo "  --last-n-steps      Steps to use for aggregation (default: 20)"
            echo "  --max-traces        Max files to process, 0=all (default: 0)"
            echo "  --help, -h          Show this help message"
            exit 0
            ;;
        -*)
            echo -e "${RED}Error: Unknown option $1${NC}"
            exit 1
            ;;
        *)
            if [ -z "$TRACE_DIR" ]; then
                TRACE_DIR="$1"
            fi
            shift
            ;;
    esac
done

# Validate input directory
if [ -z "$TRACE_DIR" ]; then
    echo -e "${RED}Error: Please specify a trace directory${NC}"
    echo "Usage: $0 <trace_directory> [options]"
    exit 1
fi

if [ ! -d "$TRACE_DIR" ]; then
    echo -e "${RED}Error: Directory not found: $TRACE_DIR${NC}"
    exit 1
fi

# Convert to absolute path
TRACE_DIR=$(cd "$TRACE_DIR" && pwd)

# Set output directories (under input directory)
ANALYSIS_DIR="${TRACE_DIR}/analysis_results"
AGGREGATED_DIR="${TRACE_DIR}/aggregated_results"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}MOE Trace Analysis Pipeline${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo "Configuration:"
echo "  Input directory:    $TRACE_DIR"
echo "  Analysis output:    $ANALYSIS_DIR"
echo "  Aggregated output:  $AGGREGATED_DIR"
echo "  Gap threshold:      ${GAP_THRESHOLD_US}us"
echo "  Last N steps:       $LAST_N_STEPS"
echo "  Max traces:         $MAX_TRACES"
echo ""

# Check for Python
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

PYTHON_CMD=$(command -v python3 || command -v python)

# Check for required Python packages
echo -e "${YELLOW}Checking dependencies...${NC}"
if ! $PYTHON_CMD -c "import ijson" 2>/dev/null; then
    echo -e "${YELLOW}Warning: ijson not found. Installing...${NC}"
    $PYTHON_CMD -m pip install ijson
fi

echo -e "${GREEN}Dependencies OK${NC}"
echo ""

# Count trace files
echo -e "${YELLOW}Scanning for trace files...${NC}"
TRACE_COUNT=$(find "$TRACE_DIR" -name "*.trace.json" -o -name "*.trace.json.gz" | wc -l)
echo "  Found $TRACE_COUNT trace files"
echo ""

if [ "$TRACE_COUNT" -eq 0 ]; then
    echo -e "${RED}Error: No trace files found in $TRACE_DIR${NC}"
    exit 1
fi

# =============================================================================
# Step 1: Run individual trace analysis
# =============================================================================
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}Step 1: Analyzing Individual Trace Files${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

# Clean up previous analysis results
if [ -d "$ANALYSIS_DIR" ]; then
    echo -e "${YELLOW}Removing previous analysis results...${NC}"
    rm -rf "$ANALYSIS_DIR"
fi

mkdir -p "$ANALYSIS_DIR"

# Run analysis
$PYTHON_CMD "${SCRIPT_DIR}/run_analysis.py" \
    --trace-dir "$TRACE_DIR" \
    --output-dir "$ANALYSIS_DIR" \
    --gap-threshold-us "$GAP_THRESHOLD_US" \
    --max-traces "$MAX_TRACES"

echo ""

# Check if analysis produced results
ANALYSIS_COUNT=$(find "$ANALYSIS_DIR" -name "*.analysis.csv" | wc -l)
if [ "$ANALYSIS_COUNT" -eq 0 ]; then
    echo -e "${RED}Error: No analysis results generated${NC}"
    exit 1
fi

echo -e "${GREEN}Analysis complete: $ANALYSIS_COUNT files generated${NC}"
echo ""

# =============================================================================
# Step 2: Aggregate results across ranks
# =============================================================================
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}Step 2: Aggregating Results Across Ranks${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

# Clean up previous aggregated results
if [ -d "$AGGREGATED_DIR" ]; then
    echo -e "${YELLOW}Removing previous aggregated results...${NC}"
    rm -rf "$AGGREGATED_DIR"
fi

mkdir -p "$AGGREGATED_DIR"

# Run aggregation
$PYTHON_CMD "${SCRIPT_DIR}/aggregate_analysis.py" \
    --input-dir "$ANALYSIS_DIR" \
    --output-dir "$AGGREGATED_DIR" \
    --last-n-steps "$LAST_N_STEPS"

echo ""

# =============================================================================
# Step 3: Generate component time visualization
# =============================================================================
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}Step 3: Generating Component Time Visualization${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

COMPONENT_PLOTS_DIR="${TRACE_DIR}/component_plots"

# Clean up previous plots
if [ -d "$COMPONENT_PLOTS_DIR" ]; then
    echo -e "${YELLOW}Removing previous plots...${NC}"
    rm -rf "$COMPONENT_PLOTS_DIR"
fi

mkdir -p "$COMPONENT_PLOTS_DIR"

# Run component time analysis
$PYTHON_CMD "${SCRIPT_DIR}/analyze_rank_component_time.py" \
    "$ANALYSIS_DIR" \
    "$COMPONENT_PLOTS_DIR"

echo ""

# =============================================================================
# Summary
# =============================================================================
echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}Analysis Pipeline Complete!${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""
echo "Output locations:"
echo "  1. Individual analysis:  $ANALYSIS_DIR"
echo "     - $ANALYSIS_COUNT CSV files (*.analysis.csv)"
echo ""
echo "  2. Aggregated results:   $AGGREGATED_DIR"
echo "     - aggregated_stats_per_layer.csv  (per-layer statistics)"
echo "     - aggregated_stats_averaged.csv   (averaged across layers)"
echo "     - aggregation_summary.txt         (human-readable summary)"
echo ""
echo "  3. Component plots:      $COMPONENT_PLOTS_DIR"
echo "     - rank_X_component_time.png (one plot per rank)"
echo ""

# Show quick summary if available
if [ -f "${AGGREGATED_DIR}/aggregated_stats_averaged.csv" ]; then
    echo -e "${BLUE}Quick Summary (Averaged across layers):${NC}"
    echo ""
    # Skip header and format output
    tail -n +2 "${AGGREGATED_DIR}/aggregated_stats_averaged.csv" | while IFS=',' read -r layer_idx layer_type stage count mean std min max p50 p90 p95 layer_std; do
        printf "  %-12s: %8.3f ms (±%6.3f ms)\n" "$stage" "$mean" "$std"
    done
    echo ""
fi

echo -e "${GREEN}Done!${NC}"
