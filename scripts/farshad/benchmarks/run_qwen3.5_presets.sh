#!/bin/bash
# Quick launcher for Qwen3.5 benchmarks with common configurations
# Uses bench_one_batch (not bench_offline_throughput)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
    echo "Quick launcher for Qwen3.5 bench_one_batch benchmarks"
    echo ""
    echo "Usage: $0 <preset>"
    echo ""
    echo "Available presets:"
    echo "  30b-quick           Qwen3.5-30B, quick test (few batch sizes)"
    echo "  30b-full            Qwen3.5-30B, full sweep"
    echo "  30b-tp6             Qwen3.5-30B with TP=6 (all NUMA nodes)"
    echo "  2b-quick            Qwen3.5-2B, quick test"
    echo "  2b-full             Qwen3.5-2B, full sweep"
    echo "  both-quick          Run both 30B and 2B (quick)"
    echo "  30b-no-compile      Qwen3.5-30B without torch.compile"
    echo "  30b-varying-io      Qwen3.5-30B with varying input/output lengths"
    echo ""
    echo "Examples:"
    echo "  $0 30b-quick"
    echo "  $0 both-quick"
    echo "  OUTPUT_DIR_TAG=exp1 $0 30b-full"
    exit 1
}

if [ $# -eq 0 ]; then
    usage
fi

PRESET="$1"

case "$PRESET" in
    30b-quick)
        echo "Running Qwen3.5-30B (quick test)..."
        MODEL_NAME=qwen3.5-30b \
        BATCH_SIZES="11 22" \
        bash "${SCRIPT_DIR}/bench_qwen3.5_offline.sh"
        ;;
    
    30b-full)
        echo "Running Qwen3.5-30B (full sweep)..."
        MODEL_NAME=qwen3.5-30b \
        bash "${SCRIPT_DIR}/bench_qwen3.5_offline.sh"
        ;;
    
    30b-tp6)
        echo "Running Qwen3.5-30B with TP=6 (all NUMA nodes)..."
        MODEL_NAME=qwen3.5-30b \
        TP=6 \
        BATCH_SIZES="5 11 22" \
        bash "${SCRIPT_DIR}/bench_qwen3.5_offline.sh"
        ;;
    
    2b-quick)
        echo "Running Qwen3.5-2B (quick test)..."
        MODEL_NAME=qwen3.5-2b \
        BATCH_SIZES="32 64" \
        bash "${SCRIPT_DIR}/bench_qwen3.5_offline.sh"
        ;;
    
    2b-full)
        echo "Running Qwen3.5-2B (full sweep)..."
        MODEL_NAME=qwen3.5-2b \
        bash "${SCRIPT_DIR}/bench_qwen3.5_offline.sh"
        ;;
    
    both-quick)
        echo "Running both Qwen3.5-30B and Qwen3.5-2B (quick)..."
        echo ""
        echo "=== Starting Qwen3.5-30B ==="
        MODEL_NAME=qwen3.5-30b \
        BATCH_SIZES="11 22" \
        bash "${SCRIPT_DIR}/bench_qwen3.5_offline.sh"
        
        echo ""
        echo "=== Starting Qwen3.5-2B ==="
        MODEL_NAME=qwen3.5-2b \
        BATCH_SIZES="32 64" \
        bash "${SCRIPT_DIR}/bench_qwen3.5_offline.sh"
        ;;
    
    30b-no-compile)
        echo "Running Qwen3.5-30B without torch.compile..."
        MODEL_NAME=qwen3.5-30b \
        ENABLE_TORCH_COMPILE=0 \
        BATCH_SIZES="11 22" \
        bash "${SCRIPT_DIR}/bench_qwen3.5_offline.sh"
        ;;
    
    30b-varying-io)
        echo "Running Qwen3.5-30B with varying input/output lengths..."
        MODEL_NAME=qwen3.5-30b \
        BATCH_SIZES="22" \
        INPUT_LENS="512 1024 2048" \
        OUTPUT_LENS="512 1024 2048" \
        bash "${SCRIPT_DIR}/bench_qwen3.5_offline.sh"
        ;;
    
    *)
        echo "Unknown preset: $PRESET"
        usage
        ;;
esac

echo ""
echo "==================================================="
echo "Benchmark completed!"
echo "Check results in bench-one-batch-logs-* directories"
echo "==================================================="
