#!/bin/bash
# Collect expert statistics for all Qwen3.5/3.6 configurations from the measurements table
# Run this script INSIDE the farshad-sglang-new container
# Usage: 
#   bash collect_qwen35_expert_stats.sh [NUMA_NODE]
#   Example: bash collect_qwen35_expert_stats.sh 0
#   DRY_RUN=1 bash collect_qwen35_expert_stats.sh 0  # Dry run mode

set -euo pipefail

# Set HuggingFace cache to /data2 partition (403GB available) instead of default ~/.cache (32GB)
export HF_HOME=/model/qwen-models
export HUGGINGFACE_HUB_CACHE=/model/qwen-models/hub

# Create cache directory if it doesn't exist (skip if symlink/directory already exists)
if [ ! -e "${HF_HOME}" ]; then
    mkdir -p "${HF_HOME}"
fi

# Dry run mode (set DRY_RUN=1 to only show what would be run)
DRY_RUN=${DRY_RUN:-0}

# NUMA node selection (default to node 0 if not specified)
NUMA_NODE=${1:-${NUMA_NODE:-0}}

# Set CPU thread binding for the selected NUMA node to prevent thread migration across nodes
# This MUST be set before any Python/SGLang processes start
case "${NUMA_NODE}" in
    0) export SGLANG_CPU_OMP_THREADS_BIND="0-42" ;;
    1) export SGLANG_CPU_OMP_THREADS_BIND="43-85" ;;
    2) export SGLANG_CPU_OMP_THREADS_BIND="86-127" ;;
    3) export SGLANG_CPU_OMP_THREADS_BIND="128-170" ;;
    4) export SGLANG_CPU_OMP_THREADS_BIND="171-213" ;;
    5) export SGLANG_CPU_OMP_THREADS_BIND="214-255" ;;
    *) echo "ERROR: Invalid NUMA node ${NUMA_NODE}"; exit 1 ;;
esac

echo "Using NUMA node: ${NUMA_NODE}"
echo "CPU thread binding: ${SGLANG_CPU_OMP_THREADS_BIND}"

OUTPUT_DIR_TAG=${OUTPUT_DIR_TAG:-$(date +%Y%m%d)-expert-stats}
BASE_OUTPUT_DIR=/code/qwen35-expert-stats-${OUTPUT_DIR_TAG}-node${NUMA_NODE}
mkdir -p "${BASE_OUTPUT_DIR}"

echo "Verifying NUMA configuration..."
numactl --hardware | grep "node ${NUMA_NODE}" || { echo "ERROR: NUMA node ${NUMA_NODE} not found!"; exit 1; }
echo ""

# Enable expert distribution recording if supported
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=${BASE_OUTPUT_DIR}/expert_logs
mkdir -p ${SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR}

# Configuration array: model_name, model_path, tp, batch_size, input_len, output_len
# Based on the measurements table in QWEN35.md
# Models not yet downloaded will be skipped automatically
# NOTE: Using TP=1 for all configs since expert activation patterns don't depend on TP
#       (TP only affects performance, not routing decisions)
declare -a CONFIGS=(
    # Format: "model_name|model_path|tp|batch_size|input_len|output_len|description"
    # Using HuggingFace model IDs - will auto-download to HF_HOME=/model/qwen-models
    # NOTE: Qwen3.5 models do NOT have -Instruct variants on HuggingFace
    
    # Dense models (no MoE, skip for expert stats)
    # "qwen3.5-2B-bf16|Qwen/Qwen3.5-2B|1|61|1024|1024|Qwen3.5-2B BF16 BS=61"
    # "qwen3.5-9B-bf16|Qwen/Qwen3.5-9B|1|35|1024|1024|Qwen3.5-9B BF16 BS=35"
    # "qwen3.5-27B-fp8|Qwen/Qwen3.5-27B-fp8|1|13|1024|1024|Qwen3.5-27B FP8 BS=13"
    # "qwen3.5-27B-bf16|Qwen/Qwen3.5-27B|1|13|1024|1024|Qwen3.5-27B BF16 BS=13"
    
    # MoE models - Qwen3.5-35B (only A3B variants exist)
    "qwen3.5-35B-A3B-fp8|Qwen/Qwen3.5-35B-A3B-FP8|1|22|1024|1024|Qwen3.5-35B-A3B FP8 BS=22"
    "qwen3.5-35B-A3B-bf16|Qwen/Qwen3.5-35B-A3B|1|20|1024|1024|Qwen3.5-35B-A3B BF16 BS=20"
    
    # Qwen3.5-122B (A10B variants)
    "qwen3.5-122B-A10B-fp8|Qwen/Qwen3.5-122B-A10B-FP8|1|5|1024|1024|Qwen3.5-122B-A10B FP8 BS=5"
    "qwen3.5-122B-A10B-bf16|Qwen/Qwen3.5-122B-A10B|1|10|1024|1024|Qwen3.5-122B-A10B BF16 BS=10"
    
    # NOTE: 397B models exist but are too large (~800GB each) to download
    
    # Dense model (no MoE, skip for expert stats)
    # "qwen3.5-0.8B|Qwen/Qwen3.5-0.8B|1|32|1024|1024|Qwen3.5-0.8B BF16 BS=32"
)

# Function to run a single benchmark configuration
run_benchmark() {
    local model_name="$1"
    local model_path="$2"
    local tp="$3"
    local batch_size="$4"
    local input_len="$5"
    local output_len="$6"
    local description="$7"
    
    local output_subdir="${BASE_OUTPUT_DIR}/${model_name}_tp${tp}_bs${batch_size}"
    mkdir -p "${output_subdir}"
    
    # Set expert log directory for this specific run
    export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR="${output_subdir}/expert_logs"
    mkdir -p "${SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR}"
    
    echo "=========================================="
    echo "Running: ${description}"
    echo "  Model: ${model_path}"
    echo "  TP: ${tp}, Batch Size: ${batch_size}"
    echo "  Input: ${input_len}, Output: ${output_len}"
    echo "  NUMA Node: ${NUMA_NODE}"
    echo "  Expert logs: ${SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR}"
    echo "=========================================="
    
    # Safety check: ensure TP=1
    if [ "$tp" != "1" ]; then
        echo "ERROR: TP=${tp} detected! Only TP=1 is allowed for expert stats collection."
        echo "Configuration: ${description}"
        exit 1
    fi
    
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[DRY RUN] Would execute:"
        echo "  numactl --cpunodebind=${NUMA_NODE} --membind=${NUMA_NODE} \\"
        echo "  python3 -m sglang.bench_one_batch \\"
        echo "    --model \"${model_path}\" \\"
        echo "    --trust-remote-code \\"
        echo "    --device cpu \\"
        echo "    --tp ${tp} \\"
        echo "    --batch-size ${batch_size} \\"
        echo "    --input ${input_len} \\"
        echo "    --output ${output_len} \\"
        echo "    --mem-fraction-static 0.8 \\"
        echo "    --max-total-tokens 65536 \\"
        echo "    --enable-torch-compile"
        echo ""
        echo "[DRY RUN] Output would be saved to: ${output_subdir}/benchmark.log"
        return
    fi
    
    # Run bench_one_batch with numactl to bind to specific NUMA node
    numactl --cpunodebind=${NUMA_NODE} --membind=${NUMA_NODE} \
    /opt/.venv/bin/python3 -m sglang.bench_one_batch \
        --model "${model_path}" \
        --trust-remote-code \
        --device cpu \
        --tp ${tp} \
        --batch-size ${batch_size} \
        --input ${input_len} \
        --output ${output_len} \
        --mem-fraction-static 0.8 \
        --max-total-tokens 65536 \
        --expert-distribution-recorder-mode per_token \
        --expert-distribution-recorder-buffer-size 5000 \
        2>&1 | tee "${output_subdir}/benchmark.log"
    
    echo ""
    echo "Completed: ${description}"
    echo "Results saved to: ${output_subdir}"
    echo ""
    
    # Small delay between runs
    sleep 3
}

# Main execution
echo "====================================================="
echo "Qwen3.5/3.6 Expert Statistics Collection"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "MODE: DRY RUN (no actual execution)"
fi
echo "====================================================="
echo "Base output directory: ${BASE_OUTPUT_DIR}"
echo "NUMA Node: ${NUMA_NODE}"
echo "Number of configurations: ${#CONFIGS[@]}"
echo "====================================================="
echo ""

# Validate all configs use TP=1
echo "Validating TP settings..."
all_tp1=true
for config in "${CONFIGS[@]}"; do
    IFS='|' read -r model_name model_path tp batch_size input_len output_len description <<< "$config"
    if [ "$tp" != "1" ]; then
        echo "ERROR: Found TP=${tp} in config: ${description}"
        all_tp1=false
    fi
done

if [ "$all_tp1" = true ]; then
    echo "✓ All configurations use TP=1 (verified)"
    echo ""
else
    echo "✗ Some configurations do not use TP=1!"
    echo "Aborting."
    exit 1
fi

# Run all configurations (HuggingFace will auto-download as needed)
run_count=0
skip_count=0
for config in "${CONFIGS[@]}"; do
    IFS='|' read -r model_name model_path tp batch_size input_len output_len description <<< "$config"
    
    run_benchmark "$model_name" "$model_path" "$tp" "$batch_size" "$input_len" "$output_len" "$description"
    run_count=$((run_count + 1))
done

echo ""
echo "====================================================="
if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run completed!"
    echo "====================================================="
    echo "Configurations that would run: ${run_count}"
    echo "Configurations skipped (model not found): ${skip_count}"
    echo ""
    echo "All configurations verified to use TP=1 ✓"
    echo ""
    echo "To actually run the benchmarks, execute without DRY_RUN:"
    echo "  bash collect_qwen35_expert_stats.sh ${NUMA_NODE}"
else
    echo "All benchmarks completed!"
    echo "====================================================="
    echo "Benchmarks run: ${run_count}"
    echo "Configurations skipped: ${skip_count}"
    echo "Results saved to: ${BASE_OUTPUT_DIR}"
    echo "NUMA Node used: ${NUMA_NODE}"
    echo ""
    echo "Next steps:"
    echo "1. Check for expert logs in: ${BASE_OUTPUT_DIR}/*/expert_logs/"
    echo "2. Analyze with: python3 /code/farshad_expert_stats_scripts/analyze_prefill_from_log.py"
fi
echo "====================================================="
