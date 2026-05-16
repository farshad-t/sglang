#!/bin/bash
# Benchmark script for Qwen3.5 models using bench_one_batch
# Usage:
#   MODEL_NAME=qwen3.5-30b bash bench_qwen3.5_offline.sh
#   MODEL_NAME=qwen3.5-2b TP=1 bash bench_qwen3.5_offline.sh

MODEL_NAME=${MODEL_NAME:-qwen3.5-30b}
ENABLE_TORCH_COMPILE=${ENABLE_TORCH_COMPILE:-1}
OUTPUT_DIR_TAG=${OUTPUT_DIR_TAG:-$(date +%Y%m%d)}

MODEL_EXTRA_ARGS=""

# Model configuration
if [ "$MODEL_NAME" == "qwen3.5-30b" ]; then
    MODEL_PATH=/model/Qwen3.5-30B-Instruct
    HF_MODEL_NAME=Qwen/Qwen3.5-30B-Instruct
    # Default settings for 30B model
    MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-65536}
    TP=${TP:-2}
    DEFAULT_BATCH_SIZES="5 11 22 48"
    
elif [ "$MODEL_NAME" == "qwen3.5-2b" ]; then
    MODEL_PATH=/model/Qwen3.5-2B-Instruct
    HF_MODEL_NAME=Qwen/Qwen3.5-2B-Instruct
    # Default settings for 2B model (smaller, can handle higher concurrency)
    MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-131072}
    TP=${TP:-1}
    DEFAULT_BATCH_SIZES="16 32 64 128"
    
else
    echo "Unsupported model: $MODEL_NAME"
    echo "Supported models: qwen3.5-30b, qwen3.5-2b"
    exit 1
fi

if [ "$ENABLE_TORCH_COMPILE" -eq 1 ]; then
    MODEL_EXTRA_ARGS="${MODEL_EXTRA_ARGS} --enable-torch-compile"
fi

# Create output directory
OUTPUT_DIR=bench-one-batch-logs-${MODEL_NAME}-${OUTPUT_DIR_TAG}
mkdir -p "${OUTPUT_DIR}"

# Create output directory
OUTPUT_DIR=bench-one-batch-logs-${MODEL_NAME}-${OUTPUT_DIR_TAG}
mkdir -p "${OUTPUT_DIR}"

# Model arguments
model_args="--model ${HF_MODEL_NAME} \
    --trust-remote-code \
    --device cpu \
    --tp ${TP} \
    --mem-fraction-static 0.8 \
    --max-total-tokens ${MAX_TOTAL_TOKENS} \
    ${MODEL_EXTRA_ARGS}"

bench_one_batch_run() {
    local batch_size="$1"
    local input_len="$2"
    local output_len="$3"

    benchmark_args="--batch-size ${batch_size} \
        --input ${input_len} \
        --output ${output_len}"

    echo "=========================================="
    echo "Running bench_one_batch:"
    echo "  Model: ${MODEL_NAME} (${HF_MODEL_NAME})"
    echo "  Batch Size: ${batch_size}"
    echo "  Input Length: ${input_len}"
    echo "  Output Length: ${output_len}"
    echo "  TP: ${TP}"
    echo "  Torch Compile: ${ENABLE_TORCH_COMPILE}"
    echo "=========================================="
    
    python3 -m sglang.bench_one_batch \
        $model_args \
        $benchmark_args \
        2>&1 | tee "${OUTPUT_DIR}/sglang_${MODEL_NAME}_bench_one_batch_bs${batch_size}_input${input_len}_output${output_len}_tp${TP}-${OUTPUT_DIR_TAG}.log"
}

bench_sweep() {
    # Configure sweep parameters
    # Override these by setting environment variables before running the script
    local -a batch_sizes=(${BATCH_SIZES:-$DEFAULT_BATCH_SIZES})
    local -a input_lens=(${INPUT_LENS:-1024})
    local -a output_lens=(${OUTPUT_LENS:-1024})
    
    echo "Starting benchmark sweep with:"
    echo "  Batch sizes: ${batch_sizes[@]}"
    echo "  Input lengths: ${input_lens[@]}"
    echo "  Output lengths: ${output_lens[@]}"
    echo ""
    
    local i
    for ((i=0; i<${#input_lens[@]}; i++)); do
        local input_len=${input_lens[$i]}
        local output_len=${output_lens[$i]}
        
        for batch_size in "${batch_sizes[@]}"; do
            bench_one_batch_run "$batch_size" "$input_len" "$output_len"
            echo ""
            # Small delay between runs
            sleep 2
        done
    done
}

# Main execution
echo "====================================================="
echo "Qwen3.5 bench_one_batch Benchmark"
echo "====================================================="
echo "Configuration:"
echo "  Model: ${MODEL_NAME}"
echo "  HF Model: ${HF_MODEL_NAME}"
echo "  Model Path: ${MODEL_PATH}"
echo "  Output Directory: ${OUTPUT_DIR}"
echo "  Max Total Tokens: ${MAX_TOTAL_TOKENS}"
echo "  TP: ${TP}"
echo "  Torch Compile: ${ENABLE_TORCH_COMPILE}"
echo "====================================================="
echo ""

# Run the benchmark sweep
bench_sweep

echo ""
echo "====================================================="
echo "Benchmark completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "====================================================="
