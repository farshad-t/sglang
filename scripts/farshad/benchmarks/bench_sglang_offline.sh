#!/bin/bash

MODEL_NAME=${MODEL_NAME:-qwen3}
QUANTIZATION=${QUANTIZATION:-w4a8}
PROFILE=${PROFILE:-0}
OUTPUT_DIR_TAG=${OUTPUT_DIR_TAG:-$(date +%Y%m%d)}

ENDPOINT_URL=http://localhost:30001

MODEL_EXTRA_ARGS=""
BENCHMARK_EXTRA_ARGS=""
if [ "$MODEL_NAME" == "qwen3" ]; then
    MODEL_PATH=/model/Qwen3-30B-A3B-Instruct-2507-w4g128/
    GROUP_SIZE=${GROUP_SIZE:-128}
    if [ "$QUANTIZATION" == "w4a8" ]; then
        MODEL_PATH=/model/Qwen3-30B-A3B-Instruct-2507-w4g${GROUP_SIZE}/
    elif [ "$QUANTIZATION" == "w8a8" ]; then
        MODEL_PATH=/model/Qwen3-30B-A3B-Instruct-2507-quantized.w8a8/
        MODEL_EXTRA_ARGS="--quantization w8a8_int8 --dtype bfloat16"
    else
        echo "Unsupported quantization: $QUANTIZATION"
        exit 1
    fi
elif [ "$MODEL_NAME" == "llama3" ]; then
    MODEL_PATH=/model/Llama-3.1-8B-Instruct-autoround-w4g128-iters128-cpu
    if [ "$QUANTIZATION" == "w4a8" ]; then
        MODEL_PATH=/model/Llama-3.1-8B-Instruct-autoround-w4g128-iters128-cpu
    elif [ "$QUANTIZATION" == "w8a8" ]; then
        MODEL_PATH=/model/Meta-Llama-3.1-8B-Instruct-quantized.w8a8
        MODEL_EXTRA_ARGS="--quantization w8a8_int8 --dtype bfloat16"
    else
        echo "Unsupported quantization: $QUANTIZATION"
        exit 1
    fi
else
    echo "Unsupported model: $MODEL_NAME"
    exit 1
fi

if [ "$PROFILE" -eq 1 ]; then
    MODEL_EXTRA_ARGS="${MODEL_EXTRA_ARGS} --profile"
fi
export SGLANG_USE_CPU_W4A8=1
NUM_PROMPTS=64

MAX_TOTAL_TOKENS=65536
MAX_PREFILL_TOKENS=67584
CHUNKED_PREFILL_SIZE=32768

OUTPUT_DIR=bench-throughput-logs-${MODEL_NAME}-exps-${QUANTIZATION}-${OUTPUT_DIR_TAG}
mkdir -p "${OUTPUT_DIR}"

export SGLANG_TORCH_PROFILER_DIR=${OUTPUT_DIR}/sglang_torch_profiler_${MODEL_NAME}_${OUTPUT_DIR_TAG}
mkdir -p "${SGLANG_TORCH_PROFILER_DIR}"

export SGLANG_CPU_OMP_THREADS_BIND="0-42" #|43-85|86-127"
# Set tp size to the size of CPU_OMP_THREADS_BIND (split by "|") to maximize throughput in offline benchmark
TP_SIZE=$(echo $SGLANG_CPU_OMP_THREADS_BIND | awk -F'|' '{print NF}')
echo "Using TP_SIZE=${TP_SIZE} for offline benchmark based on SGLANG_CPU_OMP_THREADS_BIND=${SGLANG_CPU_OMP_THREADS_BIND}"

# Set output directory for expert distribution logs
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=${OUTPUT_DIR}/expert_logs
mkdir -p ${SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR}

model_args="--model ${MODEL_PATH} \
    --dtype bfloat16 \
    --device cpu \
    --chunked-prefill-size ${CHUNKED_PREFILL_SIZE} \
    --mem-fraction-static 0.9 \
    --disable-piecewise-cuda-graph \
    --disable-cuda-graph \
    --disable-radix-cache \
    --expert-distribution-recorder-mode per_token \
    --expert-distribution-recorder-buffer-size 5000 \
    ${MODEL_EXTRA_ARGS}"

bench_throughput() {
    local concurrency="$1"
    local input_len="$2"
    local output_len="$3"
    model_args="${model_args} --max-running-requests ${concurrency} --max-prefill-tokens ${MAX_PREFILL_TOKENS}"

    benchmark_args="--num-prompts ${concurrency} \
        --dataset-name random \
        --random-input-len ${input_len} \
        --random-output-len ${output_len} \
        --random-range-ratio 1.0"

    echo "Running offline throughput benchmark with bs=${concurrency}, input_len=${input_len}, output_len=${output_len}"
    python3 -m sglang.bench_offline_throughput \
        $model_args \
        $benchmark_args \
        2>&1 | tee "${OUTPUT_DIR}/sglang_${MODEL_NAME}_bench_offline_${NUM_PROMPTS}prompts_input_len-${input_len}_output_len-${output_len}_conc-${concurrency}-${OUTPUT_DIR_TAG}.log"
}

bench_sweep() {
    local -a concurrencies=(3 5 11 22 48 86)
    local -a input_lens=(1024) #(64 64 128 128 512 512 1024 1024) #(8192 1024)
    local -a output_lens=(8192) #(32 1024 32 1024 32 1024 32 1024) #(1024 8192)
    local i
    local concurrency

    for i in "${!input_lens[@]}"; do
        for concurrency in "${concurrencies[@]}"; do
            echo "Running benchmark with concurrency=${concurrency}, input_len=${input_lens[$i]}, output_len=${output_lens[$i]}"
            bench_throughput "$concurrency" "${input_lens[$i]}" "${output_lens[$i]}"
            sleep 10
        done
    done
}

bench_sweep
#bench_serving 64 8192 1024

# Fix permissions for host access
# Change ownership to host user (farshad uid=1008, gid=1008)
echo "Fixing permissions for ${OUTPUT_DIR}..."
chown -R 1008:1008 "${OUTPUT_DIR}" 2>/dev/null || true
echo "Permissions fixed for ${OUTPUT_DIR}"