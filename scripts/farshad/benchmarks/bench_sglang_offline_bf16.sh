#!/bin/bash

MODEL_NAME=${MODEL_NAME:-qwen3}
PROFILE=${PROFILE:-0}
EXPERT_METRICS=${EXPERT_METRICS:-0}
OUTPUT_DIR_TAG=${OUTPUT_DIR_TAG:-$(date +%Y%m%d)}

ENDPOINT_URL=http://localhost:30001

MODEL_EXTRA_ARGS=""
BENCHMARK_EXTRA_ARGS=""
if [ "$MODEL_NAME" == "qwen3" ]; then
    MODEL_PATH=/model/Qwen3-30B-A3B-Instruct-2507
elif [ "$MODEL_NAME" == "llama3" ]; then
    MODEL_PATH=/model/Llama-3.1-8B-Instruct
else
    echo "Unsupported model: $MODEL_NAME"
    exit 1
fi

if [ "$PROFILE" -eq 1 ]; then
    MODEL_EXTRA_ARGS="${MODEL_EXTRA_ARGS} --profile"
    export SGLANG_TORCH_PROFILER_DIR=${OUTPUT_DIR}/sglang_torch_profiler_${MODEL_NAME}_${OUTPUT_DIR_TAG}
    mkdir -p "${SGLANG_TORCH_PROFILER_DIR}"
    if [ "$EXPERT_METRICS" -eq 1 ]; then
        MODEL_EXTRA_ARGS="${MODEL_EXTRA_ARGS} --expert-distribution-recorder-mode per_token --expert-distribution-recorder-buffer-size 5000 --enable-expert-distribution-metrics"
        export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=${OUTPUT_DIR}/expert_logs
        mkdir -p ${SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR}
    else
        unset SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR
    fi
else
    unset SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR
fi
export SGLANG_USE_CPU_W4A8=0
NUM_PROMPTS=64

MAX_TOTAL_TOKENS=65536
MAX_PREFILL_TOKENS=67584
CHUNKED_PREFILL_SIZE=32768

OUTPUT_DIR=bench-throughput-logs-${MODEL_NAME}-exps-bf16-${OUTPUT_DIR_TAG}
mkdir -p "${OUTPUT_DIR}"

export SGLANG_CPU_OMP_THREADS_BIND="128-170|171-213|214-255"
# Set tp size to the size of CPU_OMP_THREADS_BIND (split by "|") to maximize throughput in offline benchmark
TP_SIZE=$(echo $SGLANG_CPU_OMP_THREADS_BIND | awk -F'|' '{print NF}')
echo "Using TP_SIZE=${TP_SIZE} for offline benchmark based on SGLANG_CPU_OMP_THREADS_BIND=${SGLANG_CPU_OMP_THREADS_BIND}"

model_args="--model ${MODEL_PATH} \
    --dtype bfloat16 \
    --device cpu \
    --tp-size ${TP_SIZE} \
    --chunked-prefill-size ${CHUNKED_PREFILL_SIZE} \
    --mem-fraction-static 0.9 \
    --disable-piecewise-cuda-graph \
    --disable-cuda-graph \
    --disable-radix-cache \
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
    local -a concurrencies=(23) # 3 22) #(32 16 8)
    local -a input_lens=(1024) #(1024 1024 1024) #(8192 1024)
    local -a output_lens=(1024) #(2048 4096 6144) #(1024 8192)
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
