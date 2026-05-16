#!/bin/bash

MODEL_NAME=${1:-qwen3}
DP_SIZE=${DP_SIZE:-1}

PORT=8080
if [ "$DP_SIZE" -ge 2 ]; then
    PORT=30001
fi
ENDPOINT_URL=http://localhost:${PORT}
QUANTIZATION=${QUANTIZATION:-w4a8}

if [ "$MODEL_NAME" == "qwen3" ]; then
    MODEL_PATH=/model/Qwen3-30B-A3B-Instruct-2507-w4g128/
    if [ "$QUANTIZATION" == "w4a8" ]; then
        MODEL_PATH=/model/Qwen3-30B-A3B-Instruct-2507-w4g128/
    elif [ "$QUANTIZATION" == "w8a8" ]; then
        MODEL_PATH=/model/Qwen3-30B-A3B-Instruct-2507-quantized.w8a8/
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
    else
        echo "Unsupported quantization: $QUANTIZATION"
        exit 1
    fi
else
    echo "Unsupported model: $MODEL_NAME"
    exit 1
fi

NUM_PROMPTS=128
OUTPUT_DIR_TAG=${OUTPUT_DIR_TAG:-$(date +%Y%m%d)}
OUTPUT_DIR=bench-serving-${MODEL_NAME}-dp-size-${DP_SIZE}-${OUTPUT_DIR_TAG}-${QUANTIZATION}
mkdir -p "${OUTPUT_DIR}"
bench_serving() {
    local concurrency="$1"
    local input_len="$2"
    local output_len="$3"
    
    python3 -m sglang.bench_serving \
        --backend sglang \
        --base-url "$ENDPOINT_URL" \
        --model "${MODEL_PATH}" \
        --served-model-name "${MODEL_NAME}" \
        --tokenizer "$MODEL_PATH" \
        --dataset-name random \
        --num-prompts "$concurrency" \
        --random-input-len "$input_len" \
        --random-output-len "$output_len" \
        --random-range-ratio 1.0 \
        --max-concurrency "$concurrency" \
        2>&1 | tee "${OUTPUT_DIR}/sglang_${MODEL_NAME}_bench_serving_${concurrency}prompts_input_len-${input_len}_output_len-${output_len}_conc-${concurrency}-${OUTPUT_DIR_TAG}.log"
}

bench_sweep() {
    local -a concurrencies=(3 5 11 22 48)
    local -a input_lens=(8192 1024) #(64 64 128 128 512 512 1024 1024) #(8192 1024)
    local -a output_lens=(1024 8192) #(32 1024 32 1024 32 1024 32 1024) #(1024 8192)
    local i
    local concurrency

    for i in "${!input_lens[@]}"; do
        for concurrency in "${concurrencies[@]}"; do
            echo "Running benchmark with concurrency=${concurrency}, input_len=${input_lens[$i]}, output_len=${output_lens[$i]}"
            bench_serving "$concurrency" "${input_lens[$i]}" "${output_lens[$i]}"
            sleep 10
        done
    done
}

bench_sweep
#bench_serving 64 8192 1024