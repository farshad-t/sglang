#!/bin/bash

MODEL_NAME=${MODEL_NAME:-qwen3}
QUANTIZATION=${QUANTIZATION:-w4a8}
CPU_AFFINITIES=("0-42") #"43-85" "86-127") # 3 instances with 43 cores each on NUMA node 0 and 1


EXTRA_ARGS=""
if [ "$MODEL_NAME" == "qwen3" ]; then
    MODEL_PATH=/model/Qwen3-30B-A3B-Instruct-2507-w4g128/
    if [ "$QUANTIZATION" == "w4a8" ]; then
        GROUP_SIZE="${GROUP_SIZE:-128}"
        MODEL_PATH=/model/Qwen3-30B-A3B-Instruct-2507-w4g${GROUP_SIZE}/
        echo "Using GROUPS=${GROUP_SIZE}, MODEL_PATH=${MODEL_PATH}"
    elif [ "$QUANTIZATION" == "w8a8" ]; then
        MODEL_PATH=/model/Qwen3-30B-A3B-Instruct-2507-quantized.w8a8/
        EXTRA_ARGS="--quantization w8a8_int8 --dtype bfloat16"
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
        EXTRA_ARGS="--quantization w8a8_int8 --dtype bfloat16"
    else
        echo "Unsupported quantization: $QUANTIZATION"
        exit 1
    fi
    #EXTRA_ARGS="--quantization w8a8_int8"
else
    echo "Unsupported model: $MODEL_NAME"
    exit 1
fi
#MODEL_PATH=/model/Llama-3.1-8B-Instruct-autoround-w4g128-iters128-cpu
#MODEL_PATH=/model/Meta-Llama-3.1-8B-Instruct-quantized.w8a8
#MODEL_NAME=llama3

# Needed for mlperf 6.0 container
#export SGLANG_USE_CPU_W4A8=1

MAX_RUNNING_REQUESTS=128
MAX_PREFILL_TOKENS=16384
MAX_TOTAL_TOKENS=131072 # Experimental. Some request do not complete
CHUNKED_PREFILL_SIZE=131072 # This is applied to the entire batch (i.e sum[tokens]), so it doesn't apply on a per-request basis. https://github.com/sgl-project/sglang/issues/20018
                            #  Should cover all (64) requests with 1024 input tokens
ROUTER_PORT=30001

# Get the size of CPU_AFFINITIES
DP_SIZE=${#CPU_AFFINITIES[@]}
OUTPUT_DIR=/logs/sglang-router-logs-dp-size-${DP_SIZE}-model-${MODEL_NAME}-quant-${QUANTIZATION}-$(date +%Y%m%d)
mkdir -p "${OUTPUT_DIR}"
probe_endpoint() {
        local url="$1"
        local http_code

        http_code=$(curl -s -o /dev/null -w "%{http_code}" "${url}/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -d '{
                            "model": "'${MODEL_NAME}'",
                            "messages": [
                                {"role": "user", "content": "What is the capital of France?"}
                            ],
                            "max_tokens": 256,
                            "temperature": 0.7
                        }')

        [[ "$http_code" == "200" ]]
}

get_numa_node_from_cpu() {
    local cpu="$1"
    local sysfs_node="/sys/devices/system/cpu/cpu${cpu}/numa_node"
    local node

    if [[ -f "$sysfs_node" ]]; then
        cat "$sysfs_node"
        return 0
    fi

    node=$(lscpu -p=CPU,NODE | awk -F, -v target="$cpu" 'NR>1 && $1==target {print $2; exit}')
    if [[ -z "$node" ]]; then
        printf "Failed to resolve NUMA node for CPU %s\n" "$cpu" >&2
        return 1
    fi

    printf "%s" "$node"
}
start_dp() {
    local cpu_affinity="$1"
    local port="$2"
    local first_cpu
    local numa_node

    #export SGLANG_USE_CPU_W4A8=1
    first_cpu="${cpu_affinity%%,*}"
    first_cpu="${first_cpu%%-*}"
    numa_node=$(get_numa_node_from_cpu "$first_cpu")

    export SGLANG_CPU_OMP_THREADS_BIND="$cpu_affinity"
    #numactl -m "${numa_node}" -C "${cpu_affinity}" \
    python3 -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" \
        --dtype bfloat16 \
        --device cpu \
        --max-running-requests "$MAX_RUNNING_REQUESTS" \
        --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
        --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
        --mem-fraction-static 0.9 \
        --disable-radix-cache \
        --disable-piecewise-cuda-graph \
        --host 127.0.0.1 \
        --port "$port" $EXTRA_ARGS 2>&1 | tee \
        "${OUTPUT_DIR}/sglang_${MODEL_NAME}-cpus-${cpu_affinity}-max-requests-${MAX_RUNNING_REQUESTS}-chunks-${CHUNKED_PREFILL_SIZE}-${QUANTIZATION}.log" &
    
    # --max-total-tokens "$MAX_TOTAL_TOKENS" \ # Removed to match inference-max's
}

check_for_health() {

    local worker_urls

}

start_router() {
    local port_base=8080
    local i=0
    local -a launched_urls=()
    local port
    local url
    local attempt
    local worker_urls=""

    for cpu_affinity in "${CPU_AFFINITIES[@]}"; do
        port=$((port_base + i))
        url="http://localhost:${port}"
        printf "Launching dp: cpu_affinity=%s port=%s\n" "$cpu_affinity" "$port"
        start_dp "$cpu_affinity" "$port"
        launched_urls+=("$url")
        i=$((i + 1))
        sleep 5
    done

    sleep 10
    echo "Waiting for endpoints to be healthy..."
    echo "Probing endpoints for health..."

    for url in "${launched_urls[@]}"; do
        printf "Checking health for endpoint: %s\n" "$url"

        for attempt in 1 2 3 4 5 6 7 8 9 10; do
            if probe_endpoint "$url"; then
                printf "Probe succeeded: url=%s\n" "$url"
                if [[ -z "$worker_urls" ]]; then
                    worker_urls="$url"
                else
                    worker_urls+=" $url"
                fi
                break
            fi

            if [[ "$attempt" -eq 10 ]]; then
                printf "Probe failed after %s tries: url=%s\n" "$attempt" "$url" >&2
                exit 1
            fi

            sleep 5
        done
    done
    router_cmd="python3 -m sglang_router.launch_router --worker-urls $worker_urls --port $ROUTER_PORT --policy round_robin --request-timeout-secs 7200"
    echo "All endpoints are healthy. Starting router with command: $router_cmd"
    eval "$router_cmd" 2>&1 | tee "${OUTPUT_DIR}/sglang_router_round_robin.log"
    echo "Router started and logging to ${OUTPUT_DIR}/sglang_router_round_robin.log"
}

start_router