#!/usr/bin/env bash
set -euo pipefail

# Wrapper: start endpoint container, verify readiness, then run client (script or YAML).
# Usage:
#   bash run_endpoint_client.sh -c llama3.1-8b-endpoint-config.yaml -y llama3.1-8b-client-config.yaml [-w llama3.1-8b-client-warmup-config.yaml]
#
# Endpoint selection:
#   - The launcher is auto-detected from the endpoint config (vLLM vs SGLang).
#   - Override detection by exporting ENDPOINT_NAME (e.g., vllm or sglang).
# Options:
#   -c CONFIG_YAML   Endpoint config YAML path (required)
#   -y CLIENT_YAML   Client config YAML to run via inference-endpoint (preferred)
#   -w WARMUP_YAML   Optional warmup client YAML to run first (256 samples)
#   -n NAME          Container name (default: mlperf-endpoints-6.0)
#   -l LOG_DIR       Host logs dir (default: ./logs)
#   -i IMAGE         Docker image (default: vllm_xpu:ww04)
#   -t TIMEOUT       Extra readiness wait seconds (default: 0; wrapper trusts start script)
#
# Exits non-zero if container or endpoint launch fails. Prints clear diagnostics.

CONFIG_YAML=""
CLIENT_YAML=""
WARMUP_YAML=""
CONTAINER_NAME="mlperf-endpoints-6.0"
LOG_DIR="$(pwd)/logs"
DOCKER_IMAGE="vllm_xpu:ww04"
EXTRA_WAIT=0

while getopts ":c:y:w:n:l:i:t:" opt; do
  case "$opt" in
    c) CONFIG_YAML="$OPTARG" ;;
    y) CLIENT_YAML="$OPTARG" ;;
    w) WARMUP_YAML="$OPTARG" ;;
    n) CONTAINER_NAME="$OPTARG" ;;
    l) LOG_DIR="$OPTARG" ;;
    i) DOCKER_IMAGE="$OPTARG" ;;
    t) EXTRA_WAIT="$OPTARG" ;;
    :) echo "[error] Option -$OPTARG requires an argument" >&2; exit 2 ;;
    \?) echo "[error] Unknown option -$OPTARG" >&2; exit 2 ;;
  esac
done

if [[ -z "$CONFIG_YAML" || -z "$CLIENT_YAML" ]]; then
  echo "[usage] bash run_endpoint_client.sh -c <endpoint.yaml> -y <client.yaml> [-w warmup.yaml] [-n name] [-l logdir] [-i image] [-t seconds]" >&2
  exit 2
fi

if [[ ! -f "$CONFIG_YAML" ]]; then
  echo "[error] Config YAML not found: $CONFIG_YAML" >&2
  exit 1
fi
if [[ -n "$CLIENT_YAML" && ! -f "$CLIENT_YAML" ]]; then
  echo "[error] Client YAML not found: $CLIENT_YAML" >&2
  exit 1
fi
if [[ -n "$WARMUP_YAML" && ! -f "$WARMUP_YAML" ]]; then
  echo "[error] Warmup YAML not found: $WARMUP_YAML" >&2
  exit 1
fi
 

# Export overrides for the start script when desired
export ENDPOINT_CONFIG="$CONFIG_YAML"
export LOG_DIR="$LOG_DIR"
export DOCKER_IMAGE="$DOCKER_IMAGE"
export CONTAINER_NAME="$CONTAINER_NAME"

# Start container + endpoint (the script waits for readiness)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "[info] Starting endpoint container with config: $ENDPOINT_CONFIG"
if ! bash "$SCRIPT_DIR/start_endpoint_container.sh"; then
  echo "[error] Endpoint container launch failed." >&2
  echo "[hint] Check logs in $LOG_DIR and docker logs for $CONTAINER_NAME" >&2
  exit 1
fi

# Optional extra wait (wrapper-level)
if [[ "$EXTRA_WAIT" -gt 0 ]]; then
  echo "[info] Extra wait ${EXTRA_WAIT}s before running client..."
  sleep "$EXTRA_WAIT"
fi

 # Run inference-endpoint client from YAML
 if ! command -v inference-endpoint >/dev/null 2>&1; then
   echo "[error] 'inference-endpoint' not found in PATH." >&2
   echo "[hint] Ensure mlperf client tools are installed and PATH is set. See https://github.com/mlcommons/endpoints/tree/main?tab=readme-ov-file#installation" >&2
   docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
   exit 1
 fi
 mkdir -p "$LOG_DIR" || true
 CLIENT_LOG="$LOG_DIR/inference_endpoint_client.log"
 WARMUP_LOG="$LOG_DIR/inference_endpoint_warmup.log"
 # Warmup run if provided
 if [[ -n "$WARMUP_YAML" ]]; then
   echo "[info] Warmup: inference-endpoint benchmark from-config -c $WARMUP_YAML"
   if ! inference-endpoint benchmark from-config -c "$WARMUP_YAML" 2>&1 | tee "$WARMUP_LOG"; then
     echo "[error] Warmup run failed; see $WARMUP_LOG" >&2
     docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
     exit 1
   fi
   echo "[info] Warmup complete. Pausing 10s before performance run..."
   sleep 10
 fi
 echo "[info] Running: inference-endpoint benchmark from-config -c $CLIENT_YAML"
 if ! inference-endpoint benchmark from-config -c "$CLIENT_YAML" 2>&1 | tee "$CLIENT_LOG"; then
   echo "[error] inference-endpoint failed; see $CLIENT_LOG" >&2
   docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
   exit 1
 fi

echo "[done] Endpoint client completed successfully."

# Stop endpoint and container
# The lauched endpoint process will exit on container stop
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
