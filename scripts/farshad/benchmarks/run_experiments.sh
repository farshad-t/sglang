#!/usr/bin/env bash
set -euo pipefail

# Automate experiments: update configs and run client for each combo.
# Combos:
#  - devices "0,1,2,3" with all concurrencies
#  - devices "0,1" with all concurrencies
#
# Behavior per run:
#  - Update endpoint config `devices`
#  - Update client config `target_concurrency` and `report_dir`
#  - Run `run_endpoint_client.sh -c <endpoint.yaml> -y <client.yaml> -l <report_dir> -i <docker_image>`
#  - Copy the two modified configs into `<report_dir>`
#
# Endpoint selection:
#  - The endpoint launcher is auto-detected from the endpoint config (vLLM vs SGLang).
#  - Override by exporting ENDPOINT_NAME before running if needed.
#
# Optional: set DRY_RUN=1 to skip invoking the client and only print actions.
# Optional: override configs/logs via CLI flags (see -h).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENDPOINT_YAML="${SCRIPT_DIR}/llama3.1-8b-endpoint-config.yaml"
CLIENT_YAML="${SCRIPT_DIR}/llama3.1-8b-client-config.yaml"
CLIENT_WARMUP_YAML="${SCRIPT_DIR}/llama3.1-8b-client-warmup-config.yaml"
RUN_WRAPPER="${SCRIPT_DIR}/run_endpoint_client.sh"
DOCKER_IMAGE_DEFAULT="vllm_xpu:ww06"
DOCKER_IMAGE="${DOCKER_IMAGE:-$DOCKER_IMAGE_DEFAULT}"

usage() {
  echo "Usage: $0 [-c endpoint.yaml] [-y client.yaml] [-w warmup.yaml] [-l logs_dir] [-i docker_image] [-d device_type]" >&2
  echo "  -c endpoint.yaml   Endpoint config YAML (default: $ENDPOINT_YAML)" >&2
  echo "  -y client.yaml     Client config YAML (default: $CLIENT_YAML)" >&2
  echo "  -w warmup.yaml     Warmup client YAML (default: $CLIENT_WARMUP_YAML)" >&2
  echo "  -l logs_dir        Output logs dir (default: \$LOGS_DIR or ${SCRIPT_DIR}/run-logs-MMDD)" >&2
  echo "  -i docker_image    Docker image (default: \$DOCKER_IMAGE or $DOCKER_IMAGE_DEFAULT)" >&2
  echo "  -d device_type     Device type override (default: xpu; e.g., xpu, cpu)" >&2
  echo "  -h                 Show this help" >&2
}

dry_run="${DRY_RUN:-0}"
LOGS_DIR="${LOGS_DIR:-${SCRIPT_DIR}/run-logs-$(date +%m%d)}"

DEVICE_TYPE="xpu"
WARMUP_PROVIDED=0

while getopts ":c:y:w:l:i:d:h" opt; do
  case "$opt" in
    c) ENDPOINT_YAML="$OPTARG" ;;
    y) CLIENT_YAML="$OPTARG" ;;
    w) CLIENT_WARMUP_YAML="$OPTARG"; WARMUP_PROVIDED=1 ;;
    l) LOGS_DIR="$OPTARG" ;;
    i) DOCKER_IMAGE="$OPTARG" ;;
    d) DEVICE_TYPE="$OPTARG" ;;
    h)
      usage
      exit 0
      ;;
    :) echo "[error] Option -$OPTARG requires an argument" >&2; usage; exit 2 ;;
    \?) echo "[error] Unknown option -$OPTARG" >&2; usage; exit 2 ;;
  esac
done

if [[ "$LOGS_DIR" != /* ]]; then
  LOGS_DIR="${SCRIPT_DIR}/${LOGS_DIR}"
fi

if [[ ! -f "$ENDPOINT_YAML" || ! -f "$CLIENT_YAML" || ! -f "$RUN_WRAPPER" ]]; then
  echo "[error] Required files not found."
  echo "        ENDPOINT: $ENDPOINT_YAML"
  echo "        CLIENT  : $CLIENT_YAML"
  echo "        WRAPPER : $RUN_WRAPPER"
  exit 1
fi
if [[ ! -f "$CLIENT_WARMUP_YAML" ]]; then
  warmup_parent="$(dirname "$CLIENT_WARMUP_YAML")"
  mkdir -p "$warmup_parent"
fi
# Define experiment matrix as devices -> concurrencies.
# Format: map devices mask -> space-separated list of concurrencies to run.
declare -A DEVICE_CONCURRENCY_MAP=()
declare -a DEVICE_ORDER=("0,1,2,3,4,5" "0,1,2,3" "0,1" "0")
DEVICE_CONCURRENCY_MAP["0,1,2,3,4,5"]="32 16 8"
DEVICE_CONCURRENCY_MAP["0,1,2,3"]="64 4"
DEVICE_CONCURRENCY_MAP["0,1"]="64 4"
DEVICE_CONCURRENCY_MAP["0"]="64 4"

# Count and display planned device/concurrency pairs
total_runs=0
for d in "${DEVICE_ORDER[@]}"; do
  for c in ${DEVICE_CONCURRENCY_MAP[$d]}; do
    total_runs=$((total_runs + 1))
  done
done

echo "[info] Planning $total_runs experiments"
echo "[info] Planned device/concurrency pairs:"
for d in "${DEVICE_ORDER[@]}"; do
  for c in ${DEVICE_CONCURRENCY_MAP[$d]}; do
    echo "  - devices=${d} concurrency=${c}"
  done
done

# Utility: count devices from mask (comma-separated)
count_devices() {
  local mask="$1"
  if [[ -z "$mask" ]]; then echo 0; return; fi
  awk -F',' 'BEGIN{n=0} {print NF}' <<< "$mask"
}

# Update YAML helpers (in-place)
set_endpoint_devices() {
  local devices_mask="$1"
  # Replace top-level devices: <line starting with devices:>
  sed -i -E "s/^devices:.*/devices: ${devices_mask}/" "$ENDPOINT_YAML"
}

set_endpoint_platform() {
  local device_type="$1"
  if [[ -n "$device_type" ]]; then
    sed -i -E "s/^platform:.*/platform: ${device_type}/" "$ENDPOINT_YAML"
  fi
}

set_sglang_device_type() {
  local device_type="$1"
  if [[ -n "$device_type" ]]; then
    sed -i -E "/^[[:space:]]*- name: device$/ {n; s/^([[:space:]]*)value:.*/\1value: ${device_type}/}" "$ENDPOINT_YAML"
  fi
}

set_client_concurrency() {
  local conc="$1"
  # Replace line containing target_concurrency under load_pattern
  sed -i -E "s/^([[:space:]]*)target_concurrency:.*/\1target_concurrency: ${conc}/" "$CLIENT_YAML"
}

set_client_report_dir() {
  local dp_size="$1" conc="$2"
  local dir="${LOGS_DIR}/llama3.1-8b-dp-${dp_size}-concurrency-${conc}"
  # Replace report_dir line
  sed -i -E "s#^report_dir:.*#report_dir: ${dir}#" "$CLIENT_YAML"
  # Ensure directory exists
  mkdir -p "${dir}"
  echo "$dir"
}

# Create/update warmup YAML (256 samples) and its report_dir
prepare_warmup_yaml() {
  local dp_size="$1" conc="$2"
  local warm_dir="${LOGS_DIR}/llama3.1-8b-dp-${dp_size}-concurrency-${conc}-warmup"
  if [[ "$dry_run" == "1" ]]; then
    echo "[dry-run] Would create warmup config: ${CLIENT_WARMUP_YAML} from ${CLIENT_YAML}"
    echo "[dry-run] Would set n_samples_to_issue: 256 and report_dir: ${warm_dir}"
  else
    mkdir -p "$(dirname "$CLIENT_WARMUP_YAML")"
    cp -f "$CLIENT_YAML" "$CLIENT_WARMUP_YAML"
    sed -i -E "s/^([[:space:]]*)n_samples_to_issue:.*/\1n_samples_to_issue: 256/" "$CLIENT_WARMUP_YAML"
    sed -i -E "s/^([[:space:]]*)target_concurrency:.*/\1target_concurrency: 256/" "$CLIENT_WARMUP_YAML"
    sed -i -E "s#^report_dir:.*#report_dir: ${warm_dir}#" "$CLIENT_WARMUP_YAML"
    mkdir -p "${warm_dir}"
  fi
}

run_one() {
  local devices_mask="$1" conc="$2" idx="$3"
  local dp_size
  dp_size=$(count_devices "$devices_mask")

  echo "[run ${idx}] devices=${devices_mask} (dp=${dp_size}) concurrency=${conc}"
  if [[ "$DEVICE_TYPE" == "xpu" ]]; then
    export ZE_AFFINITY_MASK=$(printf '%s' "$devices_mask" | tr ',' '\n' | awk '{printf "%d,", $0}' | sed 's/,$//')
    echo "[info] Setting ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK}"
  else
      export CPU_VISIBLE_MEMORY_NODES=$(printf '%s' "$devices_mask" | tr ',' '\n' | awk '{printf "%d,", $0}' | sed 's/,$//')
      echo "[info] Setting CPU_VISIBLE_MEMORY_NODES=${CPU_VISIBLE_MEMORY_NODES}"
  fi
  set_endpoint_devices "$devices_mask"
  set_endpoint_platform "$DEVICE_TYPE"
  set_sglang_device_type "$DEVICE_TYPE"
  set_client_concurrency "$conc"
  local report_dir
  report_dir=$(set_client_report_dir "$dp_size" "$conc")
  # Prepare warmup client YAML with 256 samples
  prepare_warmup_yaml "$dp_size" "$conc"
  local abs_report_dir
  if [[ "$report_dir" == /* ]]; then
    abs_report_dir="$report_dir"
  else
    abs_report_dir="${SCRIPT_DIR}/${report_dir}"
  fi

  # Pre-run cleanup (try sudo if available/non-interactive)
  if [[ "$dry_run" == "1" ]]; then
    if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
      echo "[dry-run] Would run cleanup with sudo: sudo bash ${SCRIPT_DIR}/run_clean.sh"
    else
      echo "[dry-run] Would run cleanup: bash ${SCRIPT_DIR}/run_clean.sh (sudo not available or not cached)"
    fi
  else
    echo "[info] Running cleanup script before the test..."
    #if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    #  sudo bash "${SCRIPT_DIR}/run_clean.sh" || echo "[warn] sudo cleanup failed; continuing"
    #else
    #  echo "[warn] sudo requires a password or is unavailable; running cleanup without sudo (may partially fail). To enable non-interactive sudo, run 'sudo -v' before starting."
    #  bash "${SCRIPT_DIR}/run_clean.sh" || echo "[warn] cleanup script failed; continuing"
    #fi
  fi

  if [[ "$dry_run" == "1" ]]; then
    echo "[dry-run] Would run: bash $RUN_WRAPPER -c $ENDPOINT_YAML -y $CLIENT_YAML -w $CLIENT_WARMUP_YAML -l ${abs_report_dir} -i ${DOCKER_IMAGE}"
  else
    bash "$RUN_WRAPPER" -c "$ENDPOINT_YAML" -y "$CLIENT_YAML" -w "$CLIENT_WARMUP_YAML" -l "${abs_report_dir}" -i "${DOCKER_IMAGE}"

    # Archive the exact configs used
    cp -f "$ENDPOINT_YAML" "${abs_report_dir}/llama3.1-8b-endpoint-config.used.yaml"
    cp -f "$CLIENT_YAML" "${abs_report_dir}/llama3.1-8b-client-config.used.yaml"
    cp -f "$CLIENT_WARMUP_YAML" "${abs_report_dir}/llama3.1-8b-client-warmup-config.used.yaml"
  fi

  # Inter-run pause (20s) except after final run
  if [[ "$idx" -lt "$total_runs" ]]; then
    if [[ "$dry_run" == "1" ]]; then
      echo "[dry-run] Would pause for ~20s before next run"
    else
      echo "[info] Pausing 20s before next run..."
      sleep 20
    fi
  fi
}

# Main loop
run_idx=0
for d in "${DEVICE_ORDER[@]}"; do
  for c in ${DEVICE_CONCURRENCY_MAP[$d]}; do
    run_idx=$((run_idx + 1))
    run_one "$d" "$c" "$run_idx"
  done
done

echo "[done] Completed $total_runs experiments."
