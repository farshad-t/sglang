#!/usr/bin/env bash

set -euo pipefail

# Starts the vLLM endpoint inside a Docker container and waits for readiness.
# On failure (docker or launcher), prints diagnostics and exits non-zero.

DATA_DIR=${DATA_DIR:-/data/dataset}
MODEL_DIR=${MODEL_DIR:-/data/model}
LOG_DIR=${LOG_DIR:-$(pwd)/logs}
DOCKER_IMAGE=${DOCKER_IMAGE:-vllm_xpu:ww04}
CONTAINER_NAME=${CONTAINER_NAME:-mlperf-endpoints-6.0}
ENDPOINT_CONFIG=${ENDPOINT_CONFIG:-llama3.1-8b-endpoint-config.yaml}
ENDPOINT_NAME=${ENDPOINT_NAME:-}
LOG_FILE_BASENAME=${LOG_FILE_BASENAME:-vllm_serve.log}

mkdir -p "${LOG_DIR}"

if [[ ! -f "${ENDPOINT_CONFIG}" ]]; then
        echo "[error] Config YAML not found: ${ENDPOINT_CONFIG}" >&2
        exit 1
fi

detect_endpoint_name() {
        local name
        name=$(awk '
                /^endpoint:/ {in_ep=1; next}
                in_ep && $1 == "-" && $2 == "name:" {print $3; exit}
        ' "${ENDPOINT_CONFIG}" | tr -d '"')
        if [[ -n "$name" ]]; then
                echo "$name"
                return 0
        fi
        return 1
}

detect_platform() {
        local platform
        platform=$(awk -F': *' '/^platform:/ {print $2; exit}' "${ENDPOINT_CONFIG}" | tr -d '"')
        if [[ -n "$platform" ]]; then
                echo "$platform"
                return 0
        fi
        return 1
}

resolve_endpoint_name() {
        if [[ -n "${ENDPOINT_NAME}" ]]; then
                echo "${ENDPOINT_NAME}"
                return 0
        fi
        if detect_endpoint_name; then
                return 0
        fi
        echo "vllm"
}

cleanup_container() {
        local name="$1"
        if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
                docker rm -f "$name" >/dev/null 2>&1 || true
        fi
}

echo "[start] Image=${DOCKER_IMAGE} Container=${CONTAINER_NAME} Config=${ENDPOINT_CONFIG}"

# Clean up any stale container with same name
cleanup_container "${CONTAINER_NAME}"

# Start a base container detached that stays alive
# Raise file descriptor limit for high-concurrency router/endpoint connections.
platform=""
detect_platform >/dev/null 2>&1 && platform="$(detect_platform)"

device_mounts=()
affinity_env=()
if [[ "$platform" != "cpu" ]]; then
        device_mounts+=(--device /dev/dri:/dev/dri -v /dev/dri/by-path:/dev/dri/by-path)
        affinity_env=(-e ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-})
fi

if ! docker run --privileged -d -it -u root \
                                --ulimit nofile=262144:262144 \
                                --init \
                                --ipc=host --net=host --cap-add=ALL \
                                "${device_mounts[@]}" \
                                -e http_proxy=${http_proxy:-} \
                                -e https_proxy=${https_proxy:-} \
                                -e no_proxy="${no_proxy:-},localhost,127.0.0.1" \
                                "${affinity_env[@]}" \
                                -v "${DATA_DIR}":/dataset \
                                -v "${MODEL_DIR}":/model \
                                -v "${LOG_DIR}":/logs \
                                -v "$(pwd)":/workspace \
                                --workdir /workspace \
                                --entrypoint /bin/bash \
                                --name "${CONTAINER_NAME}" \
                                "${DOCKER_IMAGE}" -lc 'tail -f /dev/null' >/dev/null; then
        echo "[error] Failed to start base container ${CONTAINER_NAME}" >&2
        exit 1
fi

# Inside the running container, launch the endpoint and wait for readiness
set +e
echo "[info] Launching endpoint inside container..."
endpoint_name=$(resolve_endpoint_name)
if [[ "$endpoint_name" == "sglang" ]]; then
        docker exec -u root -w /workspace "${CONTAINER_NAME}" \
                /bin/bash -lc "python3 launch_sglang_endpoint.py -c /workspace/${ENDPOINT_CONFIG} --endpoint-name ${endpoint_name} --detach-router"
else
        docker exec -u root -w /workspace "${CONTAINER_NAME}" \
                /bin/bash -lc "python3 launch_vllm_endpoint.py -c /workspace/${ENDPOINT_CONFIG} --endpoint-name ${endpoint_name} --wait-ready --log-file /logs/${LOG_FILE_BASENAME}"
fi
status=$?
set -e

if [[ $status -ne 0 ]]; then
        echo "[error] Endpoint launcher failed with exit code ${status}" >&2
        echo "[hint] Recent container logs (docker logs):" >&2
        docker logs --tail 200 "${CONTAINER_NAME}" >&2 || true
        echo "[hint] Recent endpoint log (/logs/${LOG_FILE_BASENAME}) if present:" >&2
        docker exec "${CONTAINER_NAME}" bash -lc "test -f /logs/${LOG_FILE_BASENAME} && tail -n 200 /logs/${LOG_FILE_BASENAME}" >&2 || true
        cleanup_container "${CONTAINER_NAME}"
        exit ${status}
fi

echo "[ready] Endpoint is healthy and container is running: ${CONTAINER_NAME}"
echo "[info] Logs: ${LOG_DIR}/${LOG_FILE_BASENAME}"
exit 0