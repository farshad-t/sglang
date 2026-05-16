#!/bin/bash
# Wrapper script to run Qwen3.5 expert statistics collection inside farshad-sglang-new container
# Run this script on the HOST (not inside container)
# Usage: 
#   ./run_qwen35_expert_stats_in_container.sh [NUMA_NODE]
#   Example: ./run_qwen35_expert_stats_in_container.sh 0
#   DRY_RUN=1 ./run_qwen35_expert_stats_in_container.sh 0  # Dry run mode

set -euo pipefail

CONTAINER_NAME="farshad-sglang-new"
SCRIPT_PATH="/code/collect_qwen35_expert_stats.sh"

# Dry run mode
DRY_RUN=${DRY_RUN:-0}

# Get NUMA node from argument or prompt
NUMA_NODE=${1:-}
if [ -z "$NUMA_NODE" ]; then
    echo "====================================================="
    echo "NUMA Node Selection"
    echo "====================================================="
    echo "Available NUMA nodes on this system: 0-5"
    echo ""
    echo "NUMA Node Layout:"
    echo "  Node 0: CPUs 0-42"
    echo "  Node 1: CPUs 43-85"
    echo "  Node 2: CPUs 86-127"
    echo "  Node 3: CPUs 128-170"
    echo "  Node 4: CPUs 171-213"
    echo "  Node 5: CPUs 214-255"
    echo ""
    read -p "Enter NUMA node to use (0-5) [default: 0]: " NUMA_NODE
    NUMA_NODE=${NUMA_NODE:-0}
fi

# Validate NUMA node
if ! [[ "$NUMA_NODE" =~ ^[0-5]$ ]]; then
    echo "ERROR: Invalid NUMA node: $NUMA_NODE"
    echo "Must be between 0 and 5"
    exit 1
fi

echo ""
echo "====================================================="
echo "Qwen3.5 Expert Statistics Collection"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "MODE: DRY RUN"
fi
echo "====================================================="
echo "Container: ${CONTAINER_NAME}"
echo "NUMA Node: ${NUMA_NODE}"
echo ""

# Check if container exists
if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "ERROR: Container '${CONTAINER_NAME}' does not exist!"
    echo ""
    echo "Create it with:"
    echo "docker run -it --privileged \\"
    echo "    --name farshad-sglang-new \\"
    echo "    --ipc=host --network=host \\"
    echo "    -v ~/.cache/huggingface:/root/.cache/huggingface \\"
    echo "    -v /data2/llama:/model \\"
    echo "    -v /data/farshad/github/vllm-llama-endpoints/closed/Intel/code/endpoints:/code \\"
    echo "    -e http_proxy=\$http_proxy \\"
    echo "    -e https_proxy=\$https_proxy \\"
    echo "    -e no_proxy=\$no_proxy \\"
    echo "    sglang-cpu:ww20-farshad /bin/bash"
    exit 1
fi

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Starting container '${CONTAINER_NAME}'..."
    docker start "${CONTAINER_NAME}"
    sleep 2
    echo "Container started."
    echo ""
fi

# Check if script exists in container
if ! docker exec "${CONTAINER_NAME}" test -f "${SCRIPT_PATH}"; then
    echo "ERROR: Script not found in container: ${SCRIPT_PATH}"
    echo "Make sure the /code volume is mounted correctly."
    exit 1
fi

echo "Running expert statistics collection inside container..."
echo ""

# Execute the script inside the container with NUMA node parameter and DRY_RUN env
# Use bash -l (login shell) to load .bashrc which activates the venv
docker exec -it -e DRY_RUN="${DRY_RUN}" "${CONTAINER_NAME}" bash -l "${SCRIPT_PATH}" "${NUMA_NODE}"

echo ""
echo "====================================================="
if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run completed!"
else
    echo "Collection completed!"
fi
echo "====================================================="
echo ""
if [ "$DRY_RUN" -eq 0 ]; then
    echo "To view results:"
    echo "  docker exec -it ${CONTAINER_NAME} ls -lh /code/qwen35-expert-stats-*-node${NUMA_NODE}"
    echo ""
fi
echo "To enter container:"
echo "  docker exec -it ${CONTAINER_NAME} bash"
echo ""
echo "To stop container:"
echo "  docker stop ${CONTAINER_NAME}"
echo "====================================================="
