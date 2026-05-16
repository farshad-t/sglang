#!/bin/bash

# Download Qwen3.5 models only (no benchmarking)
# Models will be downloaded to HF_HOME=/model/qwen-models

set -e

export HF_HOME=/model/qwen-models
export HUGGINGFACE_HUB_CACHE=/model/qwen-models/hub

echo "=================================================="
echo "Qwen3.5 Model Download Script"
echo "=================================================="
echo "HF_HOME: ${HF_HOME}"
echo "Cache: ${HUGGINGFACE_HUB_CACHE}"
echo ""

# List of models to download (in order of size)
declare -a MODELS=(
    # Tier 1 (~182GB) - small models
    "Qwen/Qwen3.5-0.8B"
    "Qwen/Qwen3.5-2B"
    "Qwen/Qwen3.5-9B"
    
    # Tier 2 (~204GB) - 27B and 35B variants
    "Qwen/Qwen3.5-27B-fp8"
    "Qwen/Qwen3.5-27B"
    "Qwen/Qwen3.5-35B-A3B-FP8"
    "Qwen/Qwen3.5-35B-A3B"
    
    # Tier 3 (~353GB) - 122B variants (A10B)
    "Qwen/Qwen3.5-122B-A10B-FP8"
    "Qwen/Qwen3.5-122B-A10B"
    
    # Note: 397B variants exist but won't fit in available disk space
    # "Qwen/Qwen3.5-397B-fp8"
    # "Qwen/Qwen3.5-397B-A17B-fp8"
    # "Qwen/Qwen3.5-397B"
)

total=${#MODELS[@]}
current=0

for model_id in "${MODELS[@]}"; do
    current=$((current + 1))
    echo ""
    echo "=================================================="
    echo "[$current/$total] Processing: ${model_id}"
    echo "=================================================="
    
    # Check if model is already fully downloaded
    model_cache_name=$(echo "${model_id}" | sed 's|/|--|g')
    model_dir="/model/qwen-models/models--${model_cache_name}"
    
    if [ -d "${model_dir}/snapshots" ]; then
        # Check if there's a complete snapshot (has config.json)
        if find "${model_dir}/snapshots" -name "config.json" -type f 2>/dev/null | grep -q .; then
            echo "⊙ Model already downloaded, skipping"
            continue
        fi
    fi
    
    echo "Downloading ${model_id}..."
    /opt/.venv/bin/python3 -c "
from huggingface_hub import snapshot_download
import os

model_id = '${model_id}'
cache_dir = os.environ.get('HF_HOME', '/model/qwen-models')

try:
    path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        resume_download=True,
        max_workers=4
    )
    print(f'✓ Downloaded to: {path}')
except Exception as e:
    print(f'✗ Failed: {e}')
    exit(1)
"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully downloaded: ${model_id}"
    else
        echo "✗ Failed to download: ${model_id}"
        echo "Stopping download process."
        exit 1
    fi
done

echo ""
echo "=================================================="
echo "Download Complete!"
echo "=================================================="
echo "Total models downloaded: ${current}"
echo "Cache location: ${HF_HOME}"
echo ""
