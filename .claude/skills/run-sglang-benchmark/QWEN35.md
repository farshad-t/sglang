# Qwen3.5 Benchmark Guide

## Overview

Qwen3.5 benchmarking uses a **different container and benchmark tool** than Qwen3.

| | Qwen3 | Qwen3.5 |
|---|---|---|
| Container | `farshad-sglang` | `farshad-sglang-new` |
| Image | `sglang-worker-router:farshad` | `sglang-cpu:ww20-farshad` |
| SGLang | `0.5.12.dev120` (mingfeima fork) | upstream `sgl-project/sglang` main (`6c3541a91`) |
| Built from | Custom image | `docker/xeon.Dockerfile` |
| Benchmark tool | `bench_offline_throughput` | `bench_one_batch` |
| Local patches | Yes (expert dump guard) | None — clean upstream |

## Executive Report Requirements

### Models & Dtypes Needed

| Model | Dtypes | Status |
|-------|--------|--------|
| Qwen/Qwen3.5-35B-A3B | BF16, FP8 | ✅ Have both |
| Qwen/Qwen3.5-9B | BF16 | ❌ Missing |

### Existing Measurements

| Model | TP | BatchSize | TTFT (ms) | TPOT (ms) | Total Throughput |
|-------|-----|-----------|-----------|-----------|------------------|
| qwen3.5-27B-fp8 (29G) | 6 | 13 | 4944 | 87.76 | 140.50 |
| qwen3.5-27B-bf16 (52G) | 6 | 13 | 4646 | 74.47 | 164.66 |
| qwen3.5-35B-fp8 (35G) | 2 | 22 | 3681 | 87.42 | 725.70 |
| qwen3.5-35B-bf16 (67G) | 2 | 20 | 3164 | 99.94 | 582.84 |
| qwen3.5-122B-fp8 (119G) | 2 | 5 | 2715 | 98.02 | 149.13 |
| qwen3.5-122B-bf16 (234G) | 6 | 10 | 2811 | 94.10 | 103.34 |
| qwen3.5-397B-fp8 (379G) | 2 | 2 | 2551 | 93.47 | 62.58 |
| qwen3.5-397B-bf16 (807G) | 6 | 3 | 2209 | 92.41 | 31.75 |
| qwen3.5-2B-bf16 | 1 | 61 | 4998 | 65.67 | 5190.90 |

## Container: farshad-sglang-new

**Launch command (already run 2026-05-12):**
```bash
docker run -it --privileged \
    --name farshad-sglang-new \
    --ipc=host --network=host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v /data2/llama:/model \
    -v /data/farshad/github/vllm-llama-endpoints/closed/Intel/code/endpoints:/code \
    -e http_proxy=$http_proxy \
    -e https_proxy=$https_proxy \
    -e no_proxy=$no_proxy \
    sglang-cpu:ww20-farshad /bin/bash
```

**Restart if stopped:**
```bash
docker start -ai farshad-sglang-new
# or for background:
docker start farshad-sglang-new
docker exec -it farshad-sglang-new /bin/bash
```

## Models Available

| Model | Local Path | HF Name | Size |
|-------|-----------|---------|------|
| Qwen3.6-35B-A3B | `/model/Qwen3.6-35B-A3B` | `Qwen/Qwen3.6-35B-A3B` | 35B (MoE, 3B active) |
| Qwen3.5-0.8B | `/model/Qwen3.5-0.8B` | `Qwen/Qwen3.5-0.8B` | 0.8B |
| Qwen3.5-9B | Not downloaded | `Qwen/Qwen3.5-9B` | 9B |

## Benchmark: bench_one_batch

Coworker's recommended command for Qwen3.5:
```bash
python3 -m sglang.bench_one_batch \
    --batch-size 16 \
    --input 1024 \
    --output 1024 \
    --model Qwen/Qwen3.5-9B \
    --trust-remote-code \
    --device cpu \
    --tp 6 \
    --prompt-file prompt.json \
    --mem-fraction-static 0.8 \
    --max-total-tokens 65536 \
    --enable-torch-compile
```

### bench_one_batch vs bench_offline_throughput

| | `bench_one_batch` | `bench_offline_throughput` |
|---|---|---|
| Purpose | Raw forward pass latency | End-to-end throughput with scheduler |
| Scheduling | None | Full scheduler |
| Torch compile | Supported (`--enable-torch-compile`) | Not typically used |
| Best for | Kernel profiling, torch.compile testing | Realistic serving throughput |

### Key Flags for Qwen3.5

- **`--trust-remote-code`** — Required (model has custom HF code)
- **`--enable-torch-compile`** — JIT compiles model graph; first run slow, subsequent fast
- **`--tp 6`** — Uses all 6 NUMA nodes (all 256 cores)
- **`--prompt-file prompt.json`** — Custom prompt input file (needs to exist)

## Online Serving (alternative)

Coworker's server launch script (`/data/tianmu/vllm_test/start_model_server_qwen35.sh`):
```bash
python -m sglang.launch_server \
    --model /model/Qwen3.6-35B-A3B \
    --served-model-name Qwen/Qwen3.6-35B-A3B \
    --trust-remote-code \
    --disable-overlap-schedule \
    --tool-call-parser qwen3_coder \
    --device cpu \
    --host 0.0.0.0 \
    --decode-log-interval 5 \
    --tp 2 \
    --dp-size 1
```

## NUMA Topology (gnr630032)

| NUMA Node | Cores |
|-----------|-------|
| 0 | 0-42 |
| 1 | 43-85 |
| 2 | 86-127 |
| 3 | 128-170 |
| 4 | 171-213 |
| 5 | 214-255 |

TP=6 uses all nodes. TP=2 auto-selects 2 sub-NUMA nodes.

## TODO / Open Items

- [ ] Download `Qwen3.5-9B` model to `/data2/llama/`
- [ ] Create/find `prompt.json` file for bench_one_batch
- [ ] Run first bench_one_batch with torch.compile on Qwen3.5-9B
- [ ] Compare Qwen3.5 vs Qwen3 performance
