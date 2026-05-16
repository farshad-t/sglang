# Qwen3-30B-A3B Benchmark Guide

## Overview

Qwen3-30B-A3B benchmarking uses the `farshad-sglang` container with the mingfeima fork of SGLang and `bench_offline_throughput`.

| Property | Value |
|---|---|
| Container | `farshad-sglang` |
| Image | `sglang-worker-router:farshad` |
| SGLang | `0.5.12.dev120` (mingfeima/sglang fork) |
| Benchmark tool | `bench_offline_throughput` |
| Local patches | `bench_offline_throughput.py` (expert distribution dump guard) |

## Dashboard Results (vllm_0.19.1, 2026-05-06)

### SGLang

| Model | Weight | Precision | TP/PP/DP | Concurrency | TTFT (s) | TPOT (ms) | Output Throughput |
|-------|--------|-----------|----------|-------------|----------|-----------|-------------------|
| Qwen/Qwen3-30B-A3B | BF16 | BF16 | TP3DP1 | 23 | 1.899 | 98.89 | 228.50 |

### vLLM

| Model | Weight | Precision | TP/PP/DP | Concurrency | TTFT (s) | TPOT (ms) | Output Throughput |
|-------|--------|-----------|----------|-------------|----------|-----------|-------------------|
| Qwen/Qwen3-30B-A3B | BF16 | BF16 | TP2DP2 | 50 | 4.004 | 98.58 | 478.07 |

## My Run (farshad, 2026-05-11)

| Config | Value |
|---|---|
| Container | `farshad-sglang` (mingfeima fork) |
| Script | `bench_sglang_offline_bf16.sh` |
| TP | 3 (NUMA nodes 3,4,5: cores 128-255) |
| Batch size | 23 |
| Input/Output | 1024/1024 |

| Metric | Value |
|---|---|
| Successful requests | 23 |
| Benchmark duration (s) | 112.62 |
| Output throughput (tok/s) | 209.13 |
| Total throughput (tok/s) | 418.27 |
| TTFT | ~5-7s |
| TPOT | ~110 ms/token/request |

**Note**: My results differ from the dashboard — different SGLang version (mingfeima fork vs dashboard's version), different NUMA binding, and different measurement methodology.

## Container: farshad-sglang

**Restart if stopped:**
```bash
docker start farshad-sglang
docker exec -it farshad-sglang /bin/bash
```

## Models Available

| Model | Path | Dtype |
|-------|------|-------|
| Qwen3-30B-A3B bf16 | `/model/Qwen3-30B-A3B-Instruct-2507` | BF16 |
| Qwen3-30B-A3B w8a8 | `/model/Qwen3-30B-A3B-Instruct-2507-quantized.w8a8` | W8A8 |
| Qwen3-30B-A3B w4g128 | `/model/Qwen3-30B-A3B-Instruct-2507-w4g128` | W4G128 |

## Running Benchmarks

### BF16
```bash
docker exec -d farshad-sglang bash -c "cd /code && bash bench_sglang_offline_bf16.sh"
```

### W8A8
```bash
docker exec -d farshad-sglang bash -c "cd /code && QUANTIZATION=w8a8 bash bench_sglang_offline.sh"
```

### With Profiling
```bash
docker exec -d farshad-sglang bash -c "cd /code && PROFILE=1 bash bench_sglang_offline_bf16.sh"
```

### With Expert Metrics
```bash
docker exec -d farshad-sglang bash -c "cd /code && PROFILE=1 EXPERT_METRICS=1 bash bench_sglang_offline_bf16.sh"
```

## NUMA Binding

```bash
# TP3 on nodes 3,4,5 (what I used):
export SGLANG_CPU_OMP_THREADS_BIND="128-170|171-213|214-255"

# TP3 on nodes 0,1,2:
export SGLANG_CPU_OMP_THREADS_BIND="0-42|43-85|86-127"
```

## Log Location

```bash
# On host:
ls /data/farshad/github/vllm-llama-endpoints/closed/Intel/code/endpoints/bench-throughput-logs-qwen3-exps-bf16-*/
```
