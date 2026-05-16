# SGLang Offline Benchmark Skill

## When to Use This Skill

Use this skill when the user requests:
- Running SGLang offline throughput benchmarks
- Benchmarking model performance on CPU
- Testing different quantization types (w4a8, w8a8, bf16)
- Measuring throughput/latency for various concurrency levels
- Consolidating or completing incomplete benchmark runs

## Critical: Pre-flight Checklist

**ALWAYS confirm these parameters with the user BEFORE running:**

### 1. Quantization / Dtype
- **bf16**: Use `bench_sglang_offline_bf16.sh`
- **w8a8**: Use `bench_sglang_offline.sh` with `QUANTIZATION=w8a8`
- **w4a8**: Use `bench_sglang_offline.sh` with `QUANTIZATION=w4a8` (default)

### 2. NUMA Nodes & TP Size
- TP size is determined by `SGLANG_CPU_OMP_THREADS_BIND` (pipe-separated groups)
- Each group = one TP rank pinned to one NUMA node
- **CRITICAL**: Must pass `--tp-size ${TP_SIZE}` explicitly in model_args

**System NUMA topology (gnr630032):**
| NUMA Node | Cores |
|-----------|-------|
| 0 | 0-42 |
| 1 | 43-85 |
| 2 | 86-127 |
| 3 | 128-170 |
| 4 | 171-213 |
| 5 | 214-255 |

**Common configurations:**
```bash
# TP1 on node 0:
export SGLANG_CPU_OMP_THREADS_BIND="0-42"

# TP3 on nodes 3,4,5:
export SGLANG_CPU_OMP_THREADS_BIND="128-170|171-213|214-255"

# TP3 on nodes 0,1,2:
export SGLANG_CPU_OMP_THREADS_BIND="0-42|43-85|86-127"
```

### 3. Docker Container
- **Container `farshad-sglang`** (Qwen3-30B, w4a8/w8a8/bf16):
  - **Image**: `sglang-worker-router:farshad`
  - **SGLang**: `0.5.12.dev120` (mingfeima/sglang fork, commit `ece7e95b6`)
  - **Torch**: 2.9.0+cpu
  - **Python**: `/opt/venv/bin/python3`
  - **Local patches**: `bench_offline_throughput.py` (expert distribution dump guard)

- **Container `farshad-sglang-new`** (Qwen3.5/3.6):
  - **Image**: `sglang-cpu:ww20-farshad`
  - **SGLang**: upstream sgl-project/sglang main (commit `6c3541a91`)
  - **Built from**: `docker/xeon.Dockerfile` in sglang repo
  - **Python**: `/opt/.venv/bin/python3`
  - **No local patches** — clean upstream

- **Common volume mounts**:
  - `/data2/llama/` → `/model` (model weights)
  - `/data/farshad/github/vllm-llama-endpoints/closed/Intel/code/endpoints` → `/code` (scripts & logs)
- **Working directory inside container**: `/code`

### 4. Performance vs Profiling Mode
- **Performance mode** (default): `PROFILE=0` — no metrics overhead, fastest run
- **Profiling mode**: `PROFILE=1` — enables torch profiler
- **Expert metrics**: `EXPERT_METRICS=1` — only active when `PROFILE=1`; records per-token expert activation distributions

## Execution Steps

### Step 1: Ensure Container is Running

```bash
docker ps | grep farshad-sglang
# If not running:
docker start farshad-sglang
```

### Step 2: Edit Script for Target Configuration

Key parameters to set in the script before running:

```bash
# In bench_sglang_offline_bf16.sh (or bench_sglang_offline.sh):

# NUMA/TP binding — edit this line:
export SGLANG_CPU_OMP_THREADS_BIND="128-170|171-213|214-255"  # TP3 on nodes 3,4,5

# Workload — edit bench_sweep():
bench_sweep() {
    local -a concurrencies=(23)     # batch sizes to test
    local -a input_lens=(1024)      # input token lengths
    local -a output_lens=(1024)     # output token lengths
}
```

**TP size is auto-derived** from the bind string:
```bash
TP_SIZE=$(echo $SGLANG_CPU_OMP_THREADS_BIND | awk -F'|' '{print NF}')
```

And passed to the model via `--tp-size ${TP_SIZE}`.

### Step 3: Run Benchmark

**Performance mode (no profiling):**
```bash
docker exec -d farshad-sglang bash -c "cd /code && bash bench_sglang_offline_bf16.sh"
```

**With profiling:**
```bash
docker exec -d farshad-sglang bash -c "cd /code && PROFILE=1 bash bench_sglang_offline_bf16.sh"
```

**With expert metrics:**
```bash
docker exec -d farshad-sglang bash -c "cd /code && PROFILE=1 EXPERT_METRICS=1 bash bench_sglang_offline_bf16.sh"
```

**Custom date tag (to consolidate with existing results):**
```bash
docker exec -d farshad-sglang bash -c "cd /code && OUTPUT_DIR_TAG=20260509 bash bench_sglang_offline_bf16.sh"
```

### Step 4: Monitor Progress

```bash
# Check if process is running:
docker exec farshad-sglang bash -c "ps aux | grep bench_offline_throughput"

# Tail the log (from host):
tail -f /data/farshad/github/vllm-llama-endpoints/closed/Intel/code/endpoints/bench-throughput-logs-qwen3-exps-bf16-$(date +%Y%m%d)/*.log

# Verify TP size in log:
grep "tp_size" /data/farshad/github/vllm-llama-endpoints/closed/Intel/code/endpoints/bench-throughput-logs-qwen3-exps-bf16-$(date +%Y%m%d)/*.log | head -1
```

### Step 5: Kill a Running Benchmark (if needed)

```bash
docker exec farshad-sglang bash -c "pkill -f bench_offline_throughput; pkill -f sglang"
```

## Validation

### Success Markers in Log

```
====== Offline Throughput Benchmark Result =======
Successful requests:                     23
Benchmark duration (s):                  XX.XX
Output token throughput (tok/s):         XX.XX
```

### Verify TP Size

In the log's `server_args` line, confirm: `tp_size=3` (or whatever was requested).
Also look for `[TP0]`, `[TP1]`, `[TP2]` prefixes in log lines during startup.

### Failure Markers

```
Traceback
Error
ModuleNotFoundError
```

## Environment Variables Reference

| Variable | Purpose | Default | Example |
|----------|---------|---------|---------|
| `MODEL_NAME` | Model to benchmark | `qwen3` | `qwen3`, `llama3` |
| `OUTPUT_DIR_TAG` | Date tag for output dir | today's date | `20260511` |
| `PROFILE` | Enable profiling | `0` | `0`, `1` |
| `EXPERT_METRICS` | Enable expert distribution recording | `0` | `0`, `1` (requires PROFILE=1) |
| `QUANTIZATION` | Quantization type (non-bf16 script) | `w4a8` | `w4a8`, `w8a8` |
| `SGLANG_CPU_OMP_THREADS_BIND` | Core binding per TP rank | set in script | `"128-170\|171-213\|214-255"` |
| `SGLANG_USE_CPU_W4A8` | Enable w4a8 kernel | `0` in bf16 script | `0`, `1` |

## Script Files

| Script | Location (inside container at /code/) | Purpose |
|--------|--------------------------------------|---------|
| `bench_sglang_offline_bf16.sh` | `/code/bench_sglang_offline_bf16.sh` | bf16 benchmark |
| `bench_sglang_offline.sh` | `/code/bench_sglang_offline.sh` | w4a8/w8a8 benchmark |

## Model Paths (inside container at /model/)

| Model | Path |
|-------|------|
| Qwen3-30B bf16 | `/model/Qwen3-30B-A3B-Instruct-2507` |
| Qwen3-30B w8a8 | `/model/Qwen3-30B-A3B-Instruct-2507-quantized.w8a8` |
| Qwen3-30B w4g128 | `/model/Qwen3-30B-A3B-Instruct-2507-w4g128` |
| Llama-3.1-8B | `/model/Llama-3.1-8B-Instruct` |

## Troubleshooting

### Container not running
```bash
docker start farshad-sglang
# Verify:
docker ps | grep farshad-sglang
```

### Wrong TP size in log
- Check that `--tp-size ${TP_SIZE}` is in `model_args` in the script
- Verify `SGLANG_CPU_OMP_THREADS_BIND` has the right number of `|` groups
- The script auto-derives: `TP_SIZE=$(echo $SGLANG_CPU_OMP_THREADS_BIND | awk -F'|' '{print NF}')`

### ChildProcessError at end of log
- Harmless cleanup error. Check if benchmark results were printed above it.

### Benchmark uses wrong cores / interferes with other workloads
- Check `SGLANG_CPU_OMP_THREADS_BIND` doesn't overlap with other running containers
- Common conflict: nodes 0-2 may be used by other users' containers

### Benchmark runs but TP=1 despite multi-group bind
- **Root cause**: `--tp-size` was missing from model_args
- **Fix**: Ensure the script has `--tp-size ${TP_SIZE} \` in the model_args block
