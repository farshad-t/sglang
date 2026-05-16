# Qwen3.5 Inference on Intel Xeon CPUs with SGLang

## Model Support

This guide covers running two Qwen3.5 models:
- **Qwen3.5-30B-Instruct**: 30B parameter model (MoE architecture)
- **Qwen3.5-2B-Instruct**: 2B parameter dense model

## Prerequisites

### 1. Model Files
Ensure your models are downloaded and available in the following paths (inside container):
- Qwen3.5-30B: `/model/Qwen3.5-30B-Instruct/` (BF16) or quantized variants
- Qwen3.5-2B: `/model/Qwen3.5-2B-Instruct/` (BF16) or quantized variants

### 2. Quantized Model Variants (Optional but Recommended)
For better performance on CPUs, use quantized models:
- **W4A8**: Weight 4-bit, Activation 8-bit (recommended for best latency)
  - `/model/Qwen3.5-30B-Instruct-w4g128/`
  - `/model/Qwen3.5-2B-Instruct-w4g128/`
- **W8A8**: Weight 8-bit, Activation 8-bit (good balance)
  - `/model/Qwen3.5-30B-Instruct-quantized.w8a8/`
  - `/model/Qwen3.5-2B-Instruct-quantized.w8a8/`
- **BF16**: Full precision (baseline)
  - `/model/Qwen3.5-30B-Instruct/`
  - `/model/Qwen3.5-2B-Instruct/`

### 3. SGLang Installation
Follow the Qwen3-MoE-ReadMe.md for SGLang installation inside the container:
```bash
bash start_sglang_container.sh
pushd /sgl-workspace/sglang
git checkout python/pyproject.toml sgl-kernel/pyproject.toml
git remote add upstream https://github.com/sgl-project/sglang.git
git fetch upstream && git checkout upstream/main
cp python/pyproject_cpu.toml python/pyproject.toml && cp sgl-kernel/pyproject_cpu.toml sgl-kernel/pyproject.toml
pushd python && uv pip install -e . && popd
pushd sgl-kernel && bash build.sh 3.12 cpu && uv pip install . --no-build-isolation --force-reinstall && popd
popd
```

## Quick Start

### Offline Benchmarking (Static Batch)

The `bench_qwen3.5_offline.sh` script runs offline throughput benchmarks with configurable parameters.

#### Basic Usage

**Run Qwen3.5-30B with W4A8 quantization (default):**
```bash
MODEL_NAME=qwen3.5-30b QUANTIZATION=w4a8 bash bench_qwen3.5_offline.sh
```

**Run Qwen3.5-2B with W8A8 quantization:**
```bash
MODEL_NAME=qwen3.5-2b QUANTIZATION=w8a8 bash bench_qwen3.5_offline.sh
```

**Run Qwen3.5-30B with BF16 (no quantization):**
```bash
MODEL_NAME=qwen3.5-30b QUANTIZATION=bf16 bash bench_qwen3.5_offline.sh
```

#### Advanced Configuration

**Custom concurrency sweep:**
```bash
MODEL_NAME=qwen3.5-30b \
QUANTIZATION=w4a8 \
CONCURRENCIES="5 10 20 40" \
bash bench_qwen3.5_offline.sh
```

**Custom input/output lengths:**
```bash
MODEL_NAME=qwen3.5-2b \
QUANTIZATION=w8a8 \
INPUT_LENS="512 1024 2048" \
OUTPUT_LENS="512 1024 2048" \
bash bench_qwen3.5_offline.sh
```

**CPU thread binding (for multi-socket systems):**
```bash
# Bind to specific CPU cores (e.g., 0-42 for first socket)
MODEL_NAME=qwen3.5-30b \
SGLANG_CPU_OMP_THREADS_BIND="0-42" \
bash bench_qwen3.5_offline.sh

# Multi-instance with tensor parallelism (split across cores with |)
MODEL_NAME=qwen3.5-30b \
SGLANG_CPU_OMP_THREADS_BIND="0-42|43-85" \
bash bench_qwen3.5_offline.sh
```

**Enable profiling:**
```bash
MODEL_NAME=qwen3.5-30b \
PROFILE=1 \
bash bench_qwen3.5_offline.sh
```

**Custom output directory tag:**
```bash
MODEL_NAME=qwen3.5-30b \
OUTPUT_DIR_TAG="experiment1" \
bash bench_qwen3.5_offline.sh
# Results will be in: bench-throughput-logs-qwen3.5-30b-w4a8-experiment1/
```

#### Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `qwen3.5-30b` | Model to benchmark: `qwen3.5-30b` or `qwen3.5-2b` |
| `QUANTIZATION` | `w4a8` | Quantization type: `w4a8`, `w8a8`, or `bf16` |
| `PROFILE` | `0` | Enable profiling: `1` to enable, `0` to disable |
| `OUTPUT_DIR_TAG` | `YYYYMMDD` | Tag for output directory (default: current date) |
| `GROUP_SIZE` | `128` | Weight quantization group size (for W4A8) |
| `NUM_PROMPTS` | `64` | Number of prompts to generate |
| `MAX_TOTAL_TOKENS` | Model-dependent | Maximum total tokens in memory |
| `MAX_PREFILL_TOKENS` | Model-dependent | Maximum tokens in prefill phase |
| `SGLANG_CPU_OMP_THREADS_BIND` | `"0-42"` | CPU core binding (use `\|` for multi-instance) |
| `CONCURRENCIES` | Model-dependent | Space-separated list of batch sizes to test |
| `INPUT_LENS` | `1024` | Space-separated list of input lengths |
| `OUTPUT_LENS` | `8192` | Space-separated list of output lengths |

### Default Configurations

#### Qwen3.5-30B (Large MoE Model)
- **Default Concurrencies**: `3 5 11 22 48 86`
- **Max Total Tokens**: `65536`
- **Max Prefill Tokens**: `67584`

#### Qwen3.5-2B (Small Dense Model)
- **Default Concurrencies**: `8 16 32 64 128`
- **Max Total Tokens**: `131072`
- **Max Prefill Tokens**: `65536`

## Output and Logs

### Directory Structure
Results are saved to: `bench-throughput-logs-{MODEL_NAME}-{QUANTIZATION}-{TAG}/`

Example for Qwen3.5-30B with W4A8 on May 13, 2026:
```
bench-throughput-logs-qwen3.5-30b-w4a8-20260513/
├── expert_logs/                              # MoE expert activation logs
│   └── expert_distribution_recorder_*.pt
├── sglang_torch_profiler_qwen3.5-30b_20260513/  # Profiler traces (if PROFILE=1)
├── statistics/                               # Expert activation analysis
│   └── expert_activation_*.csv
└── sglang_qwen3.5-30b_bench_offline_*.log   # Detailed benchmark logs
```

### Log Files
Each benchmark run creates a log file with the naming pattern:
```
sglang_{MODEL}_bench_offline_{NUM_PROMPTS}prompts_input_len-{INPUT_LEN}_output_len-{OUTPUT_LEN}_conc-{CONCURRENCY}-{TAG}.log
```

### Extracting Results
Key metrics are logged in each file:
- **Throughput**: Tokens/second
- **Latency**: Mean, P50, P90, P95, P99
- **TTFT**: Time to first token
- **TPOT**: Time per output token

## Example Workflows

### 1. Quick Performance Test (30B Model)
```bash
# Test 30B model with default settings
MODEL_NAME=qwen3.5-30b bash bench_qwen3.5_offline.sh
```

### 2. Comprehensive Sweep (2B Model)
```bash
# Test 2B model with multiple configurations
MODEL_NAME=qwen3.5-2b \
QUANTIZATION=w8a8 \
CONCURRENCIES="4 8 16 32 64 128" \
INPUT_LENS="512 1024 2048 4096" \
OUTPUT_LENS="512 1024 2048 4096" \
bash bench_qwen3.5_offline.sh
```

### 3. Multi-Socket NUMA Optimization
```bash
# Run with 2 NUMA nodes, each using half the cores
MODEL_NAME=qwen3.5-30b \
QUANTIZATION=w4a8 \
SGLANG_CPU_OMP_THREADS_BIND="0-42|43-85" \
bash bench_qwen3.5_offline.sh
```

### 4. Profiling for Performance Analysis
```bash
# Run with profiling enabled
MODEL_NAME=qwen3.5-30b \
PROFILE=1 \
CONCURRENCIES="22" \
INPUT_LENS="1024" \
OUTPUT_LENS="8192" \
bash bench_qwen3.5_offline.sh
```

## Troubleshooting

### Model Not Found
If you see errors like "Model path does not exist", ensure:
1. The model is downloaded to the correct path
2. The container has the model directory mounted (check `-v ${MODEL_DIR}:/model`)
3. The model path in the script matches your actual model location

### Out of Memory
If you encounter OOM errors:
1. Reduce `MAX_TOTAL_TOKENS` or `MAX_PREFILL_TOKENS`
2. Reduce concurrency values
3. Use more aggressive quantization (W4A8 instead of W8A8)

### Poor Performance
To improve performance:
1. Use W4A8 quantization for best latency
2. Optimize `SGLANG_CPU_OMP_THREADS_BIND` based on your CPU topology
3. Use `numactl --hardware` to identify NUMA nodes
4. For multi-socket systems, use `|` to split workload across sockets
5. Adjust `MAX_PREFILL_TOKENS` based on your workload

### Expert Distribution Analysis (MoE Models)
For Qwen3.5-30B (MoE), expert activation logs are saved in `expert_logs/`.
Use the analysis scripts in `farshad_expert_stats_scripts/` to analyze expert utilization:
```bash
python farshad_expert_stats_scripts/analyze_prefill_from_log.py \
    bench-throughput-logs-qwen3.5-30b-w4a8-20260513/expert_logs/expert_distribution_recorder_*.pt
```

## Performance Tips

1. **Quantization**: W4A8 typically provides 2-3x speedup over BF16 with minimal accuracy loss
2. **Concurrency**: Higher concurrency improves throughput but may increase latency
3. **NUMA Binding**: Proper CPU binding is critical for multi-socket systems
4. **Memory**: Allocate sufficient memory with `--mem-fraction-static` (default 0.9)

## Next Steps

- Compare quantization schemes (W4A8 vs W8A8 vs BF16)
- Tune concurrency for your latency/throughput requirements
- Profile to identify bottlenecks
- Test with real workloads using the endpoint client (see README.llama3.1-8B.md)
