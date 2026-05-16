# Analyze Expert Distribution Skill

## When to Use This Skill

Use this skill when the user requests:
- Analyzing expert activation patterns from SGLang benchmark logs
- Generating bucketed expert distribution CSVs for ArchBench modeling
- Understanding prefill vs decode expert load balance for MoE models
- Creating expert distribution histograms from completed benchmark runs

## Important Constraints

### Chunked Prefill Requirement
**This analysis only works for benchmarks run WITHOUT chunked prefill.**
- Chunked prefill splits large prefill batches into multiple forward passes, corrupting the single-batch assumption
- The scripts expect one contiguous prefill batch per request group (batch_size × input_len tokens)
- If chunked prefill was enabled (default SGLang behavior for large batches), the prefill analysis will produce incorrect results
- To disable chunked prefill: set `--chunked-prefill-size` and `--max-prefill-tokens` larger than `batch_size × input_len`
- Example for batch 86, input_len=1024: need at least 88,064 tokens → use `--chunked-prefill-size 100000 --max-prefill-tokens 100000`

### Execution Environment
**These scripts run on the HOST machine, NOT inside the Docker container.**
- The log files and .pt files are on the host filesystem (mounted volumes)
- Python dependencies (torch, numpy, pandas) must be available on the host
- The scripts resolve .pt file paths relative to the log file location on the host
- Working directory should be: `/data/farshad/github/vllm-llama-endpoints/closed/Intel/code/endpoints/`

## Prerequisites

### Required Files
1. **SGLang benchmark log file(s)**: `sglang_*_conc-N-*.log` containing:
   - `.pt` file path (from `Write expert distribution to ...` line)
   - Token counts (`#Input tokens: X #Output tokens: Y`)
   - Batch size, input_len, output_len (from filename or log content)
2. **Expert distribution .pt file(s)**: Auto-located from log file content
   - Created by SGLang when `--enable-expert-distribution-metrics` is set
   - Contains `records` list with `topk_ids_of_layer` tensors per forward pass

### Python Environment
**Use the `sglang` conda environment** — the base environment does NOT have the required packages.

```bash
# All commands must be prefixed with:
conda run -n sglang python3 ...

# Verify packages are available:
conda run -n sglang python3 -c "import numpy, pandas, torch; print('OK')"
```

Required packages (already installed in `sglang` conda env):
- `torch` (for loading .pt files)
- `numpy` (for K-means clustering)
- `pandas` (for CSV output)

### Analysis Scripts Location
```
/data/farshad/github/vllm-llama-endpoints/closed/Intel/code/endpoints/farshad_expert_stats_scripts/
├── analyze_prefill_from_log.py   # Latest prefill analysis (May 10 2026)
├── analyze_decode_from_log.py    # Latest decode analysis (May 10 2026)
└── archive_older_versions/       # Previous versions (do NOT use)
```

## Execution Steps

### Step 1: Identify Available Log Files

```bash
cd /data/farshad/github/vllm-llama-endpoints/closed/Intel/code/endpoints
ls -la bench-throughput-logs-*/sglang_*.log
```

Verify each log has a corresponding .pt file:
```bash
grep "Write expert distribution to" bench-throughput-logs-*/sglang_*.log
```

### Step 2: Analyze Prefill Phase

```bash
cd /data/farshad/github/vllm-llama-endpoints/closed/Intel/code/endpoints

conda run -n sglang python3 farshad_expert_stats_scripts/analyze_prefill_from_log.py \
    bench-throughput-logs-<dir>/sglang_*_conc-*.log \
    --num-buckets 5 \
    --output-dir results/
```

**Arguments:**
- `log_files` (positional, one or more): Path(s) to SGLang benchmark log files
- `--num-buckets N`: Number of K-means buckets (default: 5)
- `--output-dir DIR`: Output directory for CSV (default: `<log_dir>/statistics/`)

**What the script does:**
1. Parses each log file to extract: .pt file path, batch_size, input_len, output_len, dtype, quantization
2. Loads the .pt file and separates prefill vs decode records (using forward_mode or token count threshold)
3. For each layer, counts expert activations across all prefill tokens
4. Applies K-means clustering (tries absolute, log-space, and quantile methods, picks lowest CV)
5. Groups experts into N buckets by similar activation load

**Output CSV columns:** `Layer`, `Bucket`, `Expert_IDs`, `Avg_Activations`, `Batch_Size`, `TopK`, `Dtype`, `Quantization`

**Output file:** `results/expert_activation_histogram_prefill_5buckets.csv`

### Step 3: Analyze Decode Phase

```bash
conda run -n sglang python3 farshad_expert_stats_scripts/analyze_decode_from_log.py \
    bench-throughput-logs-<dir>/sglang_*_conc-*.log \
    --output-dir results/
```

**Arguments:**
- `log_files` (positional, one or more): Path(s) to SGLang benchmark log files  
- `--output-dir DIR`: Output directory for CSV (default: `<log_dir>/statistics/`)

**What the script does:**
1. Parses each log file to extract metadata (same as prefill script)
2. Loads .pt file and separates prefill vs decode records
3. Reshapes decode tokens into steps of batch_size
4. For each layer and each decode step, creates histogram: "how many experts activated X times"
5. Averages histograms across all decode steps
6. Rounds to integers for clean output

**Output CSV columns:** `Layer`, `Activations_Per_Expert`, `Num_Experts`, `Batch_Size`, `TopK`, `Dtype`, `Quantization`

**Output file:** `results/expert_activation_histogram.csv`

### Step 4: Verify Results

Check output integrity:
```bash
# Verify all batch sizes are present
python3 -c "
import pandas as pd
df = pd.read_csv('results/expert_activation_histogram.csv')
print('Decode batch sizes:', sorted(df['Batch_Size'].unique()))
print('Rows per batch:', df.groupby('Batch_Size').size().to_dict())

df_p = pd.read_csv('results/expert_activation_histogram_prefill_5buckets.csv')
print('Prefill batch sizes:', sorted(df_p['Batch_Size'].unique()))
print('Rows per batch:', df_p.groupby('Batch_Size').size().to_dict())
"
```

## Example: Full Analysis Pipeline

```bash
# Working directory (HOST, not docker)
cd /data/farshad/github/vllm-llama-endpoints/closed/Intel/code/endpoints

# Analyze all batch sizes from a benchmark run
LOG_DIR="bench-throughput-logs-qwen3-exps-w8a8-20260511"

# Prefill analysis with 5 buckets
conda run -n sglang python3 farshad_expert_stats_scripts/analyze_prefill_from_log.py \
    ${LOG_DIR}/sglang_*.log \
    --num-buckets 5 \
    --output-dir ${LOG_DIR}/statistics/

# Decode analysis
conda run -n sglang python3 farshad_expert_stats_scripts/analyze_decode_from_log.py \
    ${LOG_DIR}/sglang_*.log \
    --output-dir ${LOG_DIR}/statistics/

# Verify
ls -la ${LOG_DIR}/statistics/
```

## Outputs for ArchBench

The two CSV files produced are inputs for architecture benchmarking:

1. **Decode CSV** (`expert_activation_histogram.csv`): Per-decode-step expert load distribution
2. **Prefill CSV** (`expert_activation_histogram_prefill_5buckets.csv`): K-means bucketed prefill expert load

Copy both to the archbench modeling directory for expert parallelism analysis.

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `Could not find .pt file path in log` | Log file doesn't contain expert distribution dump | Re-run benchmark with `--enable-expert-distribution-metrics` |
| `.pt file not found` | Path mismatch between docker and host | Check if .pt path in log is docker-relative; adjust mount point |
| `No prefill records found` | All records classified as decode | Check batch_size × input_len; may need to lower prefill threshold |
| Prefill results look wrong | Chunked prefill was enabled | Re-run benchmark with `--chunked-prefill-size` larger than batch_size × input_len |
| `Batch size mismatch` warning | One request finished early (N-1 pattern) | Normal for async processing; script uses most common batch size |
| Missing python packages | Wrong conda env or running inside docker | Use `conda run -n sglang python3 ...` on the host machine |

## Reference

- **Expert Distribution Guide**: `.claude/expert-distribution-guide.md` - Configuration switches, concepts, source code locations
- **Benchmark Skill**: `.claude/skills/run-sglang-benchmark/SKILL.md` - How to run benchmarks that produce the log files
- **Script Archive**: `farshad_expert_stats_scripts/archive_older_versions/README.md` - Evolution of analysis scripts
