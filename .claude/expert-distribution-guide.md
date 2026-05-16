# SGLang Expert Distribution Tracking Guide

## Summary of Current Findings

### What You Observed
- **Log output**: `[Expert Balancedness] forward_pass_id=691 current_pass_balancedness=1.000 last_10_average_balancedness=1.000 last_100_average_balancedness=1.000 last_1000_average_balancedness=1.000 gpu_physical_count_sum=8448`
- **Model**: Qwen3-30B-A3B (MoE with **128 experts**)
- **Model specs** (from config.json):
  - `num_experts: 128`
  - `num_experts_per_tok: 8` (top_k)
  - `num_hidden_layers: 48`
- **Routing**: top_k=8 (each token routes to 8 experts)
- **Configuration**: ep_size=1 (all experts on single rank)

### What Balancedness=1.000 Actually Means

**Formula** (from `expert_distribution.py:1036-1055`):
```python
balancedness = (avg_expert_count_per_GPU) / (max_expert_count_per_GPU)
```

**With ep_size=1:**
- All 64 experts reside on a single GPU/rank
- Only one "GPU" in the metric calculation
- `max_count = avg_count = total_count`
- **Result: balancedness = 1.0 is mathematically guaranteed**

**This metric does NOT tell you:**
- ❌ Which expert IDs (0-127) are being activated
- ❌ Distribution across the 128 experts
- ❌ Whether some experts are "dead" (never used)
- ❌ Load imbalance between individual experts

**This metric DOES tell you:**
- ✅ How evenly work is distributed across EP ranks (meaningless when EP=1)

---

## Switches to Get Per-Expert Distribution Data

### 1. Basic Per-Pass Statistics (Recommended Starting Point)

**Add to your model args:**
```bash
--expert-distribution-recorder-mode stat \
--expert-distribution-recorder-buffer-size 1000 \
--enable-expert-distribution-metrics
```

**Environment variables (alternative/additional):**
```bash
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_MODE=stat
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_BUFFER_SIZE=1000
```

**What you get:**
- Aggregated statistics per layer showing expert activation counts
- Less overhead than detailed per-token tracking
- Good for understanding overall expert utilization patterns

---

### 2. Detailed Per-Token Recording (Maximum Detail)

**Add to your model args:**
```bash
--expert-distribution-recorder-mode per_token \
--expert-distribution-recorder-buffer-size 10000 \
--enable-expert-distribution-metrics
```

**What you get:**
- Complete record of which experts were selected for every token
- Can reconstruct exact routing decisions
- Useful for debugging routing behavior
- ⚠️ Higher memory and performance overhead

---

### 3. Return Expert IDs in API Responses

**Add to your model args:**
```bash
--enable-return-routed-experts \
--enable-return-indexer-topk
```

**What you get:**
- Each API response includes which experts were routed
- Format: base64-encoded int32 array (shape: [num_layers, num_tokens, top_k])
- Decode with: `np.frombuffer(pybase64.b64decode(routed_experts_base64.encode('utf-8')), dtype=np.int32)`

**Example usage:**
```python
import numpy as np
import pybase64

response = requests.post("http://localhost:30001/generate", json={...})
data = response.json()
routed_experts_base64 = data["meta_info"]["routed_experts"]
routed_experts = np.frombuffer(
    pybase64.b64decode(routed_experts_base64.encode("utf-8")), 
    dtype=np.int32
)
# Shape: (num_layers * num_tokens * top_k,) - need to reshape
```

---

### 4. Control Recording via API Endpoints

**Start recording:**
```bash
curl -X POST http://localhost:30001/start_expert_distribution_record
```

**Stop recording:**
```bash
curl -X POST http://localhost:30001/stop_expert_distribution_record
```

**Dump recorded data:**
```bash
curl -X POST http://localhost:30001/dump_expert_distribution_record
```

**What you get:**
- On-demand control of recording (reduces overhead when not needed)
- Dump creates a `.pt` file with full distribution data
- File location: `$SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR` (default: current directory)

---

### 5. Additional Debugging Options

**For approximate statistics (lower overhead in DeepEP):**
```bash
--expert-distribution-recorder-mode stat_approx
```

**For per-pass recording without per-token detail:**
```bash
--expert-distribution-recorder-mode per_pass
```

**Enable heatmap metrics collection:**
```bash
export SGLANG_EPLB_HEATMAP_COLLECTION_INTERVAL=10  # Collect every 10 passes
```

---

## Recommended Configuration for Your Use Case

### To Understand Expert Utilization with ep_size=1

**Minimal approach (low overhead):**
```bash
model_args="--model ${MODEL_PATH} \
    --dtype bfloat16 \
    --device cpu \
    --expert-distribution-recorder-mode stat \
    --expert-distribution-recorder-buffer-size 1000 \
    --enable-expert-distribution-metrics \
    --enable-return-routed-experts \
    ${MODEL_EXTRA_ARGS}"
```

**Detailed approach (complete per-token data):**
```bash
model_args="--model ${MODEL_PATH} \
    --dtype bfloat16 \
    --device cpu \
    --expert-distribution-recorder-mode per_token \
    --expert-distribution-recorder-buffer-size 10000 \
    --enable-expert-distribution-metrics \
    --enable-return-routed-experts \
    --enable-return-indexer-topk \
    ${MODEL_EXTRA_ARGS}"

# Set output directory
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=./expert_distribution_logs
```

---

## How to Analyze the Results

### 1. From Dumped Files

After running with recording enabled and calling the dump endpoint:

```python
import torch
import numpy as np

# Load the dumped file
data = torch.load("expert_distribution_recorder_<timestamp>.pt")

# Extract counts
logical_count = data["logical_count"]  # Shape: (buffer_size, num_layers, num_experts)

# Analyze per-expert distribution
total_activations_per_expert = logical_count.sum(dim=(0, 1))  # Sum across all passes and layers
expert_usage_percentage = (total_activations_per_expert / total_activations_per_expert.sum()) * 100

print("Expert Usage Distribution:")
for expert_id in range(len(total_activations_per_expert)):
    count = total_activations_per_expert[expert_id].item()
    pct = expert_usage_percentage[expert_id].item()
    print(f"Expert {expert_id:2d}: {count:6d} activations ({pct:5.2f}%)")

# Find "dead" experts (never activated)
dead_experts = (total_activations_per_expert == 0).nonzero(as_tuple=True)[0]
print(f"\nDead experts (never activated): {dead_experts.tolist()}")

# Calculate load imbalance
std_dev = total_activations_per_expert.std().item()
mean_activations = total_activations_per_expert.mean().item()
cv = std_dev / mean_activations if mean_activations > 0 else 0
print(f"\nLoad Imbalance (Coefficient of Variation): {cv:.3f}")
print(f"  (0.0 = perfect balance, higher = more imbalanced)")
```

### 2. From API Responses (per-request)

```python
import requests
import numpy as np
import pybase64
from collections import Counter

response = requests.post("http://localhost:30001/generate", json={
    "input_ids": [[1, 2, 3, ...]],
    "sampling_params": {"temperature": 0.0, "top_k": 1, "max_new_tokens": 128}
})

data = response.json()
if "routed_experts" in data["meta_info"]:
    routed_experts_base64 = data["meta_info"]["routed_experts"]
    routed_experts = np.frombuffer(
        pybase64.b64decode(routed_experts_base64.encode("utf-8")), 
        dtype=np.int32
    )
    
    # Count expert frequency for this request
    expert_counts = Counter(routed_experts.flatten())
    print("Expert activations for this request:")
    for expert_id, count in sorted(expert_counts.items()):
        print(f"  Expert {expert_id}: {count} times")
```

---

## Expected Results with Proper Configuration

With the switches above, you should see:

1. **Per-expert activation counts** - showing which of the 128 experts are actually used
2. **Load distribution** - variance/std deviation in expert usage
3. **Dead experts** - experts that are never activated
4. **Per-layer statistics** - how expert usage varies across layers

### Example of Good Output

```
Expert Usage Distribution (total across all layers/passes):
Expert   0:   1234 activations (0.73%)
Expert   1:   1189 activations (0.70%)
Expert   2:    892 activations (0.53%)
...
Expert  85:    156 activations (0.09%)
Expert  86:      0 activations (0.00%)  ← Dead expert!
...
Expert 127:   1421 activations (0.84%)

Dead experts (never activated): [12, 23, 86, 121]

Load Imbalance (CV): 0.234
  (0.0 = perfect balance, higher = more imbalanced)
```

---

## Practical Expert Distribution Analysis Workflow

### Analysis Scripts Location
Scripts in: `closed/Intel/code/endpoints/farshad_expert_stats_scripts/`

### Complete Analysis Pipeline

#### Step 1: Run Benchmark with Expert Recording

```bash
# In benchmark script, enable expert recording
model_args="--model ${MODEL_PATH} \
    --expert-distribution-recorder-mode per_token \
    --expert-distribution-recorder-buffer-size 10000 \
    --enable-expert-distribution-metrics \
    ${MODEL_EXTRA_ARGS}"

# Or set environment variables
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_MODE=per_token
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=./expert_logs
```

**Output**: Benchmark creates `expert_distribution_recorder_*.pt` file in log directory

---

#### Step 2: Analyze Decode Phase Distribution

**Script**: `analyze_decode_from_log.py` (Latest - May 10 2026)

**What it does**:
- Parses SGLang server log file directly (no need for .pt files)
- Extracts decode batch information from log lines
- Creates histogram: "how many experts activated X times"
- Applies K-means clustering to group experts into buckets (default: 5)
- Outputs CSV with bucketed expert distribution

**Usage**:
```bash
cd closed/Intel/code/endpoints
python3 farshad_expert_stats_scripts/analyze_decode_from_log.py \
    bench-throughput-logs-qwen3-exps-w8a8-20260509/sglang_*.log \
    --num-buckets 5 \
    --output-dir results/
```

**Example output**:
```
Layer  0:
  Histogram (averaged across 366 decode steps):
    10 activations/expert:   5 experts
    11 activations/expert:  15 experts
    12 activations/expert:  28 experts
    13 activations/expert:  35 experts
    14 activations/expert:  25 experts
    15 activations/expert:  15 experts
    16 activations/expert:   5 experts
  Total unique experts: 128
  Average activations per step: 184.0 (expected: 184.0) ✓
```

**CSV Output**: `statistics/expert_activation_histogram.csv`

**⭐ Key Output for ArchBench**: This CSV file contains the decode-phase expert activation distribution needed for architecture modeling.

---

#### Step 3: Analyze Prefill Phase with Optimal Bucketing

**Script**: `analyze_prefill_from_log.py` (Latest - May 10 2026)

**What it does**:
- Parses SGLang server log file directly (no need for .pt files)
- Extracts prefill batch information from log lines
- Creates histogram across entire prefill phase
- Applies K-means clustering to group experts with similar loads (default: 5 buckets)
- Minimizes within-bucket variance for optimal load balancing
- Outputs bucketed CSV ready for architecture modeling

**Usage**:
```bash
python3 farshad_expert_stats_scripts/analyze_prefill_from_log.py \
    bench-throughput-logs-qwen3-exps-w8a8-20260509/sglang_*.log \
    --num-buckets 5 \
    --output-dir results/
```

**Example output from Step 3a** (raw histogram):
```
PREFILL HISTOGRAM ANALYSIS
Main prefill batch: 23552 prefill tokens

Layer  0:
  Histogram (main prefill batch):
    1821 activations/expert:   1 expert
    1822 activations/expert:   2 experts
    1823 activations/expert:   5 experts
    ...
    1850 activations/expert:   3 experts
  Total unique experts: 128
  Total activations: 188416 (expected: 188416) ✓
```

**Example output from Step 3b** (bucketed):
```
APPROACH 1: K-Means Clustering (k=5)

Bucket 0: [8 - 10] activations
  Experts: 2,304
  Mean: 9.2 ± 0.6

Bucket 1: [11 - 12] activations
  Experts: 4,608
  Mean: 11.5 ± 0.5

Bucket 2: [13 - 14] activations
  Experts: 5,376
  Mean: 13.4 ± 0.5

Bucket 3: [15 - 16] activations
  Experts: 3,072
  Mean: 15.6 ± 0.5

Bucket 4: [17 - 20] activations
  Experts: 768
  Mean: 18.2 ± 1.1

Total variance: 0.42
(Lower is better - experts within each bucket have similar load)
```

**CSV Outputs**:
- `results/prefill_expert_distribution_buckets.csv` - K-means bucketed (ArchBench ready)
- Columns: `Layer`, `Bucket_ID`, `Bucket_Range`, `Avg_Activations`, `Num_Experts`, `Total_Activations`

**Use case**: The bucketed CSV is optimized for architecture modeling - experts grouped by similar load for parallelism strategies

---

#### Step 4: Analyze Multiple Batches (Multi-file Support)

**Both scripts support multiple log files** - useful for comparing different batch sizes:

```bash
# Analyze all batch sizes at once
python3 farshad_expert_stats_scripts/analyze_decode_from_log.py \
    bench-throughput-logs-*/sglang_*_conc-*.log \
    --num-buckets 5 \
    --output-dir results_all_batches/

# Results organized by batch size:
# results_all_batches/batch_3_decode_buckets.csv
# results_all_batches/batch_5_decode_buckets.csv
# results_all_batches/batch_22_decode_buckets.csv
```

**Script Evolution** (archived in `farshad_expert_stats_scripts/archive_older_versions/`):
- Phase 1 (May 8): `analyze_expert_histogram.py`, `analyze_expert_histogram_prefill.py` - .pt file based
- Phase 2 (May 10 morning): `analyze_prefill_bucketed.py` - directory-based input
- Phase 3 (May 10 evening): `analyze_*_from_log.py` - **CURRENT** log-based approach

---

### Key Insights from Analysis

**What histogram-based analysis reveals**:

1. **Load Balance**: How evenly work is distributed among experts
   - Ideal: Narrow histogram (all experts activated ~equal times)
   - Problem: Wide histogram (some experts overloaded, others idle)

2. **Dead Experts**: Experts never activated
   - Appear as gaps in histogram (missing activation counts)
   - May indicate routing inefficiency or model pruning opportunity

3. **Layer Patterns**: How expert usage evolves through model
   - Early layers: Often more balanced
   - Late layers: May specialize (less balanced)

4. **Prefill vs Decode**: Different distribution patterns
   - Prefill: Processes full context (higher variance)
   - Decode: Single token per step (more stable)

**Example interpretation**:
```
Layer 24 Decode Histogram:
  10 activations: 5 experts   } Low utilization (15 experts)
  11 activations: 10 experts  }
  
  12 activations: 30 experts  } Medium utilization (88 experts)
  13 activations: 35 experts  }
  14 activations: 23 experts  }
  
  15 activations: 18 experts  } High utilization (25 experts)
  16 activations: 7 experts   }

Interpretation:
- 25 experts (19.5%) handle most work → bottleneck candidates
- 15 experts (11.7%) underutilized → could be pruned or load-balanced
- 88 experts (68.8%) have medium load → good balance
```

---

### Critical Outputs for ArchBench Modeling

**These CSV files are required for architecture benchmarking and expert modeling:**

1. **Decode Phase Distribution**: `results/batch_N_decode_buckets.csv`
   - Shows K-means clustered expert activation patterns during decode phase
   - Format: `Layer`, `Bucket_ID`, `Bucket_Range`, `Avg_Activations`, `Num_Experts`, `Batch_Size`, `TopK`
   - Used to model per-decode-step expert load distribution

2. **Prefill Phase Distribution (Bucketed)**: `results/batch_N_prefill_buckets.csv`
   - Shows K-means clustered expert activation patterns during prefill phase
   - Format: `Layer`, `Bucket_ID`, `Bucket_Range`, `Avg_Activations_Per_Expert`, `Num_Experts`, `Total_Activations`, `Total_Prefill_Tokens`, `TopK`
   - Experts grouped into 5 buckets by similar load for optimal parallelism modeling
   - Used to model prefill batch expert load distribution with load balancing

**Workflow**: 
- Run Steps 2 and 3 above to generate both CSV files
- Copy these two files to your archbench modeling directory
- The prefill 5-bucket CSV provides optimized grouping for expert parallelism analysis

---

## Files Created in This Session

1. **CPU-SGLang Agent**: `.github/agents/cpu-sglang.agent.md`
   - Custom agent for CPU-specific SGLang work
   
2. **CPU Instructions**: `.claude/cpu-instructions.md`
   - Comprehensive CPU development guide
   
3. **This Guide**: `.claude/expert-distribution-guide.md`
   - Expert distribution tracking reference
   
4. **Benchmark Skill**: `.claude/skills/run-sglang-benchmark/SKILL.md`
   - Complete SGLang benchmarking workflow

---

## Key Takeaways

1. **Balancedness=1.0 with ep=1 is expected** - it measures GPU-level balance, not per-expert distribution
2. **Use `--expert-distribution-recorder-mode stat` or `per_token`** to get actual per-expert data
3. **With top_k=8**, expect ~8 expert activations per token across all layers
4. **For Qwen3-30B-A3B**, model has **128 experts** total (48 layers, 128 experts per layer, top-8 routing)
5. **API endpoints provide programmatic control** over recording without server restart

### Verification of Expert Count
```bash
# From HuggingFace config.json:
curl -s "https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/raw/main/config.json" | \
  python3 -c "import sys, json; d=json.load(sys.stdin); \
  print(f\"num_experts: {d['num_experts']}\"); \
  print(f\"num_experts_per_tok: {d['num_experts_per_tok']}\"); \
  print(f\"num_hidden_layers: {d['num_hidden_layers']}\")"

# Output:
# num_experts: 128
# num_experts_per_tok: 8
# num_hidden_layers: 48
```

### Math Check
With your log data: `gpu_physical_count_sum=8448`
- Calculation: 8448 / (48 layers × 8 experts per token) = 22 tokens
- This matches your `concurrency=22` setting ✓

---

## Source Code Locations

- **Expert distribution recording**: `python/sglang/srt/eplb/expert_distribution.py`
- **Balancedness calculation**: Line 1036-1055
- **Logging output**: Lines 732-738
- **Server args**: `python/sglang/srt/server_args.py` (lines 608-613, 733, 6345-6347)
- **Routed experts capturer**: `python/sglang/srt/state_capturer/routed_experts.py`
