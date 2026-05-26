# CLAUDE.md — SGLang (Farshad's branch: farshad/expert-stats-scripts)

## MoE Expert Activation Distribution — Measured Findings

### Key Correction: Tokens Cluster on Experts

The assumption "no duplicated experts for tokens when BS < 16" is **wrong**. Measured data from SGLang's expert distribution recorder (1,023 decode steps per model) shows significant overlap:

| Model | Total Experts | top_k | BS | Naive (BS×k) | Measured Unique | Overlap |
|-------|--------------|-------|-----|-------------|-----------------|---------|
| Qwen3.5-397B-A17B | 512 | 10 | 2 | 20 | 20 | ~0% |
| Qwen3.5-122B-A10B | 128 | 8 | 5 | 40 | 36 | **10%** |
| Qwen3.5-122B-A10B | 128 | 8 | 10 | 80 | 61 | **24%** |
| Qwen3.5-35B-A3B | 256 | 8 | 20 | 160 | 86 | **46%** |
| Qwen3.5-35B-A3B | 256 | 8 | 22 | 176 | 100 | **43%** |
| Qwen3-30B-A3B | 128 | 8 | 23 | 184 | 93 | **49%** |

At BS=22 (Qwen3.5-35B): 38 out of 100 activated experts handle multiple tokens, carrying **65% of all activations**. Matmul shapes are `[M, hidden] × [hidden, ffn]` with M=2..8, not always M=1.

### Prefill vs Decode — Different Performance Regimes

**PREFILL**: ALL experts active (tokens = BS × input_len >> num_experts). Question is load imbalance, not uniqueness.
- 397B (BS=2, 2048 tokens): 21–112 tokens/expert across all 512 experts
- 35B (BS=22, 22528 tokens): 296–2042 tokens/expert across all 256 experts
- Compute-bound regime. Weight loaded once, reused 100s of times. Bottleneck = slowest expert.

**DECODE at low BS** (1–4): Near-1:1 mapping confirmed. Performance question is batched/grouped GEMM dispatch of many tiny memory-bound matmuls, not weight reuse.

**DECODE at moderate BS** (5–22): Uniform assumption breaks. Clustering enables M>1 batched matmuls for majority of work.

### Methodology

1. Run `sglang.bench_one_batch` with `--expert-distribution-recorder-mode per_token --expert-distribution-recorder-buffer-size 5000`
2. Separate prefill/decode using `forward_mode` flags
3. Decode: histogram activations per expert per step, average across all steps and layers, round with residual correction ensuring `Σ(act_per_expert × num_experts) == BS × top_k`
4. Prefill: K-means bucket experts by activation count, enforce conservation `Σ(avg_act × num_experts) == tokens × top_k`

### Expert Stats Scripts

Location: `scripts/farshad/expert_stats/`
- `collect_qwen35_expert_stats.sh` — Collection (runs inside docker container, NUMA-bound)
- `analyze_decode_from_log.py` — Per-layer decode histogram
- `analyze_prefill_from_log.py` — K-means bucketed prefill
- `analyze_all_layers_averaged.py` — Single aggregate across all layers

### Constraints

- Always use NUMA node 0 for benchmarks (`numactl --cpunodebind=0 --membind=0`)
- Expert activation patterns are independent of TP (routing is pre-TP), so collect with TP=1
- Disable chunked prefill during collection: `--chunked-prefill-size 100000 --max-prefill-tokens 100000`
