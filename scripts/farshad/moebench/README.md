# Standalone CPU MoE benchmark

Measures one Qwen3.5 MoE layer on real silicon using **the same expert
distribution the archbench projection consumes**, so a projected MoE time can be
compared against a measured one instead of against an end-to-end model time that
also contains attention, KV, collectives and framework overhead.

bf16 only. Design + test on GNR; the same scripts run on DMR unchanged.

## What is being compared

The projection turns a measured expert histogram into groups:

```
groups = [expert_dist(num_experts, avg_tokens_per_expert) for each histogram bucket]
```

and charges one batched GEMM per group — `[nexp, tokens, K] x [nexp, K, 2N]`
(fused gate+up) plus `[nexp, tokens, N] x [nexp, N, K]` (down).

This harness reproduces that distribution two ways from one histogram:

| `--mode`  | what runs                                                        |
|-----------|------------------------------------------------------------------|
| `fused`   | sglang `fused_experts_cpu` — what production actually executes    |
| `batched` | the projection's own decomposition, as `torch.bmm` per bucket     |
| `both`    | both, plus the ratio                                              |

`fused` vs `batched` separates two error sources that a single number conflates:

* **kernel quality** — does sglang's sorted-token AMX path beat a plain batched
  GEMM at these shapes?
* **model error** — does the bucket-averaged decomposition predict the real kernel?

## Where the stats come from

Fetched from the archbench repo, branch `farshad/llama4-qwen3-qwen35`, resolved
through the same `tools/cpu/config/qwen35/expert_stats_mapping.json` the projection
uses — so filenames are never hardcoded here and follow the mapping if it changes.

The resolved **commit sha is printed and written into every output row**. A measured
number that cannot be tied to a stats revision cannot be compared against a
projection run later.

Cached under `~/.cache/moebench/archbench/` keyed by sha (override with
`$MOEBENCH_CACHE`). First fetch is a blobless shallow fetch of the one branch,
~1 s; re-runs on a cached sha do no network I/O.

Escapes for a box with no access to the repo:

```
--ab-prefetch                cache every CSV the mapping references, then exit
--ab-commit <sha>            pin an exact stats revision
--ab-offline                 never touch the network (uses the last resolved sha)
--ab-local-repo DIR          read blobs from an existing clone/worktree
--stats-dir DIR              plain directory of CSVs, no git at all
--decode-csv / --prefill-csv point at one specific file
```

**The `qwen35-bkc` container has no network.** A blobless clone fetches blobs
lazily, so a cache miss inside it means a ~135 s connect timeout and then a
confusing promisor error. So warm the cache on the host first, once:

```bash
python3 bench_moe_cpu.py --ab-prefetch      # 6 CSVs for model_key 2048-256-512
```

then `--ab-offline` inside the container is a hard guarantee: a cache miss fails
immediately with the prefetch command in the message, and never reaches the network.
Verified with `docker run --network none`.

Note the qwen35 checkout at `/data/farshad/github/abench-qwen35-dmr` on the GNR box
is a git **worktree whose `.git` points into another user's home**, so it cannot be
updated or attributed to a revision. Its CSVs match the repo byte-for-byte modulo
CRLF, but prefer the repo.

### Verified clone

A clean shallow clone of the branch lives at
`/data/farshad/github/archbench-claude-moebench` (329 MB, `--depth 1
--single-branch`), at the branch tip `5ae86154`. Use it as the offline backend:

```bash
docker run --rm --network none \
  -v /data/farshad/github/sglang/scripts/farshad/moebench:/wk \
  -v /data/farshad/github/archbench-claude-moebench:/ab:ro \
  -w /wk qwen35-bkc:latest \
  python bench_moe_cpu.py --phase decode --batch 1,64 --layer 0 --mode both \
      --ab-local-repo /ab --threads 32
```

What was checked against that clone:

* its checked-out CSVs are **byte-identical** to what the network fetcher pulled;
* **1800 cells** (920 decode + 880 prefill, every shipped batch × 40 layers) produce
  groups **identical to archbench's own** `collect_expert_distribution_from_csv(...,
  expert_dist_mode='stats')`, loaded straight out of the clone — 0 differences. This
  is the guarantee that "same design and same values" actually holds;
* **0 validation violations** across all 1800 cells. Decode mass is exact everywhere
  (slack 0, 8–242 experts active, 1–43 groups). Prefill gap runs +0.0% (bs100 L21) to
  **+19.7% (bs320 L3)**, which is what pins archbench's 20% band, with slack 0–7
  slots — bounded by `topk-1` as designed. The bs320 decode defect recorded earlier
  (1763 experts in a 256-expert layer) is **not present** on this branch;
* the branch's own `tests/test_expert_dist_validation.py` and
  `tests/test_expert_stats_audit.py` pass — 18 tests (run with the `abench` conda env,
  which is the one that has pytest).

## Decode vs prefill

The two phases carry deliberately different stat structures, and the harness keeps
them distinct rather than normalising them:

| | decode | prefill |
|---|---|---|
| schema | `Layer,Activations_Per_Expert,Num_Experts,Batch_Size,TopK` | `Layer,Bucket,Expert_IDs,Avg_Activations,Batch_Size,TopK,Num_Experts` |
| shape | full per-layer histogram | 5 load buckets, one averaged count each |
| expert count | `Num_Experts` column | `len(Expert_IDs)` (`Num_Experts` is empty in the real-prompt files) |
| mass | exact: `Σ act·nexp == batch·topk` | overshoots `batch·seq·topk`, checked against a ±20% band |

The overshoot is real, not a bug: the real-prompt stats were collected on prompts
longer than the modelled 1024. Measured +1.4% at bs1 L0, +8.1% at bs64 L0.

## Building the routing table

`fused_experts_cpu` takes a `[num_tokens, topk]` table, not a histogram, so the
histogram is expanded to per-expert token counts and laid into the table
**column-major**: `ids[r, c] = flat[c * num_tokens + r]`, where `flat` is each
expert id repeated by its count, experts in descending-count order.

That cannot put an expert twice in one row: each expert is a contiguous run in
`flat`, and column-major placement collides only at index gaps that are multiples
of `num_tokens`, while a run is shorter than that (`max(count) <= num_tokens` is
enforced). So every row holds `topk` distinct experts — a table a real router could
have produced. Both properties are re-checked at runtime per cell.

A `[num_tokens, topk]` table holds exactly `num_tokens · topk` assignments, so the
mass must be divisible by `topk`. Decode always is. Prefill usually is not (it is a
5-bucket average), so the token count is rounded **up** and the ≤`topk-1` leftover
slots go to the busiest experts — 2 slots out of 566,990 at bs64 prefill.

## Trusting the number

`--check` runs the kernel against a plain fp32 per-expert reference on the *same*
routing table before timing. This is what rules out a kernel — or a harness bug —
that quietly does less work than the histogram says, which would otherwise show up
as a flattering GFLOP/s rather than as a failure. Measured rel_err 3.5e-3 at
decode bs64 L0, consistent with bf16 accumulated over K=2048.

Topk weights are randomised and row-normalised rather than uniform `1/topk`, so a
mis-attributed routed slot cannot cancel out.

## Portability of the kernel call

`fused_experts_cpu`'s signature drifts between builds — the `qwen35-bkc` container
carries an extra `a1_scale` that upstream `main` does not. Arguments are therefore
bound **by name** from the registered schema, and an argument the harness does not
recognise fails loudly with the schema printed. No positional assumptions.

## Running

Locally (dry-run needs no torch and no sgl_kernel):

```bash
python3 bench_moe_cpu.py --phase prefill --batch 64 --layer 0 --dry-run
```

In the container that has a working `sgl_kernel` on the GNR box:

```bash
docker run --rm \
  -v /data/farshad/github/sglang/scripts/farshad/moebench:/wk \
  -v ~/.cache/moebench:/cache -e MOEBENCH_CACHE=/cache/archbench \
  -w /wk qwen35-bkc:latest \
  python bench_moe_cpu.py --phase decode --batch 1,64,128,320 --layer all \
      --mode both --check --threads 96 --out /wk/decode.csv
```

Warm the cache on the host once (`--dry-run` is enough) if the container has no
network, then pass `--ab-offline` inside it.

## Dense models, and TP

**`--dense-ffn`** benchmarks a dense SwiGLU FFN instead of an MoE layer, as the
`num_experts=1` case — which is how archbench itself models it
(`common/swiglu_block.py`: *"num_experts=1 is the dense FFN case"*), using the same
`BatchedSwiGLU` primitive the expert groups are built from. It reads no stats, because
a dense model has no experts to collect. Needed for the dense lanes of the daily run:

```bash
python3 bench_moe_cpu.py --dense-ffn --hidden-size 4096 --intermediate-size 12288 \
    --phase prefill --batch 96
```

**`--tp N`** shards `moe_intermediate_size` by `N`, i.e. what ONE rank executes (all
`E` experts kept, `N -> N/tp`). The stats are still looked up under the *unsharded*
model key, since routing is a property of the model, not of the split.

This matters because **archbench's `Qwen3_5MoeExperts` does not shard the MoE at all** —
it passes `config.moe_intermediate_size` and `config.num_experts` straight through with
no `model_parallel_size` division, unlike attention (`n_local_kv_heads =
ceil(n_kv_heads/MP)`) and vocab. So `--tp 1` is what the projection charges per rank and
`--tp 4` is what a real sglang TP4 rank runs. Measured at the rt prefill cell (bs1 L0,
32 threads): 17.65 ms unsharded vs **3.81 ms** at `--tp 4`, a 4.6× gap.

## Running the daily-run lanes

`run_dmr_lanes.sh` encodes the bf16 lanes of the qwen35 DMR-X4 daily run
(`qwen35_dmrx4_lanes_20260903_1119_collmodes_shmprofile.csv`) so a whole sweep is one
command. Lanes: `35b_rt` (bs1, TP4 rank, 56c), `35b_rt_tp4shard` (same with `--tp 4`),
`35b_thr` (bs320, TP1, 224c), `9b_rt` / `9b_thr` (dense, bs1 / bs96).

```bash
OUT=run1 LANES="35b_rt 35b_thr" LAYERS=all THREADS_RT=56 THREADS_THR=224 \
    bash run_dmr_lanes.sh
```

Knobs: `OUT` `THREADS_RT` `THREADS_THR` `LAYERS` `AB` `PY` `LANES`. It records
`hostname`, `numactl --hardware`, `lscpu` and the git sha into `$OUT/env.txt` — the DMR
box's SNC state flips across reboots, so the topology has to be captured per run, not
assumed.

Cost to plan around, measured on GNR96C: **35B bs320 prefill is 19.74 TFLOP/layer**
(3.14 M routed slots, 392,275 tokens fed) at 6.48 s/iter — so it gets few iterations on
purpose. It is also the **+19.7% mass-overshoot** cell. That overshoot inflates the
absolute time, but *not* the projection-vs-measured ratio: both sides consume the
identical histogram.

## Files

* `bench_moe_cpu.py` — driver: CLI, routing table, timing, CSV output
* `moe_stats.py` — CSV readers + validation, ported from archbench
  `qwen3_moe_expert_dist_utils.py`
* `archbench_stats.py` — resolves and caches stats files out of the archbench repo
* `run_dmr_lanes.sh` — the daily-run lanes as one sweep

## Measured on GNR96C, `qwen35-bkc:latest`, 32 threads, 5 iters

Smoke numbers only — few iterations, threads not tuned. Recorded to show the
harness works end to end, not as results.

| cell | groups | active E | GFLOP | fused ms | batched ms | fused/batched |
|---|---|---|---|---|---|---|
| decode bs1 L0   | 1  | 8/256   | 0.05   | 0.263 | 0.647  | 0.41 |
| decode bs64 L0  | 8  | 202/256 | 3.22   | 6.379 | 16.878 | 0.38 |
| decode bs128 L0 | 14 | 233/256 | 6.44   | 9.628 | 25.379 | 0.38 |
| decode bs320 L0 | 27 | 233/256 | 16.11  | 6.418 | 27.732 | 0.23 |
| prefill bs1 L0  | 5  | 252/256 | 52.29  | 12.922 | 24.782 | 0.52 |
| prefill bs1 L20 | 5  | 190/256 | 51.79  | 7.153 | 13.819 | 0.52 |

`layers_averaged` (different CSVs, bs20 collection), fused only:
decode bs20 → 7 groups, 86/256 experts, 1.01 GFLOP, 2.526 ms;
prefill bs20 → 6 groups, 256/256 experts, 1030.94 GFLOP, 83.3 ms.

`--no-prepack` at decode bs64 L0: 66.158 ms vs 6.379 ms prepacked — **10× slower**.
`convert_weight_packed` is not optional for a meaningful measurement; the non-VNNI
path is a fallback, not a variant worth projecting.

Two things worth chasing with proper iteration counts, not conclusions yet:

* decode bs320 is *faster* than bs128 despite 2.5× the FLOPs. Both activate 233
  experts, so both stream the same ~1.4 GB of expert weights; decode at these
  batches looks weight-bandwidth bound, not compute bound, which is exactly the
  regime the projection has to get right.
* `fused` beats the batched decomposition by 2–4× everywhere. If that holds, the
  projection's per-bucket batched-GEMM cost is an upper bound on the real kernel,
  and the gap is a kernel-quality term the model does not currently carry.
