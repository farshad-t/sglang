#!/usr/bin/env python3
"""Four-way per-lane comparison for ONE MoE layer of qwen35-35B-A3B on DMRX4PT.

  col 1  moebench FUSED    -- `fused_experts_cpu`, the shipping kernel. Carries the expert
                             GEMMs PLUS SwiGLU and the token gather/scatter.
  col 2  moebench UNFUSED  -- the harness's `batched` leg: one torch.bmm pair per histogram
                             bucket. Same SHAPE SET the projection models. GEMMs only.
  col 3  benchdnn          -- per-expert-GEMM sum, oneDNN brg_matmul, wtag=any (B VNNI-
                             prepacked). GEMMs only.
  col 4  archbench PROJECTION -- the projection's charge for those same GEMMs.

INSTANCE COLLAPSING -- one rule, every column, every cell: **MEDIAN over the four 56-core
instances**, at tp=4 exactly as at tp=1. Not the max. A max folds a straggler term into the
cost, and a straggler is a scale-up effect this experiment does not study; it also makes the
tp=4 rows incomparable to the tp=1 rows, which have no collective at all. The spread is
carried separately as `imb` = worst instance / median.

For benchdnn that means a layer's kernels are summed WITHIN one 56-core group first, and
only then are the four groups collapsed -- collapsing per case and summing after is a
different statistic. `parse_bdnnB.py` therefore keeps the per-instance min-times and
`join_and_report.py` emits `bdnn_gemm_ms_median_inst`.

Over the 40 layers: MEDIAN, matching the moebench run's own leg join.
"""
import argparse, csv, os, statistics
from collections import defaultdict

LANES = {
    ("prefill", "320", "1"): "thr_prefill_bs320_TP1",
    ("decode", "320", "1"): "thr_decode_bs320_TP1",
    ("prefill", "1", "4"): "rt_prefill_bs1_TP4",
    ("decode", "1", "4"): "rt_decode_bs1_TP4",
}
LANE_ORDER = list(dict.fromkeys(LANES.values()))

ap = argparse.ArgumentParser()
ap.add_argument("--moebench-dir", required=True)
ap.add_argument("--per-layer", required=True, help="bdnn ..._per_layer.csv (bdnn + proj)")
ap.add_argument("--label", default="")
ap.add_argument("--out-md", required=True)
ap.add_argument("--out-csv", required=True)
a = ap.parse_args()


def load_moebench(d):
    """-> {(lane, leg, stat): {layer: [one value per instance]}}"""
    got = defaultdict(lambda: defaultdict(list))
    files = []
    for fn in sorted(f for f in os.listdir(d) if f.startswith("results") and f.endswith(".csv")):
        rows = list(csv.DictReader(open(os.path.join(d, fn))))
        if not rows or "label" not in rows[0]:
            continue                      # only the one-leg-per-row schema is uniform
        n = 0
        for r in rows:
            lane = LANES.get((r["phase"], r["batch"], r["tp"]))
            if lane is None:
                continue
            leg = "fused" if r["label"].endswith("_fused") else "batched"
            for stat in ("median", "min"):
                v = r.get(f"{leg}_{stat}_ms", "")
                if v in ("", "0", "0.0"):
                    continue
                got[(lane, leg, stat)][int(r["layer"])].append(float(v))
                n += 1
        if n:
            files.append(f"{fn}({n})")
        break                             # first schema-matching file is the run's own CSV
    return got, files


def series(got, lane, leg, stat):
    """{layer: median over the four instances}, plus the worst-layer imbalance."""
    per = got.get((lane, leg, stat))
    if not per:
        return None, None, 0
    med = {L: statistics.median(per[L]) for L in per}
    imb = max(max(per[L]) / statistics.median(per[L]) for L in per)
    return med, imb, min(len(per[L]) for L in per)


mb, mb_files = load_moebench(a.moebench_dir)

bdnn = defaultdict(dict)
for r in csv.DictReader(open(a.per_layer)):
    bdnn[r["lane"]][int(r["layer"])] = r


def per_layer_series(lane, col):
    per = bdnn.get(lane, {})
    return {L: float(per[L][col]) for L in per} if per else None


def med(s):
    """Lane value: median over the 40 layers."""
    return statistics.median(s.values()) if s else None


def ratio(num, den):
    """Median over layers of the PER-LAYER ratio -- not a ratio of the two medians.

    This is the moebench leg join's own convention, and applying it to the cross-column
    ratios too keeps one rule for every number in the table.
    """
    if not num or not den:
        return None
    shared = sorted(set(num) & set(den))
    return statistics.median(num[L] / den[L] for L in shared) if shared else None


def fmt(v):
    return f"{v:.3f}x" if v else "-"


rows = []
for lane in LANE_ORDER:
    f, f_imb, n_inst = series(mb, lane, "fused", "median")
    b, b_imb, _ = series(mb, lane, "batched", "median")
    f_min, _, _ = series(mb, lane, "fused", "min")
    b_min, _, _ = series(mb, lane, "batched", "min")
    if not (f and b):
        continue
    d = per_layer_series(lane, "bdnn_gemm_ms_median_inst")
    d_max = per_layer_series(lane, "bdnn_gemm_ms")
    p = per_layer_series(lane, "proj_gemm_ms")
    fb = [f[L] / b[L] for L in sorted(set(f) & set(b))]
    rows.append(dict(
        lane=lane, layers=len(f), instances=n_inst,
        fused_ms=med(f), unfused_ms=med(b), bdnn_ms=med(d), proj_ms=med(p),
        bdnn_over_unfused=ratio(d, b), bdnn_over_proj=ratio(d, p),
        proj_over_fused=ratio(p, f), fused_over_unfused=statistics.median(fb),
        fb_lo=min(fb), fb_hi=max(fb),
        fused_imb=f_imb, unfused_imb=b_imb,
        bdnn_imb=max(per_layer_series(lane, "bdnn_instance_imbalance").values()),
        fused_ms_min=med(f_min), unfused_ms_min=med(b_min),
        bdnn_ms_max_inst=med(d_max), bdnn_max_over_median=ratio(d_max, d),
    ))

with open(a.out_csv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

NAME = a.label or os.path.basename(a.moebench_dir.rstrip("/"))
md = []
A = md.append
A("# The full picture: moebench fused vs moebench unfused vs benchdnn vs projection")
A("")
A(f"qwen35-35B-A3B, ONE MoE layer, DMRX4PT (224c / 1 socket / SNC off), free-landing "
  f"frequency, 4 concurrent 56-core instances in every column. moebench run: `{NAME}`.")
A("")
A("| column | what it is | expert GEMMs | SwiGLU | gather/scatter + topk weighting | "
  "B VNNI-prepacked |")
A("|---|---|:-:|:-:|:-:|:-:|")
A("| moebench FUSED | `fused_experts_cpu` -- the shipping kernel | yes | yes | **yes** | yes |")
A("| moebench UNFUSED | `torch.bmm` per histogram bucket -- reference impl | yes | "
  "**yes** | no | **no** |")
A("| benchdnn | oneDNN `brg_matmul`, one GEMM at a time | yes | **no** | no | yes |")
A("| archbench PROJECTION | the model's charge for those GEMMs | yes | no | no | n/a |")
A("")
A("The UNFUSED leg is **not** GEMM-only: its per-bucket body is `h = bmm(a, w1)`, "
  "`gate, up = h.chunk(2, -1)`, `bmm(silu(gate) * up, w2)`, so it re-reads `h` and writes a "
  "fresh activation that the second bmm then consumes. Its weights are also NOT prepacked, "
  "where benchdnn ran `wtag=any`. Only FUSED adds the token gather/scatter and the topk "
  "weighting. The UNFUSED leg carries the projection's SHAPE SET but not its cost model, so "
  "**UNFUSED vs PROJECTION is not projection error.**")
A("")
A("## Method: one instance-collapsing rule everywhere")
A("")
A("**MEDIAN over the four 56-core instances, at tp=4 exactly as at tp=1**, then median over "
  "the 40 layers. No max anywhere. A max folds a straggler term into the cost -- a scale-up "
  "effect this experiment does not study -- and makes the tp=4 rows incomparable to the tp=1 "
  "rows, which have no collective at all. Spread stays visible as its own `imb` column "
  "(worst instance / median).")
A("")
A("For benchdnn this means a layer's kernels are summed **within one 56-core group** and "
  "only then are the four groups collapsed. Collapsing per case and summing afterwards is a "
  "different statistic, so the per-instance min-times are carried through unreduced.")
A("")
A("## The table")
A("")
A("Every ms is a median over instances then over the 40 layers; every ratio is the median "
  "over layers of the per-layer ratio, which is the moebench leg join's own convention.")
A("")
A("| lane | FUSED ms | UNFUSED ms | benchdnn ms | PROJECTION ms | bdnn/unfused | "
  "bdnn/proj | proj/fused | fused/unfused | f/b range |")
A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
for x in rows:
    A(f"| {x['lane']} | {x['fused_ms']:.3f} | {x['unfused_ms']:.3f} | {x['bdnn_ms']:.3f} | "
      f"{x['proj_ms']:.3f} | {fmt(x['bdnn_over_unfused'])} | "
      f"{fmt(x['bdnn_over_proj'])} | {fmt(x['proj_over_fused'])} | "
      f"**{x['fused_over_unfused']:.3f}x** | {x['fb_lo']:.3f}..{x['fb_hi']:.3f} |")
A("")
A("### Across-instance spread, carried not baked in")
A("")
A("`imb` = worst instance / median, taken at the WORST layer of the lane -- the same "
  "worst-case convention the moebench leg join reports.")
A("")
A("| lane | FUSED imb | UNFUSED imb | benchdnn imb |")
A("|---|---:|---:|---:|")
for x in rows:
    A(f"| {x['lane']} | {x['fused_imb']:.3f} | {x['unfused_imb']:.3f} | {x['bdnn_imb']:.3f} |")
A("")
A("benchdnn's instances are not barrier-synced where the moebench harness is, which is why "
  "its `imb` is the wide one. Under the median rule that spread no longer leaks into the "
  "headline number, which is the point of the rule.")
A("")
A("### Iteration statistic: the one asymmetry left")
A("")
A("The moebench columns are a median over iterations; benchdnn reports min over iterations "
  "and has no median. Re-running the moebench columns on their `min` shows how little that "
  "choice moves anything:")
A("")
A("| lane | FUSED median-iter | FUSED min-iter | UNFUSED median-iter | UNFUSED min-iter |")
A("|---|---:|---:|---:|---:|")
for x in rows:
    A(f"| {x['lane']} | {x['fused_ms']:.3f} | {x['fused_ms_min']:.3f} | "
      f"{x['unfused_ms']:.3f} | {x['unfused_ms_min']:.3f} |")
A("")
A("### What the old max rule was doing to the benchdnn column")
A("")
A("| lane | benchdnn median-inst | benchdnn max-inst (old) | max/median |")
A("|---|---:|---:|---:|")
for x in rows:
    A(f"| {x['lane']} | {x['bdnn_ms']:.3f} | {x['bdnn_ms_max_inst']:.3f} | "
      f"{fmt(x['bdnn_max_over_median'])} |")
A("")
A("## Provenance")
A("")
A(f"- moebench `{NAME}`: {', '.join(mb_files)}")
A(f"- benchdnn + projection: `{os.path.basename(a.per_layer)}`")
A(f"- layers x instances per lane: {rows[0]['layers']} x {rows[0]['instances']}")
A("")

open(a.out_md, "w").write("\n".join(md) + "\n")
print("\n".join(md))
