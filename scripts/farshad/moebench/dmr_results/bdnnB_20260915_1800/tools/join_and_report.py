#!/usr/bin/env python3
"""Step B join + report: benchdnn (non-fused) vs projection vs fused `fused_experts_cpu`.

Three outputs:
  --out-raw     per-case benchdnn measurement, self-contained (adds GFLOP/TF-s)
  --out-join    per-case benchdnn vs projected ms vs projected hw_eff, carrying
                lane/leg/G/M/K/N/gflop/wei_MiB/bound/effsrc
  --out-md      per-lane roll-up for ONE MoE layer, benchdnn vs fused vs projection

The per-layer roll-up is EXACT, not an average: the fused harness records each layer's
expert-bucket decomposition in its `groups` column (e.g. "2x75709;14x42272;..." = GxM
pairs), and each bucket charges exactly two batched GEMMs --
    GLU : G x M x hidden      : G x hidden x 2*mis_local
    down: G x M x mis_local   : G x mis_local x hidden
with mis_local = moe_intermediate_size already divided by TP. So a layer's benchdnn
prediction is the sum of that layer's own 2*num_groups case times. G == 1 collapses to
benchdnn's 2D form (MxK:KxN), which is why the case set has two dtag families.
"""
import argparse, os, re, statistics, sys
import pandas as pd

HIDDEN = 2048

LANES = {
    # projection lane name -> (phase, batch, tp, fused csv key)
    "rt_decode_bs1_TP4":     ("decode", 1, 4, "results"),
    "rt_prefill_bs1_TP4":    ("prefill", 1, 4, "results"),
    "thr_decode_bs320_TP1":  ("decode", 320, 1, "results"),
    "thr_prefill_bs320_TP1": ("prefill", 320, 1, "thrprefill_all40"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True, help="parse_bdnnB.py output")
ap.add_argument("--cases", required=True, help="prepared projection case CSV")
ap.add_argument("--fused-dir", required=True, help="free-landing fused run dir")
ap.add_argument("--out-raw", required=True)
ap.add_argument("--out-join", required=True)
ap.add_argument("--out-md", required=True)
ap.add_argument("--label", default="")
a = ap.parse_args()

b = pd.read_csv(a.raw)
p = pd.read_csv(a.cases)
assert not p.bdnn_shape.duplicated().any(), "bdnn_shape is expected to be globally unique"


def dims(shape):
    L = [int(x) for x in shape.split(":")[0].split("x")]
    R = [int(x) for x in shape.split(":")[1].split("x")]
    return (1, L[0], L[1], R[1]) if len(L) == 2 else (L[0], L[1], L[2], R[2])


# ---- 1. raw, self-contained -----------------------------------------------------------
d = b.copy()
gmkn = d.bdnn_shape.map(dims)
d["G"], d["M"], d["K"], d["N"] = (gmkn.map(lambda t, i=i: t[i]) for i in range(4))
d["gflop"] = 2 * d.G * d.M * d.K * d.N / 1e9
d["bdnn_TFps"] = (d.gflop / d.bdnn_min_ms).round(3)
# eff% is deliberately absent: no compute peak (AMX/Tmul) has ever been measured on this
# box, so there is no verified denominator. TF/s is the raw rate; do not divide it by a
# datasheet number and call it efficiency.
d["eff_pct"] = "no_verified_peak_on_this_box"
raw_cols = ["idx", "bdnn_shape", "cold_cache", "dtag", "G", "M", "K", "N", "gflop",
            "impl", "n_instances", "bdnn_min_ms", "bdnn_min_ms_fastest",
            "bdnn_min_ms_mean", "bdnn_min_ms_median", "bdnn_spread_pct", "bdnn_avg_ms",
            "bdnn_TFps",
            "bdnn_min_Gbw", "eff_pct", "core_mhz_mean", "core_mhz_min", "core_mhz_max",
            "core_mhz_mean_r0", "core_mhz_mean_r1", "core_mhz_mean_r2",
            "core_mhz_mean_r3", "pkg_temp_max", "n_freq_samples", "wall_s", "rc"]
d[raw_cols].to_csv(a.out_raw, index=False)

# ---- 2. join against the projection ---------------------------------------------------
j = p.merge(d[["bdnn_shape", "cold_cache", "dtag", "impl", "bdnn_min_ms",
               "bdnn_min_ms_fastest", "bdnn_min_ms_mean", "bdnn_min_ms_median",
               "bdnn_spread_pct",
               "bdnn_TFps", "bdnn_min_Gbw", "core_mhz_mean", "pkg_temp_max",
               "n_instances"]],
             on="bdnn_shape", how="left")
j = j.rename(columns={"ms": "proj_ms", "hw_eff": "proj_hw_eff",
                      "core": "proj_core_GHz", "uncore": "proj_uncore_GHz"})
j["bdnn_over_proj"] = (j.bdnn_min_ms_median / j.proj_ms).round(4)
# First-order clock normalisation: scale benchdnn's free-landed time onto the clock the
# projection charged. Exact only for a purely core-clock-bound kernel -- a MEM-bound case
# does not scale with core clock at all -- so it is a second column, never the headline.
j["bdnn_ms_at_proj_core"] = (j.bdnn_min_ms_median * j.core_mhz_mean / (j.proj_core_GHz * 1000)).round(4)
j["bdnn_at_proj_core_over_proj"] = (j.bdnn_ms_at_proj_core / j.proj_ms).round(4)
join_cols = ["bdnn_shape", "opts", "cold_cache", "dtag", "lanes", "leg", "G", "M", "K", "N",
             "gflop", "wei_MiB", "bound", "effsrc", "n_rows", "sim_rows",
             "proj_core_GHz", "proj_uncore_GHz", "proj_hw_eff", "proj_ms",
             "bdnn_min_ms", "bdnn_min_ms_fastest", "bdnn_min_ms_mean", "bdnn_min_ms_median",
             "bdnn_spread_pct",
             "bdnn_TFps", "bdnn_min_Gbw", "core_mhz_mean", "pkg_temp_max", "n_instances",
             "bdnn_over_proj", "bdnn_ms_at_proj_core", "bdnn_at_proj_core_over_proj",
             "impl"]
j[join_cols].to_csv(a.out_join, index=False)

missing = j[j.bdnn_min_ms.isna()]

# ---- 3. per-layer reconstruction and roll-up -----------------------------------------
bt = dict(zip(d.bdnn_shape, d.bdnn_min_ms))
bt_mean = dict(zip(d.bdnn_shape, d.bdnn_min_ms_mean))
# Per-instance, so a layer can be summed within one 56-core group before the four are
# collapsed -- the uniform median-over-instances rule the fused/batched legs now use.
RCOLS = ["bdnn_min_ms_r0", "bdnn_min_ms_r1", "bdnn_min_ms_r2", "bdnn_min_ms_r3"]
bt_inst = {s: list(v) for s, v in zip(d.bdnn_shape, d[RCOLS].to_numpy())}
pt = dict(zip(p.bdnn_shape, p.ms))


def shape_of(G, M, K, N):
    return f"{G}x{M}x{K}:{G}x{K}x{N}" if G > 1 else f"{M}x{K}:{K}x{N}"


fused = {}
for key, fn in [("results", "results.csv"),
                ("thrprefill_all40", "results_thrprefill_all40.csv")]:
    fp = os.path.join(a.fused_dir, fn)
    if os.path.exists(fp):
        fused[key] = pd.read_csv(fp)

lane_rows, layer_rows, unresolved, unmeasured_in_layer = [], [], [], []
for lane, (phase, batch, tp, key) in LANES.items():
    # A run that keeps every lane in one results.csv has no split thr_prefill file; the
    # split file must still win where it exists, since results.csv holds only a 5-layer
    # sample of that lane in the older layout.
    f = fused.get(key)
    if f is None:
        f = fused.get("results")
    if f is None:
        continue
    g = f[(f.phase == phase) & (f.batch == batch) & (f.tp == tp)]
    if g.empty:
        continue
    mis = int(g.moe_intermediate_size.iloc[0])          # already divided by TP
    for layer, gl in g.groupby("layer"):
        # the four concurrent 56-core instances all decode the SAME bucket set
        groups = gl.groups.iloc[0]
        bsum = bsum_mean = psum = 0.0
        bsum_inst = [0.0, 0.0, 0.0, 0.0]
        nk = n_missing_case = n_unmeasured = 0
        for tok in str(groups).split(";"):
            G, M = (int(x) for x in tok.split("x"))
            for K, N in ((HIDDEN, 2 * mis), (mis, HIDDEN)):
                s = shape_of(G, M, K, N)
                nk += 1
                if s not in pt:
                    # the layer's own bucket decomposition implies a shape the prepared
                    # case set does not carry -- a coverage gap in the INPUT, not here
                    unresolved.append((lane, int(layer), s))
                    n_missing_case += 1
                    continue
                psum += pt[s]
                if s in bt:
                    bsum += bt[s]
                    bsum_mean += bt_mean[s]
                    for r, v in enumerate(bt_inst[s]):
                        bsum_inst[r] += float(v)
                else:
                    unmeasured_in_layer.append((lane, int(layer), s))
                    n_unmeasured += 1
        layer_rows.append(dict(
            lane=lane, layer=int(layer), buckets=len(str(groups).split(";")), kernels=nk,
            kernels_not_in_caseset=n_missing_case, kernels_unmeasured=n_unmeasured,
            bdnn_gemm_ms=round(bsum, 4), bdnn_gemm_ms_mean_inst=round(bsum_mean, 4),
            bdnn_gemm_ms_median_inst=round(statistics.median(bsum_inst), 4),
            bdnn_instance_imbalance=round(max(bsum_inst) / statistics.median(bsum_inst), 4),
            proj_gemm_ms=round(psum, 4),
            # slowest of the four instances: a rank waits for its slowest peer
            fused_min_ms=round(gl.fused_min_ms.max(), 4),
            fused_min_ms_mean_inst=round(gl.fused_min_ms.mean(), 4),
            fused_min_ms_median_inst=round(gl.fused_min_ms.median(), 4),
            fused_median_ms_median_inst=round(gl.fused_median_ms.median(), 4),
            fused_instance_imbalance=round(gl.fused_min_ms.max() / gl.fused_min_ms.min(), 4),
        ))

L = pd.DataFrame(layer_rows)
if not L.empty:
    L.to_csv(a.out_md.replace(".md", "_per_layer.csv"), index=False)
    L["r_bdnn_proj"] = L.bdnn_gemm_ms_median_inst / L.proj_gemm_ms
    # The moebench run reports its legs on median-over-iterations; use the same column
    # so this report and the four-way table cannot disagree on the fused number.
    L["r_bdnn_fused"] = L.bdnn_gemm_ms_median_inst / L.fused_median_ms_median_inst
    L["r_proj_fused"] = L.proj_gemm_ms / L.fused_median_ms_median_inst
    lane_rows = L.groupby("lane").agg(
        layers=("layer", "nunique"), buckets_min=("buckets", "min"),
        buckets_max=("buckets", "max"), kernels_min=("kernels", "min"),
        kernels_max=("kernels", "max"),
        gaps=("kernels_not_in_caseset", "sum"), unmeas=("kernels_unmeasured", "sum"),
        bdnn=("bdnn_gemm_ms_median_inst", "median"),
        bdnn_maxi=("bdnn_gemm_ms", "median"),
        proj=("proj_gemm_ms", "median"),
        fused=("fused_median_ms_median_inst", "median"),
        bdnn_imbal=("bdnn_instance_imbalance", "max"),
        imbal=("fused_instance_imbalance", "max"),
        bdnn_over_proj=("r_bdnn_proj", "median"),
        bdnn_over_fused=("r_bdnn_fused", "median"),
        proj_over_fused=("r_proj_fused", "median"),
    ).reset_index()

# ---- shape-family disagreement -------------------------------------------------------
jj = j.dropna(subset=["bdnn_min_ms"]).copy()
jj["Mbin"] = pd.cut(jj.M, [0, 8, 64, 512, 4096, 32768, 10**9],
                    labels=["M<=8", "8<M<=64", "64<M<=512", "512<M<=4k", "4k<M<=32k", "M>32k"])
fam = jj.groupby(["leg", "Mbin"], observed=True).agg(
    cases=("bdnn_min_ms_median", "size"), proj_ms=("proj_ms", "sum"),
    bdnn_ms=("bdnn_min_ms_median", "sum"), med_ratio=("bdnn_over_proj", "median"),
    p10=("bdnn_over_proj", lambda s: s.quantile(.10)),
    p90=("bdnn_over_proj", lambda s: s.quantile(.90)),
    mhz=("core_mhz_mean", "median")).reset_index()
fam["sum_ratio"] = fam.bdnn_ms / fam.proj_ms

bnd = jj.groupby("bound").agg(
    cases=("bdnn_min_ms_median", "size"), proj_ms=("proj_ms", "sum"),
    bdnn_ms=("bdnn_min_ms_median", "sum"),
    med_ratio=("bdnn_over_proj", "median"), mhz=("core_mhz_mean", "median")).reset_index()
bnd["sum_ratio"] = bnd.bdnn_ms / bnd.proj_ms

# ---- markdown ------------------------------------------------------------------------
def tbl(df, cols, fmt):
    out = ["| " + " | ".join(c for c, _ in cols) + " |",
           "|" + "|".join("---:" if r else "---" for _, r in cols) + "|"]
    for _, row in df.iterrows():
        out.append("| " + " | ".join(fmt(c, row) for c, _ in cols) + " |")
    return "\n".join(out)


w = []
w.append(f"# Expert-GEMM benchdnn (non-fused) vs projection vs fused — DMRX4PT, free-landing{a.label}\n")
w.append("Step B of the moebench campaign. Question: **does a per-kernel benchdnn "
         "measurement predict the real fused MoE layer, and does it agree with what "
         "archbench charges?**\n")
w.append("## Instrument\n")
w.append(f"- benchdnn `--mode=P`, batched matmul, bf16:bf16:bf16, `wtag=any` (VNNI prepacked B), "
         f"per-case `cold-cache` and `dtag/stag` preserved from the projection's own case identity.\n"
         f"- **4 concurrent 56-core instances** (cores 0-55 / 56-111 / 112-167 / 168-223), the same "
         f"core layout the fused lanes used: the rt lanes are the 4 TP4 ranks, the thr lanes 4 "
         f"independent TP1 replicas. Instances are collapsed by **MEDIAN**, at tp=4 exactly as "
         f"at tp=1, and a layer's kernels are summed WITHIN one 56-core group before the four are "
         f"collapsed. A max would fold a straggler term into the cost and make the tp=4 rows "
         f"incomparable to the tp=1 rows, which have no collective at all; the spread is carried "
         f"as its own `imb` column instead.\n"
         f"- **Free-landing frequency, no pin.** Caps verified at baseline before the first case and "
         f"re-read after the last. `perf` is absent on this box, so achieved core MHz comes from a "
         f"1 Hz `scaling_cur_freq` sampler windowed to each case.\n"
         f"- All cases dispatched to `{'|'.join(sorted(d.impl.unique()))}`.\n"
         f"- **eff% is not reported: no compute peak has ever been measured on this box**, so there "
         f"is no verified denominator. TF/s is given raw.\n")
w.append(f"- Cases: **{len(d)} of {len(p)}** measured"
         + (f" ({len(missing)} unmeasured)" if len(missing) else " (all)") + ".\n")
w.append("- Projection: `qwen35_0903_1215_dmrx4_ancollonly` (2026-09-03, the run behind "
         "XAWPTM-481's last checkpoint). Fused: free-landing `fused_experts_cpu`, "
         f"`{os.path.basename(a.fused_dir.rstrip('/'))}`.\n")

w.append("\n## Per-lane roll-up, ONE MoE layer (median over the lane's 40 layers)\n")
w.append("`bdnn` = sum of that layer's own expert-GEMM cases, reconstructed exactly from the "
         "layer's bucket decomposition. `proj` = the projection's charge for the same GEMMs. "
         "`fused` = measured `fused_experts_cpu`, which also carries SwiGLU and the token "
         "gather/scatter, so `fused` is expected to exceed a pure GEMM sum. Every ratio is the "
         "MEDIAN OVER LAYERS OF THE PER-LAYER RATIO, not a ratio of the two medians -- the same "
         "convention the moebench leg join uses.\n")
if lane_rows is not None and len(lane_rows):
    cols = [("lane", 0), ("layers", 1), ("buckets", 1), ("kernels/layer", 1),
            ("unmeasured", 1), ("bdnn ms", 1), ("proj ms", 1), ("fused ms", 1),
            ("bdnn/proj", 1), ("bdnn/fused", 1), ("proj/fused", 1)]

    def f(c, r):
        whole = r.unmeas == 0          # a partial sum is NOT a lane result
        if c == "lane": return r.lane + ("" if whole else " *")
        if c == "layers": return str(int(r.layers))
        if c == "buckets": return f"{int(r.buckets_min)}" if r.buckets_min == r.buckets_max else f"{int(r.buckets_min)}-{int(r.buckets_max)}"
        if c == "kernels/layer": return f"{int(r.kernels_min)}" if r.kernels_min == r.kernels_max else f"{int(r.kernels_min)}-{int(r.kernels_max)}"
        if c == "unmeasured": return "-" if whole else f"{int(r.unmeas)} of {int(r.kernels_min if r.kernels_min == r.kernels_max else 0) * int(r.layers) or int(r.unmeas)}"
        if c == "bdnn ms": return f"{r.bdnn:.3f}" if whole else f"(≥{r.bdnn:.3f})"
        if c == "proj ms": return f"{r.proj:.3f}"
        if c == "fused ms": return f"{r.fused:.3f}"
        if not whole: return "n/a"
        return f"{getattr(r, {'bdnn/proj':'bdnn_over_proj','bdnn/fused':'bdnn_over_fused','proj/fused':'proj_over_fused'}[c]):.3f}x"
    w.append(tbl(lane_rows.sort_values("fused", ascending=False), cols, f))
    if (lane_rows.unmeas > 0).any():
        w.append("\n`*` = **incomplete lane: the benchdnn sum is missing kernels, so it is a "
                 "LOWER BOUND and every ratio is withheld.** `proj ms` and `fused ms` are "
                 "complete for every lane; only the benchdnn column is short.\n")

w.append("\n### Across-instance spread, carried not baked in\n")
w.append("`imb` = worst instance / median at the lane's worst layer. benchdnn's instances are "
         "not barrier-synced where the fused harness is, so benchdnn carries the wider spread -- "
         "under the median rule it no longer leaks into the headline number.\n")
if lane_rows is not None and len(lane_rows):
    c2 = [("lane", 0), ("bdnn imb", 1), ("fused imb", 1),
          ("bdnn median-inst ms", 1), ("bdnn max-inst (retired) ms", 1), ("max/median", 1)]

    def f2(c, r):
        if c == "lane": return r.lane
        if c == "bdnn imb": return f"{r.bdnn_imbal:.3f}"
        if c == "fused imb": return f"{r.imbal:.3f}"
        if c == "bdnn median-inst ms": return f"{r.bdnn:.3f}"
        if c == "bdnn max-inst (retired) ms": return f"{r.bdnn_maxi:.3f}"
        return f"{r.bdnn_maxi / r.bdnn:.3f}x"
    w.append(tbl(lane_rows.sort_values("fused", ascending=False), c2, f2))
    w.append("\nThe last column is what the retired max-over-instances rule was adding to the "
             "benchdnn column.\n")

w.append("\n## Where benchdnn and the projection disagree, by shape family\n")
w.append("`sum_ratio` weights by time (what moves a lane); `med_ratio` weights by case count "
         "(what a per-kernel eff table would see).\n")
cols = [("leg", 0), ("M range", 0), ("cases", 1), ("proj ms", 1), ("bdnn ms", 1),
        ("sum_ratio", 1), ("med_ratio", 1), ("p10", 1), ("p90", 1), ("core MHz", 1)]


def ff(c, r):
    m = {"leg": r.leg, "M range": str(r.Mbin), "cases": str(int(r.cases)),
         "proj ms": f"{r.proj_ms:.3f}", "bdnn ms": f"{r.bdnn_ms:.3f}",
         "sum_ratio": f"{r.sum_ratio:.3f}x", "med_ratio": f"{r.med_ratio:.3f}x",
         "p10": f"{r.p10:.2f}", "p90": f"{r.p90:.2f}", "core MHz": f"{r.mhz:.0f}"}
    return m[c]


w.append(tbl(fam, cols, ff))
w.append("\n### By the projection's boundness verdict\n")
cols = [("bound", 0), ("cases", 1), ("proj ms", 1), ("bdnn ms", 1), ("sum_ratio", 1),
        ("med_ratio", 1), ("core MHz", 1)]
w.append(tbl(bnd, cols, lambda c, r: {"bound": r.bound, "cases": str(int(r.cases)),
                                      "proj ms": f"{r.proj_ms:.3f}", "bdnn ms": f"{r.bdnn_ms:.3f}",
                                      "sum_ratio": f"{r.sum_ratio:.3f}x",
                                      "med_ratio": f"{r.med_ratio:.3f}x",
                                      "mhz": "", "core MHz": f"{r.mhz:.0f}"}[c]))

w.append("\n## Frequency landed\n")
w.append(f"- Across all cases: core {d.core_mhz_mean.min():.0f}-{d.core_mhz_mean.max():.0f} MHz "
         f"(median {d.core_mhz_mean.median():.0f}); package temp max {d.pkg_temp_max.max():.0f} C.\n")
w.append(f"- The projection charges core {sorted(p.core.unique())} GHz / uncore "
         f"{sorted(p.uncore.unique())} GHz. benchdnn ran unpinned, so `bdnn/proj` above mixes a "
         f"clock difference with a kernel-quality difference. The joined CSV carries "
         f"`bdnn_ms_at_proj_core` as a first-order clock normalisation; it is exact only for "
         f"core-clock-bound cases.\n")
w.append(f"- Across-instance spread of the four 56-core groups: median "
         f"{d.bdnn_spread_pct.median():.1f}%, p90 {d.bdnn_spread_pct.quantile(.9):.1f}%, "
         f"max {d.bdnn_spread_pct.max():.1f}%.\n")

if len(missing):
    w.append(f"\n## Unmeasured cases ({len(missing)})\n")
    for _, r in missing.head(30).iterrows():
        w.append(f"- `{r.bdnn_shape}` ({r.lanes}, {r.leg})")
if unresolved:
    u = sorted(set(unresolved))
    w.append(f"\n## Reconstruction gaps ({len(u)} lane/layer/shape, "
             f"{len({s for _, _, s in u})} distinct shapes)\n")
    w.append("Shapes a layer's own bucket decomposition implies that the prepared case set "
             "does not carry. Every roll-up row above is short by these:\n")
    for lane, layer, s in u[:30]:
        w.append(f"- {lane} layer {layer}: `{s}`")
if unmeasured_in_layer:
    u = sorted(set(unmeasured_in_layer))
    w.append(f"\n## Roll-up rows short a benchdnn number ({len(u)})\n")
    for lane, layer, s in u[:30]:
        w.append(f"- {lane} layer {layer}: `{s}`")

open(a.out_md, "w").write("\n".join(w) + "\n")
print(f"wrote {a.out_raw}\n      {a.out_join}\n      {a.out_md}", file=sys.stderr)
if len(missing):
    print(f"WARNING {len(missing)} cases unmeasured", file=sys.stderr)
if unresolved:
    print(f"WARNING {len(set(unresolved))} reconstruction gaps", file=sys.stderr)
