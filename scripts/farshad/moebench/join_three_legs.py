#!/usr/bin/env python3
"""Join the three legs of a lane sweep: fused mode 0, fused mode 2, and unfused.

    fused m0   fused_experts_cpu as it ships -- gather, two GEMMs with SwiGLU folded
               into the fp32 accumulator, topk-weighted scatter back
    fused m2   the same kernel with the gather and the reduce DROPPED
    unfused    torch.bmm per histogram bucket + silu(gate)*up + torch.bmm

m2 and unfused do the same work, so `unfused / m2` is the matmul comparison with nothing
else in it -- which a straight `m0 / unfused` cannot be, because m0 also carries the
gather and the scatter. `(m0 - m2) / m0` prices those two terms on their own.

Legs run as separate sweeps against one --out file, so `fused_over_batched` is empty by
construction and the pairing has to be recovered here. Rows are keyed on
(phase, batch, tp, layer) and on the `fused_mode` COLUMN rather than on a label suffix:
two rows whose only difference is the gather/scatter are otherwise indistinguishable.

Collapsing follows the campaign's uniform rule -- MEDIAN over the concurrent instances, at
tp=4 exactly as at tp=1, then MEDIAN over the layers. Spread is carried separately as
`imb` = worst instance / median, taken at the worst layer. No max anywhere: a max folds a
straggler into the cost and makes the tp=4 rows incomparable to the tp=1 rows, which have
no collective at all.
"""
import argparse
import csv
import statistics
from collections import defaultdict

LEGS = ("m0", "m2", "unfused")


def leg_of(row):
    """Which leg a row's timings belong to, or None if it timed nothing.

    A row from `--mode both` carries BOTH legs, so this yields rather than returns: the
    fused columns go to m0/m2 by `fused_mode` and the batched columns to unfused.
    """
    if row.get("fused_median_ms"):
        yield ("m2" if (row.get("fused_mode") or "0").strip() == "2" else "m0",
               "fused")
    if row.get("batched_median_ms"):
        yield ("unfused", "batched")


def load(path):
    """-> {leg: {lane: {layer: [one value per instance]}}}, and the lane's row count."""
    got = {leg: defaultdict(lambda: defaultdict(list)) for leg in LEGS}
    seen = defaultdict(int)
    for r in csv.DictReader(open(path)):
        lane = (r["phase"], r["batch"], r["tp"])
        for leg, prefix in leg_of(r):
            got[leg][lane][int(r["layer"])].append(float(r[f"{prefix}_median_ms"]))
            seen[(leg, lane)] += 1
    return got, seen


def collapse(per_layer):
    """{layer: median over instances}, and the worst layer's worst/median."""
    if not per_layer:
        return None, None, 0
    med = {L: statistics.median(v) for L, v in per_layer.items()}
    imb = max(max(v) / statistics.median(v) for v in per_layer.values())
    return med, imb, min(len(v) for v in per_layer.values())


def lane_ms(med):
    return statistics.median(med.values()) if med else None


def per_layer_ratio(num, den):
    """Median over layers of the per-layer ratio, and its range.

    A ratio of the two lane medians is a different statistic and hides the layers where
    the two legs disagree most; the campaign's leg join uses this one throughout.
    """
    if not (num and den):
        return None, None, None
    shared = sorted(set(num) & set(den))
    if not shared:
        return None, None, None
    rs = [num[L] / den[L] for L in shared]
    return statistics.median(rs), min(rs), max(rs)


def fmt(v, w=8, p=3):
    return f"{v:{w}.{p}f}" if v is not None else " " * (w - 1) + "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_csv")
    ap.add_argument("--out-md", default="")
    ap.add_argument("--out-csv", default="")
    a = ap.parse_args()

    got, seen = load(a.results_csv)
    lanes = sorted({lane for leg in LEGS for lane in got[leg]},
                   key=lambda x: (x[0], int(x[1]), int(x[2])))

    rows = []
    for lane in lanes:
        s = {}
        for leg in LEGS:
            s[leg] = collapse(got[leg].get(lane))
        m0, m2, unf = (s[leg][0] for leg in LEGS)
        gs, gs_lo, gs_hi = per_layer_ratio(
            {L: m0[L] - m2[L] for L in set(m0 or {}) & set(m2 or {})}, m0) \
            if (m0 and m2) else (None, None, None)
        uo2, uo2_lo, uo2_hi = per_layer_ratio(unf, m2)
        uo0, _, _ = per_layer_ratio(unf, m0)
        rows.append(dict(
            phase=lane[0], batch=lane[1], tp=lane[2],
            layers=max(len(x or {}) for x in (m0, m2, unf)),
            instances=min(s[leg][2] for leg in LEGS if s[leg][2]),
            m0_ms=lane_ms(m0), m2_ms=lane_ms(m2), unfused_ms=lane_ms(unf),
            gather_scatter_share=gs, gs_lo=gs_lo, gs_hi=gs_hi,
            unfused_over_m2=uo2, uo2_lo=uo2_lo, uo2_hi=uo2_hi,
            unfused_over_m0=uo0,
            m0_imb=s["m0"][1], m2_imb=s["m2"][1], unfused_imb=s["unfused"][1],
        ))

    hdr = (f"{'lane':22s} {'fused m0':>9s} {'fused m2':>9s} {'unfused':>9s} "
           f"{'g+s':>7s} {'unf/m2':>8s} {'range':>15s} {'unf/m0':>8s}")
    md = ["| lane | fused m0 ms | fused m2 ms | unfused ms | (m0-m2)/m0 | unfused/m2 | "
          "unf/m2 range | unfused/m0 |",
          "|---|---:|---:|---:|---:|---:|---|---:|"]
    print(hdr)
    print("-" * len(hdr))
    for x in rows:
        lane = f"{x['phase']}_bs{x['batch']}_TP{x['tp']}"
        rng = (f"{x['uo2_lo']:.2f}..{x['uo2_hi']:.2f}" if x["uo2_lo"] is not None else "-")
        pct = f"{x['gather_scatter_share']:.1%}" if x["gather_scatter_share"] is not None else "-"
        print(f"{lane:22s} {fmt(x['m0_ms'], 9)} {fmt(x['m2_ms'], 9)} "
              f"{fmt(x['unfused_ms'], 9)} {pct:>7s} {fmt(x['unfused_over_m2'], 8)} "
              f"{rng:>15s} {fmt(x['unfused_over_m0'], 8)}")
        md.append(f"| {lane} | {fmt(x['m0_ms'],1)} | {fmt(x['m2_ms'],1)} | "
                  f"{fmt(x['unfused_ms'],1)} | {pct} | "
                  f"**{fmt(x['unfused_over_m2'],1)}x** | {rng} | "
                  f"{fmt(x['unfused_over_m0'],1)}x |")

    missing = [(leg, lane) for lane in lanes for leg in LEGS if not got[leg].get(lane)]
    if missing:
        print("\nlegs with no rows (a `-` above): "
              + ", ".join(f"{l}@{p}_bs{b}_TP{t}" for l, (p, b, t) in missing))

    if a.out_csv:
        with open(a.out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow({k: (round(v, 5) if isinstance(v, float) else v)
                            for k, v in r.items()})
        print(f"\nwrote {a.out_csv}")
    if a.out_md:
        open(a.out_md, "w").write("\n".join(md) + "\n")
        print(f"wrote {a.out_md}")


if __name__ == "__main__":
    main()
