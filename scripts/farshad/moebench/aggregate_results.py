#!/usr/bin/env python3
"""Merge the per-instance CSVs from one or more run directories into one tidy table,
plus a per-cell summary.

Each lane writes `results.inst<N>.csv` per concurrent instance, so a single cell
appears once per instance. Two things are worth separating there:

  * the SPREAD across instances tells you whether the four 56-core groups are
    actually equivalent on this part -- if one group is consistently slower, the
    "4 concurrent ranks" model is not symmetric and that matters for TPOT;
  * the MAX across instances is what a TP rank actually waits for, since the
    all-reduce after the MoE cannot start until the slowest rank arrives. The median
    across instances flatters the lane.

Usage:
    python3 aggregate_results.py dmr_results/*/            # -> merged.csv, summary.csv
    python3 aggregate_results.py --out-dir agg dmr_results/*/
"""

import argparse
import csv
import glob
import os
import statistics
import sys
from typing import Dict, List


def lane_of(row: Dict[str, str]) -> str:
    """Recover the lane name. Runs made before bench_moe_cpu.py grew --label do not
    record it, but the lane is fully determined by the shape it ran."""
    if row.get("label"):
        return row["label"]
    K = int(row["hidden_size"])
    E = int(row["num_experts"])
    tp = int(row.get("tp") or 1)
    batch = int(row["batch"])
    if E == 1:
        base = "9b_dense" if K == 4096 else "35b_shared"
        return f"{base}_{'rt' if batch in (1,) else 'thr'}"
    if batch == 1:
        return "35b_rt" if tp > 1 else "35b_rt_unsharded"
    return "35b_thr" if tp == 1 else f"35b_thr_tp{tp}"


def load(dirs: List[str]) -> List[Dict[str, str]]:
    rows = []
    for d in dirs:
        d = d.rstrip("/")
        run = os.path.basename(d)
        for path in sorted(glob.glob(os.path.join(d, "results.inst*.csv"))):
            inst = os.path.basename(path).split("inst")[1].split(".")[0]
            with open(path) as f:
                for r in csv.DictReader(f):
                    r["run"] = run
                    r["instance"] = inst
                    r["lane"] = lane_of(r)
                    rows.append(r)
    return rows


def fnum(row, key):
    v = row.get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+", help="run directories containing results.inst*.csv")
    p.add_argument("--out-dir", default=".")
    args = p.parse_args()

    rows = load(args.dirs)
    if not rows:
        raise SystemExit(f"no results.inst*.csv found under {args.dirs}")

    # ---- tidy merge -----------------------------------------------------------
    fields = ["run", "lane", "instance"]
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    merged = os.path.join(args.out_dir, "merged.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(merged, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ---- per-cell summary across instances ------------------------------------
    cells: Dict[tuple, List[Dict[str, str]]] = {}
    for r in rows:
        key = (r["run"], r["lane"], r["phase"], int(r["batch"]), int(r["layer"]),
               int(r.get("tp") or 1))
        cells.setdefault(key, []).append(r)

    out = []
    for (run, lane, phase, batch, layer, tp), rs in sorted(cells.items()):
        rec = dict(run=run, lane=lane, phase=phase, batch=batch, layer=layer, tp=tp,
                   instances=len(rs), threads=rs[0].get("threads"),
                   num_groups=rs[0].get("num_groups"),
                   active_experts=rs[0].get("active_experts"),
                   histogram_mass=rs[0].get("histogram_mass"),
                   num_tokens=rs[0].get("num_tokens"), gflop=rs[0].get("gflop"),
                   stats_commit=rs[0].get("stats_commit"))
        gf = fnum(rs[0], "gflop")
        for mode in ("fused", "batched"):
            vals = [fnum(r, f"{mode}_median_ms") for r in rs]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            rec[f"{mode}_min_ms"] = min(vals)
            rec[f"{mode}_median_ms"] = statistics.median(vals)
            # The slowest instance is what a rank waits for.
            rec[f"{mode}_max_ms"] = max(vals)
            rec[f"{mode}_spread"] = max(vals) / min(vals) if min(vals) else None
            if gf:
                rec[f"{mode}_gflops_at_max"] = gf / (max(vals) / 1e3)
        if rec.get("fused_max_ms") and rec.get("batched_max_ms"):
            rec["fused_over_batched"] = rec["fused_max_ms"] / rec["batched_max_ms"]
        errs = [fnum(r, "check_rel_err") for r in rs]
        errs = [e for e in errs if e is not None]
        if errs:
            rec["check_rel_err_max"] = max(errs)
        out.append(rec)

    sfields = []
    for r in out:
        for k in r:
            if k not in sfields:
                sfields.append(k)
    summary = os.path.join(args.out_dir, "summary.csv")
    with open(summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sfields)
        w.writeheader()
        w.writerows(out)

    # ---- lane-level roll-up, printed ------------------------------------------
    print(f"{len(rows)} rows -> {merged}")
    print(f"{len(out)} cells -> {summary}\n")
    hdr = (f"{'lane':<20} {'phase':<8} {'bs':>4} {'tp':>2} {'lay':>4} {'GFLOP':>9} "
           f"{'fused_max':>10} {'batch_max':>10} {'f/b':>6} {'spread':>6}")
    print(hdr); print("-" * len(hdr))
    by_lane: Dict[tuple, List[Dict]] = {}
    for r in out:
        by_lane.setdefault((r["lane"], r["phase"], r["batch"], r["tp"]), []).append(r)
    for (lane, phase, batch, tp), rs in sorted(by_lane.items()):
        # Per-layer cells collapse to the layer-summed cost, which is what a whole
        # model pass pays; single-cell lanes (dense) just report themselves.
        fm = sum(r.get("fused_max_ms") or 0 for r in rs)
        bm = sum(r.get("batched_max_ms") or 0 for r in rs)
        gf = sum(float(r["gflop"]) for r in rs)
        sp = max((r.get("fused_spread") or 1) for r in rs)
        nl = len(rs)
        print(f"{lane:<20} {phase:<8} {batch:>4} {tp:>2} {nl:>4} {gf:>9.1f} "
              f"{fm:>10.3f} {bm if bm else float('nan'):>10.3f} "
              f"{(fm/bm if bm else float('nan')):>6.2f} {sp:>6.2f}")
    print("\nfused_max/batch_max are summed over the layers measured (`lay`), using the "
          "SLOWEST instance per cell.\nspread = worst across-instance max/min ratio in "
          "the lane; >>1 means the four core groups are not equivalent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
