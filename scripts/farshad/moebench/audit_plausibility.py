#!/usr/bin/env python3
"""Physical-plausibility audit of a results CSV.

Two questions no code review can answer, both pure arithmetic on numbers the harness
already records:

  1. Is the reported GFLOP/s below the machine's AMX peak? A cell above peak means the
     FLOP accounting or the timed region is wrong, not that the kernel is fast.
  2. Could the weights have come from DDR? Weight bytes per iteration x instances,
     divided by the measured time, is an implied bandwidth. If it exceeds DDR peak the
     weights were served from cache, so the cell measures a cache-resident kernel --
     which is the thing `--copies` exists to prevent. This is a VERDICT, not an error:
     it tells you whether --copies was high enough for that shape.

Peaks are per-box and must be passed in; the defaults describe DMR-X4PT (224c, 8ch
DDR8000). AMX-bf16 is taken as 1024 FLOP/cycle/core (one TDPBF16PS on 16x32 x 32x16 =
16384 FLOP per 16 cycles), which is the SPR/GNR figure -- pass --amx-flops-per-cycle
if the part differs.

Usage:
  python3 audit_plausibility.py results.csv
  python3 audit_plausibility.py results.csv --ghz 2.8 --ddr-gbs 512 --cores 56
"""

import argparse
import csv
import sys


def weight_bytes_per_iter(row) -> int:
    """Bytes of expert weight one fused iteration must read: every ACTIVE expert's
    gate+up [2N, K] and down [K, N], bf16. Idle experts are never touched, so the
    active count -- not E -- is what streams."""
    K = int(row["hidden_size"])
    N = int(row["moe_intermediate_size"])
    active = int(row["active_experts"])
    return 2 * active * (2 * N * K + K * N)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv_file")
    p.add_argument("--ghz", type=float, default=3.2,
                   help="core frequency under load; 3.2 is the X4PT max, so this is "
                        "the most GENEROUS ceiling (a lower real frequency only makes "
                        "an over-peak cell worse)")
    p.add_argument("--cores", type=int, default=None,
                   help="cores per instance (default: the row's `threads`)")
    p.add_argument("--amx-flops-per-cycle", type=float, default=1024.0)
    p.add_argument("--ddr-gbs", type=float, default=512.0,
                   help="socket DDR peak, GB/s (8ch x DDR8000 x 8 B = 512)")
    p.add_argument("--instances", type=int, default=None,
                   help="concurrent instances sharing DDR (default: max `instance`+1 "
                        "seen in the file)")
    args = p.parse_args()

    with open(args.csv_file, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("fused_median_ms")]
    if not rows:
        raise SystemExit(f"{args.csv_file}: no rows with a fused_median_ms")
    inst = args.instances or max(int(r.get("instance") or 0) for r in rows) + 1

    print(f"{len(rows)} timed rows, assuming {inst} concurrent instance(s), "
          f"{args.ghz} GHz, {args.amx_flops_per_cycle:.0f} FLOP/cycle/core AMX bf16, "
          f"DDR peak {args.ddr_gbs:.0f} GB/s\n")
    # `copies` is NOT in the harness's RESULT_FIELDS, so a results CSV does not record
    # it -- which means a `cache-resident` verdict below cannot be traced back to the
    # --copies that produced it. Print what the file has and say when it has nothing,
    # rather than defaulting to 1 and inventing a column.
    if not any(r.get("copies") for r in rows):
        print("NOTE: this CSV records no `copies` column (it is not in RESULT_FIELDS), "
              "so the `cop` column below reads '?'. A cache-resident verdict cannot be\n"
              "      tied to the --copies that produced it until the harness records it.\n")
    hdr = (f"{'phase':<8}{'bs':>5}{'tp':>3}{'L':>4}{'thr':>5}{'cop':>4}"
           f"{'GFLOP':>8}{'ms':>9}{'GFLOP/s':>10}{'%peak':>7}"
           f"{'impl GB/s':>10}{'%DDR':>7}  verdict")
    print(hdr)
    print("-" * len(hdr))
    over_peak = []
    for r in rows:
        ms = float(r["fused_median_ms"])
        gflop = float(r["gflop"])
        cores = args.cores or int(r.get("threads") or 0) or 56
        copies = r.get("copies") or "?"
        peak = cores * args.amx_flops_per_cycle * args.ghz  # GFLOP/s
        rate = gflop / (ms / 1e3)
        pct_peak = 100.0 * rate / peak
        gbs = weight_bytes_per_iter(r) * inst / (ms / 1e3) / 1e9
        pct_ddr = 100.0 * gbs / args.ddr_gbs
        if pct_peak > 100:
            verdict = "IMPOSSIBLE: over AMX peak"
            over_peak.append(r)
        elif pct_ddr > 100:
            verdict = f"cache-resident ({pct_ddr / 100:.1f}x DDR peak)"
        elif pct_ddr > 70:
            verdict = "DDR-bound"
        else:
            verdict = "compute/latency-bound"
        print(f"{r['phase']:<8}{r['batch']:>5}{r['tp']:>3}{r['layer']:>4}"
              f"{cores:>5}{str(copies):>4}{gflop:>8.2f}{ms:>9.3f}{rate:>10.1f}"
              f"{pct_peak:>7.1f}{gbs:>10.1f}{pct_ddr:>7.1f}  {verdict}")

    print()
    if over_peak:
        print(f"FAIL: {len(over_peak)} row(s) report more FLOP/s than the machine can "
              f"issue. The FLOP count or the timed region is wrong.")
        return 1
    print("OK: no row exceeds AMX peak.")
    cache = [r for r in rows
             if weight_bytes_per_iter(r) * inst / (float(r["fused_median_ms"]) / 1e3)
             / 1e9 > args.ddr_gbs]
    if cache:
        print(f"NOTE: {len(cache)}/{len(rows)} row(s) imply more weight bandwidth than "
              f"DDR can supply, so their weights were served from cache. Raise --copies "
              f"for those shapes if the intent was to measure a DDR-resident layer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
