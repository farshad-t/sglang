#!/usr/bin/env python3
"""Physical-plausibility audit of a results CSV.

Three questions no code review can answer, all arithmetic on numbers the harness
already records:

  1. Is the reported GFLOP/s below the machine's AMX peak? A cell above peak means the
     FLOP accounting or the timed region is wrong, not that the kernel is fast.
  2. Could a cell's weights have been re-read out of cache? `time_mixed` visits every
     cell of a lane once per iteration and each cell cycles `copies` private weight
     sets, so the bytes an instance walks between two visits to the SAME set -- its
     REUSE DISTANCE -- is (active weight bytes summed over the lane's cells) x copies.
     A reuse distance that fits the LLC slice an instance owns is a cache-resident
     measurement, which is the thing `--copies` exists to prevent.
  3. Is the implied weight bandwidth within DDR peak? Weight bytes per iteration x
     instances, over the measured time.

(2) is decided BEFORE (3) because bandwidth alone cannot tell a cache-served cell from
a sound measurement on a box whose DDR peak was passed in too low. A cell that walks
hundreds of times the LLC between reuses cannot have been cache-served, so an implied
bandwidth above --ddr-gbs there indicts the assumption, not the measurement: the qwen35
thr-decode lane implies 1.6x an 8-channel peak while its reuse distance is 316x the LLC
slice, and only the footprint says which of the two numbers is wrong.

--ddr-gbs is the SOCKET peak and the implied rate is a socket aggregate (weight bytes x
instances), so the comparison holds however the socket is partitioned AS LONG AS the
instances are symmetric: with SNC4 on, four node-local instances each own 4 channels, and
dividing both sides by 4 leaves %DDR unchanged. It breaks for an ASYMMETRIC run -- one
224-core instance pinned to a single SNC node, or instances whose memory is not node-local
-- where the reachable peak is a fraction of the socket's; pass --ddr-channels for the
channels that instance can actually reach. The qwen35 lanes run SNC OFF (1 NUMA node),
where all 224 cores interleave across all 16 channels.

Peaks are per-box and must be passed in; the defaults describe DMR-X4PT -- 224c, 16
channels of DDR8000, and 1280 MiB of L3 shared by four 56-core instances. AMX-bf16 is
taken as 1024 FLOP/cycle/core (one TDPBF16PS on 16x32 x 32x16 = 16384 FLOP per 16
cycles), which is the SPR/GNR figure -- pass --amx-flops-per-cycle if the part differs.

Usage:
  python3 audit_plausibility.py results.csv
  python3 audit_plausibility.py results.csv --ghz 2.8 --ddr-channels 8 --llc-mib 240
"""

import argparse
import collections
import csv
import math
import sys

MIB = 1024 ** 2


def weight_bytes_per_iter(row) -> int:
    """Bytes of expert weight one fused iteration must read: every ACTIVE expert's
    gate+up [2N, K] and down [K, N], bf16. Idle experts are never touched, so the
    active count -- not E -- is what streams."""
    K = int(row["hidden_size"])
    N = int(row["moe_intermediate_size"])
    active = int(row["active_experts"])
    return 2 * active * (2 * N * K + K * N)


def lane_key(row):
    """One timed group: `time_mixed` interleaves every cell of one lane invocation on
    one instance, so the label and the instance index bound the reuse distance."""
    return (row.get("label") or "", row.get("instance") or "0")


def reuse_distance_bytes(rows) -> int:
    """Bytes an instance walks between two visits to one cell's weight set.

    `copies` is in RESULT_FIELDS, so a current CSV records it; an older file without
    the column falls back to 1, which UNDERSTATES the distance and can therefore only
    make this audit call a streaming cell cache-resident, never the reverse.
    """
    copies = max((int(r["copies"]) for r in rows if r.get("copies")), default=1)
    return sum(weight_bytes_per_iter(r) for r in rows) * copies


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
    p.add_argument("--ddr-channels", type=int, default=16,
                   help="memory channels the timed instances can reach (X4PT: 16 with "
                        "SNC off; 4 for one instance pinned to a single SNC4 node)")
    p.add_argument("--ddr-mts", type=float, default=8000.0,
                   help="DIMM transfer rate, MT/s (X4PT: DDR8000)")
    p.add_argument("--ddr-gbs", type=float, default=None,
                   help="socket DDR peak, GB/s; overrides the peak derived from "
                        "--ddr-channels x --ddr-mts x 8 B")
    p.add_argument("--llc-mib", type=float, default=320.0,
                   help="LLC an instance owns, MiB. The default splits X4PT's 1280 MiB "
                        "of L3 across four 56-core instances; pass the whole 1280 to "
                        "ask the question against an uncontended socket")
    p.add_argument("--instances", type=int, default=None,
                   help="concurrent instances sharing DDR (default: max `instance`+1 "
                        "seen in the file)")
    args = p.parse_args()

    per_channel = args.ddr_mts * 8 / 1000.0
    ddr_gbs = args.ddr_gbs or args.ddr_channels * per_channel
    llc_bytes = args.llc_mib * MIB

    with open(args.csv_file, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("fused_median_ms")]
    if not rows:
        raise SystemExit(f"{args.csv_file}: no rows with a fused_median_ms")
    inst = args.instances or max(int(r.get("instance") or 0) for r in rows) + 1

    lanes = collections.defaultdict(list)
    for r in rows:
        lanes[lane_key(r)].append(r)
    reuse = {k: reuse_distance_bytes(v) for k, v in lanes.items()}

    src = "given" if args.ddr_gbs else f"{args.ddr_channels}ch x {args.ddr_mts:.0f} MT/s"
    print(f"{len(rows)} timed rows, assuming {inst} concurrent instance(s), "
          f"{args.ghz} GHz, {args.amx_flops_per_cycle:.0f} FLOP/cycle/core AMX bf16, "
          f"DDR peak {ddr_gbs:.0f} GB/s ({src}), LLC {args.llc_mib:.0f} MiB/instance\n")
    if not any(r.get("copies") for r in rows):
        print("NOTE: this CSV records no `copies` column, so every reuse distance below "
              "assumes one weight set. That is a LOWER BOUND: it can only overstate\n"
              "      cache residency, never hide it.\n")
    hdr = (f"{'phase':<8}{'bs':>5}{'tp':>3}{'L':>4}{'thr':>5}{'cop':>4}"
           f"{'GFLOP':>8}{'ms':>9}{'GFLOP/s':>10}{'%peak':>7}"
           f"{'impl GB/s':>10}{'%DDR':>7}{'xLLC':>8}  verdict")
    print(hdr)
    print("-" * len(hdr))
    over_peak, cached, over_ddr = [], [], []
    for r in rows:
        ms = float(r["fused_median_ms"])
        gflop = float(r["gflop"])
        cores = args.cores or int(r.get("threads") or 0) or 56
        copies = r.get("copies") or "?"
        peak = cores * args.amx_flops_per_cycle * args.ghz  # GFLOP/s
        rate = gflop / (ms / 1e3)
        pct_peak = 100.0 * rate / peak
        gbs = weight_bytes_per_iter(r) * inst / (ms / 1e3) / 1e9
        pct_ddr = 100.0 * gbs / ddr_gbs
        x_llc = reuse[lane_key(r)] / llc_bytes
        if pct_peak > 100:
            verdict = "IMPOSSIBLE: over AMX peak"
            over_peak.append(r)
        elif x_llc <= 1.0:
            verdict = "cache-resident: weight set fits the LLC"
            cached.append(r)
        elif pct_ddr > 100:
            verdict = f"over assumed DDR peak ({pct_ddr / 100:.1f}x)"
            over_ddr.append((r, gbs))
        elif pct_ddr > 70:
            verdict = "DDR-bound"
        else:
            verdict = "compute/latency-bound"
        print(f"{r['phase']:<8}{r['batch']:>5}{r['tp']:>3}{r['layer']:>4}"
              f"{cores:>5}{str(copies):>4}{gflop:>8.2f}{ms:>9.3f}{rate:>10.1f}"
              f"{pct_peak:>7.1f}{gbs:>10.1f}{pct_ddr:>7.1f}{x_llc:>8.1f}  {verdict}")

    print()
    if over_peak:
        print(f"FAIL: {len(over_peak)} row(s) report more FLOP/s than the machine can "
              f"issue. The FLOP count or the timed region is wrong.")
        return 1
    print("OK: no row exceeds AMX peak.")
    if cached:
        print(f"NOTE: {len(cached)}/{len(rows)} row(s) have a weight reuse distance "
              f"inside the {args.llc_mib:.0f} MiB LLC slice, so those cells measured a "
              f"cache-resident kernel. Raise --copies for those shapes.")
    if over_ddr:
        worst = max(g for _, g in over_ddr)
        need = math.ceil(worst / per_channel)
        print(f"NOTE: {len(over_ddr)}/{len(rows)} row(s) imply up to {worst:.0f} GB/s, "
              f"above the assumed {ddr_gbs:.0f} GB/s, while walking far more than the "
              f"LLC between reuses.\n"
              f"      Those weights came from DRAM, so it is the ASSUMED PEAK that is "
              f"too low, not --copies: {worst:.0f} GB/s needs at least {need} channels "
              f"at {args.ddr_mts:.0f} MT/s.")
    if not cached and not over_ddr:
        print("OK: every row's weights streamed from DDR within the assumed peak.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
