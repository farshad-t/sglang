#!/usr/bin/env python3
"""Parse a step-B benchdnn run directory into the raw per-case CSV.

One row per case (not per instance). The four 56-core instances are the four TP4 ranks /
TP1 replicas that ran concurrently, so the row carries:
  bdnn_min_ms   -- SLOWEST instance's min_time. This is the comparable number: a rank
                   waits for the slowest peer, and the fused lane harness reports the
                   same way.
  bdnn_min_ms_fastest / _mean  -- the spread across instances, i.e. how unequal the four
                   56-core groups were. A wide spread means uncore/DDR contention.
Achieved core MHz is windowed from the 1 Hz sysfs freq trace over each case's
[t_start, t_end], per instance core range (`perf` is absent on this box).
"""
import argparse, csv, os, re, statistics, sys

ap = argparse.ArgumentParser()
ap.add_argument("--rundir", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

# ---- freq trace: epoch mean224 r0 r1 r2 r3 pkgtemp -----------------------------------
trace = []
tp = os.path.join(a.rundir, "freq_trace.log")
if os.path.exists(tp):
    for ln in open(tp):
        f = ln.split()
        if len(f) == 7:
            try:
                trace.append(tuple(int(x) for x in f))
            except ValueError:
                pass
trace.sort()


def freq_window(t0, t1):
    """mean/max over the samples inside [t0, t1]. Falls back to the nearest sample when a
    case was shorter than the 1 Hz sampling period."""
    sel = [s for s in trace if t0 <= s[0] <= t1]
    if not sel and trace:
        sel = [min(trace, key=lambda s: abs(s[0] - t0))]
    if not sel:
        return {}
    n = len(sel)
    out = {
        "n_freq_samples": n,
        "core_mhz_mean": round(sum(s[1] for s in sel) / n / 1000, 1),
        "core_mhz_min": round(min(s[1] for s in sel) / 1000, 1),
        "core_mhz_max": round(max(s[1] for s in sel) / 1000, 1),
        "pkg_temp_max": max(s[6] for s in sel),
    }
    for r in range(4):
        out[f"core_mhz_mean_r{r}"] = round(sum(s[2 + r] for s in sel) / n / 1000, 1)
    return out


# ---- progress.log: idx t_start t_end wall shape cc tag rc0..rc3 ----------------------
prog = {}
for ln in open(os.path.join(a.rundir, "progress.log")):
    if ln.startswith("#"):
        continue
    f = ln.split()
    if len(f) != 11:
        continue
    prog[int(f[0])] = dict(t0=int(f[1]), t1=int(f[2]), wall_s=int(f[3]), shape=f[4],
                           cold_cache=f[5], dtag=f[6], rcs=[int(x) for x in f[7:11]])

# ---- per-instance perf lines --------------------------------------------------------
# perf,<impl>,<prb -- CONTAINS SPACES BUT NO COMMAS>,min,avg,max,ib,ob,iob,minGbw,avgGbw
# The prb field embeds the whole benchdnn command line, so split on commas and take the
# numeric tail from the right rather than trusting a fixed field index.
PERF = re.compile(r"^perf,(?P<impl>[^,]*),(?P<prb>.*),"
                  r"(?P<min>[\d.eE+-]+),(?P<avg>[\d.eE+-]+),(?P<max>[\d.eE+-]+),"
                  r"(?P<ib>[\d.eE+-]+),(?P<ob>[\d.eE+-]+),(?P<iob>[\d.eE+-]+),"
                  r"(?P<mingbw>[\d.eE+-]+),(?P<avggbw>[\d.eE+-]+)$")

logdir = os.path.join(a.rundir, "logs")
rows, problems = [], []
for idx in sorted(prog):
    p = prog[idx]
    inst = []
    for r in range(4):
        lp = os.path.join(logdir, f"c{idx}_r{r}.log")
        hit = None
        if os.path.exists(lp):
            for ln in open(lp):
                if ln.startswith("perf,") and not ln.startswith("perf,impl,"):
                    m = PERF.match(ln.strip())
                    if m:
                        hit = m
        if hit is None:
            inst.append(None)
        else:
            inst.append(dict(impl=hit["impl"], min_ms=float(hit["min"]),
                             avg_ms=float(hit["avg"]), max_ms=float(hit["max"]),
                             ib=float(hit["ib"]), ob=float(hit["ob"]),
                             iob=float(hit["iob"]), min_gbw=float(hit["mingbw"])))
    ok = [x for x in inst if x]
    if len(ok) != 4:
        problems.append((idx, p["shape"], f"{len(ok)}/4 instances returned a perf line"))
    if not ok:
        continue
    mins = [x["min_ms"] for x in ok]
    impls = sorted({x["impl"] for x in ok})
    row = dict(
        idx=idx, bdnn_shape=p["shape"], cold_cache=p["cold_cache"], dtag=p["dtag"],
        n_instances=len(ok), impl="|".join(impls),
        bdnn_min_ms=max(mins),                       # slowest instance -> what a rank waits for
        bdnn_min_ms_fastest=min(mins),
        bdnn_min_ms_mean=round(sum(mins) / len(mins), 4),
        bdnn_spread_pct=round(100 * (max(mins) - min(mins)) / min(mins), 2) if min(mins) else "",
        bdnn_avg_ms=max(x["avg_ms"] for x in ok),
        ibytes=ok[0]["ib"], obytes=ok[0]["ob"], iobytes=ok[0]["iob"],
        bdnn_min_Gbw=max(x["min_gbw"] for x in ok),
        bdnn_min_ms_median=round(statistics.median(mins), 4),
        wall_s=p["wall_s"], t_start=p["t0"], t_end=p["t1"],
        rc=",".join(str(x) for x in p["rcs"]),
    )
    # Per-instance min-times are kept unreduced so a consumer can sum a layer's kernels
    # WITHIN one 56-core group and only then collapse the four; collapsing per case first
    # is a different statistic.
    for r in range(4):
        row[f"bdnn_min_ms_r{r}"] = inst[r]["min_ms"] if inst[r] else ""
    row.update(freq_window(p["t0"], p["t1"]))
    rows.append(row)

cols = ["idx", "bdnn_shape", "cold_cache", "dtag", "n_instances", "impl",
        "bdnn_min_ms", "bdnn_min_ms_fastest", "bdnn_min_ms_mean", "bdnn_min_ms_median",
        "bdnn_min_ms_r0", "bdnn_min_ms_r1", "bdnn_min_ms_r2", "bdnn_min_ms_r3",
        "bdnn_spread_pct",
        "bdnn_avg_ms", "bdnn_min_Gbw", "ibytes", "obytes", "iobytes",
        "core_mhz_mean", "core_mhz_min", "core_mhz_max",
        "core_mhz_mean_r0", "core_mhz_mean_r1", "core_mhz_mean_r2", "core_mhz_mean_r3",
        "pkg_temp_max", "n_freq_samples", "wall_s", "t_start", "t_end", "rc"]
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f"parsed {len(rows)} cases -> {a.out}", file=sys.stderr)
if problems:
    print(f"PROBLEMS ({len(problems)}):", file=sys.stderr)
    for idx, shape, why in problems[:40]:
        print(f"  idx={idx} {shape}: {why}", file=sys.stderr)
