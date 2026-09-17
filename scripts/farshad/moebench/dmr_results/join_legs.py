#!/usr/bin/env python3
"""Join the fused and batched legs of one results.csv on (phase, batch, layer, tp).

fused_over_batched is empty by design -- it is computed within one invocation and the
legs are separate sweeps -- so the ratio has to be recovered by this join. The slowest
instance is taken on each side, because that is what a rank actually waits for.
"""
import collections
import csv
import statistics
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
rows = list(csv.DictReader(open(path)))


def leg(r):
    return "fused" if r["label"].endswith("_fused") else "batched"


def key(r):
    return (r["phase"], r["batch"], r["layer"], r["tp"])


m = collections.defaultdict(dict)
for r in rows:
    t = r.get("fused_median_ms") or r.get("batched_median_ms")
    if t:
        m[key(r)].setdefault(leg(r), []).append(float(t))

fam = collections.defaultdict(list)
paired = 0
for k, v in m.items():
    if "fused" in v and "batched" in v:
        f, b = max(v["fused"]), max(v["batched"])
        fam[(k[0], k[1], k[3])].append((f, b, f / b))
        paired += 1

print("per-cell medians over the layers of each cell (slowest instance per layer):")
print(f"{'phase':8s} {'bs':>5s} {'tp':>3s} {'layers':>7s} {'fused ms':>11s} {'batched ms':>12s} {'f/b':>7s} {'range f/b':>16s}")
for k, v in sorted(fam.items()):
    fs = statistics.median(x[0] for x in v)
    bs = statistics.median(x[1] for x in v)
    rr = [x[2] for x in v]
    print(f"{k[0]:8s} {k[1]:>5s} {k[2]:>3s} {len(v):7d} {fs:11.3f} {bs:12.3f} "
          f"{statistics.median(rr):7.3f} {min(rr):7.3f}..{max(rr):<7.3f}")

print()
print(f"paired cells: {paired}   total csv rows: {len(rows)}")
print("rows per label:")
for lab, n in sorted(collections.Counter(r["label"] for r in rows).items()):
    print(f"  {lab:32s} {n}")
