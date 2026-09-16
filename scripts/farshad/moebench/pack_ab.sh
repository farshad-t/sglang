#!/bin/bash
# Is the benchdnn-vs-bmm gap the missing VNNI prepack, or torch?
#
# benchdnn ran `--wtag=any`, which lets oneDNN choose a VNNI-packed B layout. The batched
# leg hands torch.bmm a plain [nexp, K, N] weight and torch has no way to pass a
# pre-reordered B, so it pays any repacking on every call. Re-running the SAME cases with
# `--wtag` pinned to the PLAIN layout measures that term inside one instrument.
#
# Same instrument as step B: 4 concurrent 56-core instances, membind 0, free-landing.
# Collapse: sum a layer's kernels WITHIN an instance, then median over the four.
set -uo pipefail
ROOT=/home/farshad/benchdnn_work/cliffbug_x4pt
BENCHDNN=$ROOT/onednn_x4/bin/benchdnn_nordpmc
NUMACTL=$ROOT/bin/numactl
export LD_LIBRARY_PATH=$ROOT/onednn_x4/lib
CASES=${1:?casefile: one "SHAPE COLD_CACHE TAG" per line}
OUT=${2:?outdir}
BUDGET=${BUDGET:-2000}
RANGES=(0-55 56-111 112-167 168-223)
mkdir -p "$OUT"
PT='perf,%impl%,%prb%,%-time%,%0time%,%+time%'

n=0
while read -r SHAPE CC TAG; do
  case "$SHAPE" in ''|'#'*) continue;; esac
  n=$((n+1))
  for WTAG in any $TAG; do
    for r in 0 1 2 3; do
      "$NUMACTL" --physcpubind="${RANGES[$r]}" --membind 0 \
        "$BENCHDNN" --mode=P --max-ms-per-prb="$BUDGET" --matmul \
        --cold-cache="$CC" --dt=bf16:bf16:bf16 --dtag="$TAG" --stag="$TAG" --wtag="$WTAG" \
        --perf-template="$PT" "$SHAPE" > "$OUT/c${n}_w${WTAG}_r${r}.log" 2>&1 &
    done
    wait
  done
  echo "[$n] $SHAPE cc=$CC tag=$TAG done"
done < "$CASES"
echo "PACK_AB_DONE cases=$n"
