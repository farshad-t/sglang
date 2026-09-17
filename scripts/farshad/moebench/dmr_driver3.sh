#!/usr/bin/env bash
# Drive the THREE legs of the routed-MoE experiment into one results.csv:
#
#   fused m0   fused_experts_cpu as it ships -- gather, GEMMs with SwiGLU folded into the
#              accumulator, topk-weighted scatter
#   fused m2   the same kernel with the gather and the reduce dropped
#   batched    torch.bmm per histogram bucket + silu(gate)*up + torch.bmm
#
# m2 is what makes the comparison attributable: it does the same work the batched leg
# does, so `batched / m2` is about the matmul and nothing else, while `m0 - m2` prices the
# kernel's own gather and reduce. A straight m0-vs-batched ratio cannot separate the two.
#
# The legs run as SEPARATE sweeps so only one leg's weight pool is ever resident, and so a
# failure in a later leg cannot cost the earlier legs' numbers. They share OUT, hence one
# results.csv; per-invocation files are copied aside per leg before the next one overwrites
# them. Detached on purpose: no interactive session owns this.
#
# The legs run in a fixed order and the box drifts over the sweep, so read `thermal.log`
# before attributing a small m0-vs-m2 difference to the kernel.
#
# Env:
#   OUT=<dir>                       output dir                        (required, arg 1)
#   HARNESS=<dir>                   where run_dmr_lanes.sh lives      (default: this dir)
#   PY=<python>                     interpreter for the sweep
#   MOEBENCH_KERNEL_SO_DIR=<dir>    a locally built common_ops, for a box with no wheel
#   AB="..."                        stats-source flags
#   LAYERS=all|0|0,10,...           --layer for the 35B lanes
#   LANES="35b_rt 35b_thr"          lane subset
#   LEGS="m0 m2 batched"            which legs to run, in order
set -uo pipefail

OUT=${1:?output dir}
HARNESS=${HARNESS:-$(cd "$(dirname "$0")" && pwd)}
PY=${PY:-python3}
AB=${AB:-}
LAYERS=${LAYERS:-all}
LANES=${LANES:-"35b_rt 35b_thr"}
LEGS=${LEGS:-"m0 m2 batched"}

# Both reach the sweep through the environment: the stats cache so the run needs no
# network, the so dir so bench_moe_cpu.py can import a kernel that has no wheel.
export MOEBENCH_CACHE=${MOEBENCH_CACHE:-$HOME/.cache/moebench/archbench}
[ -n "${MOEBENCH_KERNEL_SO_DIR:-}" ] && export MOEBENCH_KERNEL_SO_DIR

cd "$HARNESS" || exit 2
mkdir -p "$OUT"

echo "driver3 start $(date -Is)  OUT=$OUT  harness=$HARNESS  git=$(git rev-parse --short HEAD)"
echo "  PY=$PY  so_dir=${MOEBENCH_KERNEL_SO_DIR:-<none>}  LAYERS=$LAYERS  LANES=$LANES"
echo "  legs=$LEGS"

t0=$(date +%s)
rcs=""
for leg in $LEGS; do
  case "$leg" in
    m0)      mode=fused;   fmode=0 ;;
    m2)      mode=fused;   fmode=2 ;;
    batched) mode=batched; fmode=0 ;;
    *) echo "unknown leg $leg" >&2; exit 2 ;;
  esac
  echo
  echo "==================== LEG $leg (MODE=$mode FUSED_MODE=$fmode)  $(date -Is) ===================="
  t=$(date +%s)
  MODE=$mode FUSED_MODE=$fmode OUT="$OUT" PY="$PY" AB="$AB" LAYERS="$LAYERS" \
      LANES="$LANES" bash run_dmr_lanes.sh
  rc=$?
  echo "==================== LEG $leg exit=$rc  $(( $(date +%s) - t ))s  $(date -Is) ===="
  rcs="$rcs ${leg}_rc=$rc"

  # env_{before,after}.txt are per-invocation; the next leg would overwrite them.
  for f in env_before.txt env_after.txt; do
    [ -f "$OUT/$f" ] && cp -p "$OUT/$f" "$OUT/${f%.txt}_${leg}.txt"
  done
  cp -p "$OUT/results.csv" "$OUT/results_through_${leg}.csv" 2>/dev/null

  if [ "$rc" -ne 0 ]; then
    echo "LEG $leg FAILED (exit $rc) -- not starting the remaining legs, so a partial"
    echo "results.csv cannot be mistaken for a complete one."
    echo "$rcs remaining=skipped" > "$OUT/DRIVER_DONE"
    exit "$rc"
  fi
done

echo "$rcs total=$(( $(date +%s) - t0 ))s" > "$OUT/DRIVER_DONE"
echo "driver3 done $(date -Is)  total $(( $(date +%s) - t0 ))s"
