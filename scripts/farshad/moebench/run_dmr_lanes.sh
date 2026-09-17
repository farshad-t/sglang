#!/usr/bin/env bash
# Run the bf16 lanes of the qwen35 DMR-X4 daily run through the MoE/FFN kernel
# benchmark, so each lane's projected layer cost has a measured counterpart.
#
# Lanes from tools/cpu/config/qwen35/qwen35_dmrx4_lanes_20260903_1119_collmodes_shmprofile.csv
# (all share input_seq_len=1024, output_seq_len=1024, gen_start=gen_end=1536,
# hw dmr_ap_x4_16ch_224c_128cbo_ddr8000):
#
#   lane 3/4   qwen35_035b_bs001_TP4_bf16_a003b    rt   bs1    TP4  numUnits [56]
#   lane 9/10  qwen35_035b_bs320_TP1_bf16_a003b    thr  bs320  TP1  (4 replicas x 56c)
#   lane 1/2   qwen35_009b_bs001_TP4_bf16          rt   bs1    TP4  (DENSE model)
#   lane 7/8   qwen35_009b_bs096_TP1_bf16          thr  bs96   TP1  (DENSE model)
#
# ============================ WHY 4 CONCURRENT INSTANCES =======================
# A 56-core run with the other 168 cores IDLE is optimistic: that slice gets memory
# bandwidth, LLC and power headroom that do not exist when the socket is actually
# serving. So every rt cell runs as FOUR concurrent instances pinned to 0-55,
# 56-111, 112-167, 168-223, started together and joined with a `wait` barrier, all
# doing the SAME shape -- which is also physically what TP4 is: four ranks running
# the same layer at the same time. Per-instance times are reported separately; the
# slowest instance is what a rank actually waits for.
#
# The thr lanes are ALSO four concurrent 56-core instances, for a different reason:
# TP1 means no sharding, so a throughput deployment on this socket is four INDEPENDENT
# model replicas, one per 56-core group, each running the FULL unsharded layer
# (N=512). Same core layout as rt, opposite sharding. A single 224-core instance is
# NOT the throughput lane; it is available as 35b_thr_socket for contrast only.
#
# SNC is OFF on this box (1 NUMA node, cores 0-223), and numactl is not installed,
# so pinning is taskset + OMP_PLACES/OMP_PROC_BIND. There is no per-node membind to
# do, and memory is not channel-partitioned by SNC node anyway.
#
# ============================ TP SHARDING ======================================
# The projection DOES shard the MoE intermediate dim. Qwen3_5MoeExperts passes
# config.moe_intermediate_size through undivided, but the division happens one level
# down in BatchedSwiGLU.forward (common/experts.py:50):
#     local_hidden_dim = math.ceil(self.hidden_dim / self.args.model_parallel_size)
# where `hidden_dim` IS the moe intermediate size. Dense FFNs shard the same way in
# FeedForward.forward (Llama4/ffn.py:45). Each rank keeps all E experts and holds
# N/TP of the intermediate (row-parallel down_proj), and the MoE block owns ONE
# all-reduce over the routed+shared expert sum -- each expert is built with
# collective_tp_size=1 precisely so that AR is not charged per expert.
# So the rt lanes use --tp 4 (N = 512/4 = 128), which is BOTH what a real sglang TP4 rank
# runs and what the projection charges per rank -- the setting a model-vs-measurement
# compare has to use. Realtime is TP4 only; there is deliberately no --tp 1 rt lane, since
# N=512 at bs1 is a shape nothing deploys and it cost 4x the rt sweep (402 GiB of cycled
# expert weights against 100 GiB) to answer a question about split cost rather than about
# the decomposition. The bs320 thr lanes are TP1 because a throughput deployment on this
# socket is four INDEPENDENT replicas, which is a real configuration.
#
# ============================ POWER / STATE ====================================
# The governor is read and recorded before and after, and the run REFUSES to start if
# it is not `performance` or if the box is busy. It is deliberately never CHANGED:
# this is a SHARED box (other user `sdp`), the governor is system-wide, and setting it
# needs sudo. If it ever reads back as not-performance, ask the owner rather than
# forcing it. FORCE=1 skips both guards.
#
# Knobs (env):
#   OUT=<dir>            output dir                  (default results_<timestamp>)
#   CORES=56             cores per instance
#   SPREAD_RT=4          concurrent instances for rt lanes
#   SPREAD_THR=4         concurrent instances for thr lanes
#   THREADS_THR=224      threads for the single-instance 35b_thr_socket contrast
#   LAYERS=all           --layer for the 35B lanes (e.g. 0,10,20,30,39 to sample)
#   MAXOTHER_CPU=200     refuse to start if OTHER users are burning more than this
#                        much CPU (percent; 200 = 2 cores). Load average is not used
#                        for the decision -- see the guard.
#   MODE=fused|batched   override every lane's --mode, so the whole sweep can be run as
#                        ONE leg at a time (MODE=fused ... then MODE=batched ...). Default
#                        empty = each lane keeps the mode written into it. Running the legs
#                        as separate sweeps costs one extra pool build per lane but keeps
#                        only one leg's weights resident, and it means a failure in the
#                        second leg cannot cost you the first leg's numbers.
#   FUSED_MODE=0|2       fused_experts_cpu's expert_batching_mode for the whole sweep.
#                        2 drops the gather and the topk-weighted reduce, so
#                        `MODE=fused FUSED_MODE=2` measures the same work `MODE=batched`
#                        does and the two become comparable, while mode 0 minus mode 2 is
#                        what the kernel's own gather and reduce cost. Lane names get an
#                        `_m2` suffix, and --check is dropped because mode 2's output is
#                        wrong by construction. Needs a build carrying the argument.
#   AB="..."             extra stats-source flags, e.g. AB="--ab-offline"
#   PY=python3           interpreter (used when IMG is empty)
#   IMG=<image>          run each cell inside this container instead of on the metal.
#                        The BKC is what serving actually runs, so a measured number is
#                        only comparable to a published qwen35 KPI if it came from it.
#   HOSTROOT=/home/farshad/moebench   mounted at /moebench in the container; OUT and
#                        this script must live under it
#   COPIES_RT=8          private weight sets per cell for the rt lanes, cycled so the
#                        weights are read from DDR rather than out of the 320 MiB L3
#                        slice an instance owns. rt decode reads only 12 MiB of a set.
#   COPIES_THR=2         same for the thr lanes, which need far fewer: one iteration
#                        there already reads 1.37 GiB of experts
#   COOLDOWN=15          seconds of idle between cells, for the part to settle
#   TEMP_MAX=80          do not start the next cell above this package temp (C)
#   COOLDOWN_MAX=600     give up waiting to cool after this long, and say so
#   FORCE=1              skip the busy/governor guards
#   LANES="..."          subset of: 35b_rt 35b_thr 35b_thr_socket
#                        35b_shared_rt 35b_shared_thr 9b_rt 9b_thr
#                        DEFAULT is the routed-MoE deployment lanes: 35b_rt 35b_thr.
#                        The shared-expert and 9B-dense lanes are opt-in.
set -euo pipefail
cd "$(dirname "$0")"

OUT=${OUT:-results_$(date +%Y%m%d_%H%M%S)}
CORES=${CORES:-56}
SPREAD_RT=${SPREAD_RT:-4}
SPREAD_THR=${SPREAD_THR:-4}
THREADS_THR=${THREADS_THR:-224}
LAYERS=${LAYERS:-all}
MAXOTHER_CPU=${MAXOTHER_CPU:-200}
AB=${AB:-}
MODE=${MODE:-}
FUSED_MODE=${FUSED_MODE:-}
PY=${PY:-python3}
IMG=${IMG:-}
HOSTROOT=${HOSTROOT:-/home/farshad/moebench}
COPIES_RT=${COPIES_RT:-8}
COPIES_THR=${COPIES_THR:-2}
COPIES=${COPIES:-$COPIES_RT}
FORCE=${FORCE:-0}
COOLDOWN=${COOLDOWN:-15}
TEMP_MAX=${TEMP_MAX:-80}
COOLDOWN_MAX=${COOLDOWN_MAX:-600}
# DEFAULT SET = the ROUTED MoE only. The question this benchmark exists to answer is
# whether the projection's per-bucket decomposition is a good model of the ROUTED
# experts, and whether sglang's fused kernel beats that decomposition. The shared
# expert is deliberately OUT: it is the same dense SwiGLU in the archbench model and in
# sglang, so it contributes the same term to both sides and cancels out of the
# comparison -- including it can only dilute the ratio being measured. It also would
# not be measured honestly here (see the 35b_shared_* block below). Ask for it by name
# if you ever want it.
LANES=${LANES:-"35b_rt 35b_thr"}
FAILED_LANES=""

# cpath() only rewrites a path under $HOSTROOT and `docker run` passes no -w, so a
# relative OUT resolves against the IMAGE's workdir instead of the mount: write_rows
# then raises FileNotFoundError AFTER the entire sweep, in a --rm container. Refuse now.
if [ -n "$IMG" ]; then
  case "$OUT" in
    "$HOSTROOT"/*) ;;
    *) echo "IMG is set but OUT=$OUT is not under HOSTROOT=$HOSTROOT, so the container" >&2
       echo "cannot write there (it sees only $HOSTROOT mounted at /moebench)." >&2
       echo "Use OUT=$HOSTROOT/<dir>." >&2; exit 2 ;;
  esac
  case "$PWD" in
    "$HOSTROOT"/*|"$HOSTROOT") ;;
    *) echo "IMG is set but this script lives outside HOSTROOT=$HOSTROOT ($PWD), so the" >&2
       echo "container would be handed an unmounted path for bench_moe_cpu.py." >&2; exit 2 ;;
  esac
fi

mkdir -p "$OUT"

gov() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown; }

# Hottest package across the SNC die regions. x86_pkg_temp is one zone per region.
pkgtemp() {
  local m=0 t
  for z in /sys/class/thermal/thermal_zone*; do
    [ -r "$z/temp" ] || continue
    [ "$(cat "$z/type" 2>/dev/null)" = "x86_pkg_temp" ] || continue
    t=$(( $(cat "$z/temp") / 1000 )); [ "$t" -gt "$m" ] && m=$t
  done
  echo "$m"
}

# Throttle counters are the evidence that a cell was derated. A cell whose count
# moved was measured on a clock the next cell will not see, so it is not comparable.
throttles() {
  echo "core=$(cat /sys/devices/system/cpu/cpu0/thermal_throttle/core_throttle_count 2>/dev/null || echo NA)"\
"/pkg=$(cat /sys/devices/system/cpu/cpu0/thermal_throttle/package_throttle_count 2>/dev/null || echo NA)"
}

# Idle between cells so the part settles, and refuse to start the next one while it is
# still hot. Sustained 224-core AMX is the heaviest thing this box will ever run, and a
# cell begun on an already-saturated thermal budget both risks the machine and measures
# a clock the previous cell did not see.
cool_down() {
  local label="$1" t0 waited t
  t0=$(date +%s)
  while :; do
    t=$(pkgtemp); waited=$(( $(date +%s) - t0 ))
    if [ "$t" -le "$TEMP_MAX" ] && [ "$waited" -ge "$COOLDOWN" ]; then
      echo "  cooled: ${t}C after ${waited}s idle (limit ${TEMP_MAX}C, min ${COOLDOWN}s)"
      break
    fi
    if [ "$waited" -ge "$COOLDOWN_MAX" ]; then
      echo "  WARNING: still ${t}C after ${waited}s, proceeding anyway (COOLDOWN_MAX)"
      break
    fi
    sleep 10
  done
  echo "$(date -Is) after=$label temp=${t}C throttle=$(throttles)" >> "$OUT/thermal.log"
}
epp() { cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null || echo unknown; }

# Host path -> the path the same file has inside the container ($HOSTROOT is mounted
# at /moebench). A no-op when running on the metal.
cpath() { [ -n "$IMG" ] && echo "${1/#$HOSTROOT//moebench}" || echo "$1"; }

# One cell, on the metal or in the BKC container. The guards and thermal reads stay on
# the HOST: /sys is what the host exposes, and `ps` inside a container cannot see the
# other tenant the busy guard exists to catch.
runpy() {
  if [ -z "$IMG" ]; then
    $PY bench_moe_cpu.py "$@"
  else
    docker run --rm --pid=host --user "$(id -u):$(id -g)" \
      -v "$HOSTROOT:/moebench" -v "$HOME/.cache/moebench:/cache/moebench" \
      -e HOME=/tmp -e MOEBENCH_CACHE=/cache/moebench/archbench \
      -e LD_LIBRARY_PATH=/opt/conda/envs/sglang/lib \
      -e MOEBENCH_WAIT_POLICY -e MOEBENCH_BLOCKTIME \
      "$IMG" /opt/conda/envs/sglang/bin/python \
      "$(cpath "$PWD")/bench_moe_cpu.py" "$@"
  fi
}

runpy_version() {
  if [ -z "$IMG" ]; then
    $PY -c 'import torch;print(torch.__version__)' 2>/dev/null || echo n/a
  else
    docker run --rm -e LD_LIBRARY_PATH=/opt/conda/envs/sglang/lib "$IMG" \
      /opt/conda/envs/sglang/bin/python -c 'import torch;print(torch.__version__)' \
      2>/dev/null || echo n/a
  fi
}

GOV_BEFORE=$(gov)
{
  echo "host        : $(hostname)"
  echo "date        : $(date -Is)"
  echo "nproc       : $(nproc)"
  echo "uptime      : $(uptime)"
  echo "governor    : $GOV_BEFORE   (epp $(epp))"
  echo "no_turbo    : $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo n/a)"
  echo "tuned       : $(tuned-adm active 2>/dev/null | head -1 || echo n/a)"
  echo "numa        :"; (numactl --hardware 2>/dev/null || echo "  numactl absent; NUMA node(s) = $(lscpu | awk -F: '/NUMA node\(s\)/{print $2}' | tr -d ' ')") | sed 's/^/  /'
  echo "cpu         :"; lscpu 2>/dev/null | grep -E '^(Model name|Socket|Core|Thread|NUMA node\(s\)|CPU max)' | sed 's/^/  /'
  echo "bios        : $(cat /sys/class/dmi/id/bios_version 2>/dev/null || echo n/a)"
  echo "cores/inst  : $CORES"
  echo "cooldown    : ${COOLDOWN}s min between cells, wait until <=${TEMP_MAX}C (cap ${COOLDOWN_MAX}s)"
  echo "pkg temp now: $(pkgtemp)C   throttle: $(throttles)"
  echo "spread rt   : $SPREAD_RT   thr: $SPREAD_THR"
  echo "layers      : $LAYERS"
  echo "copies      : $COPIES   (mix-layers on: one visit per layer per iteration)"
  echo "runtime     : ${IMG:-bare metal $PY}"
  echo "image id    : $([ -n "$IMG" ] && docker image inspect "$IMG" --format '{{.Id}}' 2>/dev/null || echo n/a)"
  echo "git         : $(git rev-parse HEAD 2>/dev/null || echo n/a)"
  echo "torch       : $(runpy_version)"
} | tee "$OUT/env_before.txt"

# ---- guards -------------------------------------------------------------------
# "Is the box busy?" means "is someone ELSE using it", and the 1-min load average
# cannot answer that: it decays over minutes, so it still reads 60+ right after our
# own previous sweep exits. Using it refused three legitimate launches in a row on an
# otherwise idle machine. So the decision is made on CPU currently burned by other
# users; load is recorded for context but never gates.
other_cpu() {
  ps -eo user,pcpu --no-headers 2>/dev/null \
    | awk -v me="$(id -un)" '$1 != me { s += $2 } END { printf "%d", s + 0 }'
}

if [ "$FORCE" != "1" ]; then
  OTHER=$(other_cpu)
  L1=$(awk '{print $1}' /proc/loadavg)
  if [ "$OTHER" -gt "$MAXOTHER_CPU" ]; then
    echo "REFUSING: other users are burning ${OTHER}% CPU (> MAXOTHER_CPU" \
         "${MAXOTHER_CPU}%). A contended measurement is worse than none." >&2
    ps -eo user,pcpu,comm --sort=-pcpu --no-headers | head -6 >&2
    exit 1
  fi
  if [ "$GOV_BEFORE" != "performance" ]; then
    echo "REFUSING: governor is '$GOV_BEFORE', not 'performance'." >&2
    echo "          This is a SHARED box and the governor is system-wide, so this script" >&2
    echo "          will not change it. Ask the owner to set it, or FORCE=1 to measure" >&2
    echo "          anyway (and say so in the writeup)." >&2
    exit 1
  fi
  echo "guards ok: other-user CPU ${OTHER}% <= ${MAXOTHER_CPU}%, governor $GOV_BEFORE" \
       "(1-min load $L1, not gating)"
fi

# ---- one cell, N concurrent pinned instances ----------------------------------
# The concurrency lives INSIDE bench_moe_cpu.py (--instances), not in this shell, because
# instances that merely start together drift apart within a few iterations and a cell can
# then be timed while its neighbours sit between kernels -- measuring a partly-idle
# socket. The Python driver re-synchronises all instances on a barrier before EVERY
# iteration and sets OMP_WAIT_POLICY=active / KMP_BLOCKTIME=200 so idle threads BUSY-WAIT:
# the barrier keeps their idle windows short and aligned, where sleeping costs a wake-up
# on every kernel call. Each sweep prints its own across-instance imbalance; the routed
# lanes run near 1.01x median.
spread() {  # spread <name> <n_instances> <cores_per_instance> <args...>
  local name=$1 n=$2 cores=$3; shift 3
  # MODE, when set, replaces whatever --mode the lane asked for. A lane written as
  # `--mode fused` (because both legs would not fit) still runs under MODE=batched, which
  # is the point: the legs are separate sweeps, so each only ever holds its own pool.
  if [ -n "$MODE" ]; then
    local a=() skip=0
    for x in "$@"; do
      if [ "$skip" = 1 ]; then skip=0; continue; fi
      if [ "$x" = "--mode" ]; then skip=1; continue; fi
      a+=("$x")
    done
    set -- "${a[@]}" --mode "$MODE"
    name="${name}_${MODE}"
  fi
  # FUSED_MODE picks fused_experts_cpu's expert_batching_mode for this whole sweep, the
  # same way MODE picks the leg: mode 2 drops the gather and the topk-weighted reduce, so
  # `MODE=fused FUSED_MODE=2` is the sweep that pairs with `MODE=batched`. It also strips
  # --check, which the driver refuses under mode 2 because the output is unwritten.
  if [ -n "$FUSED_MODE" ] && [ "$FUSED_MODE" != 0 ]; then
    local b=()
    for x in "$@"; do
      if [ "$x" = "--check" ]; then continue; fi
      b+=("$x")
    done
    set -- "${b[@]}" --fused-mode "$FUSED_MODE"
    name="${name}_m${FUSED_MODE}"
  fi
  echo
  echo "################ $name   (${n} x ${cores}c concurrent, barrier-synced)"
  echo "$(date -Is) before=$name temp=$(pkgtemp)C throttle=$(throttles)" >> "$OUT/thermal.log"
  echo "  pre-cell: $(pkgtemp)C, throttle $(throttles)"
  # A lane that DIES must be visible and must not be mistaken for one that measured
  # nothing. `|| true` on the whole pipeline used to discard its status under
  # `set -o pipefail`, so `set -e` never fired: every lane could fail and the sweep
  # still exited 0 with an empty console. The `|| true` now belongs to grep alone
  # (it exits 1 when nothing matches), the driver's status comes from PIPESTATUS,
  # and the filter carries the strings the driver actually fails with.
  # `{ pipeline; } || rc=$?` is what keeps `set -e` from killing the sweep here while
  # still CAPTURING the driver's status: under `pipefail` the pipeline reports the
  # driver's non-zero exit, and the `||` makes it non-fatal. A bare trailing `|| true`
  # would swallow the status instead (and reset PIPESTATUS), which is how every lane
  # could fail while the sweep exited 0 with an empty console.
  local rc=0
  # shellcheck disable=SC2086
  { runpy "$@" --instances "$n" --cores-per-instance "$cores" --copies "$COPIES" $AB \
      --label "$name" --out "$(cpath "$OUT")/results.csv" 2>&1 | tee "$OUT/$name.log" \
    | { grep -E "spread:|imbalance|^ +[0-9]+\.[0-9]+x|median imbalance|fused_experts_cpu:|batched GEMMs:|check:|REFUS|Error|Traceback|SystemExit|pool needs|disagrees|instances failed|no stats rows" || true; } ; } || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "  !! LANE $name FAILED (exit $rc) -- see $OUT/$name.log" >&2
    FAILED_LANES="$FAILED_LANES $name"
  fi
  echo "  post-cell: $(pkgtemp)C, throttle $(throttles)"
  cool_down "$name"
}

has() { [[ " $LANES " == *" $1 "* ]]; }

# ---- 35B-A3B bf16 realtime: bs1, 4 concurrent TP4 ranks, N = 512/4 = 128 -------
if has 35b_rt; then
  spread 35b_rt_decode  "$SPREAD_RT" "$CORES" --phase decode  --batch 1 --layer "$LAYERS" \
      --mode both --check --tp 4 --iters 20 --warmup 5
  spread 35b_rt_prefill "$SPREAD_RT" "$CORES" --phase prefill --batch 1 --layer "$LAYERS" \
      --mode both --tp 4 --iters 10 --warmup 3
fi

# ---- 35B-A3B bf16 throughput: bs320, TP1 = 4 INDEPENDENT 56-core replicas ------
# TP1 means no sharding, so each of the four concurrent instances runs the FULL layer
# at N=512 (hence no --tp flag). Same core layout as the rt lanes, opposite sharding.
# bs320 prefill is the heaviest cell by far (~19.7 TFLOP/layer per instance, 392k
# tokens fed, x4 concurrent) -- few iterations on purpose.
if has 35b_thr; then
  COPIES=$COPIES_THR \
  spread 35b_thr_decode  "$SPREAD_THR" "$CORES" --phase decode  --batch 320 \
      --layer "$LAYERS" --mode both --iters 10 --warmup 3
  COPIES=1 \
  spread 35b_thr_prefill "$SPREAD_THR" "$CORES" --phase prefill --batch 320 \
      --layer "$LAYERS" --mode fused --iters 3 --warmup 1
fi

# ---- contrast: the same bs320 cell as ONE instance over all 224 cores ----------
# Not the throughput lane. Isolates one-big-instance vs four-replica scaling.
if has 35b_thr_socket; then
  spread 35b_thr_socket_decode 1 "$THREADS_THR" --phase decode --batch 320 \
      --layer "$LAYERS" --mode both --iters 10 --warmup 3
fi

# ---- 35B SHARED expert -- NOT IN THE DEFAULT SET, and not a like-for-like measure --
# The MoE block is routed experts PLUS a shared expert every token passes through
# (shared_expert_intermediate_size=512, verified in config_qwen3_5_35B_A3B.json; a
# Qwen3_5MLP with row-parallel down_proj whose partial output joins the same single
# all-reduce).
#
# OFF BY DEFAULT for two independent reasons:
#
#   1. It CANCELS. The shared expert is the same dense SwiGLU on both sides of the
#      comparison -- archbench models it as a BatchedSwiGLU with num_experts=1 and
#      sglang runs it as one dense FFN -- so it adds the same term to the projection and
#      to the measurement. Including it only dilutes the routed-MoE ratio this benchmark
#      exists to measure.
#   2. It would NOT be measured honestly by this harness anyway. sglang has a SEPARATE
#      op for it, `shared_expert_cpu`, which fuses the shared FFN with the multiply by
#      routed_scaling_factor AND the add of the routed experts' output (it hard-requires
#      `fused_experts_out`). Routing it through --dense-ffn instead calls
#      `fused_experts_cpu` with num_experts=1/topk=1, so it pays the routed kernel's
#      moe_align_block_size + sorted-token machinery that the real shared path does not
#      have, and it drops the fused epilogue. Different kernel, different work.
#      Measuring it properly means binding `shared_expert_cpu`, which this harness does
#      not do.
#
# Kept here rather than deleted so the shape is on record: LANES="35b_shared_rt" runs it.
SHARED35B=(--dense-ffn --hidden-size 2048 --intermediate-size 512)
if has 35b_shared_rt; then
  spread 35b_shared_rt_decode  "$SPREAD_RT" "$CORES" "${SHARED35B[@]}" --phase decode \
      --batch 1 --mode both --check --tp 4 --iters 20 --warmup 5
  spread 35b_shared_rt_prefill "$SPREAD_RT" "$CORES" "${SHARED35B[@]}" --phase prefill \
      --batch 1 --mode both --tp 4 --iters 10 --warmup 3
fi
if has 35b_shared_thr; then
  spread 35b_shared_thr_decode  "$SPREAD_THR" "$CORES" "${SHARED35B[@]}" \
      --phase decode --batch 320 --mode both --check --iters 20 --warmup 5
  spread 35b_shared_thr_prefill "$SPREAD_THR" "$CORES" "${SHARED35B[@]}" \
      --phase prefill --batch 320 --mode both --iters 3 --warmup 1
fi

# ---- 9B bf16: DENSE model, no MoE at all (descoped; off by default) ------------
DENSE9B=(--dense-ffn --hidden-size 4096 --intermediate-size 12288)
if has 9b_rt; then
  spread 9b_rt_decode  "$SPREAD_RT" "$CORES" "${DENSE9B[@]}" --phase decode --batch 1 \
      --mode both --check --tp 4 --iters 20 --warmup 5
  spread 9b_rt_prefill "$SPREAD_RT" "$CORES" "${DENSE9B[@]}" --phase prefill --batch 1 \
      --mode both --tp 4 --iters 10 --warmup 3
fi
if has 9b_thr; then
  spread 9b_thr_decode  "$SPREAD_THR" "$CORES" "${DENSE9B[@]}" --phase decode \
      --batch 96 --mode both --check --iters 20 --warmup 5
  spread 9b_thr_prefill "$SPREAD_THR" "$CORES" "${DENSE9B[@]}" --phase prefill \
      --batch 96 --mode both --iters 5 --warmup 2
fi

# ---- state after, so any drift during the run is visible ----------------------
{
  echo "date        : $(date -Is)"
  echo "governor    : $(gov)   (epp $(epp))"
  echo "uptime      : $(uptime)"
} | tee "$OUT/env_after.txt"
GOV_AFTER=$(gov)
if [ "$GOV_BEFORE" != "$GOV_AFTER" ]; then
  echo "WARNING: governor changed under us during the run: $GOV_BEFORE -> $GOV_AFTER" >&2
  echo "         (this script never sets it, so someone else did)" >&2
fi

echo
echo "done. per-instance CSVs:"
wc -l "$OUT"/results*.csv 2>/dev/null || echo "  none written"
if [ -n "$FAILED_LANES" ]; then
  echo
  echo "FAILED LANES:$FAILED_LANES" >&2
  exit 1
fi
