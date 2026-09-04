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
# So the TP4 rt lanes use --tp 4 (N = 512/4 = 128). The `_unsharded` lane is a --tp 1
# contrast that isolates what the split costs the kernel, not the reference.
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
#   AB="..."             extra stats-source flags, e.g. AB="--ab-offline"
#   PY=python3           interpreter
#   COOLDOWN=60          seconds of idle between cells, for the part to settle
#   TEMP_MAX=80          do not start the next cell above this package temp (C)
#   COOLDOWN_MAX=600     give up waiting to cool after this long, and say so
#   FORCE=1              skip the busy/governor guards
#   LANES="..."          subset of: 35b_rt 35b_rt_unsharded 35b_thr 35b_thr_socket
#                        35b_shared_rt 35b_shared_thr 9b_rt 9b_thr
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
PY=${PY:-python3}
FORCE=${FORCE:-0}
COOLDOWN=${COOLDOWN:-60}
TEMP_MAX=${TEMP_MAX:-80}
COOLDOWN_MAX=${COOLDOWN_MAX:-600}
LANES=${LANES:-"35b_rt 35b_rt_unsharded 35b_thr 35b_shared_rt 35b_shared_thr"}

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
  echo "git         : $(git rev-parse HEAD 2>/dev/null || echo n/a)"
  echo "torch       : $($PY -c 'import torch;print(torch.__version__)' 2>/dev/null || echo n/a)"
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
# The concurrency now lives INSIDE bench_moe_cpu.py (--instances), not in this shell.
# The shell version launched independent processes that started together and then
# drifted apart, so a cell could be timed while its neighbours sat between kernels --
# which measured a partly-idle socket and produced a 2.28x across-instance spread.
# The Python driver re-synchronises all instances on a barrier before EVERY iteration
# and sets OMP_WAIT_POLICY=passive / KMP_BLOCKTIME=0 so idle threads sleep instead of
# busy-waiting on cores their neighbours need. Measured imbalance after: ~1.07x median.
spread() {  # spread <name> <n_instances> <cores_per_instance> <args...>
  local name=$1 n=$2 cores=$3; shift 3
  echo
  echo "################ $name   (${n} x ${cores}c concurrent, barrier-synced)"
  echo "$(date -Is) before=$name temp=$(pkgtemp)C throttle=$(throttles)" >> "$OUT/thermal.log"
  echo "  pre-cell: $(pkgtemp)C, throttle $(throttles)"
  # shellcheck disable=SC2086
  $PY bench_moe_cpu.py "$@" --instances "$n" --cores-per-instance "$cores" $AB \
      --out "$OUT/results.csv" 2>&1 | tee "$OUT/$name.log" \
    | grep -E "spread:|imbalance|^ +[0-9]+\.[0-9]+x|median imbalance|fused_experts_cpu:|batched GEMMs:|check:|REFUS|Error" || true
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

# ---- same, UNSHARDED (--tp 1, N=512): contrast only, not the reference ---------
if has 35b_rt_unsharded; then
  spread 35b_rt_unsharded_decode  "$SPREAD_RT" "$CORES" --phase decode  --batch 1 \
      --layer "$LAYERS" --mode both --iters 20 --warmup 5
  spread 35b_rt_unsharded_prefill "$SPREAD_RT" "$CORES" --phase prefill --batch 1 \
      --layer "$LAYERS" --mode both --iters 10 --warmup 3
fi

# ---- 35B-A3B bf16 throughput: bs320, TP1 = 4 INDEPENDENT 56-core replicas ------
# TP1 means no sharding, so each of the four concurrent instances runs the FULL layer
# at N=512 (hence no --tp flag). Same core layout as the rt lanes, opposite sharding.
# bs320 prefill is the heaviest cell by far (~19.7 TFLOP/layer per instance, 392k
# tokens fed, x4 concurrent) -- few iterations on purpose.
if has 35b_thr; then
  spread 35b_thr_decode  "$SPREAD_THR" "$CORES" --phase decode  --batch 320 \
      --layer "$LAYERS" --mode both --iters 10 --warmup 3
  spread 35b_thr_prefill "$SPREAD_THR" "$CORES" --phase prefill --batch 320 \
      --layer "$LAYERS" --mode fused --iters 3 --warmup 1
fi

# ---- contrast: the same bs320 cell as ONE instance over all 224 cores ----------
# Not the throughput lane. Isolates one-big-instance vs four-replica scaling.
if has 35b_thr_socket; then
  spread 35b_thr_socket_decode 1 "$THREADS_THR" --phase decode --batch 320 \
      --layer "$LAYERS" --mode both --iters 10 --warmup 3
fi

# ---- 35B SHARED expert (dense, K=2048, N=512) ---------------------------------
# The MoE block is routed experts PLUS a shared expert every token passes through
# (shared_expert_intermediate_size=512, a Qwen3_5MLP with row-parallel down_proj whose
# partial output joins the same single all-reduce). A layer-level compare needs it.
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
wc -l "$OUT"/results.inst*.csv 2>/dev/null || echo "  none written"
