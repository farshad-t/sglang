#!/usr/bin/env bash
# Run the bf16 lanes of the qwen35 DMR-X4 daily run through the MoE/FFN kernel
# benchmark, so each lane's projected layer cost has a measured counterpart.
#
# Lanes taken from tools/cpu/config/qwen35/qwen35_dmrx4_lanes_20260903_1119_collmodes_shmprofile.csv
# (all four share input_seq_len=1024, output_seq_len=1024, gen_start=gen_end=1536,
# hw dmr_ap_x4_16ch_224c_128cbo_ddr8000):
#
#   lane 1/2   qwen35_009b_bs001_TP4_bf16          rt   bs1    TP4  numUnits [56]
#   lane 3/4   qwen35_035b_bs001_TP4_bf16_a003b    rt   bs1    TP4  numUnits [56]
#   lane 7/8   qwen35_009b_bs096_TP1_bf16          thr  bs96   TP1  (whole socket)
#   lane 9/10  qwen35_035b_bs320_TP1_bf16_a003b    thr  bs320  TP1  (whole socket)
#
# The 9B is DENSE (Qwen3_5ForConditionalGeneration, hidden 4096 / intermediate 12288,
# no num_experts), so its lanes run --dense-ffn, i.e. the num_experts=1 case that
# archbench itself uses for a dense FFN. Only the 35B has routed experts.
#
# TP: the projection DOES shard the MoE intermediate dim. `Qwen3_5MoeExperts` passes
# `config.moe_intermediate_size` through undivided, but the division happens one level
# down in `BatchedSwiGLU.forward` (common/experts.py:50):
#     local_hidden_dim = math.ceil(self.hidden_dim / self.args.model_parallel_size)
# where `hidden_dim` IS the moe intermediate size. Dense FFNs shard the same way in
# `FeedForward.forward` (Llama4/ffn.py:45). Each rank keeps all E experts and holds
# N/TP of the intermediate, i.e. row-parallel down_proj, and the MoE block owns ONE
# all-reduce on the routed+shared expert sum (each expert passes collective_tp_size=1
# precisely so the AR is not charged per expert).
#
# So for the TP4 rt lanes the faithful shape is --tp 4 (N = 512/4 = 128). The
# `_unsharded` lanes below are a --tp 1 contrast, not the reference.
#
# Knobs (env):
#   OUT=<dir>        output dir                    (default results_<timestamp>)
#   THREADS_RT=56    threads for the rt lanes = one TP rank's core budget
#   THREADS_THR=224  threads for the thr lanes = whole socket
#   LAYERS=all       --layer value for the 35B lanes (e.g. 0,10,20,30,39 to sample)
#   AB="..."         extra stats-source flags, e.g. AB="--ab-offline"
#   PY=python3       interpreter
#   LANES="..."      subset of: 35b_rt 35b_rt_unsharded 35b_thr 35b_shared_rt
#                    35b_shared_thr 9b_rt 9b_thr
set -euo pipefail
cd "$(dirname "$0")"

OUT=${OUT:-results_$(date +%Y%m%d_%H%M%S)}
THREADS_RT=${THREADS_RT:-56}
THREADS_THR=${THREADS_THR:-224}
LAYERS=${LAYERS:-all}
AB=${AB:-}
PY=${PY:-python3}
LANES=${LANES:-"35b_rt 35b_rt_unsharded 35b_thr 35b_shared_rt 35b_shared_thr"}

mkdir -p "$OUT"
echo "output dir: $OUT"
{
  echo "host       : $(hostname)"
  echo "date       : $(date -Is)"
  echo "nproc      : $(nproc)"
  echo "numactl    :"; numactl --hardware 2>&1 | sed 's/^/  /' || echo "  (numactl absent)"
  echo "cpu        :"; lscpu 2>/dev/null | grep -E '^(Model name|Socket|Core|Thread|NUMA node\(s\)|CPU max)' | sed 's/^/  /'
  echo "threads_rt : $THREADS_RT"
  echo "threads_thr: $THREADS_THR"
  echo "layers     : $LAYERS"
  echo "git        : $(git rev-parse HEAD 2>/dev/null || echo n/a)"
} | tee "$OUT/env.txt"

run() {   # run <name> <log-suffix> <args...>
  local name=$1; shift
  echo
  echo "################ $name"
  # shellcheck disable=SC2086
  $PY bench_moe_cpu.py "$@" $AB --out "$OUT/results.csv" 2>&1 | tee "$OUT/$name.log"
}

has() { [[ " $LANES " == *" $1 "* ]]; }

# ---- 35B-A3B bf16, realtime: bs1, ONE TP4 rank (56 cores, N = 512/4 = 128) -----
# THE reference lane: matches what the projection charges a rank and what a real
# sglang TP4 rank executes.
if has 35b_rt; then
  run 35b_rt_decode  --phase decode  --batch 1 --layer "$LAYERS" --mode both --check \
      --tp 4 --threads "$THREADS_RT" --iters 20 --warmup 5
  run 35b_rt_prefill --phase prefill --batch 1 --layer "$LAYERS" --mode both \
      --tp 4 --threads "$THREADS_RT" --iters 10 --warmup 3
fi

# ---- same lane UNSHARDED (--tp 1, N=512) -- contrast only, not the reference ----
# Isolates what the TP split costs the kernel: same routing, 4x the N.
if has 35b_rt_unsharded; then
  run 35b_rt_unsharded_decode  --phase decode  --batch 1 --layer "$LAYERS" --mode both \
      --threads "$THREADS_RT" --iters 20 --warmup 5
  run 35b_rt_unsharded_prefill --phase prefill --batch 1 --layer "$LAYERS" --mode both \
      --threads "$THREADS_RT" --iters 10 --warmup 3
fi

# ---- 35B-A3B bf16, throughput: bs320, TP1, whole socket -----------------------
# bs320 prefill is the heaviest cell by far (~19.7 TFLOP/layer, 392k tokens fed) and
# is also the +19.7% mass-overshoot cell. Few iterations on purpose.
if has 35b_thr; then
  run 35b_thr_decode  --phase decode  --batch 320 --layer "$LAYERS" --mode both \
      --threads "$THREADS_THR" --iters 10 --warmup 3
  run 35b_thr_prefill --phase prefill --batch 320 --layer "$LAYERS" --mode fused \
      --threads "$THREADS_THR" --iters 3 --warmup 1
fi

# ---- 35B SHARED expert (dense, K=2048 N=512) ----------------------------------
# The 35B MoE block is routed experts PLUS a shared expert every token passes through
# (shared_expert_intermediate_size=512, a Qwen3_5MLP whose down_proj is row-parallel and
# whose partial output is summed into the same single all-reduce). A layer-level compare
# against the projection needs it, so measure it as its own dense cell.
SHARED35B="--dense-ffn --hidden-size 2048 --intermediate-size 512"
if has 35b_shared_rt; then
  # shellcheck disable=SC2086
  run 35b_shared_rt_decode  $SHARED35B --phase decode  --batch 1 --mode both --check \
      --tp 4 --threads "$THREADS_RT" --iters 20 --warmup 5
  # shellcheck disable=SC2086
  run 35b_shared_rt_prefill $SHARED35B --phase prefill --batch 1 --mode both \
      --tp 4 --threads "$THREADS_RT" --iters 10 --warmup 3
fi
if has 35b_shared_thr; then
  # shellcheck disable=SC2086
  run 35b_shared_thr_decode  $SHARED35B --phase decode  --batch 320 --mode both --check \
      --threads "$THREADS_THR" --iters 20 --warmup 5
  # shellcheck disable=SC2086
  run 35b_shared_thr_prefill $SHARED35B --phase prefill --batch 320 --mode both \
      --threads "$THREADS_THR" --iters 3 --warmup 1
fi

# ---- 9B bf16: DENSE model, no MoE at all --------------------------------------
# Qwen3_5ForConditionalGeneration, hidden 4096 / intermediate 12288, no num_experts.
# Off by default (the 9B lanes were descoped); kept correct in case they come back.
# FeedForward.forward shards the intermediate the same way, hence --tp 4 on the rt lane.
DENSE9B="--dense-ffn --hidden-size 4096 --intermediate-size 12288"
if has 9b_rt; then
  # shellcheck disable=SC2086
  run 9b_rt_decode  $DENSE9B --phase decode  --batch 1 --mode both --check \
      --tp 4 --threads "$THREADS_RT" --iters 20 --warmup 5
  # shellcheck disable=SC2086
  run 9b_rt_prefill $DENSE9B --phase prefill --batch 1 --mode both \
      --tp 4 --threads "$THREADS_RT" --iters 10 --warmup 3
fi
if has 9b_thr; then
  # shellcheck disable=SC2086
  run 9b_thr_decode  $DENSE9B --phase decode  --batch 96 --mode both --check \
      --threads "$THREADS_THR" --iters 20 --warmup 5
  # shellcheck disable=SC2086
  run 9b_thr_prefill $DENSE9B --phase prefill --batch 96 --mode both \
      --threads "$THREADS_THR" --iters 5 --warmup 2
fi

echo
echo "done. rows in $OUT/results.csv:"
wc -l < "$OUT/results.csv"
