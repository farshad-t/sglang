#!/usr/bin/env python3
"""Run N iterations of ONE fused_experts_cpu mode, for perf to count AMX bf16 ops around.

Single-threaded and single-instance on purpose: the point is a deterministic op count, not
a time. argv: <so_dir> <mode> <iters>
"""
import os, sys

os.sched_setaffinity(0, {100})  # socket 1, clear of the socket-0 benchmark ranges
sys.path.insert(0, sys.argv[1])
mode, iters = int(sys.argv[2]), int(sys.argv[3])
import torch

torch.set_num_threads(1)
import common_ops  # noqa: F401

E, K, N, TOPK, M = 256, 2048, 512, 8, 4096
g = torch.Generator().manual_seed(11)
# balanced routing: expert (TOPK*i + j) % E, so each expert gets exactly M*TOPK/E tokens
tid = torch.tensor([[(TOPK * i + j) % E for j in range(TOPK)] for i in range(M)],
                   dtype=torch.int32)
tw = (torch.rand((M, TOPK), generator=g, dtype=torch.float32) + 0.5) / TOPK
hs = (torch.randn((M, K), generator=g, dtype=torch.float32) / 8).to(torch.bfloat16)
w1 = (torch.randn((E, 2 * N, K), generator=g, dtype=torch.float32) / 24).to(torch.bfloat16)
w2 = (torch.randn((E, K, N), generator=g, dtype=torch.float32) / 24).to(torch.bfloat16)

fe = torch.ops.sgl_kernel.fused_experts_cpu
args = [hs, w1, w2, tw, tid, False, 0, None, None, None, None, None, None, None,
        None, None, False, None, mode]
for _ in range(iters):
    fe(*args)
print(f"mode={mode} iters={iters} M={M} routed={M * TOPK} tokens_per_expert={M * TOPK // E}",
      file=sys.stderr)
