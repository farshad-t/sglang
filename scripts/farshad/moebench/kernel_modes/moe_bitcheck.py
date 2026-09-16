#!/usr/bin/env python3
"""Bit-compare fused_experts_cpu mode 0 between a candidate build and the pristine build.

Runs as a subprocess against ONE .so (sys.argv[1]) so the two extensions never share a
process -- both register the same sgl_kernel:: op names and the second import would lose.
Dumps the raw output bytes to sys.argv[2].
"""
import sys, os

so_dir, out_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, so_dir)
import torch

torch.set_num_threads(16)
import common_ops  # noqa: F401

E, K, N, topk = 32, 512, 128, 4
M = 96
g = torch.Generator().manual_seed(20260916)
hs = (torch.randn((M, K), generator=g, dtype=torch.float32) / 8).to(torch.bfloat16)
w1 = (torch.randn((E, 2 * N, K), generator=g, dtype=torch.float32) / 16).to(torch.bfloat16)
w2 = (torch.randn((E, K, N), generator=g, dtype=torch.float32) / 16).to(torch.bfloat16)
logits = torch.randn((M, E), generator=g, dtype=torch.float32)
tw, tid = torch.topk(torch.softmax(logits, dim=-1), topk, dim=-1)
tid = tid.to(torch.int32)

args = [hs, w1, w2, tw, tid, False, 0, None, None, None, None, None, None, None,
        None, None, False, None]
# the pristine build predates expert_batching_mode, so its schema takes 18 args
schema = str(torch.ops.sgl_kernel.fused_experts_cpu.default._schema)
if "expert_batching_mode" in schema:
    args.append(0)
out = torch.ops.sgl_kernel.fused_experts_cpu(*args)
with open(out_path, "wb") as f:
    f.write(out.contiguous().view(torch.uint8).numpy().tobytes())
print(f"{so_dir}: M={M} K={K} N={N} E={E} topk={topk} sum={out.float().sum().item():.9g}")
