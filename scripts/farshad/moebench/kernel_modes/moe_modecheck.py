#!/usr/bin/env python3
"""Per-mode output hashes at the campaign shape, for one .so given as argv[1]."""
import hashlib, sys

sys.path.insert(0, sys.argv[1])
import torch

torch.set_num_threads(32)
import common_ops  # noqa: F401

E, K, N, topk, M = 256, 2048, 512, 8, 96
g = torch.Generator().manual_seed(7)
hs = (torch.randn((M, K), generator=g, dtype=torch.float32) / 8).to(torch.bfloat16)
w1 = (torch.randn((E, 2 * N, K), generator=g, dtype=torch.float32) / 24).to(torch.bfloat16)
w2 = (torch.randn((E, K, N), generator=g, dtype=torch.float32) / 24).to(torch.bfloat16)
tw, tid = torch.topk(torch.softmax(torch.randn((M, E), generator=g), dim=-1), topk, dim=-1)
tid = tid.to(torch.int32)

schema = str(torch.ops.sgl_kernel.fused_experts_cpu.default._schema)
modes = [0, 1, 2] if "expert_batching_mode" in schema else [None]
for mode in modes:
    args = [hs, w1, w2, tw, tid, False, 0, None, None, None, None, None, None, None,
            None, None, False, None]
    if mode is not None:
        args.append(mode)
    try:
        out = torch.ops.sgl_kernel.fused_experts_cpu(*args)
    except Exception as e:  # noqa: BLE001
        print(f"mode={mode}: RAISED {type(e).__name__}: {e}")
        continue
    b = out.contiguous().view(torch.uint8).numpy().tobytes()
    print(f"mode={mode}: md5={hashlib.md5(b).hexdigest()} "
          f"sum={out.float().sum().item():.9g} finite={bool(out.isfinite().all())}")
