#!/usr/bin/env python3
"""Price fused_experts_cpu's own gather+scatter, and the unfused GEMM+SwiGLU against it.

Three rungs, one process, same weights / activation / warmup / iteration count:

    mode0     fused_experts_cpu(expert_batching_mode=0)  -- gather + gemms + scatter
    mode2     fused_experts_cpu(expert_batching_mode=2)  -- gemms only, wrong result
    unfused   bmm_cpu + silu(gate)*up + bmm_cpu          -- what mode2 does, separate ops

mode0 - mode2  is the fused kernel's own gather + [M,topk,K]->[M,K] reduce.
unfused/mode2  compares the two matmul implementations with gather/scatter held out.

Fair-footprint note: mode2 reads its activation out of hidden_states, an [M, K] tensor,
so the routed rows it streams come from an M*K footprint. A genuine pre-gathered
[G, t, K] batch would be topk times bigger and would fall out of LLC when mode2's does
not, so the unfused rung reuses ONE A buffer sized to M*K across chunks of the bucket.
Its h / act / y buffers are full [M*topk, *] slabs, matching the fused kernel's ic1/ic2.
`unfused_wide` is the same rung with the honest full-size A, to show what that costs.

Routing is built with exact per-expert token counts so the unfused rung can batch each
count into one bmm_cpu call; mode0/mode2 see the identical routing table.
"""
import argparse, json, os, statistics, sys, time
import multiprocessing as mp

# One instance per SNC domain of socket 0. --skip-first-core drops cpu 0/32/64, which
# carry another user's hard-pinned inference schedulers when that server is up.
FULL = {0: (0, 32), 1: (32, 64), 2: (64, 96)}

E, K, N, TOPK = 256, 2048, 512, 8

REGIMES = {
    # name: (num_tokens, per-expert token counts as {count: n_experts})
    "decode": (72, {1: 64, 2: 64, 3: 64, 6: 32}),
    "prefill": (16384, {512: 256}),
    "prefill_hot": (32768, {32768: 8}),
}


def routing(num_tokens, counts, topk):
    """topk_ids with exactly `counts[c]` experts holding c tokens each, rows all distinct.

    Each expert's c copies are consecutive in the flat slot list and are dealt round-robin
    over the rows, so with c < num_tokens no row can draw the same expert twice.
    """
    import torch

    slots, e = [], 0
    for c in sorted(counts):
        for _ in range(counts[c]):
            slots += [e] * c
            e += 1
    assert e <= E, f"{e} experts needed, only {E} exist"
    assert len(slots) == num_tokens * topk, f"{len(slots)} slots != {num_tokens * topk}"
    rows = [[] for _ in range(num_tokens)]
    for i, ex in enumerate(slots):
        rows[i % num_tokens].append(ex)
    assert all(len(r) == topk and len(set(r)) == topk for r in rows), "duplicate expert in a row"
    return torch.tensor(rows, dtype=torch.int32)


def buckets_of(counts, topk_ids):
    """{token_count: [expert ids]} plus each expert's row slice into the sorted layout."""
    import torch

    per_expert = torch.bincount(topk_ids.flatten().to(torch.int64), minlength=E)
    out = {}
    for c in sorted(counts):
        ids = (per_expert == c).nonzero().flatten().tolist()
        out[c] = ids
    return out


def build(num_tokens, counts, seed, ops):
    import torch

    g = torch.Generator().manual_seed(seed)
    tid = routing(num_tokens, counts, TOPK)
    tw = (torch.rand((num_tokens, TOPK), generator=g, dtype=torch.float32) + 0.5) / TOPK
    hs = (torch.randn((num_tokens, K), generator=g, dtype=torch.float32) / 8).to(torch.bfloat16)
    w1 = (torch.randn((E, 2 * N, K), generator=g, dtype=torch.float32) / 24).to(torch.bfloat16)
    w2 = (torch.randn((E, K, N), generator=g, dtype=torch.float32) / 24).to(torch.bfloat16)

    routed = num_tokens * TOPK
    # Full-size slabs, so the unfused rung's output footprint matches the fused ic1/ic2.
    hbuf = torch.empty((routed, 2 * N), dtype=torch.bfloat16).uniform_(-1, 1, generator=g)
    actbuf = torch.empty((routed, N), dtype=torch.bfloat16).uniform_(-1, 1, generator=g)
    ybuf = torch.empty((routed, K), dtype=torch.bfloat16).uniform_(-1, 1, generator=g)

    # w1 is [E, 2N, K] and w2 is [E, K, N] -- both already output-channel-major, which is
    # the layout convert_weight_packed wants, so no transpose here.
    work = []
    for c, ids in buckets_of(counts, tid).items():
        if not ids:
            continue
        sel = torch.tensor(ids, dtype=torch.int64)
        p1 = ops.convert_weight_packed(w1.index_select(0, sel).contiguous())
        p2 = ops.convert_weight_packed(w2.index_select(0, sel).contiguous())
        gc = max(1, min(len(ids), num_tokens // c))
        a_small = torch.empty((gc, c, K), dtype=torch.bfloat16).uniform_(-1, 1, generator=g)
        a_wide = torch.empty((len(ids), c, K), dtype=torch.bfloat16).uniform_(-1, 1, generator=g)
        work.append(dict(c=c, G=len(ids), gc=gc, p1=p1, p2=p2, a_small=a_small, a_wide=a_wide))
    return dict(tid=tid, tw=tw, hs=hs, w1=w1, w2=w2, routed=routed,
                hbuf=hbuf, actbuf=actbuf, ybuf=ybuf, work=work)


def make_rungs(st, ops):
    import torch
    import torch.nn.functional as F

    fe = torch.ops.sgl_kernel.fused_experts_cpu

    has_mode = "expert_batching_mode" in str(fe.default._schema)

    def fused(mode):
        args = [st["hs"], st["w1"], st["w2"], st["tw"], st["tid"], False, 0,
                None, None, None, None, None, None, None, None, None, False, None]
        # the pristine build predates expert_batching_mode; its schema takes 18 args
        if has_mode:
            args = args + [mode]
        elif mode != 0:
            return None

        def run():
            fe(*args)
        return run

    def unfused(a_key, alloc=False):
        def run():
            # alloc=True re-allocates the h/act/y slabs per call, as fused_experts_cpu does
            # with its internal buffer, so the page-fault cost lands on both sides.
            if alloc:
                hb = torch.empty((st["routed"], 2 * N), dtype=torch.bfloat16)
                ab = torch.empty((st["routed"], N), dtype=torch.bfloat16)
                yb = torch.empty((st["routed"], K), dtype=torch.bfloat16)
            else:
                hb, ab, yb = st["hbuf"], st["actbuf"], st["ybuf"]
            for wk in st["work"]:
                c, G, p1, p2 = wk["c"], wk["G"], wk["p1"], wk["p2"]
                gc = G if a_key == "a_wide" else wk["gc"]
                a_all = wk[a_key]
                base = 0
                for s in range(0, G, gc):
                    n = min(gc, G - s)
                    a = a_all[:n] if a_key == "a_small" else a_all[s:s + n]
                    rows = n * c
                    h = hb[base:base + rows].view(n, c, 2 * N)
                    ops.bmm_cpu(h, a, p1[s:s + n], True, None)
                    gate, up = h.chunk(2, dim=-1)
                    act = ab[base:base + rows].view(n, c, N)
                    torch.mul(F.silu(gate), up, out=act)
                    y = yb[base:base + rows].view(n, c, K)
                    ops.bmm_cpu(y, act, p2[s:s + n], True, None)
                    base += rows
        return run

    # what fused_experts_cpu's own at::empty costs: same byte count, same first touch
    nbytes = st["routed"] * N * 2 + st["routed"] * K * 2
    nthreads = torch.get_num_threads()
    nbytes += nthreads * 32 * K * 2 + nthreads * 2 * 32 * 32 * 4

    def allocfloor():
        b = torch.empty(nbytes, dtype=torch.uint8)
        b.view(-1)[::4096] = 1

    return dict(mode0=fused(0), mode2=fused(2), allocfloor=allocfloor,
                unfused=unfused("a_small"), unfused_wide=unfused("a_wide"),
                unfused_alloc=unfused("a_small", alloc=True))


def instance(idx, cores, barrier, q, a):
    os.sched_setaffinity(0, set(cores))
    os.environ["OMP_NUM_THREADS"] = str(len(cores))
    sys.path.insert(0, a.so_dir)
    import torch

    torch.set_num_threads(len(cores))
    import common_ops  # noqa: F401

    # fused_experts_cpu at::empty's its [M*topk, N] + [M*topk, K] intermediate on EVERY
    # call, so its timing depends on whether glibc's arena already holds faulted pages of
    # that size -- worth 25% between runs. Grow and fault the arena to a fixed size up
    # front (with MALLOC_TRIM_THRESHOLD_=-1 it is never returned) so every rung, and every
    # run, starts from the same allocator state.
    warm = torch.empty(a.arena_gib * 1024**3, dtype=torch.uint8)
    warm.view(-1)[::4096] = 1
    del warm

    ops = torch.ops.sgl_kernel
    num_tokens, counts = REGIMES[a.regime]
    st = build(num_tokens, counts, a.seed + idx, ops)
    rungs = make_rungs(st, ops)
    names = [n for n in a.rungs.split(",") if rungs.get(n) is not None]
    # Warm every rung before timing any of them: oneDNN init and OMP thread wake-up are
    # one-time, and whichever rung ran first would otherwise absorb all of it.
    for nm in names:
        for _ in range(a.warmup):
            rungs[nm]()
    out = {}
    for nm in names:
        fn = rungs[nm]
        barrier.wait()
        s = []
        for _ in range(a.iters):
            barrier.wait()
            t0 = time.perf_counter()
            fn()
            s.append((time.perf_counter() - t0) * 1000)
        out[nm] = dict(median_ms=statistics.median(s), min_ms=min(s), max_ms=max(s))
    hist = {wk["c"]: wk["G"] for wk in st["work"]}
    q.put((idx, dict(out, _meta=dict(cores=len(cores), num_tokens=num_tokens,
                                     routed=st["routed"], hist=hist))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--so-dir", default="/tmp/moe_cm_mode2")
    p.add_argument("--regime", default="prefill", choices=sorted(REGIMES))
    p.add_argument("--instances", type=int, default=3)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=9)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--rungs", default="mode0,mode2,unfused,unfused_wide")
    p.add_argument("--skip-first-core", action="store_true")
    p.add_argument("--arena-gib", type=int, default=4)
    p.add_argument("--out-json", default="")
    a = p.parse_args()

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(a.instances)
    q = ctx.Queue()
    ranges = {k: range(lo + (1 if a.skip_first_core else 0), hi) for k, (lo, hi) in FULL.items()}
    procs = [ctx.Process(target=instance, args=(i, list(ranges[i]), barrier, q, a))
             for i in range(a.instances)]
    for pr in procs:
        pr.start()
    got = dict(q.get() for _ in range(a.instances))
    for pr in procs:
        pr.join()

    m = got[0]["_meta"]
    print(f"regime={a.regime} K={K} N={N} E={E} topk={TOPK} tokens={m['num_tokens']} "
          f"routed={m['routed']} tokens_per_expert={m['hist']} "
          f"instances={a.instances}x{m['cores']}c iters={a.iters}")
    print(f"{'rung':14s} {'median ms':>11s} {'min ms':>10s} {'worst/med':>10s}")
    res = {}
    for nm in [n for n in a.rungs.split(",") if n in got[0]]:
        med = [got[i][nm]["median_ms"] for i in got]
        mm = statistics.median(med)
        res[nm] = dict(median_ms=mm, min_ms=statistics.median([got[i][nm]["min_ms"] for i in got]),
                       imb=max(med) / mm, per_instance=med)
        print(f"{nm:14s} {mm:>11.3f} {res[nm]['min_ms']:>10.3f} {res[nm]['imb']:>10.3f}")

    if "mode0" in res and "mode2" in res:
        d = res["mode0"]["median_ms"] - res["mode2"]["median_ms"]
        print(f"\nmode0 - mode2   = {d:>9.3f} ms  "
              f"({100 * d / res['mode0']['median_ms']:.1f}% of mode0)  "
              f"-- the fused kernel's own gather + reduce")
    for nm in ("unfused", "unfused_wide"):
        if nm in res and "mode2" in res:
            print(f"{nm:13s} / mode2 = {res[nm]['median_ms'] / res['mode2']['median_ms']:>7.3f}x")
    if a.out_json:
        json.dump(dict(regime=a.regime, meta=m, rungs=res), open(a.out_json, "w"), indent=2)


if __name__ == "__main__":
    main()
