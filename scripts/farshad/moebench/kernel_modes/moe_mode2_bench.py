#!/usr/bin/env python3
"""Price each thing the fused MoE kernel does that an unfused decomposition does not.

One process, same weights / activation / warmup / iteration count:

    mode0     fused_experts_cpu(expert_batching_mode=0)  -- as it ships
    mode2     fused_experts_cpu(expert_batching_mode=2)  -- minus gather and reduce
    mode3     fused_experts_cpu(expert_batching_mode=3)  -- minus the scatter as well
    unfused   routing prep + per-call alloc + bmm_cpu + silu(gate)*up + bmm_cpu
    unfused_bare   the same without the routing prep or the per-call alloc

    mode0 - mode2   the gather's row copy + the [M,topk,K]->[M,K] topk reduce
    mode2 - mode3   the topk-weighted scatter: gemm2's permuted store back to routed-slot
                    order, which mode 2 still pays and the unfused rung never does
    unfused / mode3 the two implementations with every term that is not the matmul or the
                    activation fusion held out of, or charged to, BOTH sides

Two steps used to sit on the fused side only, and both are now on the unfused rung as well:
the routing sort (`moe_routing_prep_cpu`, the identical function fused_experts_cpu calls --
the harness's precomputed buckets are a shortcut a real unfused route does not get), and the
per-call allocation of the intermediates plus a block matching out_hidden_states and the
thread scratch, first-touched, since what that costs is page faults and not the malloc.
`allocfloor` and `routing_prep` price those two on their own; `unfused_bare` shows the leg
without them.

What is left on one side only is the SwiGLU fusion, deliberately: the fused kernel keeps
gate|up in its fp32 accumulator and stores bf16 straight out of `silu_and_mul_stub`, where
the decomposition has to materialise `h` and read it back. That is a real property of the
two implementations, so it stays counted.

The unfused rung issues ONE bmm_cpu pair per bucket over all its experts, which is the
shape set the projection models: its benchdnn cases are `<G>x<tokens>x2048:<G>x2048x1024`,
one batched matmul per bucket with the batch dim equal to the bucket's expert count. There
is deliberately no chunked variant -- nothing upstream chunks, so a chunked rung would be
measuring an implementation nobody has.

Its h / act / y buffers are full [M*topk, *] slabs, matching the fused kernel's ic1/ic2.
Its A is the honest pre-gathered [G, tokens, K], which is topk times the [M, K] that mode
2 and 3 read their activation slices out of -- an asymmetry that stays, because a
pre-gathered batch is what the unfused route actually holds.

Routing is built with exact per-expert token counts so the unfused rung can batch each
count into one bmm_cpu call; every fused mode sees the identical routing table.
"""
import argparse, json, os, statistics, sys, time
import multiprocessing as mp

# One instance per SNC domain of socket 0. --skip-first-core drops cpu 0/32/64, which
# carry another user's hard-pinned inference schedulers when that server is up.
FULL = {0: (0, 32), 1: (32, 64), 2: (64, 96)}

E, K, TOPK = 256, 2048, 8
N = 512  # per-regime; set by instance() before any build

REGIMES = {
    # name: (num_tokens, per-expert token counts as {count: n_experts}, N)
    "decode": (72, {1: 64, 2: 64, 3: 64, 6: 32}, 512),
    "prefill": (16384, {512: 256}, 512),
    "prefill_hot": (32768, {32768: 8}, 512),
    # 8192 tokens/expert across all 256 experts: the "tens of thousands per expert"
    # end WITHOUT concentrating routing onto a handful of experts. ~29 GiB/instance.
    "prefill_big": (262144, {8192: 256}, 512),
    # The four cells run on DMR (224c, SNC OFF, 4x56c), replayed here. rt is TP4, so
    # N = 512/4; thr is TP1 at the full N. Counts are balanced rather than the run's real
    # histogram, because the unfused rung needs equal-token buckets to batch a bmm_cpu.
    "dmr_rt_decode": (1, {1: 8}, 128),
    "dmr_rt_prefill": (1024, {32: 256}, 128),
    "dmr_thr_decode": (320, {10: 256}, 512),
    "dmr_thr_prefill": (327680, {10240: 256}, 512),
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
    silubuf = torch.empty((routed, N), dtype=torch.bfloat16).uniform_(-1, 1, generator=g)

    # fused_experts_cpu repacks w1/w2 on EVERY call unless is_vnni=True, and that repack is
    # 60-96% of its runtime here -- so pack once at setup and pass is_vnni=True, matching how
    # the unfused rung gets its weights and how bench_moe_cpu.py's --prepack default runs.
    pw1 = ops.convert_weight_packed(w1)
    pw2 = ops.convert_weight_packed(w2)

    # w1 is [E, 2N, K] and w2 is [E, K, N] -- both already output-channel-major, which is
    # the layout convert_weight_packed wants, so no transpose here.
    work = []
    for c, ids in buckets_of(counts, tid).items():
        if not ids:
            continue
        sel = torch.tensor(ids, dtype=torch.int64)
        p1 = ops.convert_weight_packed(w1.index_select(0, sel).contiguous())
        p2 = ops.convert_weight_packed(w2.index_select(0, sel).contiguous())
        # The honest pre-gathered batch: one [G, tokens, K] A per bucket, which is what a
        # single bmm_cpu call over the bucket's experts consumes.
        a = torch.empty((len(ids), c, K), dtype=torch.bfloat16).uniform_(-1, 1, generator=g)
        work.append(dict(c=c, G=len(ids), p1=p1, p2=p2, a=a))
    return dict(tid=tid, tw=tw, hs=hs, pw1=pw1, pw2=pw2, routed=routed,
                hbuf=hbuf, actbuf=actbuf, ybuf=ybuf, silubuf=silubuf, work=work)


def make_rungs(st, ops):
    import torch
    import torch.nn.functional as F

    fe = torch.ops.sgl_kernel.fused_experts_cpu
    M = st["hs"].size(0)
    nthreads = torch.get_num_threads()

    has_mode = "expert_batching_mode" in str(fe.default._schema)

    def fused(mode):
        args = [st["hs"], st["pw1"], st["pw2"], st["tw"], st["tid"], False, 0,
                None, None, None, None, None, None, None, None, None, True, None]
        # the pristine build predates expert_batching_mode; its schema takes 18 args
        if has_mode:
            args = args + [mode]
        elif mode != 0:
            return None

        def run():
            fe(*args)
        return run

    has_prep = hasattr(ops, "moe_routing_prep_cpu")
    # fused_experts_cpu allocates out_hidden_states plus the thread scratch (A_tmp, C_tmp)
    # that ic1/ic2 do not correspond to; charging the unfused side an equal-sized block, and
    # FIRST TOUCHING it, is what makes the two pay the same page-fault work. The cost is the
    # faults, not the malloc, so an untouched torch.empty would be free and match nothing.
    extra_bytes = M * K * 2 + nthreads * 32 * K * 2 + nthreads * 2 * 32 * 32 * 4

    def unfused(alloc=True, prep=True):
        """The decomposition, charged the two steps only the fused side used to pay.

        `prep` runs the SAME routing sort fused_experts_cpu runs; the harness's buckets are
        precomputed at setup, which a real unfused route could not do. `alloc` allocates the
        intermediates per call instead of reusing setup buffers, as fused_experts_cpu does.
        Both default ON: with them the two legs' op lists differ only in the SwiGLU fusion.
        """
        def run():
            if prep and has_prep:
                ops.moe_routing_prep_cpu(st["tid"], E)
            if alloc:
                hb = torch.empty((st["routed"], 2 * N), dtype=torch.bfloat16)
                ab = torch.empty((st["routed"], N), dtype=torch.bfloat16)
                yb = torch.empty((st["routed"], K), dtype=torch.bfloat16)
                sb = torch.empty((st["routed"], N), dtype=torch.bfloat16)
                extra = torch.empty(extra_bytes, dtype=torch.uint8)
                extra.view(-1)[::4096] = 1
            else:
                hb, ab, yb, sb = st["hbuf"], st["actbuf"], st["ybuf"], st["silubuf"]
            base = 0
            for wk in st["work"]:
                c, G, p1, p2 = wk["c"], wk["G"], wk["p1"], wk["p2"]
                rows = G * c
                h = hb[base:base + rows].view(G, c, 2 * N)
                ops.bmm_cpu(h, wk["a"], p1, True, None)
                gate, up = h.chunk(2, dim=-1)
                act = ab[base:base + rows].view(G, c, N)
                # silu(gate) * up without F.silu's per-call temporary: that temporary was an
                # allocation inside the timed loop that the fused side does not have.
                s = sb[base:base + rows].view(G, c, N)
                torch.sigmoid(gate, out=s)
                s.mul_(gate)
                torch.mul(s, up, out=act)
                y = yb[base:base + rows].view(G, c, K)
                ops.bmm_cpu(y, act, p2, True, None)
                base += rows
        return run

    # what fused_experts_cpu's own at::empty costs: same byte count, same first touch
    nbytes = st["routed"] * N * 2 + st["routed"] * K * 2 + extra_bytes

    def allocfloor():
        b = torch.empty(nbytes, dtype=torch.uint8)
        b.view(-1)[::4096] = 1

    def routing_prep():
        ops.moe_routing_prep_cpu(st["tid"], E)

    r = dict(mode0=fused(0), mode2=fused(2), mode3=fused(3), allocfloor=allocfloor,
             unfused=unfused(), unfused_bare=unfused(alloc=False, prep=False))
    if has_prep:
        r["routing_prep"] = routing_prep
    return r


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
    global N
    num_tokens, counts, N = REGIMES[a.regime]
    st = build(num_tokens, counts, a.seed + idx, ops)
    rungs = make_rungs(st, ops)
    names = [n for n in a.rungs.split(",") if rungs.get(n) is not None]
    # Warm every rung before timing any of them: oneDNN init and OMP thread wake-up are
    # one-time, and whichever rung ran first would otherwise absorb all of it.
    for nm in names:
        for _ in range(a.warmup):
            rungs[nm]()
    samples = {nm: [] for nm in names}
    if a.interleave:
        # Round-robin the rungs INSIDE the iteration loop, so a drift over the run (AMX
        # frequency ramp, page placement settling, allocator growth) lands on every rung
        # equally instead of on whichever one is timed first.
        barrier.wait()
        for it in range(a.iters):
            order = names if it % 2 == 0 else names[::-1]
            for nm in order:
                barrier.wait()
                t0 = time.perf_counter()
                rungs[nm]()
                samples[nm].append((time.perf_counter() - t0) * 1000)
    else:
        for nm in names:
            barrier.wait()
            for _ in range(a.iters):
                barrier.wait()
                t0 = time.perf_counter()
                rungs[nm]()
                samples[nm].append((time.perf_counter() - t0) * 1000)
    out = {nm: dict(median_ms=statistics.median(v), min_ms=min(v), max_ms=max(v))
           for nm, v in samples.items()}
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
    p.add_argument("--rungs",
                   default="mode0,mode2,mode3,unfused,allocfloor,routing_prep")
    p.add_argument("--skip-first-core", action="store_true")
    p.add_argument("--interleave", action="store_true")
    p.add_argument("--arena-gib", type=int, default=4)
    p.add_argument("--malloc-tune", type=int, default=4,
                   help="GiB; sets MALLOC_MMAP_THRESHOLD_ above the fused kernel's per-call "
                        "intermediate so it stays on the heap, plus never-trim. 0 disables, "
                        "which makes the fused rungs measure page faults.")
    p.add_argument("--out-json", default="")
    a = p.parse_args()

    # glibc reads MALLOC_* once at process start, so these have to be set BEFORE the spawn --
    # putting them in instance() is too late. fused_experts_cpu at::empty's its [M*topk, *]
    # intermediates on every call (~671 MiB at the prefill shapes); on the defaults a block
    # that size is mmap'd and munmap'd per call, so each call page-faults the whole slab and
    # the fused rungs measure page-zeroing instead of the kernel. Keeping the block on the
    # heap and never trimming is what makes instance()'s arena pre-warm actually hold, and
    # without it the fused rungs came out 5x slower with allocfloor at 88% of mode0.
    if a.malloc_tune:
        os.environ["MALLOC_TRIM_THRESHOLD_"] = "-1"
        os.environ["MALLOC_MMAP_THRESHOLD_"] = str(a.malloc_tune * 1024 ** 3)
        os.environ["MALLOC_TOP_PAD_"] = str(1024 ** 3)
    print("malloc_tune=" + (f"{a.malloc_tune} GiB" if a.malloc_tune else "OFF"))

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
    print(f"regime={a.regime} K={K} N={REGIMES[a.regime][2]} E={E} topk={TOPK} tokens={m['num_tokens']} "
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

    m0 = res.get("mode0", {}).get("median_ms")
    for lo, hi, what in (("mode2", "mode0", "the gather's row copy + the topk reduce"),
                         ("mode3", "mode2", "the topk-weighted scatter"),
                         ("mode3", "mode0", "all three, i.e. everything unfused does not do")):
        if lo in res and hi in res:
            d = res[hi]["median_ms"] - res[lo]["median_ms"]
            print(f"\n{hi} - {lo}   = {d:>9.3f} ms  ({100 * d / m0:+.1f}% of mode0)  -- {what}")
    for nm in ("mode0", "mode2", "mode3"):
        if "unfused" in res and nm in res:
            print(f"unfused / {nm} = {res['unfused']['median_ms'] / res[nm]['median_ms']:>7.3f}x"
                  + ("   <-- the matmul comparison, all three terms out of both sides"
                     if nm == "mode3" else ""))
    # An allocation-dominated cell makes every mode difference unattributable, and that is
    # not visible from the mode numbers themselves -- it has to be said out loud.
    if "allocfloor" in res and m0:
        frac = res["allocfloor"]["median_ms"] / m0
        warn = ("   <-- ALLOCATION-DOMINATED: the fused rungs are timing page faults, not "
                "the kernel. Do not quote the mode differences." if frac > 0.25 else "")
        print(f"\nallocfloor / mode0 = {frac:.1%}{warn}")
    if a.out_json:
        json.dump(dict(regime=a.regime, meta=m, malloc_tune=a.malloc_tune, rungs=res),
                  open(a.out_json, "w"), indent=2)


if __name__ == "__main__":
    main()
