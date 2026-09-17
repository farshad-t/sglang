#!/usr/bin/env python3
"""Decompose the benchdnn-vs-batched gap into named, separately-measured terms.

benchdnn times two GEMMs. The batched leg's per-bucket body is

    h = torch.bmm(a, w1)             # [nexp, tokens, 2N]
    gate, up = h.chunk(2, dim=-1)    # two STRIDED views of h
    torch.bmm(F.silu(gate) * up, w2)

so the difference is not one thing. This runs a ladder on ONE cell, adding one
ingredient per rung, under the run's own instrument (4 concurrent 56-core instances,
barrier-synced, weights cloned across `copies` so they stream from DDR):

    bmm2        two bmm calls, no SwiGLU        -- vs benchdnn: bmm quality + no VNNI
                                                  prepack + per-call cost
    full        + chunk + silu*up               -- the batched leg as measured
    contig      + silu*up on CONTIGUOUS halves  -- isolates the strided-chunk penalty
    swiglu      silu*up alone, no bmm           -- SwiGLU's own cost
    gather      index_select of the routed rows -- the term UNFUSED was missing
    scatter     weighted index_add_ back        -- the other term UNFUSED was missing
    e2e         gather + full + scatter         -- APPLES-TO-APPLES UNFUSED: everything
                                                  fused_experts_cpu does, unfused
    opfloor     same op COUNT on 1-element      -- per-call floor (dispatch + OMP
                tensors                            fork/join at 56 threads)

Read it as: full-bmm2 should land on swiglu; bmm2-benchdnn is what torch costs over
oneDNN on identical shapes; opfloor bounds how much of either is just per-call cost; and
e2e is the number to put next to FUSED, since only e2e does the same work.

The gather indices are drawn across the whole token axis rather than replayed from the
run's routing table: this reproduces the gather's COST, not the exact routed slots.

Instance collapsing follows the campaign's uniform rule: MEDIAN over the four
instances, spread reported separately.
"""
import argparse, csv, json, os, statistics, sys
import multiprocessing as mp

RANGES = [range(0, 56), range(56, 112), range(112, 168), range(168, 224)]
LANES = {
    "thr_prefill": ("prefill", "320", "1"),
    "thr_decode": ("decode", "320", "1"),
    "rt_prefill": ("prefill", "1", "4"),
    "rt_decode": ("decode", "1", "4"),
}


def groups_for(results_csv, lane, layer):
    """The cell's bucket histogram, copies and token count, from the run's own CSV."""
    phase, batch, tp = LANES[lane]
    for r in csv.DictReader(open(results_csv)):
        if (r["phase"], r["batch"], r["tp"]) == (phase, batch, tp) and int(r["layer"]) == layer:
            g = [tuple(int(x) for x in tok.split("x")) for tok in r["groups"].split(";")]
            return (g, int(r["copies"]), int(r["moe_intermediate_size"]),
                    int(r["num_tokens"]), int(r["topk"]))
    raise SystemExit(f"no row for lane={lane} layer={layer} in {results_csv}")


def build(groups, mis, copies, seed, hidden, num_tokens):
    import torch
    K, N = hidden, mis
    g = torch.Generator().manual_seed(seed)
    acts = torch.randn(max(n * t * K for n, t in groups), generator=g,
                       dtype=torch.bfloat16) / 10
    # Stand-in for bmm1's output, so the `swiglu` rung can be timed without a GEMM.
    # Filled, not torch.empty: uninitialised memory can hold NaN/denormals, and those
    # change how long the elementwise pass takes.
    hbuf = torch.empty(max(n * t * 2 * N for n, t in groups),
                       dtype=torch.bfloat16).uniform_(-1, 1, generator=g)
    # The gather source and scatter destination the batched leg does not have: the block's
    # real hidden states, one row per token, which every bucket reads from and writes back.
    hidden_states = torch.randn((num_tokens, K), generator=g, dtype=torch.bfloat16) / 10
    out = torch.zeros((num_tokens, K), dtype=torch.bfloat16)
    # Row indices per bucket. A router does not hand one expert a contiguous token range,
    # so these are drawn across the whole token axis -- the access pattern is what costs.
    # This models the gather's COST, not the run's exact routing table.
    idx, wts = [], []
    for nexp, tokens in groups:
        n = nexp * tokens
        idx.append(torch.randint(0, num_tokens, (n,), generator=g, dtype=torch.int64))
        wts.append((torch.rand((n, 1), generator=g, dtype=torch.float32) + 0.5)
                   .to(torch.bfloat16))
    # torch.bmm wants B as [G, K, N]; convert_weight_packed wants the output-channel-major
    # [G, N, K] that fused_experts_cpu already uses. Both are built so the packed and
    # unpacked rungs differ ONLY in the weight layout.
    work = []
    for nexp, tokens in groups:
        w1 = (torch.randn((nexp, K, 2 * N), generator=g, dtype=torch.float32) / 10).to(torch.bfloat16)
        w2 = (torch.randn((nexp, N, K), generator=g, dtype=torch.float32) / 10).to(torch.bfloat16)
        work.append((nexp, tokens, w1, w2))
    sets = [work] + [[(n, t, a.clone(), b.clone()) for n, t, a, b in work]
                     for _ in range(copies - 1)]
    return acts, hbuf, sets, hidden_states, out, idx, wts


def make_rungs(groups, mis, copies, seed, hidden, num_tokens):
    import torch
    import torch.nn.functional as F
    K, N = hidden, mis
    acts, hbuf, sets, hidden_states, out, idx, wts = build(
        groups, mis, copies, seed, hidden, num_tokens)
    state = dict(i=0)

    def cur():
        s = sets[state["i"] % len(sets)]
        state["i"] += 1
        return s

    def a_of(nexp, tokens):
        return acts[:nexp * tokens * K].view(nexp, tokens, K)

    def bmm2():
        for nexp, tokens, w1, w2 in cur():
            h = torch.bmm(a_of(nexp, tokens), w1)
            torch.bmm(h[:, :, :N], w2)          # a view, no elementwise work

    def full():
        for nexp, tokens, w1, w2 in cur():
            h = torch.bmm(a_of(nexp, tokens), w1)
            gate, up = h.chunk(2, dim=-1)
            torch.bmm(F.silu(gate) * up, w2)

    def contig():
        for nexp, tokens, w1, w2 in cur():
            h = torch.bmm(a_of(nexp, tokens), w1)
            gate = h[:, :, :N].contiguous()
            up = h[:, :, N:].contiguous()
            torch.bmm(F.silu(gate) * up, w2)

    def swiglu():
        for nexp, tokens, _w1, _w2 in cur():
            h = hbuf[:nexp * tokens * 2 * N].view(nexp, tokens, 2 * N)
            gate, up = h.chunk(2, dim=-1)
            F.silu(gate) * up

    def gather():
        for b, (_nexp, _tokens, _w1, _w2) in enumerate(cur()):
            hidden_states.index_select(0, idx[b])

    def scatter():
        for b, (nexp, tokens, _w1, _w2) in enumerate(cur()):
            y = acts[:nexp * tokens * K].view(nexp * tokens, K)
            out.index_add_(0, idx[b], y * wts[b])

    def e2e():
        """The apples-to-apples UNFUSED: everything fused_experts_cpu does, unfused."""
        for b, (nexp, tokens, w1, w2) in enumerate(cur()):
            a = hidden_states.index_select(0, idx[b])
            h = torch.bmm(a.view(nexp, tokens, K), w1)
            gate, up = h.chunk(2, dim=-1)
            y = torch.bmm(F.silu(gate) * up, w2).view(nexp * tokens, K)
            out.index_add_(0, idx[b], y * wts[b])

    # ---- packed-weight variants -----------------------------------------------------
    # torch.bmm cannot take a pre-reordered B. sgl_kernel's bmm_cpu can: pass is_vnni=True
    # with a weight run through convert_weight_packed, which expects [G, N, K].
    ops = getattr(__import__("sgl_kernel"), "common_ops", None)
    packed = None
    if ops is not None and hasattr(ops, "bmm_cpu") and hasattr(ops, "convert_weight_packed"):
        packed = []
        for si, st in enumerate(sets):
            one = []
            for bi, (nexp, tokens, w1, w2) in enumerate(st):
                # w1 [G,K,2N] -> [G,2N,K]; w2 [G,N,K] is already output-channel-major
                p1 = ops.convert_weight_packed(w1.transpose(1, 2).contiguous())
                # GEMM2's torch-B is w2 [G,N,K] (K_in=N, N_out=K), so the packer wants
                # [G,K,N] -- output-channel major, same rule as GEMM1.
                p2 = ops.convert_weight_packed(w2.transpose(1, 2).contiguous())
                one.append((nexp, tokens, p1, p2))
            packed.append(one)
        pstate = dict(i=0)

        def pcur():
            st = packed[pstate["i"] % len(packed)]
            pstate["i"] += 1
            return st

        def bmm2_packed():
            for nexp, tokens, p1, p2 in pcur():
                h = torch.empty((nexp, tokens, 2 * N), dtype=torch.bfloat16)
                ops.bmm_cpu(h, a_of(nexp, tokens), p1, True, None)
                y = torch.empty((nexp, tokens, K), dtype=torch.bfloat16)
                # h[:, :, :N] is strided; bmm_cpu needs it contiguous, and the unpacked
                # bmm2 hands torch.bmm the view. Copy the SAME slice the packed second
                # GEMM will read, so the rung is not paying for a slice bmm2 skips.
                ops.bmm_cpu(y, h.chunk(2, dim=-1)[0].contiguous(), p2, True, None)

        def full_packed():
            for nexp, tokens, p1, p2 in pcur():
                h = torch.empty((nexp, tokens, 2 * N), dtype=torch.bfloat16)
                ops.bmm_cpu(h, a_of(nexp, tokens), p1, True, None)
                gate, up = h.chunk(2, dim=-1)
                y = torch.empty((nexp, tokens, K), dtype=torch.bfloat16)
                ops.bmm_cpu(y, (F.silu(gate) * up).contiguous(), p2, True, None)

        def e2e_packed():
            """Apples-to-apples UNFUSED with a VNNI-packed B: the fairest non-fused number."""
            for b, (nexp, tokens, p1, p2) in enumerate(pcur()):
                a = hidden_states.index_select(0, idx[b])
                h = torch.empty((nexp, tokens, 2 * N), dtype=torch.bfloat16)
                ops.bmm_cpu(h, a.view(nexp, tokens, K), p1, True, None)
                gate, up = h.chunk(2, dim=-1)
                y = torch.empty((nexp, tokens, K), dtype=torch.bfloat16)
                ops.bmm_cpu(y, (F.silu(gate) * up).contiguous(), p2, True, None)
                out.index_add_(0, idx[b], y.view(nexp * tokens, K) * wts[b])

    tiny_a = torch.ones((1, 1, 1), dtype=torch.bfloat16)
    tiny_w = torch.ones((1, 1, 2), dtype=torch.bfloat16)
    tiny_w2 = torch.ones((1, 1, 1), dtype=torch.bfloat16)

    def opfloor():
        for _nexp, _tokens, _w1, _w2 in cur():
            h = torch.bmm(tiny_a, tiny_w)
            gate, up = h.chunk(2, dim=-1)
            torch.bmm(F.silu(gate) * up, tiny_w2)

    r = dict(bmm2=bmm2, full=full, contig=contig, swiglu=swiglu,
             gather=gather, scatter=scatter, opfloor=opfloor)
    if packed is None:
        r["e2e"] = e2e
    else:
        # `e2e` is the headline apples-to-apples number, so it takes the DEFAULT weight
        # layout, which is packed -- the same choice fused_experts_cpu is measured under.
        r.update(e2e=e2e_packed, e2e_unpacked=e2e, bmm2_packed=bmm2_packed,
                 full_packed=full_packed)
    return r


def instance(idx, cores, barrier, q, args):
    os.sched_setaffinity(0, set(cores))
    os.environ["OMP_NUM_THREADS"] = str(len(cores))
    import torch
    torch.set_num_threads(len(cores))
    groups, copies, mis, num_tokens, _topk = groups_for(
        args.results_csv, args.lane, args.layer)
    rungs = make_rungs(groups, mis, copies, args.seed + idx, args.hidden, num_tokens)
    names = args.rungs.split(",")
    # Warm EVERY rung before timing ANY of them: lazy oneDNN init, the caching allocator
    # and OMP thread wake-up are one-time costs, and whichever rung ran first would
    # otherwise absorb them all and look slow.
    for name in names:
        for _ in range(args.warmup):
            rungs[name]()
    out = {}
    for name in names:
        fn = rungs[name]
        barrier.wait()
        samples = []
        import time
        for _ in range(args.iters):
            barrier.wait()
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1000)
        out[name] = dict(median_ms=statistics.median(samples), min_ms=min(samples),
                         iters=len(samples))
    q.put((idx, dict(out, _meta=dict(buckets=len(groups), copies=copies, mis=mis,
                                     num_tokens=num_tokens, cores=len(cores)))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-csv", required=True)
    p.add_argument("--lane", default="thr_prefill", choices=sorted(LANES))
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--instances", type=int, default=4)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--rungs",
                   default="bmm2,full,contig,swiglu,gather,scatter,e2e,opfloor")
    p.add_argument("--out-json", default="")
    a = p.parse_args()

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(a.instances)
    q = ctx.Queue()
    procs = [ctx.Process(target=instance, args=(i, list(RANGES[i]), barrier, q, a))
             for i in range(a.instances)]
    for pr in procs:
        pr.start()
    got = dict(q.get() for _ in range(a.instances))
    for pr in procs:
        pr.join()

    meta = got[0]["_meta"]
    print(f"lane={a.lane} layer={a.layer} buckets={meta['buckets']} copies={meta['copies']} "
          f"mis={meta['mis']} instances={a.instances}x{meta['cores']}c iters={a.iters}")
    groups, _c, _m, ntok, topk = groups_for(a.results_csv, a.lane, a.layer)
    slots = sum(n * t for n, t in groups)
    routed = ntok * topk
    print(f"routed slots: bucketed={slots} vs exact={routed} -> the bucketed decomposition "
          f"does {slots/routed:.3f}x the GEMM work fused does (padding to a common token "
          f"count per bucket); experts covered={sum(n for n, _ in groups)}")
    print(f"{'rung':10s} {'median ms':>11s} {'min ms':>10s} {'imb':>6s}")
    res = {}
    for name in a.rungs.split(","):
        med = [got[i][name]["median_ms"] for i in got]
        mn = [got[i][name]["min_ms"] for i in got]
        m = statistics.median(med)
        res[name] = dict(median_ms=m, min_ms=statistics.median(mn),
                         imb=max(med) / m)
        print(f"{name:10s} {m:>11.3f} {res[name]['min_ms']:>10.3f} {res[name]['imb']:>6.3f}")

    if "full" in res and "bmm2" in res:
        d = res["full"]["median_ms"] - res["bmm2"]["median_ms"]
        print(f"\nfull - bmm2      = {d:>10.3f} ms   (what the SwiGLU step adds in place)")
    if "swiglu" in res:
        print(f"swiglu alone     = {res['swiglu']['median_ms']:>10.3f} ms   "
              f"(SwiGLU measured on its own)")
    if "contig" in res and "full" in res:
        print(f"full - contig    = "
              f"{res['full']['median_ms'] - res['contig']['median_ms']:>10.3f} ms   "
              f"(strided chunk penalty)")
    if "e2e" in res and "full" in res:
        print(f"e2e - full      = "
              f"{res['e2e']['median_ms'] - res['full']['median_ms']:>10.3f} ms   "
              f"(gather + weighted scatter, the terms UNFUSED was missing)")
        print(f"e2e (apples-to-apples UNFUSED) = {res['e2e']['median_ms']:.3f} ms   "
              f"-- compare against FUSED directly")
    if "gather" in res and "scatter" in res:
        print(f"gather alone     = {res['gather']['median_ms']:>10.3f} ms\n"
              f"scatter alone    = {res['scatter']['median_ms']:>10.3f} ms")
    if "opfloor" in res:
        n = meta["buckets"] * 4
        print(f"opfloor          = {res['opfloor']['median_ms']:>10.3f} ms over ~{n} op "
              f"calls = {res['opfloor']['median_ms']*1000/n:.1f} us/op")

    if a.out_json:
        json.dump(dict(lane=a.lane, layer=a.layer, meta=meta, rungs=res),
                  open(a.out_json, "w"), indent=2)
        print(f"\nwrote {a.out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
