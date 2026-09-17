#!/usr/bin/env python3
"""All 40 layers of a lane: the apples-to-apples UNFUSED expert block, timed.

One rung only, the one that is comparable to `fused_experts_cpu`:

    a = hidden_states.index_select(0, routed_rows)     gather
    h = bmm_cpu(a, packed_w1)                          GEMM 1, VNNI-packed B
    y = bmm_cpu(silu(gate) * up, packed_w2)            SwiGLU + GEMM 2
    out.index_add_(0, routed_rows, y * topk_weight)    weighted scatter

Same input (one [num_tokens, K] hidden-state tensor), same output ([num_tokens, K]),
same packed weight layout, same op set. Packing runs at BUILD time, never inside the
timed region -- exactly where make_fused_runner puts it.

Why this is a separate script from gap_ladder.py: per-layer setup dominated a 40-layer
sweep of 40 separate invocations. Here the process is entered once per lane, the
activation / hidden-state / output buffers are sized to the lane's LARGEST layer and
re-used as prefix views, and only the weights (which are per-layer) are rebuilt.
Weights are drawn with uniform_ on bf16 rather than a fp32 randn plus a cast, which is
what made the build the slow part.

Layers are visited in a STRIDED order (0, 20, 10, 30, 5, ...) and each layer's row is
appended and flushed as it completes, so a run cut short still covers the whole range
instead of layers 0..k.
"""
import argparse, csv, os, statistics, sys, time
import multiprocessing as mp

RANGES = [range(0, 56), range(56, 112), range(112, 168), range(168, 224)]
LANES = {
    "thr_prefill": ("prefill", "320", "1"),
    "thr_decode": ("decode", "320", "1"),
    "rt_prefill": ("prefill", "1", "4"),
    "rt_decode": ("decode", "1", "4"),
}


def cells(results_csv, lane):
    """{layer: (groups, copies, mis, num_tokens, topk)} for the lane, one row per layer."""
    phase, batch, tp = LANES[lane]
    out = {}
    for r in csv.DictReader(open(results_csv)):
        if (r["phase"], r["batch"], r["tp"]) != (phase, batch, tp):
            continue
        L = int(r["layer"])
        if L in out:
            continue
        out[L] = ([tuple(int(x) for x in t.split("x")) for t in r["groups"].split(";")],
                  int(r["copies"]), int(r["moe_intermediate_size"]),
                  int(r["num_tokens"]), int(r["topk"]))
    return out


def strided(layers):
    """0, 20, 10, 30, 5, 25, ... so a truncated run still spans the range."""
    layers = sorted(layers)
    out, step = [], len(layers)
    while step > 1:
        step //= 2
        for i in range(0, len(layers), step):
            if layers[i] not in out:
                out.append(layers[i])
    for L in layers:
        if L not in out:
            out.append(L)
    return out


def instance(inst_idx, cores, barrier, q, args):
    os.sched_setaffinity(0, set(cores))
    os.environ["OMP_NUM_THREADS"] = str(len(cores))
    import torch
    import torch.nn.functional as F
    import sgl_kernel
    ops = sgl_kernel.common_ops
    torch.set_num_threads(len(cores))

    cs = cells(args.results_csv, args.lane)
    K = args.hidden
    g = torch.Generator().manual_seed(args.seed + inst_idx)

    # Buffers sized to the lane's largest layer, reused as prefix views for every layer.
    max_slots = max(max(n * t for n, t in gr) for gr, *_ in cs.values())
    max_mis = max(m for _gr, _c, m, _nt, _tk in cs.values())
    max_tok = max(nt for *_x, nt, _tk in cs.values())
    hidden_states = torch.empty((max_tok, K), dtype=torch.bfloat16).uniform_(-1, 1, generator=g)
    # Pre-gathered activation, the way make_batched_runner supplies it: each routed slot's
    # row already materialised. The `swiglu_only` rung reads THIS, so it does the GEMMs and
    # the SwiGLU but never the gather or the scatter.
    acts = torch.empty(max_slots * K, dtype=torch.bfloat16).uniform_(-1, 1, generator=g)
    out = torch.zeros((max_tok, K), dtype=torch.bfloat16)
    hbuf = torch.empty(max_slots * 2 * max_mis, dtype=torch.bfloat16)

    for L in strided(cs):
        groups, copies, mis, num_tokens, _topk = cs[L]
        N = mis
        idx, wts, packed = [], [], []
        for nexp, tokens in groups:
            n = nexp * tokens
            idx.append(torch.randint(0, num_tokens, (n,), generator=g, dtype=torch.int64))
            wts.append(torch.empty((n, 1), dtype=torch.bfloat16).uniform_(0.5, 1.5, generator=g))
        sets = []
        for _ in range(copies):
            one = []
            for nexp, tokens in groups:
                w1 = torch.empty((nexp, 2 * N, K), dtype=torch.bfloat16).uniform_(-.1, .1, generator=g)
                w2 = torch.empty((nexp, K, N), dtype=torch.bfloat16).uniform_(-.1, .1, generator=g)
                one.append((nexp, tokens, ops.convert_weight_packed(w1),
                            ops.convert_weight_packed(w2)))
            sets.append(one)
        state = dict(i=0)

        def body(st, b, nexp, tokens, p1, p2, a):
            h = hbuf[:nexp * tokens * 2 * N].view(nexp, tokens, 2 * N)
            ops.bmm_cpu(h, a.view(nexp, tokens, K), p1, True, None)
            gate, up = h.chunk(2, dim=-1)
            y = torch.empty((nexp, tokens, K), dtype=torch.bfloat16)
            ops.bmm_cpu(y, (F.silu(gate) * up).contiguous(), p2, True, None)
            return y

        def swiglu_only():
            st = sets[state["i"] % len(sets)]
            state["i"] += 1
            for b, (nexp, tokens, p1, p2) in enumerate(st):
                body(st, b, nexp, tokens, p1, p2, acts[:nexp * tokens * K])

        def e2e():
            st = sets[state["i"] % len(sets)]
            state["i"] += 1
            for b, (nexp, tokens, p1, p2) in enumerate(st):
                a = hidden_states.index_select(0, idx[b])
                y = body(st, b, nexp, tokens, p1, p2, a)
                out.index_add_(0, idx[b], y.view(nexp * tokens, K) * wts[b])

        rungs = {"swiglu_only": swiglu_only, "e2e": e2e}
        names = [n for n in args.rungs.split(",") if n in rungs]
        for n in names:                      # warm every rung before timing any
            for _ in range(args.warmup):
                rungs[n]()
        res = {}
        for n in names:
            barrier.wait()
            samples = []
            for _ in range(args.iters):
                barrier.wait()
                t0 = time.perf_counter()
                rungs[n]()
                samples.append((time.perf_counter() - t0) * 1000)
            res[n] = (statistics.median(samples), min(samples))
        q.put((inst_idx, L, res, len(groups), sum(n * t for n, t in groups)))
        del sets, idx, wts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-csv", required=True)
    p.add_argument("--lane", required=True, choices=sorted(LANES))
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--instances", type=int, default=4)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--rungs", default="swiglu_only,e2e")
    p.add_argument("--out-csv", required=True)
    a = p.parse_args()

    n_layers = len(cells(a.results_csv, a.lane))
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(a.instances)
    q = ctx.Queue()
    procs = [ctx.Process(target=instance, args=(i, list(RANGES[i]), barrier, q, a))
             for i in range(a.instances)]
    for pr in procs:
        pr.start()

    fh = open(a.out_csv, "w", newline="")
    w = csv.writer(fh)
    RN = [n for n in a.rungs.split(",")]
    w.writerow(["lane", "layer", "buckets", "slots"]
               + [f"{n}_{k}" for n in RN for k in ("median_ms", "min_ms", "imb")])
    fh.flush()
    pend = {}
    done = 0
    # One message per (instance, layer); a layer's row is written once all four report.
    for _ in range(a.instances * n_layers):
        inst_idx, L, res, nb, slots = q.get()
        pend.setdefault(L, []).append((res, nb, slots))
        if len(pend[L]) == a.instances:
            row = [a.lane, L, nb, slots]
            shown = []
            for n in RN:
                meds = [x[0][n][0] for x in pend[L]]
                m = statistics.median(meds)
                row += [round(m, 4), round(statistics.median(x[0][n][1] for x in pend[L]), 4),
                        round(max(meds) / m, 4)]
                shown.append(f"{n}={m:.3f}")
            w.writerow(row)
            fh.flush()
            done += 1
            print(f"[{done}/{n_layers}] layer {L}: " + "  ".join(shown), flush=True)
            del pend[L]
    for pr in procs:
        pr.join()
    fh.close()

    rows = list(csv.DictReader(open(a.out_csv)))
    for n in RN:
        ms = sorted(float(r[f"{n}_median_ms"]) for r in rows)
        print(f"{a.lane} {n}: layers={len(rows)} median={statistics.median(ms):.3f} ms "
              f"range={ms[0]:.3f}..{ms[-1]:.3f}", file=sys.stderr)
    if set(RN) >= {"swiglu_only", "e2e"}:
        d = sorted(float(r["e2e_median_ms"]) - float(r["swiglu_only_median_ms"]) for r in rows)
        print(f"{a.lane} gather+scatter (e2e - swiglu_only): median={statistics.median(d):.3f} ms "
              f"range={d[0]:.3f}..{d[-1]:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
