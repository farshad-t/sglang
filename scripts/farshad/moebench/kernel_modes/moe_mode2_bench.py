#!/usr/bin/env python3
"""Price each thing the fused MoE kernel does beyond the matmuls themselves.

One process, same weights / activation / warmup, one WALL BUDGET per rung:

    mode0     fused_experts_cpu(expert_batching_mode=0)  -- as it ships
    mode2     fused_experts_cpu(expert_batching_mode=2)  -- minus gather and reduce
    mode3     fused_experts_cpu(expert_batching_mode=3)  -- minus the scatter as well
    *_hoist   the same three with routing_buffer + workspace handed in by the caller
    allocfloor / routing_prep   the per-call at::empty and the routing sort, on their own

    mode0 - mode2   the gather's row copy + the [M,topk,K]->[M,K] topk reduce
    mode2 - mode3   the topk-weighted scatter: gemm2's permuted store back to routed-slot
                    order, which mode 2 still pays

THE NON-FUSED REFERENCE IS BENCHDNN, AND THERE IS NO UNFUSED RUNG IN HERE. Owner's call,
2026-10-07: the comparable to the fused-sglang table is the per-layer sum of benchdnn's own
matmul kernels. The `unfused`/`unfused_bare` rungs were REMOVED, not just disabled -- besides
being the wrong reference they actively corrupted the fused numbers, because the rungs are
interleaved inside one iteration loop and their per-iteration buffer allocations churned the
glibc arena for every OTHER rung timed beside them. Dropping them cut rt_decode's mode0 from
0.206 to 0.056 ms. Removing them also removes the per-bucket weight packing from setup.

To be comparable with benchdnn the sampling has to match it, so each rung gets `--budget-ms`
of wall time (benchdnn's `--max-ms-per-prb`, 2000 by default) rather than a fixed call count.

The three instances are three COPIES of the same kernel run at once, to measure a kernel on a
BUSY socket -- they are not TP ranks and nothing waits for a peer. So across instances the
reduction is the MEDIAN (the representative domain under load), after taking each instance's
own min across iterations (benchdnn's min_time convention). The benchdnn rollup must reduce
its three SNC nodes the same way or the ratio compares two different statistics.

What is left on one side only is the SwiGLU fusion, deliberately: the fused kernel keeps
gate|up in its fp32 accumulator and stores bf16 straight out of `silu_and_mul_stub`. That
is a real property of the implementation, so it stays counted.

Routing is the REAL per-layer expert histogram, read from the archbench expert-stats CSVs
through moe_stats at the revision the projection uses; every rung sees the identical table.
"""
import argparse, json, os, statistics, sys, time
import multiprocessing as mp

# One instance per SNC domain of socket 0. --skip-first-core drops cpu 0/32/64, which
# carry another user's hard-pinned inference schedulers when that server is up.
FULL = {0: (0, 32), 1: (32, 64), 2: (64, 96)}

E, K, TOPK = 256, 2048, 8
N = 512  # per-lane; set by instance() before any build
MOE_INTERMEDIATE_SIZE = 512

#   lane: (phase, batch, tp).  N is MOE_INTERMEDIATE_SIZE / tp -- rt is TP4, thr is TP1.
# The per-expert token counts are NOT synthetic: they are read per LAYER from the archbench
# expert-stats CSVs through moe_stats, the same reader and the same pinned stats revision
# the projection uses. (The balanced histograms that used to live here existed only so the
# unfused rung could batch each bucket into one bmm_cpu call. There is no unfused rung any
# more -- benchdnn is the non-fused reference -- so nothing forces equal-token buckets and
# the real distribution is used directly.)
LANES = {
    "rt_decode":   ("decode", 1, 4),
    "rt_prefill":  ("prefill", 1, 4),
    "thr_decode":  ("decode", 320, 1),
    "thr_prefill": ("prefill", 320, 1),
}
STATS_FILES = {
    "decode": "qwen3.5-35B-A3B_realprompt_decode.csv",
    "prefill": "qwen3.5-35B-A3B_realprompt_prefill_5buckets.csv",
}


def lane_histogram(lane, layer, stats_dir):
    """(num_tokens, {tokens_per_expert: n_experts}, N, slack) for one lane, one layer."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import collections
    import moe_stats

    phase, batch, tp = LANES[lane]
    csv_path = os.path.join(stats_dir, STATS_FILES[phase])
    hist = moe_stats.read_histogram(csv_path, phase, batch, layer)
    if not hist:
        raise SystemExit(f"no histogram for lane={lane} layer={layer} in {csv_path}")
    # Decode mass is exact; prefill is a 5-bucket average and only has to land in the band.
    moe_stats.validate(hist, E, batch * TOPK if phase == "decode" else None, phase,
                       f"{lane} layer {layer}")
    # Expand to per-expert counts so prefill's non-topk-divisible mass gets its slack
    # applied, then fold back to a histogram, which is what routing() consumes.
    counts, num_tokens, slack = moe_stats.expert_token_counts(hist, E, TOPK)
    folded = dict(collections.Counter(c for c in counts if c > 0))
    return num_tokens, folded, MOE_INTERMEDIATE_SIZE // tp, slack


def instance_cache_mib(cores):
    """L2 + L3 MiB visible to these cores, read from sysfs: each distinct cache (by its
    shared_cpu_list) counted once. On DMRX4 one 56-core SNC node is 28 x 4 MiB L2 + 320 MiB L3."""
    seen = {}
    for c in cores:
        for idx in ("index2", "index3"):
            d = f"/sys/devices/system/cpu/cpu{c}/cache/{idx}"
            try:
                key = (idx, open(f"{d}/shared_cpu_list").read().strip())
                size = open(f"{d}/size").read().strip()
            except OSError:
                continue
            unit = {"K": 1 / 1024, "M": 1, "G": 1024}[size[-1]]
            seen[key] = float(size[:-1]) * unit
    return sum(seen.values())


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


def build(num_tokens, counts, seed, ops, copies=1):
    import torch

    g = torch.Generator().manual_seed(seed)
    tid = routing(num_tokens, counts, TOPK)
    tw = (torch.rand((num_tokens, TOPK), generator=g, dtype=torch.float32) + 0.5) / TOPK
    hs = (torch.randn((num_tokens, K), generator=g, dtype=torch.float32) / 8).to(torch.bfloat16)
    w1 = (torch.randn((E, 2 * N, K), generator=g, dtype=torch.float32) / 24).to(torch.bfloat16)
    w2 = (torch.randn((E, K, N), generator=g, dtype=torch.float32) / 24).to(torch.bfloat16)

    routed = num_tokens * TOPK
    # fused_experts_cpu repacks w1/w2 on EVERY call unless is_vnni=True, and that repack is
    # 60-96% of its runtime here -- so pack once at setup and pass is_vnni=True, matching how
    # bench_moe_cpu.py's --prepack default runs. w1 is [E, 2N, K] and w2 is [E, K, N], both
    # already output-channel-major, which is the layout convert_weight_packed wants.
    pw1 = ops.convert_weight_packed(w1)
    pw2 = ops.convert_weight_packed(w2)
    # COLD WEIGHTS: `copies` private packed weight sets, and every fused call -- whatever the
    # rung -- takes the NEXT one from one shared counter. With copies-1 other sets walked in
    # between, a set's weights have been evicted from L2+L3 by the time it is used again, the
    # way the 39 other layers evict them end to end. copies=1 is the old warm-weight bench.
    # Same scheme as bench_moe_cpu.py --copies (make_fused_runner).
    wsets = [(pw1, pw2)] + [(pw1.clone(), pw2.clone()) for _ in range(copies - 1)]

    # Pre-built routing buffer and workspace for the hoisted rungs. The workspace is sized
    # generously (the kernel only checks >=) and first-touched here, so its page faults are
    # paid at setup rather than inside the timed call.
    nthreads = torch.get_num_threads()
    rbuf = ops.moe_routing_prep_cpu(tid, E) if hasattr(ops, "moe_routing_prep_cpu") else None
    # buffer_size_nbytes for bf16: ic1 + ic2 + A_tmp + C_tmp, at BLOCK_M = BLOCK_N = 32, plus a
    # MiB of slack. A blanket 2x factor would have cost 13 GiB per instance at thr prefill.
    ws_bytes = (routed * N * 2 + routed * K * 2 + nthreads * 32 * K * 2
                + nthreads * 2 * 32 * 32 * 4 + (1 << 20))
    ws = torch.empty(ws_bytes, dtype=torch.int8)
    ws.view(-1)[::4096] = 1
    return dict(tid=tid, tw=tw, hs=hs, wsets=wsets, ctr=[0], routed=routed, rbuf=rbuf, ws=ws)


def make_rungs(st, ops):
    import torch
    import torch.nn.functional as F

    fe = torch.ops.sgl_kernel.fused_experts_cpu
    M = st["hs"].size(0)
    nthreads = torch.get_num_threads()

    has_mode = "expert_batching_mode" in str(fe.default._schema)

    has_hoist = "routing_buffer" in str(fe.default._schema)

    def fused(mode, hoist=False):
        """hoist=True passes a pre-built routing buffer and workspace, and runs in place, so
        the routing sort and the per-call at::empty are out of the timed region. Everything
        else is untouched, which is what makes mode3 - mode3(hoist) their measured cost."""
        if hoist and (not has_hoist or st["rbuf"] is None):
            return None
        if not has_mode and mode != 0:
            return None
        argsets = []
        for pw1, pw2 in st["wsets"]:
            args = [st["hs"], pw1, pw2, st["tw"], st["tid"], bool(hoist), 0,
                    None, None, None, None, None, None, None, None, None, True, None]
            # the pristine build predates expert_batching_mode; its schema takes 18 args
            if has_mode:
                args = args + [mode]
            if has_hoist:
                args = args + ([st["rbuf"], st["ws"]] if hoist else [None, None])
            argsets.append(args)
        ctr = st["ctr"]  # ONE counter shared by every rung, so interleaved rungs stay cold

        def run():
            fe(*argsets[ctr[0] % len(argsets)])
            ctr[0] += 1
        return run

    has_prep = hasattr(ops, "moe_routing_prep_cpu")
    # fused_experts_cpu allocates out_hidden_states plus the thread scratch (A_tmp, C_tmp)
    # on top of ic1/ic2. What that costs is the FIRST TOUCH, not the malloc, so allocfloor
    # has to touch every page -- an untouched torch.empty would be free and match nothing.
    extra_bytes = M * K * 2 + nthreads * 32 * K * 2 + nthreads * 2 * 32 * 32 * 4

    # what fused_experts_cpu's own at::empty costs: same byte count, same first touch
    nbytes = st["routed"] * N * 2 + st["routed"] * K * 2 + extra_bytes

    def allocfloor():
        b = torch.empty(nbytes, dtype=torch.uint8)
        b.view(-1)[::4096] = 1

    def routing_prep():
        ops.moe_routing_prep_cpu(st["tid"], E)

    r = dict(mode0=fused(0), mode2=fused(2), mode3=fused(3), allocfloor=allocfloor)
    if has_prep:
        r["routing_prep"] = routing_prep
    # The subtractive ladder: each rung removes ONE more step from the real kernel, so its
    # cost is measured in situ. Subtracting the standalone `allocfloor` / `routing_prep`
    # instead would over-count -- at bs1 those two alone already exceed mode0.
    # The hoisted variants are available but are NOT in the default rung list: hoisting is not
    # "mode0 minus the allocation", it penalises the modes that do the permuted store into ic2
    # (thr_prefill mode0 +23%, mode2 +14%) while helping mode3 (-7%), so it changes the
    # kernel's memory behaviour rather than isolating one term.
    for name, fn in (("mode0_hoist", fused(0, hoist=True)),
                     ("mode2_hoist", fused(2, hoist=True)),
                     ("mode3_hoist", fused(3, hoist=True))):
        if fn is not None:
            r[name] = fn
    return r


def instance(idx, cores, barrier, q, a, nit):
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
    num_tokens, counts, N, slack = lane_histogram(a.lane, a.layer, a.stats_dir)
    st = build(num_tokens, counts, a.seed + idx, ops, a.copies)
    # bytes ONE call reads from the weights: only the routed experts' rows are touched
    touched = sum(counts.values()) * (2 * N * K + K * N) * 2
    rungs = make_rungs(st, ops)
    names = [n for n in a.rungs.split(",") if rungs.get(n) is not None]
    # Warm every rung before timing any of them: oneDNN init and OMP thread wake-up are
    # one-time, and whichever rung ran first would otherwise absorb all of it. The warmup is
    # also timed, to size the measured loop the way benchdnn sizes its own: a per-rung WALL
    # BUDGET rather than a fixed call count, so a 0.1 ms rung and an 800 ms rung each get a
    # comparable amount of sampling instead of a comparable number of calls.
    est = {}
    for nm in names:
        t0 = time.perf_counter()
        for _ in range(a.warmup):
            rungs[nm]()
        est[nm] = (time.perf_counter() - t0) * 1000 / max(1, a.warmup)

    # The iteration counts MUST be identical across instances: every timed call sits behind a
    # barrier, so an instance that ran one more or one fewer would hang its peers. Instance 0
    # sizes them from its own estimate and publishes; everyone reads the same array.
    if idx == 0:
        for i, nm in enumerate(names):
            n = int(a.budget_ms / est[nm]) if est[nm] > 0 else a.max_iters
            nit[i] = max(a.min_iters, min(a.max_iters, n))
    barrier.wait()
    iters = {nm: nit[i] for i, nm in enumerate(names)}

    samples = {nm: [] for nm in names}
    if a.interleave:
        # Round-robin the rungs INSIDE the iteration loop, so a drift over the run (AMX
        # frequency ramp, page placement settling, allocator growth) lands on every rung
        # equally instead of on whichever one is timed first. A rung whose budget is already
        # spent is skipped -- the schedule is a pure function of the shared `iters`, so every
        # instance skips the same rung in the same round and the barriers stay aligned.
        barrier.wait()
        for it in range(max(iters.values())):
            order = names if it % 2 == 0 else names[::-1]
            for nm in order:
                if it >= iters[nm]:
                    continue
                barrier.wait()
                t0 = time.perf_counter()
                rungs[nm]()
                samples[nm].append((time.perf_counter() - t0) * 1000)
    else:
        for nm in names:
            barrier.wait()
            for _ in range(iters[nm]):
                barrier.wait()
                t0 = time.perf_counter()
                rungs[nm]()
                samples[nm].append((time.perf_counter() - t0) * 1000)
    out = {nm: dict(median_ms=statistics.median(v), min_ms=min(v), max_ms=max(v),
                    iters=len(v), est_ms=est[nm]) for nm, v in samples.items()}
    q.put((idx, dict(out, _meta=dict(cores=len(cores), num_tokens=num_tokens, N=N,
                                     routed=st["routed"], hist=counts, slack=slack,
                                     active_experts=sum(counts.values()), copies=a.copies,
                                     touched_mib=touched / 2**20,
                                     reuse_distance_mib=(a.copies - 1) * touched / 2**20,
                                     cache_mib=instance_cache_mib(cores)))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--so-dir", default="/tmp/moe_cm_mode2")
    p.add_argument("--lane", default="rt_prefill", choices=sorted(LANES))
    p.add_argument("--layer", type=int, default=0, help="which MoE layer's real histogram")
    p.add_argument("--stats-dir", default="/home/farshad/.cache/moebench/archbench/files/"
                   "5ae86154d545244f6239bf966aeb694899fe53ab/tools/cpu/config/qwen35/expert_stats",
                   help="archbench expert_stats dir; the pinned revision the projection uses")
    p.add_argument("--instances", type=int, default=3)
    p.add_argument("--copies", type=int, default=1,
                   help="packed weight sets rotated per call; 1 = warm weights. Pick it so "
                        "(copies-1) x touched MiB exceeds the instance's L2+L3 (cold weights)")
    p.add_argument("--cores-per-instance", type=int, default=32,
                   help="one instance per SNC domain: GNR 32, DMRX4 56")
    p.add_argument("--warmup", type=int, default=3)
    # Matches benchdnn's --max-ms-per-prb: give every rung the same wall budget, not the same
    # call count, so the two sides are sampled the same way before their mins are compared.
    p.add_argument("--budget-ms", type=float, default=2000.0)
    p.add_argument("--min-iters", type=int, default=3)
    p.add_argument("--max-iters", type=int, default=5000)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--rungs", default="mode0,mode2,mode3,allocfloor,routing_prep")
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
    c = a.cores_per_instance
    ranges = {i: range(i * c + (1 if a.skip_first_core else 0), (i + 1) * c)
              for i in range(a.instances)}
    nit = ctx.Array("i", len(a.rungs.split(",")))
    procs = [ctx.Process(target=instance, args=(i, list(ranges[i]), barrier, q, a, nit))
             for i in range(a.instances)]
    for pr in procs:
        pr.start()
    got = dict(q.get() for _ in range(a.instances))
    for pr in procs:
        pr.join()

    m = got[0]["_meta"]
    ph, bs, tp = LANES[a.lane]
    print(f"lane={a.lane} layer={a.layer} phase={ph} batch={bs} tp={tp} K={K} N={m['N']} "
          f"E={E} topk={TOPK} tokens={m['num_tokens']} routed={m['routed']} "
          f"active_experts={m['active_experts']} slack={m['slack']} "
          f"instances={a.instances}x{m['cores']}c budget={a.budget_ms:.0f}ms/rung")
    print(f"  weights: copies={m['copies']} touched/call={m['touched_mib']:.1f} MiB "
          f"reuse distance={m['reuse_distance_mib']:.0f} MiB  "
          f"instance L2+L3={m['cache_mib']:.0f} MiB")
    # End to end, each MoE layer's weights are evicted by the other 39 layers before that
    # layer runs again, so they come from DDR. A reuse distance that fits the instance's
    # L2+L3 lets the weights stay cache-resident between calls, which is not that.
    warm = m["reuse_distance_mib"] <= m["cache_mib"]
    if warm:
        print(f"WARNING: WEIGHTS ARE CACHE-RESIDENT. --copies {m['copies']} gives a reuse distance "
              f"of {m['reuse_distance_mib']:.0f} MiB <= the instance's L2+L3 "
              f"({m['cache_mib']:.0f} MiB), so the weights stay in cache between calls. That is "
              f"NOT how end-to-end inference runs the MoE layers: there each layer's weights are "
              f"evicted by the other 39 layers and come from DDR. Raise --copies above "
              f"{1 + m['cache_mib'] / m['touched_mib']:.0f} for DDR-resident weights.")
    print(f"  real histogram {{tokens_per_expert: n_experts}} = {m['hist']}")
    # `best_ms`: each instance's own min across iterations (benchdnn's min_time convention),
    # reduced across instances by the MEDIAN. The three instances are three copies of the same
    # kernel loading the socket, not TP ranks -- nothing waits for a peer -- so the question
    # they answer is "what does this kernel cost on a busy socket", and the representative
    # domain is the median one. `max` would report the unluckiest domain; `min` would report
    # the luckiest and so partly undo the loading the three instances exist to create.
    # The benchdnn rollup applies the identical reduction over its three SNC nodes.
    print(f"{'rung':14s} {'best ms':>10s} {'median ms':>11s} {'worst/med':>10s} {'iters':>7s}")
    res = {}
    for nm in [n for n in a.rungs.split(",") if n in got[0]]:
        med = [got[i][nm]["median_ms"] for i in got]
        mins = [got[i][nm]["min_ms"] for i in got]
        mm = statistics.median(med)
        res[nm] = dict(best_ms=statistics.median(mins), median_ms=mm, min_ms=min(mins),
                       max_ms=max(mins), imb=max(med) / mm,
                       iters=got[0][nm]["iters"], per_instance=med, per_instance_min=mins)
        print(f"{nm:14s} {res[nm]['best_ms']:>10.3f} {mm:>11.3f} {res[nm]['imb']:>10.3f} "
              f"{res[nm]['iters']:>7d}")

    m0 = res.get("mode0", {}).get("best_ms")
    for lo, hi, what in (("mode2", "mode0", "the gather's row copy + the topk reduce"),
                         ("mode3", "mode2", "the topk-weighted scatter"),
                         ("mode3", "mode0", "all three: gather + topk reduce + weighted scatter")):
        if lo in res and hi in res:
            d = res[hi]["best_ms"] - res[lo]["best_ms"]
            print(f"\n{hi} - {lo}   = {d:>9.3f} ms  ({100 * d / m0:+.1f}% of mode0)  -- {what}")
    # An allocation-dominated cell makes every mode difference unattributable, and that is
    # not visible from the mode numbers themselves -- it has to be said out loud.
    if "allocfloor" in res and m0:
        frac = res["allocfloor"]["best_ms"] / m0
        warn = ("   <-- ALLOCATION-DOMINATED: the fused rungs are timing page faults, not "
                "the kernel. Do not quote the mode differences." if frac > 0.25 else "")
        print(f"\nallocfloor / mode0 = {frac:.1%}{warn}")
    if a.out_json:
        json.dump(dict(lane=a.lane, layer=a.layer, phase=ph, batch=bs, tp=tp,
                       weights_cache_resident=warm,
                       stats_dir=a.stats_dir, meta=m, malloc_tune=a.malloc_tune, rungs=res),
                  open(a.out_json, "w"), indent=2)


if __name__ == "__main__":
    main()
