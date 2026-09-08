#!/usr/bin/env python3
"""Standalone CPU MoE benchmark driven by archbench expert-stats CSVs.

WHAT THIS IS FOR
----------------
The archbench projection models one MoE layer as a set of batched GEMMs, one per
bucket of the measured expert histogram:

    groups = [expert_dist(num_experts, avg_tokens_per_expert) for each bucket]

and each group costs one batched GEMM [nexp, tokens, K] x [nexp, K, 2N] (fused
gate+up) plus [nexp, tokens, N] x [nexp, N, K] (down). This benchmark runs the
SAME distribution on real silicon, so the projection can be compared against a
measured number rather than against a whole-model end-to-end time that mixes in
attention, KV, comms and framework overhead.

It measures two things from one histogram, which is the point:

  --mode fused     sglang's `fused_experts_cpu` -- what production actually runs.
  --mode batched   the projection's own decomposition, as torch batched GEMMs.
                   Same FLOPs, same shapes, one bmm per histogram bucket.

fused vs batched isolates KERNEL QUALITY (does sglang's sorted-token AMX path
beat/lag a plain batched GEMM at these shapes?) from MODEL ERROR (does the
bucket-averaged decomposition predict the real kernel?). Running only one of them
conflates the two.

bf16 only, by design: this is the numeric the projection covers here, and the
design/test box has no AMX-fp8.

WHERE THE STATS COME FROM
-------------------------
Straight out of the archbench repo (branch `farshad/llama4-qwen3-qwen35`), resolved
through the same `expert_stats_mapping.json` the projection uses, so the filename is
never hardcoded here. The resolved commit sha is printed and written into the output
CSV, so a measurement can always be tied back to the exact stats revision it used.
See archbench_stats.py for the offline escapes.

PORTABILITY
-----------
`fused_experts_cpu`'s signature drifts between sglang builds (the qwen35-bkc
container carries an extra `a1_scale` that upstream main does not). The call is
therefore assembled by INTROSPECTING the registered schema and binding arguments
BY NAME, so the same script runs unmodified on another box with another build.

USAGE
-----
  # single cell
  python bench_moe_cpu.py --phase decode --batch 64 --layer 0

  # sweep, both modes, write a CSV
  python bench_moe_cpu.py --phase decode --batch 1,64,128,320 --layer 0,10,20,39 \
      --mode both --out results.csv

  # describe the decomposition without touching the kernel (no sgl_kernel needed)
  python bench_moe_cpu.py --phase prefill --batch 64 --layer 0 --dry-run
"""

import argparse
import csv
import os
import statistics
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import archbench_stats  # noqa: E402
import moe_stats  # noqa: E402

# Qwen3.5-35B-A3B. hidden_size / num_experts / moe_intermediate_size /
# num_experts_per_tok / num_hidden_layers, from the model config.json.
DEFAULT_MODEL = dict(hidden_size=2048, num_experts=256, moe_intermediate_size=512,
                     topk=8, num_layers=40)

# Only used with --stats-dir, where there is no mapping json to consult.
STATS_DIR_FILES = {
    ("per_layer", "decode"): "qwen3.5-35B-A3B_realprompt_decode.csv",
    ("per_layer", "prefill"): "qwen3.5-35B-A3B_realprompt_prefill_5buckets.csv",
    ("layers_averaged", "decode"): "qwen3.5-35B-A3B-bf16_tp1_bs20_decode_layers_averaged.csv",
    ("layers_averaged", "prefill"):
        "qwen3.5-35B-A3B-bf16_tp1_bs20_prefill_layers_averaged_5buckets.csv",
}


# ---------------------------------------------------------------------------
# routing table
# ---------------------------------------------------------------------------

def build_topk_ids(counts: List[int], num_tokens: int, topk: int):
    """Build a [num_tokens, topk] routing table whose per-expert token tallies equal
    `counts` exactly.

    Construction: lay the routed-slot list (each expert id repeated count times,
    experts in descending-count order) into the table COLUMN-MAJOR, i.e.
    table[r, c] = flat[c * num_tokens + r].

    Why this cannot put the same expert twice in one row: each expert occupies a
    CONTIGUOUS run in `flat`, and column-major placement sends flat indices i and j
    to the same row only when (i - j) is a multiple of num_tokens. Within one run
    the index gap is < count <= num_tokens, so the only such pair is i == j.
    `expert_token_counts` already enforces max(count) <= num_tokens, so every row
    holds topk DISTINCT experts -- a table a real router could have produced.
    """
    import torch

    flat: List[int] = []
    for eid in sorted(range(len(counts)), key=lambda e: (-counts[e], e)):
        flat.extend([eid] * counts[eid])
    assert len(flat) == num_tokens * topk, (len(flat), num_tokens * topk)

    # Column-major unflatten: fill [topk, num_tokens] row-major, then transpose, so
    # ids[r, c] == flat[c * num_tokens + r].
    ids = torch.tensor(flat, dtype=torch.int32).view(topk, num_tokens).t().contiguous()

    # Self-check: the table must reproduce the histogram and route no token to the
    # same expert twice. Both are O(M*topk) and worth paying once per cell -- a
    # silent violation here would make every measured number meaningless.
    got = torch.bincount(ids.reshape(-1).to(torch.int64), minlength=len(counts))
    exp = torch.tensor(counts, dtype=got.dtype)
    if not torch.equal(got, exp):
        bad = int((got != exp).nonzero()[0])
        raise AssertionError(f"routing table does not match the histogram at expert {bad}: "
                             f"got {int(got[bad])} tokens, want {counts[bad]}")
    dup = (ids.sort(dim=1).values.diff(dim=1) == 0).any(dim=1)
    if bool(dup.any()):
        r = int(dup.nonzero()[0])
        raise AssertionError(f"token {r} routed to a duplicate expert: {ids[r].tolist()}")
    return ids


# ---------------------------------------------------------------------------
# sglang kernel, bound by name so the script survives signature drift
# ---------------------------------------------------------------------------

class FusedExpertsCaller:
    """Call `fused_experts_cpu` with arguments bound BY NAME, on whichever of the two
    registration styles the installed build uses:

      torch.ops   sgl_kernel registers through TORCH_LIBRARY, so the op carries a real
                  schema and argument names/types come from it. Upstream main and the
                  qwen35-bkc container.
      pybind      the CPU-optimised fork (llama4_optimzed_cpu and friends) registers
                  through PYBIND11_MODULE as sgl_kernel.common_ops.*, which has no
                  schema. Names come from the Python wrapper in sgl_kernel/cpu.py via
                  inspect.signature instead.

    Both are needed: the signature differs across builds in count AND in kind (the fork
    takes `use_int8_w8a8: bool` where newer builds take `moe_comp_method: int`, and the
    fork has no `activation` argument at all -- SiLU is compiled in). Binding by name
    off whatever the build actually exposes is what lets one script run on both.
    """

    # Everything the harness can supply, keyed by argument name. Any name a build asks
    # for that is absent here falls back to that argument's default, or None if optional.
    def _supplied(self, a, w1, w2, topk_weights, topk_ids) -> Dict[str, object]:
        return {
            "hidden_states": a,
            "x": a,              # the fork's wrapper calls it x
            "input": a,          # some older builds call it input
            "w1": w1,
            "w13_weight": w1,    # the fork's wrapper name for the fused gate+up weight
            "w2": w2,
            "w2_weight": w2,
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "inplace": self.inplace,
            # Unquantized bf16. Spelled `moe_comp_method`/`quant`/`quant_method` (0) in
            # newer builds and `use_int8_w8a8` (False) in the fork.
            "moe_comp_method": 0,
            "quant": 0,
            "quant_method": 0,
            "use_int8_w8a8": False,
            "use_fp8_w8a16": False,
            "is_vnni": self.prepack,
            "activation": self.activation,
        }

    def __init__(self, prepack: bool, inplace: bool, activation: str):
        import torch
        self.torch = torch
        self.prepack = prepack
        self.inplace = inplace
        self.activation = activation

        self.kind = None
        # TORCH_LIBRARY registration runs at `import sgl_kernel`, so probing
        # torch.ops first reports "absent" on a build that has the op: in the
        # qwen35-bkc container that sent us down the pybind path, which that build
        # does not have either. Import before looking, tolerate a build with no
        # importable package (the pybind branch reports it properly).
        try:
            import sgl_kernel  # noqa: F401
        except Exception:
            pass
        op = getattr(getattr(torch.ops, "sgl_kernel", None), "fused_experts_cpu", None)
        if op is not None:
            self.kind = "torch.ops"
            self.op = op
            self.schema = op.default._schema
            self.arg_names = [x.name for x in self.schema.arguments]
            self.pack = torch.ops.sgl_kernel.convert_weight_packed
        else:
            import inspect
            try:
                from sgl_kernel import cpu as sk_cpu
                import sgl_kernel
            except ImportError as exc:
                raise RuntimeError(
                    "no sgl_kernel CPU MoE kernel found: neither "
                    "torch.ops.sgl_kernel.fused_experts_cpu nor sgl_kernel.cpu. Install "
                    "a CPU-enabled sgl-kernel build, or use --mode batched / --dry-run."
                ) from exc
            self.kind = "pybind"
            self.op = sk_cpu.fused_experts
            self.schema = inspect.signature(self.op)
            self.arg_names = list(self.schema.parameters)
            self.pack = sgl_kernel.common_ops.convert_weight_packed
        if prepack and self.pack is None:
            raise RuntimeError("prepacking requested but convert_weight_packed is not "
                               "available in this build (pass --no-prepack)")

    def prepack_weight(self, w):
        return self.pack(w) if self.prepack else w

    def describe(self) -> str:
        if self.kind == "torch.ops":
            return f"[torch.ops] {self.schema}"
        return f"[pybind] fused_experts{self.schema}"

    def __call__(self, a, w1, w2, topk_weights, topk_ids):
        supplied = self._supplied(a, w1, w2, topk_weights, topk_ids)
        args = []
        if self.kind == "torch.ops":
            for arg in self.schema.arguments:
                if arg.name in supplied:
                    args.append(supplied[arg.name])
                elif arg.has_default_value():
                    args.append(arg.default_value)
                elif "Optional" in str(arg.type) or str(arg.type).endswith("?"):
                    args.append(None)
                else:
                    raise RuntimeError(
                        f"fused_experts_cpu argument {arg.name!r} ({arg.type}) is required "
                        f"by this build but the harness cannot fill it. Schema:\n"
                        f"  {self.schema}")
        else:
            import inspect
            for name, param in self.schema.parameters.items():
                if name in supplied:
                    args.append(supplied[name])
                elif param.default is not inspect.Parameter.empty:
                    args.append(param.default)
                else:
                    raise RuntimeError(
                        f"fused_experts_cpu argument {name!r} is required by this build "
                        f"but the harness cannot fill it. Signature:\n  {self.schema}")
        return self.op(*args)


# ---------------------------------------------------------------------------
# the two things we time
# ---------------------------------------------------------------------------

def time_mixed(runners, warmup: int, iters: int, barrier=None) -> List[Dict[str, float]]:
    """Time several cells INTERLEAVED -- one visit to each, then round again.

    Visiting every layer once per iteration is what sends a layer's weights back to
    DDR: by the time the loop returns to a layer, the other layers have walked over
    the whole cache hierarchy, which is exactly what a rank stepping through 40
    layers does in production. Timing one cell to completion leaves its working set
    resident instead -- 12 MiB of active experts at decode bs1, against the 320 MiB
    L3 slice a 56-core instance owns -- and measures a cache-resident kernel.

    `barrier` re-syncs the concurrent instances before EVERY call. Without it,
    instances that start together drift apart within a few iterations (per-layer
    work differs and nothing pulls them back) and one can then be timed while its
    neighbours sit between kernels, measuring a partly-idle socket. With it, every
    timed window holds all instances on the SAME cell.
    """
    def sync():
        if barrier is not None:
            barrier.wait()

    for _ in range(warmup):
        for fn in runners:
            sync()
            fn()
    samples: List[List[float]] = [[] for _ in runners]
    for _ in range(iters):
        for i, fn in enumerate(runners):
            sync()
            t0 = time.perf_counter()
            fn()
            samples[i].append((time.perf_counter() - t0) * 1e3)
    return [summarize_samples(s) for s in samples]


def summarize_samples(samples: List[float]) -> Dict[str, float]:
    return dict(median_ms=statistics.median(samples),
                min_ms=min(samples),
                mean_ms=statistics.fmean(samples),
                p90_ms=sorted(samples)[max(0, int(0.9 * len(samples)) - 1)],
                iters=len(samples))


def time_it(fn, warmup: int, iters: int, barrier=None) -> Dict[str, float]:
    return time_mixed([fn], warmup, iters, barrier)[0]


def fused_tensors(counts, num_tokens, model, caller, seed):
    import torch
    K, N, E, topk = (model["hidden_size"], model["moe_intermediate_size"],
                     model["num_experts"], model["topk"])
    g = torch.Generator().manual_seed(seed)
    a = (torch.randn((num_tokens, K), generator=g, dtype=torch.float32) / 10).to(torch.bfloat16)
    w1 = (torch.randn((E, 2 * N, K), generator=g, dtype=torch.float32) / 10).to(torch.bfloat16)
    w2 = (torch.randn((E, K, N), generator=g, dtype=torch.float32) / 10).to(torch.bfloat16)
    topk_ids = build_topk_ids(counts, num_tokens, topk)
    # Uneven weights, so a kernel that drops or mis-attributes a routed slot cannot
    # cancel out against a uniform 1/topk. Rows still sum to 1 as a router's would.
    tw = torch.rand((num_tokens, topk), generator=g, dtype=torch.float32) + 0.5
    topk_weights = tw / tw.sum(dim=1, keepdim=True)
    return a, w1, w2, topk_weights, topk_ids


def make_fused_runner(counts, num_tokens, model, caller, seed, copies: int = 1):
    """Cycle the call over `copies` private prepacked weight sets.

    The extra sets are clones, not fresh randoms: what has to differ between them is
    the ADDRESS, not the values -- an AMX GEMM's cost does not depend on the bits it
    multiplies, and a clone is one memcpy where a randn is a fp32 draw plus a cast.
    The activations are deliberately shared: they are small and are cache-resident in
    production too.
    """
    a, w1, w2, topk_weights, topk_ids = fused_tensors(counts, num_tokens, model,
                                                      caller, seed)
    first = (caller.prepack_weight(w1), caller.prepack_weight(w2))
    del w1, w2
    sets = [first] + [(first[0].clone(), first[1].clone()) for _ in range(copies - 1)]
    state = dict(i=0)

    def run():
        pw1, pw2 = sets[state["i"] % len(sets)]
        state["i"] += 1
        return caller(a, pw1, pw2, topk_weights, topk_ids)

    return run


def check_fused(counts, num_tokens, model, caller, seed, activation="silu"):
    """Compare fused_experts_cpu against a plain per-expert fp32 reference driven by
    the SAME routing table. This is what rules out a kernel (or a harness bug) that
    quietly does less work than the histogram says -- which would show up as a
    flattering GFLOP/s rather than as a failure."""
    import torch
    import torch.nn.functional as F
    a, w1, w2, topk_weights, topk_ids = fused_tensors(counts, num_tokens, model,
                                                      caller, seed)
    # Snapshot the activations BEFORE the call. With --inplace the kernel writes its
    # output into `a` and hands the same storage back, so an `a.float()` taken
    # afterwards would build the reference out of the kernel's own output and the
    # check would grade the kernel against itself.
    af = a.float()
    got = caller(a, caller.prepack_weight(w1), caller.prepack_weight(w2),
                 topk_weights, topk_ids).to(torch.float32)

    ref = torch.zeros_like(af)
    N = model["moe_intermediate_size"]
    # Upcast one expert's weights at a time: the whole stack in fp32 would be ~3 GB
    # and is only read once each.
    for e in range(model["num_experts"]):
        hit = (topk_ids == e).nonzero()
        if hit.numel() == 0:
            continue
        rows, slots = hit[:, 0], hit[:, 1]
        h = af[rows] @ w1[e].float().t()        # [n, 2N], gate|up fused
        gate, up = h[:, :N], h[:, N:]
        act = F.silu(gate) if activation == "silu" else F.gelu(gate)
        out = (act * up) @ w2[e].float().t()    # [n, K]
        ref.index_add_(0, rows, out * topk_weights[rows, slots].unsqueeze(1))

    err = (got - ref).abs().max().item()
    scale = max(ref.abs().max().item(), 1e-6)
    return err / scale, err


def make_batched_runner(groups, model, seed, copies: int = 1):
    """The projection's own decomposition: one batched GEMM pair per histogram bucket.

    Weights are cloned across `copies` so this leg streams from DDR on the same terms
    as the fused one -- comparing a cache-resident bmm against a DDR-bound kernel
    would attribute a memory-hierarchy difference to kernel quality.
    """
    import torch
    import torch.nn.functional as F
    K, N = model["hidden_size"], model["moe_intermediate_size"]
    g = torch.Generator().manual_seed(seed)
    work = []
    for nexp, tokens in groups:
        a = (torch.randn((nexp, tokens, K), generator=g, dtype=torch.float32) / 10).to(torch.bfloat16)
        w1 = (torch.randn((nexp, K, 2 * N), generator=g, dtype=torch.float32) / 10).to(torch.bfloat16)
        w2 = (torch.randn((nexp, N, K), generator=g, dtype=torch.float32) / 10).to(torch.bfloat16)
        work.append((a, w1, w2))
    sets = [work] + [[(a, w1.clone(), w2.clone()) for a, w1, w2 in work]
                     for _ in range(copies - 1)]
    state = dict(i=0)

    def run():
        for a, w1, w2 in sets[state["i"] % len(sets)]:
            h = torch.bmm(a, w1)
            gate, up = h.chunk(2, dim=-1)
            torch.bmm(F.silu(gate) * up, w2)
        state["i"] += 1

    return run


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def resolve_csv(args, phase: str, model_key: str) -> Tuple[str, str, str]:
    """Return (local_csv_path, provenance_string, commit_sha_or_empty)."""
    override = args.decode_csv if phase == "decode" else args.prefill_csv
    if override:
        return override, f"--{phase}-csv override", ""

    if args.stats_dir:
        key = (args.expert_stats_mode, phase)
        if key not in STATS_DIR_FILES:
            raise SystemExit(f"--stats-dir has no known filename for {key}; pass "
                             f"--{phase}-csv explicitly")
        path = os.path.join(args.stats_dir, STATS_DIR_FILES[key])
        if not os.path.exists(path):
            raise SystemExit(f"stats CSV not found: {path}")
        return path, f"--stats-dir {args.stats_dir}", ""

    src = archbench_stats.from_args(args)
    decode_path, prefill_path, model_name = src.resolve_stats(
        model_key, args.dtype, args.expert_stats_mode)
    repo_path = decode_path if phase == "decode" else prefill_path
    local = src.materialize(repo_path)
    return local, f"{src.origin} ({model_name}) :: {repo_path}", src.commit


def routed_tokens(phase: str, batch: int, seq_len: int, topk: int) -> int:
    """Routed-token slots this cell implies: decode emits 1 token/seq, prefill seq_len."""
    return topk * batch * (seq_len if phase == "prefill" else 1)


def int_list(s: str) -> List[int]:
    return [int(x) for x in s.replace(" ", "").split(",") if x]


def layer_list(s: str) -> List[int]:
    """Layer indices, or "all" -- resolved against --num-layers after parsing."""
    return ["all"] if s.strip().lower() == "all" else int_list(s)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Not argparse-required, only because --ab-prefetch needs neither.
    p.add_argument("--phase", choices=["decode", "prefill"], default=None)
    p.add_argument("--batch", type=int_list, default=None,
                   help="comma-separated batch sizes")
    p.add_argument("--layer", type=layer_list, default=[0],
                   help='comma-separated layer indices, or "all" for 0..num_layers-1')
    p.add_argument("--num-layers", type=int, default=DEFAULT_MODEL["num_layers"],
                   help='only used to expand --layer all')
    p.add_argument("--mode", choices=["fused", "batched", "both"], default="fused",
                   help="fused = sglang fused_experts_cpu; batched = the projection's "
                        "per-bucket batched GEMMs; both = run each and report the ratio")
    archbench_stats.add_args(p)
    p.add_argument("--decode-csv", default=None, help="override the decode stats CSV "
                                                      "with a local file")
    p.add_argument("--prefill-csv", default=None, help="override the prefill stats CSV "
                                                       "with a local file")
    p.add_argument("--expert-stats-mode", choices=["per_layer", "layers_averaged"],
                   default="per_layer",
                   help="matches archbench's expert_stats_mode")
    p.add_argument("--dtype", default="bf16",
                   help="expert_stats_mapping dtype key. bf16 only is supported here; "
                        "the mapping's fp8 per_layer entry points at the same CSVs "
                        "anyway, but the kernel path would not be bf16.")
    p.add_argument("--seq-len", type=int, default=1024,
                   help="modelled prefill input_seq_len; only used for the mass check")
    p.add_argument("--hidden-size", type=int, default=DEFAULT_MODEL["hidden_size"])
    p.add_argument("--num-experts", type=int, default=DEFAULT_MODEL["num_experts"])
    p.add_argument("--moe-intermediate-size", type=int,
                   default=DEFAULT_MODEL["moe_intermediate_size"])
    p.add_argument("--topk", type=int, default=DEFAULT_MODEL["topk"])
    p.add_argument("--tp", type=int, default=1,
                   help="tensor-parallel size to shard moe_intermediate_size by, i.e. what "
                        "ONE rank executes (N -> N/tp, all E experts kept). The expert "
                        "stats are still looked up under the unsharded model key, since "
                        "routing does not depend on the split. Note archbench's own "
                        "Qwen3_5MoeExperts does NOT shard the MoE, so --tp 1 is what the "
                        "projection charges per rank and --tp 4 is what a real sglang TP4 "
                        "rank runs.")
    p.add_argument("--dense-ffn", action="store_true",
                   help="benchmark a DENSE SwiGLU FFN instead of an MoE layer, as the "
                        "num_experts=1 case (which is how archbench models it). Reads no "
                        "stats; use for the dense models in the daily run, e.g. 9B: "
                        "--dense-ffn --hidden-size 4096 --intermediate-size 12288")
    p.add_argument("--intermediate-size", type=int, default=12288,
                   help="dense FFN intermediate size; only used with --dense-ffn")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--copies", type=int, default=1,
                   help="private prepacked weight sets per cell, cycled one per call so "
                        "the weights are read from DDR. One set's ACTIVE experts are only "
                        "12 MiB at decode bs1 TP4, against the 320 MiB L3 slice a 56-core "
                        "instance owns, so without copies that cell measures cache.")
    p.add_argument("--no-mix-layers", dest="mix_layers", action="store_false",
                   help="time each cell to completion instead of visiting every layer "
                        "once per iteration. Interleaving is the default because it is "
                        "what a rank walking 40 layers does, and it is what keeps a "
                        "layer's weights out of cache between two visits.")
    p.add_argument("--threads", type=int, default=None, help="torch.set_num_threads")
    p.add_argument("--no-prepack", action="store_true",
                   help="skip convert_weight_packed (measures the non-VNNI path)")
    p.add_argument("--inplace", action="store_true",
                   help="let fused_experts_cpu write into its input (as production does). "
                        "Off by default so the input stays stable across iterations.")
    p.add_argument("--activation", default="silu")
    p.add_argument("--check", action="store_true",
                   help="before timing, verify fused_experts_cpu against an fp32 "
                        "per-expert reference on the same routing table (slow; use on "
                        "small cells to validate a new build or a new box)")
    p.add_argument("--check-rtol", type=float, default=2e-2,
                   help="max relative error the --check comparison tolerates. bf16 "
                        "inputs accumulated over K=2048 make ~1e-2 normal.")
    p.add_argument("--instances", type=int, default=1,
                   help="run this many pinned instances CONCURRENTLY, re-synchronised on "
                        "a barrier before every iteration. 4 with --cores-per-instance 56 "
                        "is one TP4 rank per 56-core group on a 224c socket.")
    p.add_argument("--cores-per-instance", type=int, default=None,
                   help="cores each instance is pinned to (default: nproc // instances)")
    p.add_argument("--core-offset", type=int, default=0,
                   help="first core of instance 0")
    p.add_argument("--instance-index", type=int, default=0,
                   help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="append results to this CSV")
    p.add_argument("--dry-run", action="store_true",
                   help="print the decomposition and shapes; run no kernel, import no torch")
    p.add_argument("--skip-invalid", action="store_true",
                   help="warn and continue when a cell's stats fail validation "
                        "(default: stop, so bad data cannot be quietly measured)")
    args = p.parse_args()
    if not args.ab_prefetch and (args.phase is None or args.batch is None):
        p.error("--phase and --batch are required (except with --ab-prefetch)")
    if args.layer == ["all"]:
        args.layer = list(range(args.num_layers))

    if args.dense_ffn:
        # A dense SwiGLU FFN is the num_experts=1 case -- archbench says so itself
        # (common/swiglu_block.py: "num_experts=1 is the dense FFN case"), and it is the
        # same BatchedSwiGLU primitive the MoE expert groups are built from. So route it
        # through the identical path with a synthetic one-expert, topk=1 histogram
        # instead of bolting on a second kernel. Needed for the dense models in the
        # daily run (9B), which have no expert stats because they have no experts.
        args.num_experts, args.topk = 1, 1
        args.moe_intermediate_size = args.intermediate_size
        args.expert_stats_mode = "dense_ffn"
        args.layer = [0]  # every dense layer is the same shape; nothing to sweep

    # The stats key is always the UNSHARDED model: routing is a property of the model,
    # not of how its experts are split across ranks. Only the kernel shapes shrink.
    model_key = f"{args.hidden_size}-{args.num_experts}-{args.moe_intermediate_size}"
    if args.moe_intermediate_size % args.tp:
        p.error(f"--tp {args.tp} does not divide moe_intermediate_size "
                f"{args.moe_intermediate_size}")

    model = dict(hidden_size=args.hidden_size, num_experts=args.num_experts,
                 moe_intermediate_size=args.moe_intermediate_size // args.tp,
                 topk=args.topk)
    K, N, E, topk = (model["hidden_size"], model["moe_intermediate_size"],
                     model["num_experts"], model["topk"])

    if args.ab_prefetch:
        src = archbench_stats.from_args(args)
        if src is None:
            raise SystemExit("--ab-prefetch is meaningless with --stats-dir")
        print(f"prefetching stats for {model_key} from {src.origin}\n"
              f"commit {src.commit}")
        n = src.prefetch(model_key)
        print(f"cached {n} files under {src.files_dir}")
        return 0

    if args.instances > 1:
        return run_spread(args)
    rows = run_cells(args, model, model_key)
    write_rows(args, rows)
    return 0


def _instance_main(idx: int, cores: List[int], barrier, queue, args, model, model_key):
    """One pinned instance. Affinity and the OpenMP policy are set BEFORE torch is
    imported, because both are read when the thread pool is first built."""
    os.sched_setaffinity(0, set(cores))
    os.environ["OMP_NUM_THREADS"] = str(len(cores))
    os.environ["OMP_PROC_BIND"] = "close"
    os.environ["OMP_PLACES"] = "cores"
    # Sleeping idle OpenMP threads (passive / KMP_BLOCKTIME=0) costs a wake-up on
    # every kernel call: measured on X4PT, decode bs1 TP4 went 0.093 ms busy-wait ->
    # 5.0 ms passive, 54x, for 0.0126 GFLOP of work. The per-iteration barrier already
    # keeps the instances in step, so their idle windows are short and aligned and
    # busy-waiting does not steal a neighbour's frequency. Overridable to re-measure.
    os.environ["OMP_WAIT_POLICY"] = os.environ.get("MOEBENCH_WAIT_POLICY", "active")
    os.environ["KMP_BLOCKTIME"] = os.environ.get("MOEBENCH_BLOCKTIME", "200")
    # KMP_AFFINITY would override OMP_PLACES/OMP_PROC_BIND and uses absolute proc
    # ids, which mis-place under a restricted affinity mask.
    os.environ.pop("KMP_AFFINITY", None)
    args.threads = len(cores)
    args.instance_index = idx
    try:
        rows = run_cells(args, model, model_key, barrier=barrier,
                         tag_prefix=f"[inst{idx} cores {cores[0]}-{cores[-1]}] ")
        queue.put((idx, rows, None))
    except BaseException as exc:  # noqa: BLE001 - reported to the parent verbatim
        import traceback
        queue.put((idx, [], traceback.format_exc()))
        raise SystemExit(1) from exc


def run_spread(args) -> int:
    """Run `--instances` copies of the same cells concurrently, each pinned to its own
    core group, all re-synchronised before every timed iteration."""
    import multiprocessing as mp

    ncpu = len(os.sched_getaffinity(0))
    per = args.cores_per_instance or ncpu // args.instances
    need = args.core_offset + per * args.instances
    if need > ncpu:
        raise SystemExit(f"--instances {args.instances} x --cores-per-instance {per} "
                         f"(+offset {args.core_offset}) needs {need} cores but only "
                         f"{ncpu} are available")
    groups = [list(range(args.core_offset + i * per, args.core_offset + (i + 1) * per))
              for i in range(args.instances)]
    print(f"spread: {args.instances} instances x {per} cores, barrier-synced per "
          f"iteration: " + ", ".join(f"{g[0]}-{g[-1]}" for g in groups))

    model_key = f"{args.hidden_size}-{args.num_experts}-{args.moe_intermediate_size}"
    model = dict(hidden_size=args.hidden_size, num_experts=args.num_experts,
                 moe_intermediate_size=args.moe_intermediate_size // args.tp,
                 topk=args.topk)

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(args.instances)
    queue = ctx.Queue()
    procs = [ctx.Process(target=_instance_main,
                         args=(i, groups[i], barrier, queue, args, model, model_key))
             for i in range(args.instances)]
    for p in procs:
        p.start()
    collected, failures = {}, {}
    for _ in procs:
        idx, rows, err = queue.get()
        collected[idx] = rows
        if err:
            failures[idx] = err
    for p in procs:
        p.join()

    if failures:
        for idx, err in sorted(failures.items()):
            print(f"\ninstance {idx} FAILED:\n{err}", file=sys.stderr)
        raise SystemExit(f"{len(failures)}/{args.instances} instances failed")

    rows: List[Dict] = []
    for idx in sorted(collected):
        rows.extend(collected[idx])
    write_rows(args, rows)
    summarize_spread(rows, args.instances)
    return 0


def summarize_spread(rows: List[Dict], instances: int) -> None:
    """Per cell, report the across-instance imbalance. With a per-iteration barrier
    every instance ran in the same window, so a spread well above 1.0 is genuine
    hardware asymmetry and not scheduling drift."""
    cells: Dict[tuple, List[Dict]] = {}
    for r in rows:
        cells.setdefault((r["phase"], r["batch"], r["layer"]), []).append(r)
    worst = []
    for key, rs in sorted(cells.items()):
        for mode in ("fused", "batched"):
            vals = [r[f"{mode}_median_ms"] for r in rs if r.get(f"{mode}_median_ms")]
            if len(vals) < 2:
                continue
            worst.append((max(vals) / min(vals), key, mode, min(vals), max(vals)))
    if not worst:
        return
    worst.sort(reverse=True)
    print(f"\nacross-instance imbalance ({instances} instances, barrier-synced), "
          f"worst 5 cells:")
    for spread, (phase, batch, layer), mode, lo, hi in worst[:5]:
        print(f"  {spread:5.2f}x  {phase} bs{batch} L{layer} {mode:<8} "
              f"{lo:.3f} .. {hi:.3f} ms")
    med = statistics.median([w[0] for w in worst])
    print(f"  median imbalance across all cells/modes: {med:.3f}x")


def run_cells(args, model, model_key, barrier=None, tag_prefix="") -> List[Dict]:
    K, N, E, topk = (model["hidden_size"], model["moe_intermediate_size"],
                     model["num_experts"], model["topk"])
    if args.dense_ffn:
        csv_file, provenance, stats_commit = "", "dense FFN (synthetic, no stats)", ""
    else:
        csv_file, provenance, stats_commit = resolve_csv(args, args.phase, model_key)

    caller = None
    if not args.dry_run:
        import torch
        try:
            import sgl_kernel  # noqa: F401  (registers torch.ops.sgl_kernel)
        except ImportError:
            pass
        if args.threads:
            torch.set_num_threads(args.threads)
        print(f"torch {torch.__version__}  threads={torch.get_num_threads()}")
        if args.mode in ("fused", "both"):
            caller = FusedExpertsCaller(prepack=not args.no_prepack, inplace=args.inplace,
                                        activation=args.activation)
            print(f"kernel schema: {caller.describe()}")

    print(f"stats: {provenance}")
    print(f"       commit {stats_commit or '(local file, unversioned)'}")
    print(f"       cached at {csv_file}")
    print(f"model: model_key={model_key} K={K} N={N} E={E} topk={topk}  "
          f"phase={args.phase} stats_mode={args.expert_stats_mode} dtype={args.dtype}")

    cells = []
    for batch in args.batch:
        for layer in args.layer:
            cell = prepare_cell(args, model, model_key, batch, layer, csv_file,
                                stats_commit, tag_prefix)
            if cell is not None:
                cells.append(cell)

    if args.dry_run:
        return [c["row"] for c in cells]
    if args.check and args.mode in ("fused", "both"):
        for cell in cells:
            check_cell(args, model, caller, cell)
    time_cells(args, model, caller, cells, barrier)
    return [c["row"] for c in cells]


def mem_available_bytes() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


def pool_bytes(leg: str, cells: List[Dict], model, copies: int) -> int:
    K, N, E = (model["hidden_size"], model["moe_intermediate_size"],
               model["num_experts"])
    if leg == "fused":
        # Activations are not copied, but at thr prefill one cell's [num_tokens, K] is
        # 1.6 GiB, so leaving them out of the estimate understates the pool by more
        # than the weights of a whole extra copy.
        weights = 2 * (E * 2 * N * K + E * K * N) * copies * len(cells)
        return weights + sum(2 * c["num_tokens"] * K for c in cells)
    total = 0
    for c in cells:
        weights = sum(nexp * (K * 2 * N + N * K) for nexp, _ in c["groups"])
        acts = sum(nexp * tokens * K for nexp, tokens in c["groups"])
        total += 2 * (weights * copies + acts)
    return total


def announce_pool(leg: str, cells: List[Dict], args, model) -> None:
    """Print the weight pool this leg holds, and refuse a plan that cannot fit.

    The pool is deliberately large -- it is what forces the weight traffic onto DDR --
    so the guard matters: --copies multiplies it by the concurrent instances, and the
    failure mode without a check is an OOM kill part-way through a cell.
    """
    gib = 1 << 30
    instances = max(args.instances, 1)
    need = pool_bytes(leg, cells, model, args.copies)
    avail = mem_available_bytes()
    if not args.instance_index:
        print(f"\n{leg} weight pool: {need / gib:.1f} GiB per instance x {instances} "
              f"= {need * instances / gib:.1f} GiB of {avail / gib:.1f} GiB available "
              f"({len(cells)} cells x {args.copies} copies)")
    if avail and need * instances > 0.7 * avail:
        raise SystemExit(
            f"{leg} pool needs {need * instances / gib:.1f} GiB across {instances} "
            f"instances but only {avail / gib:.1f} GiB is available. Lower --copies "
            f"or --layer, or run the legs as separate invocations.")


def check_cell(args, model, caller, cell: Dict) -> None:
    rel, absolute = check_fused(cell["counts"], cell["num_tokens"], model, caller,
                                args.seed, args.activation)
    cell["row"]["check_rel_err"] = rel
    status = "OK" if rel <= args.check_rtol else "FAIL"
    print(f"  {cell['label']} check: rel_err {rel:.3e} (abs {absolute:.3e}) vs fp32 "
          f"reference -- {status}")
    if rel > args.check_rtol:
        raise SystemExit(f"{cell['tag']}: fused_experts_cpu disagrees with the fp32 "
                         f"reference by {rel:.3e} > --check-rtol {args.check_rtol:g}; "
                         f"timings from this build are not trustworthy.")


def time_cells(args, model, caller, cells: List[Dict], barrier) -> None:
    """Time the fused leg over every cell, then the batched leg over every cell.

    The legs run as separate passes so only one leg's weight pool is resident: held
    together they double the peak, which at TP1 over 40 layers times --copies decides
    whether the pool fits in memory at all.
    """
    batches = [cells] if args.mix_layers else [[c] for c in cells]
    for leg, make in (("fused", make_fused_runner), ("batched", make_batched_runner)):
        if args.mode not in (leg, "both"):
            continue
        for group in batches:
            announce_pool(leg, group, args, model)
            t0 = time.perf_counter()
            if leg == "fused":
                runners = [make(c["counts"], c["num_tokens"], model, caller, args.seed,
                                copies=args.copies) for c in group]
            else:
                runners = [make(c["groups"], model, args.seed, copies=args.copies)
                           for c in group]
            if not args.instance_index:
                print(f"  pool built in {time.perf_counter() - t0:.1f} s")
            stats = time_mixed(runners, args.warmup, args.iters, barrier)
            for c, r in zip(group, stats):
                c["row"].update({f"{leg}_{k}": v for k, v in r.items()})
            del runners
    for cell in cells:
        report_cell(cell)


def report_cell(cell: Dict) -> None:
    row, flops = cell["row"], cell["flops"]
    print(f"\n{cell['label']}")
    for leg, label in (("fused", "fused_experts_cpu:"), ("batched", "batched GEMMs:    ")):
        med = row.get(f"{leg}_median_ms")
        if med is None:
            continue
        print(f"  {label} {med:.3f} ms median (min {row[f'{leg}_min_ms']:.3f}, "
              f"p90 {row[f'{leg}_p90_ms']:.3f})  {flops / 1e9 / (med / 1e3):.1f} GFLOP/s")
    if row.get("fused_median_ms") and row.get("batched_median_ms"):
        ratio = row["fused_median_ms"] / row["batched_median_ms"]
        row["fused_over_batched"] = ratio
        print(f"  fused / batched:   {ratio:.2f}x "
              f"({'fused wins' if ratio < 1 else 'batched wins'})")


def prepare_cell(args, model, model_key, batch: int, layer: int, csv_file: str,
                 stats_commit: str, tag_prefix: str) -> Optional[Dict]:
    """Read one cell's histogram, print its decomposition, and build its result row."""
    K, N, E, topk = (model["hidden_size"], model["moe_intermediate_size"],
                     model["num_experts"], model["topk"])
    tag = f"{args.phase} bs{batch} L{layer}"
    if args.dense_ffn:
        # One "expert" seeing every token: decode routes 1 token/seq, prefill seq_len.
        tokens = batch * (args.seq_len if args.phase == "prefill" else 1)
        tag = f"{args.phase} bs{batch} dense"
        hist = {tokens: 1}
    else:
        hist = moe_stats.read_histogram(csv_file, args.phase, batch, layer,
                                        args.expert_stats_mode)
    if hist is None:
        print(f"\n{tag}: no stats rows -- skipped")
        return None
    R = routed_tokens(args.phase, batch, args.seq_len, topk)
    try:
        moe_stats.validate(hist, E, R, args.phase, where=tag)
    except ValueError as exc:
        if not args.skip_invalid:
            raise SystemExit(f"\n{exc}\n\n(pass --skip-invalid to measure it anyway)")
        print(f"\n{tag}: INVALID STATS, measured anyway -- {exc}")

    groups = moe_stats.groups(hist)
    counts, num_tokens, slack = moe_stats.expert_token_counts(hist, E, topk)
    mass = moe_stats.histogram_mass(hist)
    active = sum(hist.values())
    # 2 flops/MAC; gate+up is K x 2N and down is N x K per routed slot.
    flops = 6.0 * (mass + slack) * K * N

    print(f"\n{tag_prefix}{tag}")
    print(f"  histogram: {len(groups)} groups, {active}/{E} experts active, "
          f"mass {mass} slots (routed_tokens {R}, {(mass - R) / R:+.1%})")
    print(f"  tokens fed: {num_tokens}"
          + (f"  (+{slack} slack slots for topk divisibility)" if slack else ""))
    for nexp, tokens in groups:
        print(f"    group: {nexp:4d} experts x {tokens:6d} tokens  -> bmm "
              f"[{nexp},{tokens},{K}]x[{nexp},{K},{2 * N}] + "
              f"[{nexp},{tokens},{N}]x[{nexp},{N},{K}]")
    print(f"  work: {flops / 1e9:.2f} GFLOP")

    row = dict(phase=args.phase, batch=batch, layer=layer,
               instance=args.instance_index,
               stats_commit=stats_commit,
               stats_ref=args.ab_ref if stats_commit else "",
               dtype=args.dtype,
               stats_mode=args.expert_stats_mode, hidden_size=K,
               moe_intermediate_size=N, num_experts=E, topk=topk,
               tp=args.tp, threads=args.threads or 0,
               num_groups=len(groups), active_experts=active,
               histogram_mass=mass, routed_tokens=R, slack=slack,
               num_tokens=num_tokens, gflop=flops / 1e9,
               groups=";".join(f"{n}x{t}" for n, t in groups))
    return dict(tag=tag, label=f"{tag_prefix}{tag}", row=row, counts=counts,
                num_tokens=num_tokens, groups=groups, flops=flops)


# Every column the results CSV can carry, in a FIXED order.
#
# The header MUST NOT be derived from the rows one invocation happens to produce.
# A lane sweep appends several invocations to the SAME --out file, and they do not
# all fill the same columns: `check_rel_err` only exists under --check, and the
# batched_* / fused_over_batched columns only under --mode batched/both. The first
# invocation writes the header; a later one whose rows lack an earlier column used
# to shift every following value one cell to the LEFT, silently, under a header that
# still said otherwise. That is how a 3.295 ms `fused_median_ms` was read back as a
# check_rel_err of 3.295 (i.e. "330% error") for the lanes that never ran --check at
# all -- the kernel and the fp32 reference were both fine. Fixed order + restval ""
# keeps every value under its own label whatever a given cell measured.
RESULT_FIELDS: List[str] = [
    "phase", "batch", "layer", "instance",
    "stats_commit", "stats_ref", "dtype", "stats_mode",
    "hidden_size", "moe_intermediate_size", "num_experts", "topk", "tp", "threads",
    "num_groups", "active_experts", "histogram_mass", "routed_tokens", "slack",
    "num_tokens", "gflop", "groups",
    "check_rel_err",
    "fused_median_ms", "fused_min_ms", "fused_mean_ms", "fused_p90_ms", "fused_iters",
    "batched_median_ms", "batched_min_ms", "batched_mean_ms", "batched_p90_ms",
    "batched_iters",
    "fused_over_batched",
]


def write_rows(args, rows: List[Dict]) -> None:
    if not (args.out and rows):
        return
    keys = {k for r in rows for k in r}
    unknown = sorted(keys - set(RESULT_FIELDS))
    if unknown:
        raise SystemExit(f"write_rows: new result column(s) {unknown} are not in "
                         f"RESULT_FIELDS. Add them there (at the END, so old CSVs stay "
                         f"readable) rather than letting the header float.")

    # Honour a header already on disk -- including one written by an older revision
    # with fewer columns -- and refuse to append rows it cannot represent, instead of
    # writing values into the wrong columns.
    fields = RESULT_FIELDS
    new = not os.path.exists(args.out)
    if not new:
        with open(args.out, newline="") as f:
            existing = next(csv.reader(f), None)
        if existing:
            missing = sorted(keys - set(existing))
            if missing:
                raise SystemExit(
                    f"write_rows: {args.out} was opened with a header that has no "
                    f"{missing} column(s), so these rows cannot be appended without "
                    f"shifting the ones already there. Write to a fresh --out file.")
            fields = existing

    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
