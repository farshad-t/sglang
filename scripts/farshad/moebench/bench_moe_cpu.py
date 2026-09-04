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
    """Bind `fused_experts_cpu` arguments by NAME from its registered schema."""

    # Values we supply, keyed by the schema's argument names. Anything the schema
    # asks for that is not here is left at its default (or None if optional).
    def __init__(self, prepack: bool, inplace: bool, activation: str):
        import torch
        self.torch = torch
        try:
            self.op = torch.ops.sgl_kernel.fused_experts_cpu
        except (AttributeError, RuntimeError) as exc:
            raise RuntimeError(
                "torch.ops.sgl_kernel.fused_experts_cpu is not registered. Import the "
                "build that provides it (`import sgl_kernel`) before running, or use "
                "--mode batched / --dry-run.") from exc
        self.schema = self.op.default._schema
        self.names = [a.name for a in self.schema.arguments]
        self.prepack = prepack
        self.inplace = inplace
        self.activation = activation
        self.pack = getattr(torch.ops.sgl_kernel, "convert_weight_packed", None)
        if prepack and self.pack is None:
            raise RuntimeError("--prepack requested but sgl_kernel.convert_weight_packed "
                               "is not registered in this build")

    def prepack_weight(self, w):
        return self.pack(w) if self.prepack else w

    def describe(self) -> str:
        return str(self.schema)

    def __call__(self, a, w1, w2, topk_weights, topk_ids):
        supplied = {
            "hidden_states": a,
            "input": a,          # older builds name it `input`
            "w1": w1,
            "w2": w2,
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "inplace": self.inplace,
            # UNQUANT. Named `moe_comp_method` in the container build and `quant` /
            # `quant_method` upstream; all take the same 0 == unquantized bf16.
            "moe_comp_method": 0,
            "quant": 0,
            "quant_method": 0,
            "use_int8_w8a8": False,
            "use_fp8_w8a16": False,
            "is_vnni": self.prepack,
            "activation": self.activation,
        }
        args = []
        for arg in self.schema.arguments:
            if arg.name in supplied:
                args.append(supplied[arg.name])
            elif arg.has_default_value():
                args.append(arg.default_value)
            elif "Optional" in str(arg.type) or str(arg.type).endswith("?"):
                args.append(None)
            else:
                raise RuntimeError(
                    f"fused_experts_cpu argument {arg.name!r} ({arg.type}) is required by "
                    f"this build but the harness does not know how to fill it. Schema:\n"
                    f"  {self.schema}")
        return self.op(*args)


# ---------------------------------------------------------------------------
# the two things we time
# ---------------------------------------------------------------------------

def time_it(fn, warmup: int, iters: int) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return dict(median_ms=statistics.median(samples),
                min_ms=min(samples),
                mean_ms=statistics.fmean(samples),
                p90_ms=sorted(samples)[max(0, int(0.9 * len(samples)) - 1)],
                iters=len(samples))


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


def make_fused_runner(counts, num_tokens, model, caller, seed):
    a, w1, w2, topk_weights, topk_ids = fused_tensors(counts, num_tokens, model,
                                                      caller, seed)
    pw1, pw2 = caller.prepack_weight(w1), caller.prepack_weight(w2)
    return lambda: caller(a, pw1, pw2, topk_weights, topk_ids)


def check_fused(counts, num_tokens, model, caller, seed, activation="silu"):
    """Compare fused_experts_cpu against a plain per-expert fp32 reference driven by
    the SAME routing table. This is what rules out a kernel (or a harness bug) that
    quietly does less work than the histogram says -- which would show up as a
    flattering GFLOP/s rather than as a failure."""
    import torch
    import torch.nn.functional as F
    a, w1, w2, topk_weights, topk_ids = fused_tensors(counts, num_tokens, model,
                                                      caller, seed)
    got = caller(a, caller.prepack_weight(w1), caller.prepack_weight(w2),
                 topk_weights, topk_ids).to(torch.float32)

    af = a.float()
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


def make_batched_runner(groups, model, seed):
    """The projection's own decomposition: one batched GEMM pair per histogram bucket."""
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

    def run():
        for a, w1, w2 in work:
            h = torch.bmm(a, w1)
            gate, up = h.chunk(2, dim=-1)
            torch.bmm(F.silu(gate) * up, w2)

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

    rows = []
    for batch in args.batch:
        for layer in args.layer:
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
                continue
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

            print(f"\n{tag}")
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

            if args.dry_run:
                rows.append(row)
                continue

            if args.check and args.mode in ("fused", "both"):
                rel, absolute = check_fused(counts, num_tokens, model, caller,
                                            args.seed, args.activation)
                row["check_rel_err"] = rel
                status = "OK" if rel <= args.check_rtol else "FAIL"
                print(f"  check: rel_err {rel:.3e} (abs {absolute:.3e}) vs fp32 "
                      f"reference -- {status}")
                if rel > args.check_rtol:
                    raise SystemExit(f"{tag}: fused_experts_cpu disagrees with the fp32 "
                                     f"reference by {rel:.3e} > --check-rtol "
                                     f"{args.check_rtol:g}; timings from this build are "
                                     f"not trustworthy.")

            if args.mode in ("fused", "both"):
                r = time_it(make_fused_runner(counts, num_tokens, model, caller, args.seed),
                            args.warmup, args.iters)
                row.update({f"fused_{k}": v for k, v in r.items()})
                print(f"  fused_experts_cpu: {r['median_ms']:.3f} ms median "
                      f"(min {r['min_ms']:.3f}, p90 {r['p90_ms']:.3f})  "
                      f"{flops / 1e9 / (r['median_ms'] / 1e3):.1f} GFLOP/s")
            if args.mode in ("batched", "both"):
                r = time_it(make_batched_runner(groups, model, args.seed),
                            args.warmup, args.iters)
                row.update({f"batched_{k}": v for k, v in r.items()})
                print(f"  batched GEMMs:     {r['median_ms']:.3f} ms median "
                      f"(min {r['min_ms']:.3f}, p90 {r['p90_ms']:.3f})  "
                      f"{flops / 1e9 / (r['median_ms'] / 1e3):.1f} GFLOP/s")
            if args.mode == "both":
                ratio = row["fused_median_ms"] / row["batched_median_ms"]
                row["fused_over_batched"] = ratio
                print(f"  fused / batched:   {ratio:.2f}x "
                      f"({'fused wins' if ratio < 1 else 'batched wins'})")
            rows.append(row)

    if args.out and rows:
        fields: List[str] = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
        new = not os.path.exists(args.out)
        with open(args.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
