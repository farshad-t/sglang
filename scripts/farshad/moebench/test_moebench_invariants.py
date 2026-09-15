#!/usr/bin/env python3
"""Verification suite for the parts of the harness that decide WHAT gets measured.

Three groups, each answering a question that reading the code cannot:

  INVARIANTS   does the routing table actually realise the measured histogram, and
               does the GFLOP number describe the shapes that really run?
  DIFFERENTIAL is moe_stats.py really "a faithful port" of archbench's reader, as its
               docstring claims? Compared against the ORIGINAL, pulled from the pinned
               archbench commit, over every (batch, layer) in every shipped CSV.
  GATE POWER   can --check FAIL? A correct stub kernel plus injected faults, to measure
               which classes of kernel defect the rel_err metric can see.

Run:  python3 test_moebench_invariants.py            (fast subset)
      MOEBENCH_TEST_FULL=1 python3 test_moebench_invariants.py   (every cell)

Skips rather than fails when torch or the archbench stats cache is unavailable, so it
is runnable on a box with no kernel build.
"""

import csv
import importlib.util
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_moe_cpu as B
import moe_stats as M

FULL = bool(os.environ.get("MOEBENCH_TEST_FULL"))
# Building a routing table for a bs320 prefill layer is ~2.6M slots; fine once, too
# slow across 880 cells. The cap keeps the default run in seconds while the sampled
# layers still cover every batch. MOEBENCH_TEST_FULL=1 removes it.
MAX_SLOTS = 10 ** 9 if FULL else 200_000
SAMPLE_LAYERS = (0, 1, 20, 39)

AB_SHA = os.environ.get("MOEBENCH_AB_SHA", "5ae86154d545244f6239bf966aeb694899fe53ab")
AB_ROOT = os.path.expanduser(f"~/.cache/moebench/archbench/files/{AB_SHA}")
STATS_DIR = os.path.join(AB_ROOT, "tools/cpu/config/qwen35/expert_stats")
AB_READER = os.path.join(AB_ROOT, "networks/MLP/Qwen/Qwen3/qwen3_moe_expert_dist_utils.py")

CSV_DECODE = os.path.join(STATS_DIR, "qwen3.5-35B-A3B_realprompt_decode.csv")
CSV_PREFILL = os.path.join(STATS_DIR, "qwen3.5-35B-A3B_realprompt_prefill_5buckets.csv")
CSV_DECODE_AVG = os.path.join(STATS_DIR,
                              "qwen3.5-35B-A3B-bf16_tp1_bs20_decode_layers_averaged.csv")
CSV_PREFILL_AVG = os.path.join(
    STATS_DIR, "qwen3.5-35B-A3B-bf16_tp1_bs20_prefill_layers_averaged_5buckets.csv")

# Qwen3.5-35B-A3B, the model every shipped CSV describes.
E, TOPK, K, N_FULL = 256, 8, 2048, 512


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def _column(path, col):
    with open(path) as f:
        return sorted({int(r[col]) for r in csv.DictReader(f) if r.get(col) not in (None, "")})


def _cells(path, phase):
    """Every (batch, layer) the CSV covers, thinned unless MOEBENCH_TEST_FULL."""
    batches, layers = _column(path, "Batch_Size"), _column(path, "Layer")
    for b in batches:
        for layer in layers:
            if not FULL and phase == "prefill" and layer not in SAMPLE_LAYERS:
                continue
            hist = M.read_histogram(path, phase, b, layer, "per_layer")
            if hist:
                yield b, layer, hist


class Invariants(unittest.TestCase):
    """Properties that must hold for the benchmark to be measuring the right work."""

    def test_expert_token_counts_conserves_mass(self):
        n = 0
        for path, phase in ((CSV_DECODE, "decode"), (CSV_PREFILL, "prefill")):
            if not os.path.exists(path):
                self.skipTest(f"stats cache missing: {path}")
            for b, layer, hist in _cells(path, phase):
                mass = M.histogram_mass(hist)
                counts, num_tokens, slack = M.expert_token_counts(hist, E, TOPK)
                where = f"{phase} bs{b} L{layer}"
                self.assertEqual(len(counts), E, where)
                self.assertEqual(sum(counts), mass + slack, f"{where}: mass not conserved")
                self.assertEqual(slack, (-mass) % TOPK, f"{where}: wrong slack")
                self.assertLess(slack, TOPK, where)
                self.assertEqual(num_tokens * TOPK, mass + slack, where)
                self.assertLessEqual(max(counts), num_tokens,
                                     f"{where}: an expert wants more tokens than exist")
                self.assertEqual(sum(1 for c in counts if c), sum(hist.values()),
                                 f"{where}: active-expert count changed")
                n += 1
        self.assertGreater(n, 100, "suspiciously few cells exercised")

    def test_topk_ids_realise_the_histogram(self):
        """The routing table is the whole experiment. If it does not reproduce the
        measured per-expert tallies, every timing describes a different model."""
        torch = _torch()
        if torch is None:
            self.skipTest("torch unavailable")
        if not os.path.exists(CSV_DECODE):
            self.skipTest("stats cache missing")
        n = 0
        for path, phase in ((CSV_DECODE, "decode"), (CSV_PREFILL, "prefill")):
            for b, layer, hist in _cells(path, phase):
                counts, num_tokens, _ = M.expert_token_counts(hist, E, TOPK)
                if num_tokens * TOPK > MAX_SLOTS:
                    continue
                ids = B.build_topk_ids(counts, num_tokens, TOPK)
                where = f"{phase} bs{b} L{layer}"
                self.assertEqual(tuple(ids.shape), (num_tokens, TOPK), where)
                self.assertEqual(ids.dtype, torch.int32, where)
                got = torch.bincount(ids.reshape(-1).to(torch.int64), minlength=E)
                self.assertTrue(torch.equal(got, torch.tensor(counts, dtype=got.dtype)),
                                f"{where}: table does not match the histogram")
                dup = (ids.sort(dim=1).values.diff(dim=1) == 0).any(dim=1)
                self.assertFalse(bool(dup.any()),
                                 f"{where}: a token is routed to one expert twice")
                self.assertTrue(bool((ids >= 0).all() and (ids < E).all()), where)
                n += 1
        self.assertGreater(n, 100, "suspiciously few cells exercised")

    def test_gflop_matches_the_shapes_actually_executed(self):
        """`flops = 6*(mass+slack)*K*N` is the harness's one-liner. Recompute it the
        long way -- per bucket, from the GEMM shapes the decomposition prints -- so a
        typo in either derivation shows up as a mismatch."""
        if not os.path.exists(CSV_DECODE):
            self.skipTest("stats cache missing")
        for path, phase in ((CSV_DECODE, "decode"), (CSV_PREFILL, "prefill")):
            for b, layer, hist in _cells(path, phase):
                for tp in (1, 2, 4):
                    N = N_FULL // tp
                    mass = M.histogram_mass(hist)
                    _, _, slack = M.expert_token_counts(hist, E, TOPK)
                    short = 6.0 * (mass + slack) * K * N
                    # long way: gate+up is [tokens,K]x[K,2N], down is [tokens,N]x[N,K],
                    # 2 flops per MAC, summed over every expert in every bucket.
                    long = 0.0
                    for nexp, tokens in M.groups(hist):
                        long += nexp * (2.0 * tokens * K * 2 * N + 2.0 * tokens * N * K)
                    long += 6.0 * slack * K * N   # the padding slots are executed too
                    self.assertAlmostEqual(short / long, 1.0, places=12,
                                           msg=f"{phase} bs{b} L{layer} tp{tp}")

    def test_pool_bytes_matches_an_independent_derivation(self):
        """`pool_bytes` drives the OOM guard, so a wrong value either kills a sweep
        part-way through or refuses a plan that would have fitted. Derive it by hand."""
        N = N_FULL // 4
        model = dict(hidden_size=K, moe_intermediate_size=N, num_experts=E, topk=TOPK)
        groups = [(2, 154), (24, 88), (65, 49), (73, 25), (88, 10)]
        cell = dict(groups=groups, num_tokens=1039)
        for copies in (1, 2, 8):
            # FUSED allocates the whole expert stack (fused_tensors: w1 [E,2N,K],
            # w2 [E,K,N]) once per copy per cell, plus one [num_tokens,K] activation
            # per cell (not copied -- make_fused_runner shares `a` across copies).
            # activations: ONE buffer sized to the largest cell, not one per cell
            want = 2 * E * (2 * N * K + K * N) * copies + 2 * cell["num_tokens"] * K
            self.assertEqual(B.pool_bytes("fused", [cell], model, copies), want,
                             f"fused pool wrong at copies={copies}")
            # BATCHED allocates only the experts each bucket names, plus ONE activation
            # buffer sized to the LARGEST group -- shared by every group and every cell
            # since 8a40e4268d, so it is a max, not a sum, and is not multiplied by
            # copies either.
            w = sum(n * (K * 2 * N + N * K) for n, _ in groups)
            a = max(n * t * K for n, t in groups)
            self.assertEqual(B.pool_bytes("batched", [cell], model, copies),
                             2 * (w * copies + a), f"batched pool wrong at copies={copies}")

        # The two legs must hold comparable WEIGHT pools, or `fused / batched` compares
        # a DDR-resident kernel against a cache-resident bmm. Active experts here are
        # 252 of 256, so the weights land within a few percent.
        fw = 2 * E * (2 * N * K + K * N)
        bw = 2 * sum(n * (K * 2 * N + N * K) for n, _ in groups)
        self.assertLess(abs(fw - bw) / fw, 0.05,
                        f"leg WEIGHT pools differ by {abs(fw - bw) / fw:.1%}")
        # and the fused weight pool for this cell is the 384 MiB make_fused_runner's
        # docstring reasons about
        self.assertEqual(fw >> 20, 384)

    def test_both_legs_share_one_activation_buffer(self):
        """Both legs must hold ONE activation buffer for the whole leg, not one per cell.

        8a40e4268d did this for the batched leg only, which left the FUSED leg holding
        one [num_tokens, K] per cell -- 60 GiB of the 120 GiB fused pool at a 40-layer
        thr-prefill lane, making the leg that has to fit 1.85x the one that already did.
        Asserted here so the two legs cannot drift apart again."""
        N = N_FULL   # tp1, the throughput lane
        model = dict(hidden_size=K, moe_intermediate_size=N, num_experts=E, topk=TOPK)
        # 40 cells that all route about the same number of tokens, i.e. one layer sweep
        groups = [(1, 329), (1, 194), (30, 80), (80, 42), (140, 14)]
        cells = [dict(groups=groups, num_tokens=1031) for _ in range(40)]

        # neither leg's activation term may scale with the number of cells
        one, forty = cells[:1], cells
        for leg in ("fused", "batched"):
            grew = B.pool_bytes(leg, forty, model, 1) - B.pool_bytes(leg, one, model, 1)
            weights_only = (B.pool_bytes(leg, one, model, 1)
                            - B.pool_bytes(leg, one, model, 0))
            self.assertAlmostEqual(grew / (39 * weights_only), 1.0, places=6,
                                   msg=f"{leg}: going 1 -> 40 cells grew the pool by "
                                       f"more than 39 weight sets, so something is "
                                       f"still allocated per cell")

        # and the fused leg must no longer dominate the batched one
        f = B.pool_bytes("fused", forty, model, 1)
        b = B.pool_bytes("batched", forty, model, 1)
        self.assertLess(f / b, 1.15, f"fused pool is {f / b:.2f}x batched; the legs are "
                                     f"asymmetric again")

        # the batched leg's buffer is the largest BUCKET's expert-replicated shape, which
        # is not the model's activation footprint -- the fused leg's now is
        self.assertGreater(2 * max(B.act_elems(c["groups"], K) for c in cells),
                           2 * cells[0]["num_tokens"] * K)

    def test_routed_tokens(self):
        self.assertEqual(B.routed_tokens("decode", 320, 1024, 8), 8 * 320)
        self.assertEqual(B.routed_tokens("prefill", 1, 1024, 8), 8 * 1024)
        self.assertEqual(B.routed_tokens("prefill", 64, 1024, 8), 8 * 64 * 1024)

    def test_dense_ffn_degenerate_case(self):
        """--dense-ffn runs the MoE path at num_experts=1, topk=1. Nothing else covers it."""
        torch = _torch()
        if torch is None:
            self.skipTest("torch unavailable")
        counts, num_tokens, slack = M.expert_token_counts({1024: 1}, 1, 1)
        self.assertEqual((counts, num_tokens, slack), ([1024], 1024, 0))
        ids = B.build_topk_ids(counts, num_tokens, 1)
        self.assertEqual(tuple(ids.shape), (1024, 1))
        self.assertTrue(bool((ids == 0).all()))


class Validate(unittest.TestCase):
    """moe_stats.validate is the gate that keeps physically impossible stats out."""

    def test_rejects_more_active_experts_than_the_layer_has(self):
        # the analyzer bug it exists to catch: the activations==1 bucket emitted as a
        # token REMAINDER, so num_experts tracks the token count
        with self.assertRaises(ValueError) as cm:
            M.validate({1: 1763, 4: 60}, 256, 2560, "decode", where="synthetic")
        self.assertIn("only has 256", str(cm.exception))

    def test_decode_mass_must_be_exact(self):
        M.validate({1: 8}, 256, 8, "decode")                      # exact -> ok
        with self.assertRaises(ValueError):
            M.validate({1: 9}, 256, 8, "decode")                  # one too many

    def test_prefill_gets_a_band_not_equality(self):
        M.validate({10: 100}, 256, 1000, "prefill")               # exact
        M.validate({10: 100}, 256, 900, "prefill")                # +11%, inside 20%
        with self.assertRaises(ValueError):
            M.validate({10: 100}, 256, 700, "prefill")            # +43%, outside
        self.assertEqual(M.PREFILL_MASS_TOLERANCE, 0.20)

    def test_rejects_sub_unit_buckets(self):
        for bad in ({0: 8}, {1: 0}, {-1: 8}):
            with self.assertRaises(ValueError):
                M.validate(bad, 256, None, "decode")

    def test_every_shipped_cell_passes_its_own_validator(self):
        """A cell the harness refuses to measure is a cell the projection cannot use.
        This is a data check as much as a code check."""
        if not os.path.exists(CSV_DECODE):
            self.skipTest("stats cache missing")
        bad = []
        for path, phase in ((CSV_DECODE, "decode"), (CSV_PREFILL, "prefill")):
            for b, layer, hist in _cells(path, phase):
                R = B.routed_tokens(phase, b, 1024, TOPK)
                try:
                    M.validate(hist, E, R, phase, where=f"{phase} bs{b} L{layer}")
                except ValueError as exc:
                    bad.append(str(exc).split(":")[0] + f" (bs{b} L{layer})")
        self.assertEqual(bad, [], f"{len(bad)} shipped cells fail validation")


class DifferentialAgainstArchbench(unittest.TestCase):
    """moe_stats.py says it is "a faithful port" of archbench's reader. Prove it."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(AB_READER):
            raise unittest.SkipTest(
                f"archbench reader not cached at {AB_READER}. Warm it with:\n"
                f"  python3 -c \"import archbench_stats as A; "
                f"A.StatsSource(commit='{AB_SHA}').materialize("
                f"'networks/MLP/Qwen/Qwen3/qwen3_moe_expert_dist_utils.py')\"")
        spec = importlib.util.spec_from_file_location("ab_reader", AB_READER)
        cls.AB = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.AB)

    def test_decode_per_layer_identical(self):
        n = 0
        for b in _column(CSV_DECODE, "Batch_Size"):
            for layer in _column(CSV_DECODE, "Layer"):
                self.assertEqual(M.read_decode(CSV_DECODE, b, layer),
                                 self.AB._get_histogram_decode(b, layer, CSV_DECODE),
                                 f"decode bs{b} L{layer}")
                n += 1
        self.assertGreater(n, 500, "too few comparisons to be meaningful")

    def test_prefill_per_layer_identical(self):
        n = 0
        for b in _column(CSV_PREFILL, "Batch_Size"):
            for layer in _column(CSV_PREFILL, "Layer"):
                self.assertEqual(M.read_prefill(CSV_PREFILL, b, layer),
                                 self.AB._get_histogram_prefill(b, E, layer, CSV_PREFILL),
                                 f"prefill bs{b} L{layer}")
                n += 1
        self.assertGreater(n, 500, "too few comparisons to be meaningful")

    def test_layers_averaged_identical(self):
        for b in _column(CSV_DECODE_AVG, "Batch_Size"):
            self.assertEqual(
                M.read_decode_layers_averaged(CSV_DECODE_AVG, b),
                self.AB._get_histogram_decode_layers_averaged(b, CSV_DECODE_AVG), f"bs{b}")
        self.assertEqual(
            M.read_prefill_layers_averaged(CSV_PREFILL_AVG),
            self.AB._get_histogram_prefill_layers_averaged(CSV_PREFILL_AVG))

    def test_validate_agrees_with_archbench_on_every_shipped_cell(self):
        """Same accept/reject verdict, so the benchmark cannot measure a cell the
        projection would refuse, nor refuse one the projection accepts."""
        for path, phase in ((CSV_DECODE, "decode"), (CSV_PREFILL, "prefill")):
            for b, layer, hist in _cells(path, phase):
                R = B.routed_tokens(phase, b, 1024, TOPK)
                mine = ours = None
                try:
                    M.validate(hist, E, R, phase)
                except ValueError as exc:
                    mine = exc
                try:
                    self.AB.validate_expert_token_dist(
                        hist, E, R, phase=phase, exact_mass=(phase == "decode"))
                except ValueError as exc:
                    ours = exc
                self.assertEqual(mine is None, ours is None,
                                 f"{phase} bs{b} L{layer}: verdicts differ "
                                 f"(mine={mine!r} archbench={ours!r})")

    def test_groups_carry_the_same_work_as_archbench(self):
        """Ordering differs by design (moe_stats sorts, archbench keeps insertion
        order); the (num_experts, tokens) multiset must not."""
        for path, phase in ((CSV_DECODE, "decode"), (CSV_PREFILL, "prefill")):
            for b, layer, hist in _cells(path, phase):
                mine = sorted(M.groups(hist))
                theirs = sorted((nexp, avg) for avg, nexp in hist.items())
                self.assertEqual(mine, theirs, f"{phase} bs{b} L{layer}")


class StubCaller:
    """A CORRECT fused_experts, in fp32 with a bf16 result. `fault` injects one defect."""

    def __init__(self, model, fault=None):
        self.model = model
        self.fault = fault
        self.inplace = False
        self.prepack = False

    def prepack_weight(self, w):
        return w

    def __call__(self, a, w1, w2, topk_weights, topk_ids):
        import torch
        import torch.nn.functional as F
        N = self.model["moe_intermediate_size"]
        af = a.float()
        out = torch.zeros_like(af)
        for e in range(self.model["num_experts"]):
            if self.fault == "drop_expert" and e == 0:
                continue
            hit = (topk_ids == e).nonzero()
            if hit.numel() == 0:
                continue
            rows, slots = hit[:, 0], hit[:, 1]
            h = af[rows] @ w1[e].float().t()
            gate, up = h[:, :N], h[:, N:]
            if self.fault == "swap_gate_up":
                gate, up = up, gate
            y = (F.silu(gate) * up) @ w2[e].float().t()
            w = topk_weights[rows, slots]
            if self.fault == "no_topk_weight":
                w = torch.ones_like(w)
            elif self.fault == "uniform_topk_weight":
                w = torch.full_like(w, 1.0 / self.model["topk"])
            elif self.fault == "drop_one_slot":
                w = w.clone()
                w[0] = 0.0
            out.index_add_(0, rows, y * w.unsqueeze(1))
        if self.fault == "drop_last_token":
            out[-1] = 0.0
        if self.fault == "perturb_one_elem":
            out[0, 0] *= 1.03
        return out.to(torch.bfloat16)


class SharedActivationBuffer(unittest.TestCase):
    """The fused leg takes a contiguous PREFIX view of one shared buffer. A strided or
    wrong-shaped view would make the kernel copy, or read the wrong rows."""

    def test_prefix_views_are_contiguous_and_correctly_shaped(self):
        torch = _torch()
        if torch is None:
            self.skipTest("torch unavailable")
        model = dict(hidden_size=K, moe_intermediate_size=N_FULL // 4, num_experts=E,
                     topk=TOPK)
        acts = torch.randn(64 * K, dtype=torch.bfloat16) / 10
        seen = []
        for hist, want_tokens in (({1: 8}, 1), ({2: 8}, 2), ({8: 8}, 8)):
            counts, num_tokens, _ = M.expert_token_counts(hist, E, TOPK)
            self.assertEqual(num_tokens, want_tokens)
            a, w1, w2, tw, ids = B.fused_tensors(counts, num_tokens, model, None, 0,
                                                 acts=acts)
            self.assertEqual(tuple(a.shape), (num_tokens, K))
            self.assertTrue(a.is_contiguous(), "a strided view would force a copy")
            self.assertEqual(a.dtype, torch.bfloat16)
            self.assertEqual(a.data_ptr(), acts.data_ptr(), "not a prefix of the buffer")
            self.assertTrue(torch.equal(a.reshape(-1), acts[:num_tokens * K]))
            seen.append(num_tokens)
        self.assertEqual(seen, [1, 2, 8])

    def test_a_shared_buffer_still_produces_a_correct_result(self):
        """Same routing, same weights, shared vs private activations: the kernel result
        must match the fp32 reference either way. Guards against the view handing the
        kernel the wrong rows."""
        torch = _torch()
        if torch is None:
            self.skipTest("torch unavailable")
        model = dict(hidden_size=K, moe_intermediate_size=N_FULL // 4, num_experts=E,
                     topk=TOPK)
        counts, num_tokens, _ = M.expert_token_counts({4: 8}, E, TOPK)
        caller = StubCaller(model)
        runner = B.make_fused_runner(counts, num_tokens, model, caller, seed=0, copies=2,
                                     acts=torch.randn(16 * K, dtype=torch.bfloat16) / 10)
        out = runner()
        self.assertEqual(tuple(out.shape), (num_tokens, K))
        self.assertTrue(bool(out.abs().sum() > 0), "shared buffer produced all zeros")


class CheckGatePower(unittest.TestCase):
    """Does --check's 2e-2 gate actually FAIL when the kernel is wrong?

    Substituting a stub for the kernel is what makes this testable on a box with no
    CPU kernel build at all: the gate under test is the METRIC and the THRESHOLD,
    which are pure harness code.
    """

    RTOL = 2e-2
    # gross defects the check exists to catch: the kernel doing less, or different, work
    STRUCTURAL = ("drop_expert", "drop_last_token", "swap_gate_up", "no_topk_weight",
                  "uniform_topk_weight", "drop_one_slot")

    def _shapes(self):
        m4 = dict(hidden_size=K, moe_intermediate_size=N_FULL // 4, num_experts=E, topk=TOPK)
        m1 = dict(m4, moe_intermediate_size=N_FULL)
        one = M.expert_token_counts({1: 8}, E, TOPK)          # decode bs1
        many = M.expert_token_counts({8: 16, 4: 32}, E, TOPK)  # 32 tokens
        return (("decode bs1 tp4", m4, one), ("32 tokens tp4", m4, many),
                ("decode bs1 tp1", m1, one))

    def test_a_correct_kernel_passes(self):
        if _torch() is None:
            self.skipTest("torch unavailable")
        for label, model, (counts, num_tokens, _) in self._shapes():
            rel, _ = B.check_fused(counts, num_tokens, model, StubCaller(model), seed=0)
            self.assertLess(rel, self.RTOL, f"{label}: false alarm on a correct kernel")
            # and it should sit well clear of the gate, not scrape past it
            self.assertLess(rel, self.RTOL / 4,
                            f"{label}: correct kernel at {rel:.2e} leaves < 4x headroom")

    def test_structural_faults_are_all_caught(self):
        if _torch() is None:
            self.skipTest("torch unavailable")
        misses = []
        for label, model, (counts, num_tokens, _) in self._shapes():
            for fault in self.STRUCTURAL:
                rel, _ = B.check_fused(counts, num_tokens, model,
                                       StubCaller(model, fault), seed=0)
                if rel <= self.RTOL:
                    misses.append(f"{label}/{fault} rel={rel:.3e}")
        self.assertEqual(misses, [], f"--check did NOT catch: {misses}")

    def test_localized_error_is_below_the_metrics_resolution(self):
        """DOCUMENTED LIMITATION, asserted so it cannot drift silently.

        rel_err = max|err| / max|ref| normalises by the GLOBAL peak, so a small error
        on an element that is not near the peak is invisible. A single output element
        off by 3% is missed as soon as there is more than one token to hide in. This
        is inherent to bf16 comparison at these shapes, not a threshold that is merely
        set too loose: the per-element relative noise floor of a correct bf16 kernel is
        already ~2e-2 (measured), so no threshold on this data separates a 3% localized
        error from rounding. --check is a gross-defect gate, not a precision gate.
        """
        if _torch() is None:
            self.skipTest("torch unavailable")
        seen = {}
        for label, model, (counts, num_tokens, _) in self._shapes():
            rel, _ = B.check_fused(counts, num_tokens, model,
                                   StubCaller(model, "perturb_one_elem"), seed=0)
            seen[label] = rel
        self.assertGreater(seen["decode bs1 tp4"], self.RTOL,
                           "single-token cell should still see a 3% element error")
        self.assertLess(seen["32 tokens tp4"], self.RTOL,
                        "if this now trips, the metric changed -- re-read the docstring")


if __name__ == "__main__":
    unittest.main(verbosity=2)
