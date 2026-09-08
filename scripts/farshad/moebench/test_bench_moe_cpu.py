#!/usr/bin/env python3
"""Stdlib-only self-tests for the parts of bench_moe_cpu.py that do not need torch.

Run:  python3 test_bench_moe_cpu.py
"""

import argparse
import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_moe_cpu as B  # noqa: E402


def _row(**kw):
    """A results row with the columns run_cells always fills, plus whatever kw adds."""
    base = dict(phase="decode", batch=1, layer=0, instance=0, stats_commit="deadbeef",
                stats_ref="branch", dtype="bf16", stats_mode="per_layer",
                hidden_size=2048, moe_intermediate_size=128, num_experts=256, topk=8,
                tp=4, threads=56, num_groups=1, active_experts=8, histogram_mass=8,
                routed_tokens=8, slack=0, num_tokens=1, gflop=0.0126, groups="8x1")
    base.update(kw)
    return base


class WriteRowsTest(unittest.TestCase):
    def test_appending_rows_without_check_does_not_shift_columns(self):
        """A lane sweep appends several invocations to ONE --out file. The first lane
        ran --check and so wrote a `check_rel_err` column into the header; a later lane
        did not. The later lane's values must still land under their own labels."""
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "results.csv")
            args = argparse.Namespace(out=out)

            B.write_rows(args, [_row(check_rel_err=3.9e-3, fused_median_ms=0.09,
                                     fused_min_ms=0.08, fused_iters=20)])
            # second lane: no --check, so no check_rel_err key at all
            B.write_rows(args, [_row(phase="prefill", fused_median_ms=3.295,
                                     fused_min_ms=2.82, fused_iters=10)])

            with open(out, newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            checked, unchecked = rows
            self.assertEqual(checked["check_rel_err"], "0.0039")
            self.assertEqual(float(checked["fused_median_ms"]), 0.09)
            # the row that never measured a check must leave the cell EMPTY, not
            # slide fused_median_ms into it
            self.assertEqual(unchecked["check_rel_err"], "")
            self.assertEqual(float(unchecked["fused_median_ms"]), 3.295)
            self.assertEqual(unchecked["fused_iters"], "10")

    def test_first_write_uses_the_canonical_header(self):
        """A file opened by a no-check lane must still carry the check column, so a
        later lane that does run --check can append to it."""
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "results.csv")
            args = argparse.Namespace(out=out)
            B.write_rows(args, [_row(fused_median_ms=1.0)])
            with open(out, newline="") as f:
                header = next(csv.reader(f))
            self.assertEqual(header, B.RESULT_FIELDS)
            B.write_rows(args, [_row(check_rel_err=1e-3, fused_median_ms=2.0)])
            with open(out, newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(float(rows[1]["check_rel_err"]), 1e-3)

    def test_unknown_column_is_rejected_not_shifted(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "results.csv")
            args = argparse.Namespace(out=out)
            with self.assertRaises(SystemExit) as cm:
                B.write_rows(args, [_row(brand_new_metric=1.0)])
            self.assertIn("brand_new_metric", str(cm.exception))

    def test_row_keys_are_all_declared(self):
        """Every key _row() mimics, plus the timing/check keys run_cells adds, must be
        declared in RESULT_FIELDS -- that is what keeps the header stable."""
        keys = set(_row()) | {"check_rel_err", "fused_over_batched"}
        for mode in ("fused", "batched"):
            keys |= {f"{mode}_{k}" for k in
                     ("median_ms", "min_ms", "mean_ms", "p90_ms", "iters")}
        self.assertEqual(keys - set(B.RESULT_FIELDS), set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
