"""
Run every test script in this directory and write CSVs to ./results/.

Usage:
  cd model
  python tests/run_all_tests.py

You can also run individual tests:
  python tests/test_per_class_metrics.py
  python tests/test_calibration.py
  ...

If a checkpoint at model/checkpoints/best_model.pt is missing, the test
utilities will train a fresh model with the default config (this takes a
few minutes on a GPU) and save the checkpoint so subsequent test scripts
re-use it.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent))

from test_per_class_metrics import run as run_per_class
from test_confusion_matrix import run as run_confusion
from test_calibration import run as run_calibration
from test_threshold_sensitivity import run as run_threshold
from test_ranking_metrics import run as run_ranking
from test_cold_start import run as run_cold_start
from test_bootstrap_ci import run as run_bootstrap
from test_per_edge_type import run as run_per_edge
from test_negative_robustness import run as run_neg_robust
from test_subgroup_analysis import run as run_subgroup


TESTS = [
    ("per_class_metrics", run_per_class),
    ("confusion_matrix",  run_confusion),
    ("calibration",       run_calibration),
    ("threshold_sensitivity", run_threshold),
    ("per_edge_type",     run_per_edge),
    ("cold_start",        run_cold_start),
    ("bootstrap_ci",      run_bootstrap),
    ("subgroup_analysis", run_subgroup),
    ("negative_robustness", run_neg_robust),
    # ranking_metrics is the slowest — runs last
    ("ranking_metrics",   run_ranking),
]


def main():
    print("=" * 80)
    print("Running extended test suite")
    print("=" * 80)
    summary = []
    for name, fn in TESTS:
        print(f"\n--- {name} ---")
        t0 = time.time()
        try:
            fn()
            elapsed = time.time() - t0
            summary.append((name, "OK", round(elapsed, 1)))
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[!] {name} failed after {elapsed:.1f}s: {e}")
            traceback.print_exc()
            summary.append((name, f"FAIL: {e.__class__.__name__}", round(elapsed, 1)))

    print("\n" + "=" * 80)
    print("Test suite summary")
    print("=" * 80)
    for name, status, t in summary:
        print(f"  {name:25s}  {status:30s}  {t}s")

    n_ok = sum(1 for _, s, _ in summary if s == "OK")
    print(f"\n{n_ok}/{len(summary)} tests succeeded")
    print(f"CSV outputs in: {THIS_DIR / 'results'}")


if __name__ == "__main__":
    main()
