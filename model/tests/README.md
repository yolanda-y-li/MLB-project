# Extended Model Test Suite

This directory adds tests that go beyond the headline AUROC / AP / F1 numbers
reported by `model/main.py`. Every test loads the saved checkpoint at
`model/checkpoints/best_model.pt` (or trains one if missing) and writes a CSV
into `model/tests/results/`.

## How to run

```bash
cd model
python tests/run_all_tests.py
```

You can also run a single test:

```bash
python tests/test_per_class_metrics.py
python tests/test_calibration.py
python tests/test_ranking_metrics.py
# ...
```

All CSV outputs land in `model/tests/results/`.

## What each test produces

| Script | CSV(s) | What it answers |
|---|---|---|
| `test_per_class_metrics.py` | `per_class_metrics.csv` | Precision / recall / F1 / AUROC / AP per class. Are some classes much harder than others? |
| `test_confusion_matrix.py` | `confusion_matrix.csv` | 4x4 confusion matrix (raw + row-normalized). Which classes does the model confuse? |
| `test_calibration.py` | `calibration_summary.csv`, `calibration_bins.csv` | ECE, MCE, Brier, NLL + reliability bins. When the model says 90% confident, is it actually right 90% of the time? |
| `test_threshold_sensitivity.py` | `threshold_sensitivity.csv` | Precision/recall/F1/accuracy as a function of threshold for the binary "any interaction vs none" call. Includes best-F1 threshold. |
| `test_per_edge_type.py` | `per_edge_type.csv` | Per-edge-type accuracy (binds vs upregulates vs downregulates) and which class each is most often confused with. |
| `test_cold_start.py` | `cold_start.csv` | Performance bucketed by whether the test drug / gene was seen in training. Tests generalization to cold drugs and cold genes. |
| `test_bootstrap_ci.py` | `bootstrap_ci.csv` | 95% confidence intervals (1000 resamples) for AUROC / AP / F1, both multi-class and binary. |
| `test_subgroup_analysis.py` | `subgroup_by_drug_degree.csv`, `subgroup_by_gene_degree.csv` | Performance bucketed by drug / gene training degree. Hub effect? |
| `test_negative_robustness.py` | `negative_robustness.csv` | Re-evaluate the same trained model on edge-swap vs random negatives. |
| `test_ranking_metrics.py` | `ranking_metrics.csv` | For each test gene, rank all drugs by P(interaction) and compute Hits@K (K = 1, 5, 10, 20, 50, 100) and MRR. The most realistic deployment metric. |

## Adding new tests

Each test is a stand-alone script that exposes a top-level `run()` function.
The shared boilerplate (model loading, prediction, splits) lives in
`test_utils.py`. To add a test:

1. Create `test_<name>.py` in this directory.
2. Import helpers from `test_utils`.
3. Implement `def run(out_path: Path | None = None): ...` and write your
   CSV.
4. Register it in `run_all_tests.py`.
