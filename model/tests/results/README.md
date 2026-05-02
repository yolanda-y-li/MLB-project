# Test results — CSV reference

This folder contains the CSV outputs produced by the extended test suite at
`model/tests/`. Each test script writes one or two CSVs here. The schemas
below let you open any file and know what every column means.

A complete walk-through of *why* each test exists and what biological
question it answers is in `report/MODEL_TESTING_REPORT.md`.

If you only see synthetic numbers in `sample_outputs/`, that's the smoke
test — those are not your model's results, just placeholders that confirm
the schema renders correctly.

---

## per_class_metrics.csv

Per-class precision, recall, F1, support, plus AUROC (one-vs-rest) and AP.
Six rows per split: one for each of the four classes plus macro-average and
weighted-average rows.

| column        | meaning                                                                 |
|---------------|-------------------------------------------------------------------------|
| split         | `train` / `val` / `test`                                                |
| class_id      | `0` binds · `1` upregulates · `2` downregulates · `3` no_interaction · `-1` average row |
| class_name    | human-readable label                                                    |
| precision     | TP / (TP + FP) for this class                                           |
| recall        | TP / (TP + FN) for this class                                           |
| f1            | harmonic mean of precision and recall                                   |
| support       | number of test pairs whose true label is this class                     |
| auroc_ovr     | AUROC of "this class vs everything else"                                |
| ap            | average precision of "this class vs everything else"                    |

**What to look at:** compare F1 across classes. If `binds` F1 is much higher
than `upregulates` / `downregulates`, the model is leaning on the dominant
class.

---

## confusion_matrix.csv

Tidy long-form confusion matrix. Two rows for every (true, pred) cell — one
raw count and one row-normalized fraction.

| column      | meaning                                        |
|-------------|------------------------------------------------|
| split       | `train` / `val` / `test`                       |
| normalized  | `0` raw count · `1` fraction of the true class |
| true_class  | actual label                                   |
| pred_class  | model's argmax prediction                      |
| count       | integer count when normalized=0, fraction (0–1) when normalized=1 |

**What to look at:** the off-diagonals when `normalized=1`. If
`upregulates → downregulates` is a large fraction, the model can predict
"there's an effect" but not the sign — biologically a critical confusion.

---

## calibration_summary.csv

One row per split. Calibration is whether the model's confidence matches its
empirical accuracy.

| column   | meaning                                                                |
|----------|------------------------------------------------------------------------|
| split    | `train` / `val` / `test`                                               |
| ece      | Expected Calibration Error (lower is better; 0 means perfectly calibrated) |
| mce      | Maximum Calibration Error — worst single bin                            |
| brier    | multi-class Brier score (mean squared error of probs vs one-hot)        |
| nll      | negative log-likelihood                                                 |
| n        | number of test pairs                                                    |
| n_bins   | number of confidence bins (default 15)                                  |

**Reference points:** an ECE of ~0.02 is well calibrated, ~0.10 is poorly
calibrated.

## calibration_bins.csv

Reliability-diagram data. One row per (split × confidence bin).

| column          | meaning                                                  |
|-----------------|----------------------------------------------------------|
| split           | `train` / `val` / `test`                                 |
| bin_lo / bin_hi | confidence range of the bin (e.g. 0.6 to 0.667)           |
| mean_confidence | average max-class probability for predictions in the bin |
| mean_accuracy   | empirical accuracy in the bin                            |
| count           | number of predictions in the bin                          |

**What to look at:** plot `mean_confidence` (x) vs `mean_accuracy` (y). A
perfectly calibrated model lies on the diagonal y = x.

---

## threshold_sensitivity.csv

Binary classification ("any interaction vs no_interaction") metrics swept
over thresholds on `P(interaction) = 1 − P(no_interaction)`.

| column                | meaning                                                              |
|-----------------------|----------------------------------------------------------------------|
| split                 | `train` / `val` / `test`                                             |
| threshold             | decision threshold (or `BEST_F1@x.xxx` for the optimal threshold row) |
| precision / recall / f1 | binary-task metrics                                                |
| accuracy              | overall accuracy                                                     |
| n_predicted_positive  | how many pairs the model called "any interaction"                    |
| n_actual_positive     | how many pairs are actually positive                                 |
| n_total               | total pairs in the split                                             |

**What to look at:** the row with `threshold` starting `BEST_F1@` —
the model's optimal operating point if you only care about the binary call.

---

## per_edge_type.csv

For each true positive class (binds / upregulates / downregulates), how well
the model recovers it.

| column                  | meaning                                                            |
|-------------------------|--------------------------------------------------------------------|
| split                   | `val` / `test`                                                     |
| true_class              | one of the three positive interaction classes                      |
| n                       | support for this class                                             |
| accuracy_correct_class  | fraction of pairs of this class predicted into this class          |
| mean_prob_true_class    | average softmax mass the model puts on the correct class           |
| top1_confusion_class    | when the model is wrong, what class does it most often pick?       |
| top1_confusion_share    | what fraction of errors fall into that class?                      |
| auroc_vs_no_interaction | AUROC of this class restricted to (this class vs no_interaction)   |

---

## cold_start.csv

Performance bucketed by whether the test drug / gene was seen during training.

| column     | meaning                                                                 |
|------------|-------------------------------------------------------------------------|
| split      | `val` / `test`                                                          |
| bucket     | `SEEN_BOTH` / `UNSEEN_DRUG` / `UNSEEN_GENE` / `UNSEEN_BOTH`              |
| n          | number of test pairs in this bucket                                     |
| auroc      | macro AUROC inside the bucket (blank if support is too small)           |
| ap         | macro AP inside the bucket                                              |
| macro_f1   | macro F1 inside the bucket                                              |
| accuracy   | accuracy inside the bucket                                              |

**What to look at:** the gap between `SEEN_BOTH` and `UNSEEN_*`. With
identity-only embeddings (no chemical / protein features), cold-start
performance should drop noticeably.

---

## bootstrap_ci.csv

1000-resample bootstrap 95% confidence intervals for the headline metrics.

| column           | meaning                                                          |
|------------------|------------------------------------------------------------------|
| split            | `val` / `test`                                                   |
| metric           | `auroc_macro` / `ap_macro` / `f1_macro` / `auroc_binary` / `ap_binary` |
| point_estimate   | metric on the full split                                          |
| boot_mean        | mean across bootstrap resamples                                   |
| boot_std         | standard deviation across resamples                               |
| ci_lower_2.5     | 2.5th percentile (lower bound of 95% CI)                          |
| ci_upper_97.5    | 97.5th percentile                                                 |
| n_boot_samples   | number of valid resamples                                         |
| n_test           | size of the underlying split                                      |

**What to look at:** the CI width. A point estimate of 0.89 with CI [0.86,
0.92] means a different test split could plausibly give 0.86 or 0.92 — so
small differences between configurations are not significant.

---

## subgroup_by_drug_degree.csv / subgroup_by_gene_degree.csv

Test-set performance bucketed by the number of training-positive edges the
drug (or gene) has. `Q1` = lowest-degree quartile (rare nodes), `Q4` =
highest-degree quartile (hub nodes), plus an `ALL` row.

| column            | meaning                                          |
|-------------------|--------------------------------------------------|
| bucket            | `Q1` / `Q2` / `Q3` / `Q4` / `ALL`                |
| n                 | number of test pairs in the bucket               |
| mean_train_degree | average training-positive degree in the bucket   |
| auroc_macro       | macro AUROC                                      |
| ap_macro          | macro AP                                         |
| f1_macro          | macro F1                                         |
| accuracy          | accuracy                                         |

**What to look at:** Q1 vs Q4. A big drop on Q1 reveals the model is
hub-biased — performance is driven by well-characterized targets and
rare ones suffer.

---

## negative_robustness.csv

Re-evaluates the same trained model on test-set negatives generated by two
strategies: edge-swap (the training scheme) and uniformly random.

| column            | meaning                                       |
|-------------------|-----------------------------------------------|
| negative_strategy | `edge_swap` / `random`                        |
| n_pos             | number of positive pairs in test              |
| n_neg             | number of negative pairs                      |
| auroc_macro / ap_macro / f1_macro / accuracy  | metrics under that negative regime |

**What to look at:** if `random` is much higher than `edge_swap`, the model
is good at the easy task but struggles with biologically-plausible hard
negatives. If they're similar, the model has learned features that
generalize.

---

## ranking_metrics.csv

For each test gene with at least one positive, rank all drugs by
P(interaction) and compute hit-rate / reciprocal-rank metrics. This is the
deployment metric — "given a gene, how well does the model prioritize true
drugs at the top of the list?"

| column     | meaning                                                          |
|------------|------------------------------------------------------------------|
| metric     | `hits@1`, `hits@5`, `hits@10`, `hits@20`, `hits@50`, `hits@100`, `mrr`, `mean_rank`, `median_rank` |
| value      | the metric value                                                  |
| n_queries  | number of (gene, true-drug) ranking queries used                  |

**What to look at:** `hits@10` is the most common deployment metric — what
fraction of true drug-gene interactions show up in the top-10 of the
ranked list. `mrr` summarizes the whole ranking with a single number.
