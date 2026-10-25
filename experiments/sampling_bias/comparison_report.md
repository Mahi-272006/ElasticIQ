# Sampling-Bias Experiment — Comparison Report

Generated 2026-10-25T13:29:51.965350+00:00 · git `43ec8ed64a17` · production tree untouched (only `experiments/` files added) · **no commit or push was made**

Rerun: `.venv\Scripts\python.exe -m experiments.sampling_bias.run_experiment`

## 1. Question and scope

Does selecting the **first 2,000** product-store series (all from store CA_1, and missing the FOODS_3 department entirely) contribute to ElasticIQ's forecasting error, compared with a **reproducible stratified random sample of 2,000 series**?

This is an experiment, not a refactor: no production code, model artifact, evaluation definition, training/validation date, or WAPE calculation was changed. Everything written lives under `experiments/sampling_bias/`.

## 2. Methodology

| Arm | Series selection | Model artifact | Trained by this experiment? |
| --- | --- | --- | --- |
| **A (baseline)** | first 2,000 in file order (production `head(2000)`) | `artifacts/models/hgb_poisson_v3` | No — loaded from saved artifacts |
| **B (stratified)** | store × dept proportional sample, seed 42 | `experiments/sampling_bias/models/arm_b_stratified` | Yes — via unmodified `run_training()` |

Strata: `store_id x dept_id` (70 strata, smallest 149 series → allocation ≥ 9). Allocation: proportional to stratum size with the largest-remainder rule (integer counts sum to exactly 2000); every non-empty stratum gets >= 1.

**Held identical between arms (verified, not assumed):**

| Control | Value | Identical? |
| --- | --- | --- |
| Hyperparameters | poisson, 300 iters, lr 0.05, leaf 63, min leaf 20, seed 42 | True |
| Feature set | 24 columns | True |
| Training days | d_1..d_1885 (both arms) | True |
| Validation days | d_1886..d_1913 (both arms) | True |
| Calibration | cross-fitted, level 14, shrink 300 | yes (same code + args) |
| Recursive forecast | 28 days, 60-day tail, predictions fed back | yes (same function) |
| Scoring | src/evaluation/metrics.py wape() = sum(|actual - predicted|) / sum(|actual|) * 100 (0.0 when both sums are 0, inf otherwise) | yes (same function) |
| Library versions | 1.9.1 sklearn | True |

**Only intended difference:** which 2,000 product-store series were selected.

## 3. Verification (before any comparison)

- **Baseline reproduction:** saved WAPE `82.5704`% vs recomputed `82.5704`% → **exact match** (MAE 0.8437 vs 0.8437); cohort equals `head(2000)`: **yes**; 56,000 rows × 2000 series × d_1886..d_1913. Baseline **loaded, not retrained**.
- **Arm A recorded training time:** 280.4 s (PROGRESS.md:397 (official run #2; not re-measured)); peak memory for Arm A was not measured (never retrained here).
- **Calibration leakage check:** scale window `d_1858..d_1885` vs evaluation `d_1886..d_1913` → uses evaluated observations: **false**; leakage risk found: **false**. Calibration was left unchanged (changing it would add a variable).
- **Future window d_1914–d_1941:** `sales_train_evaluation.csv` ends at `d_1941` (28 days beyond the validation file: d_1914..d_1941); calendar covers it: true; processed outputs stop at 1913 → `candidate_final_holdout`. Repository references: 17 total, of which 0 are metric/tuning/model-selection usages (the rest are schema/data-quality scans). **This experiment did not score the holdout.**
- **Arm B panel equivalence:** for the 131 series present in both panels, every row (units, prices, calendar fields) is identical: **true** — so the panels differ only in which series they contain.

## 4. Coverage: what each arm actually sees

| Dimension (distinct levels) | Population | Arm A (head) | Arm B (stratified) |
| --- | --- | --- | --- |
| state | 3 | 1 | 3 |
| store | 10 | 1 | 10 |
| category | 3 | 3 | 3 |
| department | 7 | 6 | 7 |
| store_x_dept | 70 | 6 | 70 |

Counts by dimension/value (selected series):

**store**

| value | population | Arm A | Arm B |
| --- | --- | --- | --- |
| CA_1 | 3049 | 2000 | 200 |
| CA_2 | 3049 | 0 | 200 |
| CA_3 | 3049 | 0 | 200 |
| CA_4 | 3049 | 0 | 200 |
| TX_1 | 3049 | 0 | 200 |
| TX_2 | 3049 | 0 | 200 |
| TX_3 | 3049 | 0 | 200 |
| WI_1 | 3049 | 0 | 200 |
| WI_2 | 3049 | 0 | 200 |
| WI_3 | 3049 | 0 | 200 |

**category**

| value | population | Arm A | Arm B |
| --- | --- | --- | --- |
| FOODS | 14370 | 388 | 940 |
| HOBBIES | 5650 | 565 | 370 |
| HOUSEHOLD | 10470 | 1047 | 690 |

**department**

| value | population | Arm A | Arm B |
| --- | --- | --- | --- |
| FOODS_1 | 2160 | 216 | 140 |
| FOODS_2 | 3980 | 172 | 260 |
| FOODS_3 | 8230 | 0 | 540 |
| HOBBIES_1 | 4160 | 416 | 270 |
| HOBBIES_2 | 1490 | 149 | 100 |
| HOUSEHOLD_1 | 5320 | 532 | 350 |
| HOUSEHOLD_2 | 5150 | 515 | 340 |

## 5. Results

### 5.1 Primary — fixed common cohorts (2 × 2 matrix)

Each column is one fixed evaluation cohort (identical series AND identical dates for both models); each row is one model.

| Model | Cohort A WAPE (%) — head(2,000) | Cohort B WAPE (%) — stratified(2,000) | Cohort A MAE | Cohort B MAE |
| --- | --- | --- | --- | --- |
| Arm A (head-trained) | 82.5704 | 70.5424 | 0.8437 | 1.0053 |
| Arm B (stratified-trained) | 83.2208 | 75.2756 | 0.8503 | 1.0727 |
| **Δ (B − A), pp** | **+0.6504 pp** | **+4.7332 pp** |  |  |

### 5.2 Cohort A — head(2,000)

| Metric | Arm A | Arm B | Δ (B − A) |
| --- | --- | --- | --- |
| WAPE 28d (%) | 82.5704 | 83.2208 | +0.6504 pp |
| WAPE 7d (%) | 80.2044 | 81.2329 | +1.0285 pp |
| WAPE 14d (%) | 80.9771 | 81.8044 | +0.8273 pp |
| MAE (units) | 0.8437 | 0.8503 | +0.0066 |
| Relative WAPE change |  |  | 0.79 % |
| Series scored | 2,000 | 2,000 | identical |
| Prediction rows | 56,000 | 56,000 |  |
| Total actual units | 57,219 | 57,219 |  |
| Integrity (complete) | True | True |  |
| Winner |  |  | `arm_a` |

Per-store sweep: {"arm_b_better_in": 0, "arm_a_better_in": 1, "ties": 0, "undecided_zero_denominator": 0}; per-category sweep: {"arm_b_better_in": 1, "arm_a_better_in": 2, "ties": 0, "undecided_zero_denominator": 0}.

### 5.3 Cohort B — stratified(2,000)

| Metric | Arm A | Arm B | Δ (B − A) |
| --- | --- | --- | --- |
| WAPE 28d (%) | 70.5424 | 75.2756 | +4.7332 pp |
| WAPE 7d (%) | 69.9657 | 76.9280 | +6.9623 pp |
| WAPE 14d (%) | 69.8476 | 75.0970 | +5.2494 pp |
| MAE (units) | 1.0053 | 1.0727 | +0.0674 |
| Relative WAPE change |  |  | 6.71 % |
| Series scored | 2,000 | 2,000 | identical |
| Prediction rows | 56,000 | 56,000 |  |
| Total actual units | 79,804 | 79,804 |  |
| Integrity (complete) | True | True |  |
| Winner |  |  | `arm_a` |

Per-store sweep: {"arm_b_better_in": 1, "arm_a_better_in": 9, "ties": 0, "undecided_zero_denominator": 0}; per-category sweep: {"arm_b_better_in": 0, "arm_a_better_in": 3, "ties": 0, "undecided_zero_denominator": 0}.

### 5.4 Per-store WAPE on cohort B (where the denominator is non-zero)

| store | Arm A WAPE | Arm B WAPE | Δ pp | actual units | series | rows |
| --- | --- | --- | --- | --- | --- | --- |
| CA_1 | 76.7574 | 76.5035 | -0.2539 | 7,880 | 200 | 5600 |
| CA_2 | 72.7760 | 74.0637 | +1.2877 | 8,531 | 200 | 5600 |
| CA_3 | 58.1190 | 73.1941 | +15.0751 | 14,070 | 200 | 5600 |
| CA_4 | 96.4732 | 96.7334 | +0.2602 | 3,711 | 200 | 5600 |
| TX_1 | 74.4020 | 78.7172 | +4.3152 | 6,065 | 200 | 5600 |
| TX_2 | 68.0317 | 70.9016 | +2.8699 | 7,395 | 200 | 5600 |
| TX_3 | 60.5168 | 69.2032 | +8.6864 | 9,105 | 200 | 5600 |
| WI_1 | 73.8130 | 75.9017 | +2.0887 | 7,265 | 200 | 5600 |
| WI_2 | 74.0318 | 75.3695 | +1.3377 | 9,623 | 200 | 5600 |
| WI_3 | 76.9789 | 77.1642 | +0.1853 | 6,159 | 200 | 5600 |

### 5.5 Per-category WAPE (denominator non-zero)

**cohort_a**

| category | Arm A WAPE | Arm B WAPE | Δ pp | actual units | denom ok |
| --- | --- | --- | --- | --- | --- |
| FOODS | 84.7400 | 82.8475 | -1.8925 | 13,932 | True |
| HOBBIES | 89.1149 | 89.6635 | +0.5486 | 15,053 | True |
| HOUSEHOLD | 78.0105 | 79.9700 | +1.9595 | 28,234 | True |

**cohort_b**

| category | Arm A WAPE | Arm B WAPE | Δ pp | actual units | denom ok |
| --- | --- | --- | --- | --- | --- |
| FOODS | 64.8907 | 71.3967 | +6.5060 | 53,820 | True |
| HOBBIES | 96.0236 | 98.3384 | +2.3148 | 7,187 | True |
| HOUSEHOLD | 76.9818 | 77.5638 | +0.5820 | 18,797 | True |

### 5.6 Secondary — within-sample diagnostic (NOT apples-to-apples)

Each model on **its own** 2,000 series: Arm A 82.5704% on cohort A vs Arm B 75.2756% on cohort B (Δ -7.2948 pp, winner `arm_b`). Evaluated series differ, so this cannot carry the conclusion.

## 6. Resources

| stage | elapsed (s) | peak RSS (MB) |
| --- | --- | --- |
| verify | 1.5 | 1085.9 |
| sample | 1.0 | 221.3 |
| panel | 65.4 | 3410.2 |
| train | 200.5 | 4856.0 |
| evaluate | 148.5 | 1305.6 |
| report | 0.0 | 196.0 |

- Arm B training (production `run_training` timer): **200.4 s**; peak process RSS **4856.0 MB**.
- Arm A training: **280.4 s** (recorded in PROGRESS.md:397 (official run #2; not re-measured)); peak memory: not measured - Arm A was loaded from saved artifacts, never retrained by this experiment.

## 7. Interpretation

- **Fixed cohort A — head(2,000) (the existing selection's own series):** Arm A 82.5704% WAPE vs Arm B 83.2208% → Δ = +0.6504 pp (0.79% relative); winner = `arm_a`. MAE 0.8437 → 0.8503 units.
- **Fixed cohort B — stratified(2,000) (representative of the full catalog):** Arm A 70.5424% WAPE vs Arm B 75.2756% → Δ = +4.7332 pp (6.71% relative); winner = `arm_a`. MAE 1.0053 → 1.0727 units.
- **Descriptive cross-cohort gap (same model, different cohorts):** Arm A scores 12.0280 pp better when moved from cohort A to cohort B; Arm B scores 7.9452 pp better. The cohorts differ in composition and difficulty, so this gap is descriptive, not a causal estimate.
- **Overall reading:** Arm A wins on both fixed cohorts. On this seed and this window the experiment does NOT support sampling bias as a contributor to WAPE; the head-trained model is at least as accurate on the representative cohort.
- **What the headline WAPE represents:** the head-selected cohort carries 57,219 actual units over the same 56,000 rows, versus 79,804 units for the stratified cohort — a 28.3% smaller denominator — while its absolute error is LOWER (MAE 0.8437 vs 1.0053). The same Arm A model therefore reads 82.5704% on the head cohort and 70.5424% on the representative cohort. Selection bias is real, but it operates on *what the reported number represents* (a low-volume, one-store cohort) rather than degrading the trained model.
- **Per-store pattern on cohort B:** Arm B better in 1 store(s), Arm A better in 9, ties 0.
- **Per-category pattern on cohort B:** Arm B better in 0 category/categories, Arm A better in 3, ties 0.
- **Per-category pattern on cohort A:** Arm B better in 1, Arm A better in 2, ties 0.
- No improvement was assumed in advance: if the deltas above show Arm B equal or worse, that is the result.

## 8. Limitations and confounders

- The evaluation window d_1886–d_1913 is the same window used to develop the current model — a development result, not a confirmed result on an untouched test set. `sales_train_evaluation.csv` (d_1914–d_1941) exists and is reported as a candidate holdout; it was deliberately left unspent.
- Single seed (42) and a single stratified draw: no variance estimate. A different draw could move the deltas; treat sub-point differences as noise.
- Cross-cohort gaps mix selection effects with cohort difficulty (cohort A is one store, cohort B spans 10 stores and the FOODS_3 department). The primary claim comes from within-column comparisons only.
- Arms share 131 series, so the training sets are not disjoint — that is intentional (nested comparability) but softens the contrast.
- On cohort B, Arm A had no fitted per-series scale for 1,869 of its series and used its global scale 0.7450287366761059 with store_id unseen for 1,800 series — this is exactly what production does for out-of-selection series, but it means Arm A's cross-cell score includes that fallback by design.
- Conversely, on cohort A only 131 of 2,000 series carried a fitted scale from Arm B; the rest used Arm B's global scale. Calibration coverage is therefore asymmetric per cell.
- Arm A's training duration comes from a historical log entry and its peak memory was never measured; only Arm B's runtime/memory are instrumented.
- Arm B's `models/arm_b_stratified/metadata.json` carries the stock `sample_selection` string from the unmodified training code (it claims head selection). It is wrong text for Arm B; `sampling_manifest.csv` is the authoritative selection record. Changing the string would have required editing production code.
- Both arms are 2,000-series runs by design (16 GB laptop limit); no 6,000-series or full-catalog arm was attempted here.
- Nothing here establishes causality about *why* a cohort scores as it does (store mix, department mix, and series density all change together between cohorts).

## 9. Files, rerun, and change control

| file (under `experiments/sampling_bias/`) | size |
| --- | --- |
| `__init__.py` | 0.2 KB |
| `comparison_report.md` | 14.5 KB |
| `logs/evaluate.log` | 0.1 KB |
| `logs/panel.log` | 0.1 KB |
| `logs/report.log` | 0.1 KB |
| `logs/sample.log` | 0.1 KB |
| `logs/train.log` | 0.1 KB |
| `logs/verify.log` | 0.1 KB |
| `metrics.json` | 35.2 KB |
| `models/arm_b_stratified/bundle.joblib` | 163.3 KB |
| `models/arm_b_stratified/evaluation.json` | 3.4 KB |
| `models/arm_b_stratified/metadata.json` | 2.8 KB |
| `models/arm_b_stratified/validation_predictions.parquet` | 347.6 KB |
| `panel.py` | 6.2 KB |
| `panel_report.json` | 1.0 KB |
| `predictions/arm_a__on__cohort_b.parquet` | 561.0 KB |
| `predictions/arm_b__on__cohort_a.parquet` | 327.0 KB |
| `processed_arm_b/calendar.parquet` | 50.6 KB |
| `processed_arm_b/sample_panel.parquet` | 19283.6 KB |
| `README.md` | 5.5 KB |
| `run_experiment.py` | 70.6 KB |
| `sampling.py` | 10.9 KB |
| `sampling_coverage.csv` | 3.3 KB |
| `sampling_manifest.csv` | 408.4 KB |
| `sampling_summary.json` | 1.1 KB |
| `scoring.py` | 9.1 KB |
| `stage_resources.json` | 0.8 KB |
| `tests/__init__.py` | 0.1 KB |
| `tests/test_sampling.py` | 11.9 KB |
| `tests/test_scoring.py` | 8.7 KB |
| `train_report.json` | 1.1 KB |
| `verification.json` | 4.1 KB |

**Exact rerun command (from the repository root):**

```
.venv\Scripts\python.exe -m experiments.sampling_bias.run_experiment
```

Git: HEAD `43ec8ed64a179765fbb5851138d738b530cd695c`, branch `main`, status: only untracked `experiments/` files; no tracked file modified. **No commit and no push were performed**; only new files under `experiments/sampling_bias/` were added.

