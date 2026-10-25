# Sampling-Bias Experiment — `head(2000)` vs stratified random 2,000

Controlled experiment that tests whether ElasticIQ's series selection — the
first 2,000 product-store series in file order (`sales.head(2000)`, all from
store **CA_1**, missing the **FOODS_3** department entirely) — contributes to
the reported forecasting error, compared with a **reproducible stratified
random sample of 2,000 series** (seed 42).

Everything this experiment writes lives under `experiments/sampling_bias/`.
No production code, model, prediction, metric, report, database schema,
frontend, or deployment file is modified, and nothing is committed or pushed.

## Run

From the repository root (`E:\ElasticIQ`):

```powershell
# full pipeline: six stages, each an isolated subprocess (bounded memory)
.venv\Scripts\python.exe -m experiments.sampling_bias.run_experiment

# individual stages (useful to resume or re-inspect)
.venv\Scripts\python.exe -m experiments.sampling_bias.run_experiment --stage verify
.venv\Scripts\python.exe -m experiments.sampling_bias.run_experiment --stage sample
.venv\Scripts\python.exe -m experiments.sampling_bias.run_experiment --stage panel
.venv\Scripts\python.exe -m experiments.sampling_bias.run_experiment --stage train
.venv\Scripts\python.exe -m experiments.sampling_bias.run_experiment --stage evaluate
.venv\Scripts\python.exe -m experiments.sampling_bias.run_experiment --stage report

# focused tests (CI only collects tests/, so run these explicitly)
.venv\Scripts\python.exe -m pytest experiments/sampling_bias/tests -q
```

Runtime expectation: the `train` stage fits one 2,000-series
`HistGradientBoostingRegressor` (≈ 3 minutes, 4.9 GB peak RSS, on the
reference laptop); the other stages are a minute or less (`panel` ≈ 45 s,
`evaluate` ≈ 2 min). A full run through the command above takes ≈ 6.5
minutes. `evaluate` additionally runs two cross-model scoring passes
(each model re-scored on the other cohort).

## What each stage does

| stage      | output | notes |
|------------|--------|-------|
| `verify`   | `verification.json` | git/environment record, Arm A baseline reproduction (saved WAPE vs recomputed), calibration-window/leakage check, `d_1914–d_1941` holdout scan |
| `sample`   | `sampling_manifest.csv`, `sampling_coverage.csv`, `sampling_summary.json` | builds both cohorts; Arm B = store × department proportional stratified sample, largest-remainder allocation, `np.random.default_rng(42)` |
| `panel`    | `processed_arm_b/`, `panel_report.json` | Arm B panel built with the **unmodified** production `melt_sales_long` / `attach_calendar_fields` / `attach_prices`; asserts shared series are row-identical to the production panel |
| `train`    | `models/arm_b_stratified/`, `train_report.json` | trains Arm B by importing the **unmodified** `run_training()` with `processed_dir`/`models_dir` pointed into this directory |
| `evaluate` | `metrics.json`, `predictions/*.parquet` | scores the full 2 × 2 (model × cohort) matrix with the project's own `wape`/`mae` |
| `report`   | `comparison_report.md` | renders the narrative report from `metrics.json` |

Per-stage logs: `logs/<stage>.log`. Per-stage elapsed time and peak RSS:
`stage_resources.json`.

## The two comparisons

- **Comparison 1 (primary):** both models scored on the *same* cohort.
  Column A = cohort A (the existing head selection, 2,000 series);
  column B = cohort B (stratified, 2,000 series). Each column holds the
  cohort, dates, and denominators fixed, so the only moving part is the model.
- **Comparison 2 (secondary):** each model on *its own* cohort
  (`arm_a_on_cohort_a` vs `arm_b_on_cohort_b`). The evaluated series differ —
  **not apples-to-apples**; it is reported as a within-sample diagnostic only.

All four cells (`arm_{a,b}_on__cohort_{a,b}`) are in `metrics.json`, each with
WAPE/MAE at 7/14/28 days, series/rows/total-actuals, an integrity block
(anything a model failed to predict is reported, never dropped silently), and
per-store/per-category breakdowns.

## Controls held identical between arms

- hyperparameters: `max_iter=300, learning_rate=0.05, calibrate=True,
  level_window=14, shrink=300.0` (mirrored from the v3 model's metadata)
- feature set, training days, validation days, library versions
- recursive 28-day forecasting, 60-day inference tail, cross-fitted
  calibration, missing-value handling
- WAPE/MAE from `src/evaluation/metrics.py` (`sum|a−p| / sum|a| × 100`)
- random seed 42

**Only intended difference:** which 2,000 series each arm was trained on.
Arm A is *loaded from saved artifacts* and never retrained.

## Safety rails

- every write path passes a guard that refuses locations outside
  `experiments/sampling_bias/`;
- production artifacts under `artifacts/` and `data/processed/` are read-only
  inputs;
- `evaluate` refuses to publish if any cell's integrity block is incomplete;
- no git commit, push, reset, or clean is performed at any point.

## Known quirks (documented, not hidden)

- `models/arm_b_stratified/metadata.json` is written by the unmodified
  production code, so its `sample_selection` string repeats the stock
  head-selection wording. It is wrong text for Arm B — the authoritative
  selection record is `sampling_manifest.csv`.
- Arm A's training duration is a historical record from `PROGRESS.md`; its
  peak memory was never measured because this experiment never retrains it.
- The evaluation window `d_1886–d_1913` is the development window;
  `d_1914–d_1941` exists in `sales_train_evaluation.csv` and is reported in
  `verification.json` as a candidate holdout, deliberately left unspent.
