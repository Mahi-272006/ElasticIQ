# Diagnostic Report — ML Series Selection & Full-Scale Training Estimate

**Date:** 2026-10-25 · **Scope:** read-only inspection of the ElasticIQ training
pipeline · **Author:** agent diagnostic pass

**No code, model, evaluation methodology, or processed-data artifact was
modified.** Diagnostic scripts were kept outside the repo (temp dir), per repo
convention. This report is the only file added.

**Labeling convention used throughout:**

- **[VERIFIED]** — read directly from code or measured from the current data/artifacts
- **[ESTIMATE]** — extrapolation with stated assumptions; not measured end-to-end
- **[RECOMMENDATION]** — proposal only; nothing was executed

---

## 1. Q1 — How are the 2,000 series selected?

**[VERIFIED]** Yes — they are exactly the first 2,000 rows of the M5 series
table in file order.

- Selection code: `src/data/pipeline.py:357–359` →
  `selected = min(sample_series, len(sales))` then `sample = sales.head(selected)`,
  applied to raw `sales_train_validation.csv` before melting.
- Default N: `src/config.py:67–68` → `sample_series = 2000`, overridable via
  env `ELASTICIQ_SAMPLE_SERIES` (`.env.example:11`).
- Empirical check: the set of 2,000 ids in `data/processed/sample_panel.parquet`
  equals the set of the first 2,000 ids of the raw CSV (`panel_is_exact_head_of_raw: true`);
  raw file has 30,490 series total.
- Recorded in `hgb_poisson_v3/metadata.json` as
  `"sample_selection": "first 2,000 series in file order (deterministic head selection)"`.

## 2. Q2 — Cardinalities inside the 2,000-series sample

**[VERIFIED]** (computed by grouping the panel on `id`):

| Dimension | Count | Values (series counts) |
|---|---|---|
| Stores | **1** | CA_1 (2,000) |
| States | **1** | CA (2,000) |
| Categories | **3** | HOUSEHOLD 1,047 · HOBBIES 565 · FOODS 388 |
| Departments | **6** | HOUSEHOLD_1 532 · HOUSEHOLD_2 515 · HOBBIES_1 416 · FOODS_1 216 · FOODS_2 172 · HOBBIES_2 149 |

Notable: the head-selection is **store-clustered** — all 2,000 series come from
a single store (CA_1), and **FOODS_3 is absent entirely** (raw file order is not
alphabetical by department). 9 of 10 stores and at least one department are
outside the training population.

## 3. Q3 — Training rows after feature engineering

**[VERIFIED]** Row-removal chain (matches `hgb_poisson_v3` metadata exactly):

| Step | Rows | Change |
|---|---|---|
| Sample panel (2,000 × 1,913 days, d_1..d_1913) | 3,826,000 | — |
| Hold out validation days d_1886..d_1913 | 3,770,000 | −56,000 |
| Lag history rule `day_idx > 28` (lag_28 needs 28 prior days; d_1..d_28 dropped) | 3,714,000 | −56,000 |
| Rolling windows (7/14/28/56, `min_periods=1`, `shift(1)`) | 3,714,000 | **−0** |
| `notna()` over all 24 feature columns (20 numeric + 4 categorical) | **3,017,710** | **−696,290 (18.75 %)** |

- The 696,290 filtered rows are **100 % caused by NaN `sell_price`**
  (days before a product's introduction; `ffill` cannot backfill leading NaNs).
  `sell_price` is the only feature column with NaN inside the training window.
- 1,084 of 2,000 series are affected (median 633.5 and max 1,764 rows removed
  for affected series; removed rows span d_29..d_1792).
- Lag features as such remove only the first 28 days; rolling windows remove
  nothing; all remaining removal is the missing-price filter.
- `build_features` measured at 11–21 s for this panel (varies with machine load).
- **Final: 3,017,710 training rows — identical to `train_rows` in
  `artifacts/models/hgb_poisson_v3/metadata.json`.**

## 4. Q4 — Distinct series in train vs validation

**[VERIFIED]**

- Distinct series in training rows: **2,000**
- Distinct series in validation rows: **2,000**
- The two sets are **identical**; 0 series have zero training rows.
- Validation rows: 56,000 = 2,000 × 28 days (d_1886..d_1913).

## 5. Q5 — What does validation evaluate?

**[VERIFIED]** The split is **purely temporal on the same series** — it does
**not** evaluate unseen product-store series.

- Split: `chronological_split` → train d_1..d_1885, validate d_1886..d_1913
  (`src/forecasting/train.py`); evaluation actuals come from the same 2,000 ids
  that produced the training rows (set equality proven in Q4).
- 0 unseen series in validation ⇒ **no cold-start evaluation exists**.
- Consequence (verified from bundle): trained `categorical_levels` are
  `{store_id: 1, state_id: 1}`; at inference, `inference.py:199–202` casts
  unseen categorical levels to the training categories → any store other than
  CA_1 maps store/state to **NaN** (missing branch). Generalization to the other
  9 stores is untested by the current protocol.

## 6. Q6 — Memory & runtime for all eligible M5 series (no full run launched)

**Inputs [VERIFIED]:**

- Machine: 15.7 GB RAM installed (2.1 GB free at probe time), Intel i5-1334U
  (10 cores / 12 threads).
- Full panel size: 30,490 × 1,913 = **58,327,370 rows = 15.245×** the 2,000-series panel.
- Measured at 2,000 series (this machine): panel frame 605 MB, features frame
  699 MB, **process peak 4,032 MB**, features build 11–21 s, production-config
  HGB fit **1.29 s/iter** (100 iters, early stopping on) or 2.35–2.52 s/iter
  with early stopping off, recursive 7-day forecast 15.6 s.
- Measured at 6,000 series (in-memory raw-CSV path, no writes): panel build
  56.8 s / peak 5,963 MB, features 49.7 s, fit 5 iters 44.2 s, **process peak 9,230 MB**.
- Production-recorded totals at 2,000 series [VERIFIED from artifacts]:
  e0_base 130.6 s (100 iters), e1 69.9 s, e2 119.0 s, e3_long 322.9 s (300 iters),
  official v3 run 280.4 s (`--calibrate` = two 300-iteration fits + two
  recursive forecasts; `PROGRESS.md:397`).

**Timing reconciliation [VERIFIED]:** today's probes run ~2–5× slower than the
recorded production runs (e.g. the exact e0 fit alone took 129 s today while
e0's *entire* recorded run was 130.6 s). Cause: low free RAM at probe time
(2.1 GB free; probe peak 4.0 GB → pagefile thrash). Recorded runs are the
better "quiet machine" anchor; today's probes are the worst-case anchor.

**Estimates [ESTIMATE] — linear scaling ×15.245 (rows/series), HGB histograms
and the recursive loop are ~linear in n; assumptions stated:**

| Item | Full-scale estimate | Basis |
|---|---|---|
| Training rows | **≈ 46.0 M** (±5 %) | 3,017,710 ÷ 2,000 × 30,490; assumes CA_1's price-introduction pattern typical |
| One-time data pipeline rebuild (melt + merges + parquet) | **~5–10 min**, peak **~25–30 GB** transient | 6,000-series melt stage measured 56.8 s / 5,963 MB, scaled ×5.08 |
| Training run, 100-iteration config (e0-style) | **~30–60 min** | 130.6 s × 15.245 (recorded rates) … 224 s × 15.245 (today's rates) |
| Training run, official v3-style (calibrate, 2×300 iters) | **~1.2–4 h** | 280.4 s × 15.245 … today's-rate component sum × 15.245 |
| Recursive 28-day validation forecast | ~16 min | 63 s × 15.245 |
| **Peak RAM, training process** | **≈ 35–55 GB** | frames 9.2 + 10.7 ≈ 19.9 GB (linear from measured frame sizes) + HGB/pandas temporaries scaled from measured peaks (4,032 MB @2k; 9,230 MB @6k) |

**Verdict [ESTIMATE]:** full-scale training needs a machine with **≥64 GB
RAM**. This machine has 15.7 GB — the run is **infeasible here as-is** (it
would page heavily or be killed). Practical ceiling on this machine is roughly
the 6,000–8,000-series range (6,000 already peaked at 9.2 GB). No code change
is required to *attempt* full scale (`ELASTICIQ_SAMPLE_SERIES=30490`), but the
hardware constraint stands. Recording peak memory in `metrics.json` would require
a small instrumentation addition (not made — see §9).

## 7. Q7 — Was the 2,000-series cap intentional?

**[VERIFIED evidence] / [RECOMMENDATION conclusion]: Intentional mechanism,
unjustified value.**

Evidence that the *sampling approach* was deliberate:

- Deterministic head-selection is documented in three places: pipeline docstring
  (`pipeline.py:12–13`), the report string (`pipeline.py:403–404`), and model
  metadata (`train.py:426`).
- It is env-configurable (`ELASTICIQ_SAMPLE_SERIES`) — i.e. treated as a knob,
  not a hard constraint.
- README lists it under "Limitations & honesty contract" → "Sample vs catalog"
  (README:278), so it is consciously disclosed.
- The same head-sampling pattern (`sales.head(NUMBER_OF_PRODUCTS)`, 500) is used
  in `baseline.py`/`seasonal_baseline.py` — an established repo convention.

Evidence that the *number 2000* itself was arbitrary:

- The value appears fully formed in Milestone 1 (`9444b6e`, config default) with
  only a descriptive comment in `.env.example` ("Number of product-store series
  used for the sample training pipeline") — no rationale.
- Commit bodies for `9444b6e` and `ae1d24a` are one-liners; no design doc, ADR,
  or PROGRESS entry explains why 2,000 (no timing budget, memory budget, power
  analysis, or coverage target appears anywhere in the repo).
- The outcome — a store-clustered, department-truncated subset (CA_1 only, no
  FOODS_3) — looks accidental rather than designed; a deliberate coverage-driven
  choice would have been stratified.

**Conclusion:** limiting training was an intentional *computational/consistency*
decision (documented, deterministic, configurable sample), but the specific
threshold 2,000 was **not justified anywhere** — at the value level it is an
arbitrary subset choice.

## 8. Q8 — Recommended controlled experiment

**[RECOMMENDATION — proposal only; nothing executed]**

**Arms (single variable: series count):**

- **A:** 2,000 — `head(2000)`, bit-identical to the current model (baseline).
- **B:** 6,000 — `head(6000)`; nested (A ⊂ B) so arm A stays directly comparable;
  feasible on this machine (measured peak 9.2 GB).
- **C:** 30,490 (full) — only on a ≥64 GB machine (see Q6). If unavailable,
  substitute C′ = 20,000 and state the deviation.

**Fixed across arms (unchanged evaluation methodology):**

- Same temporal boundaries: decisions on the **dev window** (train d_1..d_1857,
  evaluate d_1858..d_1885); **official window** run once per arm (train
  d_1..d_1885, evaluate d_1886..d_1913).
- Same recursive 28-day protocol, same future-price assumption, baselines
  recomputed on identical series/dates, same config for all arms (lock the v3
  recipe: poisson, 300 iters, lr 0.05, base features, L14/shrink300, `--calibrate`).
- Selection stays deterministic `head(N)`; nested subsets keep A ≡ current model.
  Known confound to disclose: `head(N)` is store-clustered, so B/C change both
  data volume *and* store coverage. Optional robustness arm D: seeded stratified
  sample per store (same N) to separate those effects.

**Reported per arm (as requested):**

1. **WAPE** (h7/h14/h28) and **MAE** — on two populations: (i) the common
   head(2,000) subset (apples-to-apples: does more data help the *same* series?),
   (ii) each arm's full population (coverage effect), plus per-store breakdown
   in B/C (cold-store performance — the store_id→NaN issue from Q5).
2. **Training time** — `elapsed_seconds`, already recorded by the harness.
3. **Peak memory** — *not currently recorded*; add peak-RSS capture to
   `metrics.json` as part of authorizing this experiment (the only code change
   proposed, made only on approval).

**Pre-registered decision rule:** adopt the larger arm if official-window h28
WAPE on the common subset improves ≥ 1.0 pp (or overall-population WAPE
improves with no common-subset regression > 0.5 pp), and training time/memory
fit the deployment budget; otherwise keep A.

**Feasibility:** A and B runnable today; C requires the ≥64 GB machine (Q6).

## 9. Q9 — Change control & summary of the recommendation

**[VERIFIED]** Nothing was changed:

- No source file, config, model bundle, experiment metrics, or
  `data/processed/*` file was modified or regenerated.
- Evaluation methodology untouched (this report prescribes no change to it).
- Diagnostic scripts live in the OS temp dir, not the repository.
- This report is a new, separate file; git status shows only the pre-existing
  uncommitted frontend redesign plus this untracked report.

**Recommendation in one paragraph:** The 2,000-series cap is a documented,
deterministic compute sample whose specific value was never justified, and its
head-selection makes the current model single-store (CA_1) with no cold-start
evaluation. Before any further accuracy work, run the controlled A/B/C
experiment in §8 — starting with the feasible 6,000-series arm on this machine —
reporting WAPE/MAE on both the common and full populations plus training time
and peak memory, and treat full-scale training as a ≥64 GB-machine project
(≈35–55 GB peak, ≈1–4 h per official-style run).

---

### Appendix — probe measurements (this machine)

| Probe | Rows | Fit config | Fit time | Peak RSS |
|---|---|---|---|---|
| 2,000 series, features + 10-iter ES-off fit | 3,017,710 train | poisson, lr 0.05, ES off | 23.5–25.2 s (2.35–2.52 s/iter) | 4,032 MB |
| 2,000 series, exact e0 production config | 2,961,710 train | poisson, lr 0.08, 100 iters, ES on | 129.0 s (1.29 s/iter) — inflated by 2.1 GB free RAM | (not sampled) |
| 6,000 series, raw-CSV in-memory path | 8,597,479 train | poisson, 5 iters, ES off | 44.2 s (8.84 s/iter, paging-inflated) | 9,230 MB |
| Recursive forecast, 7 days | 2,000 series | — | 15.6 s | ≤4,032 MB |
