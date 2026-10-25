# ElasticIQ — Forecasting Improvement Log

Mission: reduce 28-day WAPE from **88.8277%** toward 30% while keeping the
evaluation honest (chronological split, recursive inference, no leakage,
unchanged evaluation population).

Conventions used throughout:

- **Official validation window:** d_1886–d_1913 (28 days), 2,000 series,
  56,000 prediction rows, all from store CA_1 (the pipeline samples the
  deterministic `head(2000)` of the processed panel).
- **Dev window (for decisions):** d_1858–d_1885 — carved from the *training*
  side so the official validation set is not used for repeated tuning.
- WAPE = `sum(|actual - predicted|) / sum(actual) * 100` (project definition,
  `src/evaluation/metrics.py`).

---

## Phase 1 — Repository inspection (2026-10-25)

**Finding:** Code inspection found no leakage bug:

- `src/features/engineering.py`: every lag/rolling feature is `shift(1)`-based;
  rolling windows use only days `< t`.
- `src/forecasting/inference.py`: recursive protocol feeds *predictions* back,
  never validation actuals; future prices = last observed price.
- `src/forecasting/train.py`: chronological split, baselines computed on the
  identical series/dates.
- `src/evaluation/metrics.py`: standard WAPE, zero-denominator handled.
- Artifacts present and untouched: `artifacts/models/hgb_poisson_v1/`
  (`bundle.joblib`, `evaluation.json`, `metadata.json`,
  `validation_predictions.parquet`).

**Commands run:** `git status` (clean), file reads, artifact listing.

**Changed:** nothing in the repo (diagnostic scripts live in the temp dir).

---

## Phase 2 — WAPE diagnosis (2026-10-25)

Script: `diagnose_wape.py` (console report + `artifacts/reports/diagnosis.json`).

### Metric reproduction

| metric | value |
|---|---|
| WAPE overall | **88.8277%** (exactly matches `evaluation.json`) |
| MAE | 0.9076 units |
| rows / series | 56,000 / 2,000 |
| sum(actual) / sum(pred) | 57,219 / 61,863 units → **bias +8.1%** |

### Where the error is

| cut | WAPE | notes |
|---|---|---|
| zero-sales rows (59.48% of rows) | — | model predicts **0.60 units** on them |
| positive rows (40.52%) | — | model predicts **1.85** vs actual **2.52** (under-predicts) |
| HOBBIES_2 | 162.06% | mean actual 0.254 |
| HOUSEHOLD_2 | 117.90% | mean actual 0.432 |
| HOUSEHOLD_1 | 72.78% | densest series |
| intermittent series (train mean 0.05–0.3) | 150.89% | 19,992 rows |
| low (0.3–1) | 101.28% | 22,428 rows |
| mid (1–4) | 75.20% | |
| high (≥4) | 65.52% | |
| per-series WAPE median | 117.53% | 133 series sold **0 units** in validation |

- MAE is roughly flat across forecast days (0.75–1.10) → **recursive error
  accumulation is NOT the main problem**; day-1 WAPE (90.9%) is already as bad
  as day-28 (79.0%), and day-level WAPE swings are driven by the weekday
  denominator, not by horizon.
- Alignment checks: shifting predictions ±1/±2 days makes WAPE *worse*
  (91.5% / 95.2% / 89.4% / 91.4% vs 88.83%) → **no off-by-one bug**; stored
  dates match the calendar (0 mismatches).

### Headroom and floor probes (script `diagnose_floor.py` → `diagnosis2.json`)

Leak-free baselines on the official window:

| forecast | WAPE |
|---|---|
| series train mean (constant) | 93.49% |
| series × weekday mean (shrunk) | 91.98% |
| series mean × global weekday shape | 92.21% |
| current model `hgb_poisson_v1` | **88.83%** |
| mean of last 28 days (project baseline) | 86.49% |

Leaky probes (diagnostic only — never reported as a score):

| probe | WAPE | interpretation |
|---|---|---|
| exact validation mean per series | 82.23% | best possible *constant* per series |
| validation mean per series × weekday | 68.28% | best possible weekly profile |
| best single global scale on model (s=0.661) | 83.33% | predictions ~50% too high |
| **best per-series scale on model** | **72.54%** | per-series level is mis-estimated |

### Statistical noise floor (script `diagnose_noise.py` → `diagnosis3.json`)

Assuming a perfect model knows the conditional mean exactly:

| assumption | estimated irreducible WAPE |
|---|---|
| Poisson noise only | 60.4% |
| negative binomial, median cell dispersion | 86.8% |
| negative binomial, mean cell dispersion | 98.5% |

Conditional variance/mean in training cells = 6.26 (heavily over-dispersed).

**Interpretation:** demand here is extremely intermittent (59.5% zeros, mean
1.02 units/series/day) and over-dispersed. Independent estimates put the
irreducible floor between roughly **60% and 85% WAPE**, so the 30% target
appears unreachable on this dataset/protocol. This is an evidence-based
statement, not an excuse — experiments below still push toward the lowest
honest number.

### Conclusion of diagnosis

No confirmed bug. The dominant, fixable error is **per-series level
miscalibration**: the model cannot distinguish series identities (no series-id
or per-series statistical features), so it over-predicts sparse series (0.60
units on days that are actually zero) and under-predicts dense positive days.

---

## Phase 3+ — Experiments (running)

Protocol: train on d_1..d_1857, decide on dev d_1858..d_1885, retrain chosen
config on d_1..d_1885, evaluate **once** on official d_1886..d_1913.

Harness: `python -m src.forecasting.experiment --tag <name> --eval dev|official`
(recursive inference, identical baselines, same WAPE code as `train.py`).
Outputs in `artifacts/experiments/<tag>/`.

### E0 — dev-window reference (incumbent configuration, poisson, base features)

Command: `python -m src.forecasting.experiment --tag e0_base --eval dev`

| metric | value |
|---|---|
| WAPE h7 / h14 / h28 | 84.8594% / 86.9271% / **87.9075%** |
| mean baseline h28 | 85.6236% (model still loses to it) |
| MAE | 0.8953 |
| fitted iterations / train rows | 100 / 2,961,710 |
| runtime | 130.6 s |

### E1 — WAPE-aligned loss (hypothesis: L1/median suits pooled WAPE)

Command: `--tag e1_l1 --eval dev --loss absolute_error`

| result | value |
|---|---|
| WAPE h28 | **100.0000%** (fitted only 8 iterations, predicts ~0 everywhere) |
| MAE | 1.0184 |

**Honest verdict: failed, worse than every baseline.** For this intermittent
data the conditional median is ~0 for most rows, so pure median regression
collapses to zero predictions. Confirms shrinkage must be *level-aware*, not
global. Reverted (kept as recorded negative result).

### S1 — seasonal-level baseline (leak-free, no training)

Construction: `forecast = mean(last 28 days) x normalised weekday profile`,
profile estimated from history only.

| variant | dev WAPE |
|---|---|
| level28 constant | 85.6236% |
| level28 x global weekday shape | 83.8087% |
| level28 x series shape (k=30, toward global) | 83.7909% |
| level28 x series shape x SNAP factor | **83.6578%** |
| E0 model alone | 87.9075% |
| best convex blend model+seasonal | 83.89% (weight on model = 0) |

**Finding:** a 3-line statistical baseline beats the ML model by ~4 points;
blending the two does not help yet (the model's level bias drags it down).

### C1 — calibration probe, leak-free (split dev in half)

Fit scale factors on d_1858..d_1871, score on d_1872..d_1885
(`calibration_probe.py`, uses `src/forecasting/calibration.py`):

| calibration | eval-half WAPE | bias |
|---|---|---|
| raw E0 model | 88.9336% | +7.3% |
| global scale (fitted 0.687) | 84.3018% | −26.3% |
| per-series, shrink=10 | **79.7734%** | −24.8% |
| per-series, shrink=30 | 80.8219% | −24.2% |
| per-series, shrink=100 | 82.0089% | −24.7% |
| per-series, shrink=300 | 83.0492% | −25.4% |

**Finding:** per-series level calibration is the single biggest lever found so
far (≈9 WAPE points) and it **generalises across time** (fit on one half of
dev, gains hold on the other half).

### E2 — extended features (hypothesis: per-series statistics fix level bias)

Command: `--tag e2_extended --eval dev --features extended`
(new numerics `demand_share_56`, `days_since_last_sale`, … + weekday/month as
categoricals)

| metric | value |
|---|---|
| WAPE h7 / h14 / h28 | 84.6410% / 87.0479% / **88.5849%** |
| vs E0 | **worse by +0.68 pts** |

**Honest verdict: failed.** Bundled two changes (new numerics + new
categoricals), so it also fails as an attribution study — recorded as a
negative result; extended features are not used in the final system.

### E3 — hyperparameters (hypothesis: more boosting, gentler step size)

Command: `--tag e3_long --eval dev --max-iter 300 --learning-rate 0.05`

| metric | value | vs E0 |
|---|---|---|
| WAPE h7 / h14 / h28 | 84.3877% / 86.4997% / **87.5965%** | −0.31 pts |
| MAE | 0.8921 | −0.003 |
| fitted iterations / runtime | 300 / 322.9 s | |

**Verdict: small, consistent gain.** Its *shape* is also better under
calibration (78.1890 vs 78.3152 at shrink=10) and it wins the calibration grid
(76.3856 vs 76.5442) → chosen as the final model configuration.

### Shape evaluation (model shape vs weekday shape)

`shape_eval.py`, dev fit-half → score-half, all leak-free:

| candidate | calibrated eval-half WAPE (E3) |
|---|---|
| **model shape (level14)** | **78.1890%** |
| model shape (default) | 78.5496% |
| model × weekday geometric blend | 78.9950% |
| weekday shape k=30 (84d) | 79.6985% |
| flat (constant level) | 80.0303% |

**Finding:** the model's day-to-day *pattern* is genuinely useful; only its
*level* is broken. Level window ranking: 14 > 28 > 56 days.

### Calibration design grid (level window × shrink target × shrink strength)

`calibration_grid.py` (fit d_1858–d_1871 → score d_1872–d_1885), E3 shape:

| level window | shrink → 0 | 0.5 | 1 | 3 | 5 |
|---|---|---|---|---|---|
| 3 days | 76.510 | 76.602 | 76.690 | 77.000 | 77.253 |
| **7 days** | **76.386** | 76.608 | 76.786 | 77.275 | 77.590 |
| 14 days | 76.400 | 76.703 | 76.924 | 77.464 | 77.770 |

(shrink target = *global* beats *toward 1.0* in every cell of the grid.)

- **Reverse split** (fit second half → score first half) agrees: shrink=0 is
  optimal everywhere (L3 75.566%, L7 76.022%).
- At shrink=0 the level window barely matters → mechanism identified:
  **the per-series scale absorbs the level entirely**; the real choice is how
  strongly each series' level is refitted from recent actuals.
- Zero-level fallback (series with 0 units in the last 7 days keep the
  model's level instead of forecasting 0): 76.40% vs 76.39% → **no help**,
  rejected.

### Residual structure after level refit

`residual_structure.py` (pooled weekday/category/SNAP corrections fitted on
the fit half, scored on the eval half): baseline 76.5442% → best stacked
weekday→category **76.4057%** (−0.14 pts). SNAP adds 0.00.

**Finding:** the per-series refit has already absorbed the systematic
structure; further corrections are noise-level → **not worth the complexity**.

### Final system (locked before the official run)

> **Superseded by the cross-block re-lock below** — this L7/shrink=0 design is
> what official run #1 shipped and failed with; kept verbatim as the record.

    final[id, day] = scale[id] × level[id] × shape[id, day]

- **model:** HistGradientBoosting, poisson, `--max-iter 300
  --learning-rate 0.05`, base features, seed 42 (E3 config)
- **shape:** raw prediction normalised to mean 1 within each series
- **level:** mean of the last **7** observed days (one full weekday cycle)
- **scale:** per-series weighted-median ratio fitted leak-free by inner
  cross-fit (inner model trains ≤ d_1857, fits scales on d_1858–d_1885),
  shrink=0, clipped to [0.05, 4.0]

**Leakage audit of the official evaluation:** final model trains ≤ d_1885;
scales come from a model trained ≤ d_1857 scored on ≤ d_1885; level comes
from days ≤ d_1885; shape comes from the model's own predictions; d_1886+
actuals are used **only** as the scoring target.

### Code integration (all tested, lint clean)

| file | change |
|---|---|
| `src/forecasting/postprocess.py` | **new** — `levelled_forecast`, `fit_postprocess`, `apply_postprocess` (+ zero-raw, zero-level, unknown-series guards) |
| `src/forecasting/inference.py` | `recursive_forecast` applies `bundle["postprocess"]` once, after the recursion (bundles without it behave exactly as before) |
| `src/forecasting/train.py` | `--calibrate`, `--learning-rate`, `--version`, `--level-window`; `evaluation.json` gains `raw_model`; metadata gains `postprocess` provenance; defaults unchanged |
| `src/forecasting/calibration.py` | fitted scales for the postprocess (shrink/clip documented) |
| `tests/test_postprocess.py` | **new** — 15 tests: arithmetic, degenerate cases, fitting contract, recursion wiring, `run_training(calibrate=True)` round-trip |

### Official run #1 — FAILED (2026-10-25)

Command as planned: `python -m src.forecasting.train --max-iter 300
--learning-rate 0.05 --calibrate --version hgb_poisson_v2` (481.8 s).

| metric | value |
|---|---|
| raw model (HGB only) on d_1886..d_1913 | 88.9381% (expected, ≈ incumbent) |
| **final system (scale × level × shape)** | **95.1632% — WORSE than raw and the incumbent** |
| h7 / h14 / h28 (final) | 91.77% / 93.36% / 95.16% |

**This is a genuine failure, reported as-is.** Post-mortem
(`postmortem.py`, `transfer_probe.py`):

1. **No implementation bug:** the repo's `fit_postprocess`/`apply_postprocess`
   reproduce the dev grid number to 4 decimals (76.3856% on the dev
   fit-half→score-half protocol) and the shipped predictions are exactly
   reproducible from the bundle.
2. **Decomposition on official:** prior alone (level7 × shape) = **87.7139%**
   (bias −2.4%) — better than raw and the incumbent; prior × *global* scale
   = **83.1444%** (leaky refit of the global scale on official = 83.0061% →
   the global part transfers almost perfectly); prior × dev-fitted
   *per-series* scales = 95.1632% (bias −11.7%) → **the per-series part is
   what fails.**
3. **Not model identity:** using the inner model's own shape instead of the
   final model's gives 95.1789% (same failure).
4. **Not the machinery:** leaky per-series scales fitted *on* official reach
   71.8983% → the estimator is powerful when it sees its target block.
5. **The smoking gun:** `corr(dev-fitted scale, dev need) = +0.50` but
   `corr(dev-fitted scale, official need) = **−0.14**`.

**Root cause (design flaw in the validation protocol, not in the code):**
every dev experiment fitted the scale against a level7 prior computed from
days ≤ d_1857 and *also applied it* against that same fixed level7. The
official application instead uses a *fresh* level7 (≤ d_1885). The scale
therefore absorbed dev-block-specific deviations of
`window-mean / level7(dev-dates)` — a ratio of two noisy 7-day samples of an
intermittent series — and those deviations do not survive the level-date
change (they average ≈ 1.0 in every block: dev 0.958 median, official 1.000
median, but per-series they are mostly sampling noise). Within-dev halves
agreed only because both halves shared the same fixed level7.

**Lesson (recorded for the protocol):** a validation design must reproduce
the *exact* fit→apply transform of production — including which dates the
level prior comes from — or it can only measure in-block fit, not transfer.

### Cross-block validation harness (2026-10-25) — design re-locked

`cross_block.py` measures **transfer**, not in-block fit. A probe model
trained ≤ d_1600 (HGB poisson, `max_iter=100`, seed 42) is scored on three
out-of-sample fit→score block pairs — (fit d_1601–1628 → score d_1629–1656),
(fit d_1657–1684 → score d_1685–1712), (fit d_1713–1740 → score d_1741–1768)
— plus a dev pair (e3_long fit d_1858–1871 → score d_1872–1885). Each block
gets a **fresh level prior from its own history** (production semantics), and
the grid is `level ∈ {7,14,28}` × `scale ∈ {none, med(shrink=0,10,30,100,300),
global}`. Pair predictions cached in `cross_block_cache/`.

**Harness bug found by a parity check (self-audit):** `pair_frames` originally
built *every* fit frame as `start..start+27`, so the dev fit window silently
covered d_1858–1885 — overlapping the dev score window d_1872–1885. The repo
code path gave 80.6529% on the same cell while the harness claimed 79.019%;
the 1.6-point discrepancy exposed the overlap. Fixed with explicit
`fit_end`/`score_end`. After the fix the harness dev cell reproduces the repo
code **exactly (80.6529%)**. The leak had inflated the leakiest candidate
(`med(shrink=0)`) by 8.5 points on dev (73.70% → 82.19%) — and it alone
created the illusion that dev favoured low shrink while early pairs favoured
high shrink. After the fix **every cell agrees**: shrink improves monotonically
to an interior minimum near 300, then reverses slightly at pure-global.

**Corrected results (4 cells each, leak-free):**

| candidate | mean | best | worst |
|---|---|---|---|
| **L14 \| med(shrink=300)** | **81.086%** | 79.247% | **83.791%** |
| L28 \| med(shrink=300) | 81.085% | 78.994% | 84.537% |
| L14 \| global scale | 81.211% | 79.052% | 83.775% |
| L28 \| global scale | 81.293% | 79.202% | 84.382% |
| L7 \| med(shrink=300) | 81.621% | 80.033% | 83.921% |
| L7 \| no scale (level only) | 87.491% | 84.704% | 91.138% |
| **L7 \| med(shrink=0)** *(old shipped design)* | **90.215%** | 88.775% | **91.862%** |

**Design re-lock (training-side data only — official window untouched by the
choice):** `scale = per-series weighted median, shrink=300` × `level = last
14 days` × `model shape`, clip [0.05, 4.0]. Rationale: tied-best mean, best
worst-case among mean-leaders, wins 2 of 3 early pairs, shrink=300 is an
interior optimum in every cell, and L14/L28 dominate L7 (sampling noise on
intermittent series). The old `shrink=0` design is the worst candidate in
every block — it was never a valid estimate of transferable scales.

**Code changes applied:** `DEFAULT_LEVEL_WINDOW 7→14`, `DEFAULT_SHRINK 0→300`
(`postprocess.py`, docstring records the rationale), `train.py` gains
`--shrink`, tests now assert against `DEFAULT_SHRINK`. Tests green (30 in
postprocess/train_inference/calibration), `ruff check src tests` clean.

### Official run #2 — hgb_poisson_v3 (2026-10-25)

Command (defaults now L14/shrink300, so no extra flags):
`python -m src.forecasting.train --max-iter 300 --learning-rate 0.05
--calibrate --version hgb_poisson_v3` (280.4 s; 2000 series, global scale
0.7450). `hgb_poisson_v1` (incumbent) and `hgb_poisson_v2` (failed run #1)
were left untouched — rollback = delete `hgb_poisson_v3` (and `v2`) to
return to `v1`.

**Disclosure: the official window d_1886–d_1913 has now been scored twice** —
run #1 (failed 95.16%, L7/shrink0) and this run #2 (corrected design). The
design re-lock was made exclusively on training-side cross-block data; run #2
is the second and final scoring of this window.

| metric | incumbent v1 | run #2 (v3) |
|---|---|---|
| **overall WAPE** | **88.8277%** | **82.5704%** (−6.26 pts) |
| raw HGB model (same window) | 88.8277% | 88.9381% (postprocess does all the work) |
| h7 / h14 / h28 | — | 80.20% / 80.98% / 82.57% |
| MAE (units) | 0.9076 | 0.8437 |

Prediction vs the cross-block harness: mean 81.09%, worst 83.79% → actual
82.57% lands inside the harness's expected band (official sits between the
harness's best and worst blocks).

Breakdowns (`report_breakdown.py`, same 2000 series / 56,000 rows):

| segment / group | incumbent v1 | run #2 |
|---|---|---|
| intermittent (<0.3/day) | 150.9% | **125.1%** |
| low (0.3–1) | — | 94.5% |
| mid (1–4) | — | 72.3% |
| high (≥4/day) | 65.5% | **63.8%** |
| HOBBIES_2 dept | 162.0% | **138.5%** |
| FOODS / HOBBIES / HOUSEHOLD | — | 84.7% / 89.1% / 78.0% |
| mean_baseline (same rows) | 86.49% | 86.49% (v3 beats it by 3.9 pts) |

Residual structure: every segment still under-predicts (bias −19% to −30%),
per-series median WAPE 101.3%, 133 series have zero actuals in the window
(infinite per-series WAPE), 59.5% of rows are zeros with mean final
prediction 0.372.

### Final gap analysis (honest)

- **Delivered:** 88.8277% → **82.5704%** (−6.26 pts, 7.0% relative error
  reduction) with zero leakage, no changed evaluation population, no metric
  changes, and every failed attempt recorded (E1, E2, run #1).
- **What fixed it:** the postprocess *design* re-locked by cross-block
  validation — `level 14d` (was 7d: kills sampling noise on intermittent
  series) and `shrink=300` (was 0: unshrunk per-series scales are block-specific
  artifacts). The model itself (raw 88.94%) is unchanged from run #1; the
  entire gain is honest level/scale recalibration that transfers.
- **Why not 30%:** the leaky oracle (per-series scales fitted *on* the
  official window, unusable in production) reaches only 71.90%, and the noise
  floor estimated from repeated sub-window splits is 60–85%. Reaching <30%
  would require reducing *irreducible* per-day intermittency noise (59.5%
  zero rows), which no leakage-free design in this feature set can do. The
  gap to 30% is a property of the data, not of the current model.
- **Where remaining error lives:** intermittent/low segments (bias −29%),
  HOBBIES_2 (138%), and long horizons (h28 82.57 vs h7 80.20) — all dominated
  by level-estimation noise on sparse series.

### Reproduce (final state)

```bash
# train + official evaluation (writes artifacts/models/hgb_poisson_v3)
python -m src.forecasting.train --max-iter 300 --learning-rate 0.05 --calibrate --version hgb_poisson_v3
# tests + lint
python -m pytest tests          # 166 passed
python -m ruff check src tests  # clean
# cross-block design evidence (cached, re-runs in seconds)
python <Temp>/cross_block.py
```

### Summary table

| # | hypothesis | config | dev WAPE | official WAPE | verdict |
|---|---|---|---|---|---|
| E0 | reference | poisson, base, 100 iter | 87.9075% | (incumbent 88.8277%) | baseline |
| E1 | L1 loss aligns with WAPE | absolute_error | 100.0000% | — | FAILED |
| E2 | per-series features fix level bias | poisson, extended | 88.5849% | — | FAILED (worse than E0) |
| E3 | more boosting, gentler lr | poisson, 300 iter, lr 0.05 | 87.5965% | — | mild gain → final config |
| S1 | seasonal level beats model level | level28 × weekday × SNAP | 83.6578% | — | model shape still wins after calibration |
| C1 | per-series calibration | shrink grid on dev halves | 79.7734% | — | biggest single lever |
| F | level × shape × leak-free scale | E3 + L7 + shrink0 | **76.3856%** | *hgb_poisson_v2 run* → **95.1632% FAILED** | dev protocol measured in-block fit, not transfer |
| X1 | cross-block transfer validation | L14 × shrink300 × E3 shape | 80.6529% (fresh-level dev) | run #2 `hgb_poisson_v3` | re-locked design; harness mean 81.086%, worst 83.791% |

*(experiments appended as they complete)*
