"""Tests for WAPE calibration (src/forecasting/calibration.py)."""

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import wape
from src.forecasting.calibration import (
    _weighted_median,
    apply_scales,
    evaluate,
    fit_global_scale,
    fit_per_series_scales,
)


def brute_force_best_scale(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Reference: scan scales and keep the one with the lowest abs error."""

    grid = np.linspace(0.01, 3.0, 3000)
    errors = [np.abs(actual - s * predicted).sum() for s in grid]
    return float(grid[int(np.argmin(errors))])


def test_weighted_median_matches_bruteforce_objective():
    values = np.array([0.5, 1.0, 2.5, 0.2])
    weights = np.array([1.0, 4.0, 1.0, 1.0])
    m = _weighted_median(values, weights)

    def objective(s: float) -> float:
        return float((weights * np.abs(values - s)).sum())

    assert objective(m) <= objective(1e-6) + 1e-9
    assert objective(m) <= objective(10.0) + 1e-9
    # exact optimum for this input: weighted median sits at 1.0
    assert m == 1.0


def test_global_scale_is_wape_optimal():
    rng = np.random.default_rng(0)
    actual = rng.gamma(shape=1.5, scale=2.0, size=5000).round()
    predicted = np.clip(actual * 1.8 + rng.normal(0, 0.5, 5000), 0.05, None)

    s = fit_global_scale(actual, predicted)
    reference = brute_force_best_scale(actual, predicted)

    assert s == pytest.approx(reference, abs=0.01)
    # it must actually improve on the raw predictions
    assert wape(actual, s * predicted) < wape(actual, predicted)


def test_global_scale_handles_all_zero_predictions():
    actual = np.array([1.0, 2.0, 0.0])
    predicted = np.zeros(3)
    assert fit_global_scale(actual, predicted) == 1.0  # safe fallback


def _frame() -> pd.DataFrame:
    """Two series: A is systematically 2x too high, B systematically too low."""

    rows = []
    for day in range(1, 29):
        rows.append({"id": "A", "day_idx": day, "actual": 4.0,
                     "predicted_units": 8.0})   # needs scale 0.5
        rows.append({"id": "B", "day_idx": day, "actual": 4.0,
                     "predicted_units": 1.0})   # needs scale 4.0
    return pd.DataFrame(rows)


def test_per_series_scales_learn_each_series_level():
    frame = _frame()
    # shrink=0 -> every series keeps its own estimate (shrinkage is covered
    # by test_sparse_series_shrink_toward_global_scale)
    calibration = fit_per_series_scales(frame, shrink=0.0)

    assert calibration["per_series"]["A"] == pytest.approx(0.5, abs=0.05)
    assert calibration["per_series"]["B"] == pytest.approx(4.0, abs=0.05)

    before = evaluate(frame)
    after = evaluate(apply_scales(frame, calibration))
    assert after["wape_percent"] < before["wape_percent"]
    # with a perfect per-series scale the residual error vanishes
    assert after["wape_percent"] == pytest.approx(0.0, abs=1e-6)


def test_sparse_series_shrink_toward_global_scale():
    # global scale ~ 1.0 (equal volume both ways), series "tiny" has almost
    # no volume so its own (noisy) estimate should be pulled back.
    rows = []
    for day in range(1, 29):
        rows.append({"id": "big", "day_idx": day, "actual": 40.0,
                     "predicted_units": 40.0})
        rows.append({"id": "tiny", "day_idx": day, "actual": 1.0,
                     "predicted_units": 1.0})
    frame = pd.DataFrame(rows)

    tight = fit_per_series_scales(frame, shrink=1.0)
    loose = fit_per_series_scales(frame, shrink=10_000.0)

    # with a huge shrink constant every series stays near the global scale
    assert loose["per_series"]["tiny"] == pytest.approx(
        loose["global_scale"], abs=0.05
    )
    assert tight["per_series"]["tiny"] == pytest.approx(1.0, abs=0.05)


def test_apply_scales_falls_back_to_global_for_unknown_series():
    frame = _frame()
    calibration = {"global_scale": 0.5, "per_series": {"A": 0.4}}

    out = apply_scales(frame, calibration)
    a = out[out["id"] == "A"]["predicted_units"].iloc[0]
    b = out[out["id"] == "B"]["predicted_units"].iloc[0]

    assert a == pytest.approx(8.0 * 0.4)
    assert b == pytest.approx(1.0 * 0.5)  # unknown id -> global scale
    # raw predictions are preserved for auditability
    assert out["predicted_units_raw"].iloc[0] == frame["predicted_units"].iloc[0]


def test_calibration_round_trips_through_json(tmp_path):
    calibration = fit_per_series_scales(_frame(), shrink=1.0)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration), encoding="utf-8")

    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["per_series"] == calibration["per_series"]
    assert evaluate(apply_scales(_frame(), reloaded))["wape_percent"] == pytest.approx(
        evaluate(apply_scales(_frame(), calibration))["wape_percent"]
    )


def test_evaluate_matches_project_wape_definition():
    frame = _frame()
    metrics = evaluate(frame)
    assert metrics["wape_percent"] == pytest.approx(
        wape(frame["actual"], frame["predicted_units"]), 4
    )
    assert metrics["rows"] == len(frame)
