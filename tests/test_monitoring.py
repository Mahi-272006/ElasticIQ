"""Monitoring tests: PSI drift, error scoring, staleness, evaluation replay."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import (
    classify_psi,
    demand_window_drift,
    drift_report,
    population_stability_index,
    price_window_drift,
)
from src.monitoring.tracking import (
    evaluation_report,
    score_predictions,
    staleness_report,
)


def test_psi_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    reference = rng.normal(10, 2, 5000)
    assert population_stability_index(reference, reference.copy()) == pytest.approx(
        0.0, abs=1e-9
    )


def test_psi_positive_for_shifted_distribution() -> None:
    rng = np.random.default_rng(0)
    reference = rng.normal(10, 2, 5000)
    shifted = rng.normal(14, 2, 5000)
    psi = population_stability_index(reference, shifted)
    assert psi > 0.1
    assert classify_psi(psi) in {"moderate", "significant"}


def test_psi_rejects_too_few_samples() -> None:
    with pytest.raises(ValueError, match="reference samples"):
        population_stability_index(np.arange(5.0), np.arange(50.0), bins=10)


def _tail_frame(n_series: int, days: int, shift_last: int = 0) -> pd.DataFrame:
    """Wide history-tail frame with deterministic per-series demand."""

    frame = pd.DataFrame({"id": [f"S{i:04d}" for i in range(n_series)]})
    for day in range(1, days + 1):
        values = np.array([(i + day) % 7 for i in range(n_series)], dtype=float)
        if day > days - shift_last:
            values += 5.0  # inject a recent demand lift
        frame[f"d_{day}"] = values
    return frame


def test_demand_window_drift_detects_injected_shift() -> None:
    tail = _tail_frame(n_series=300, days=56, shift_last=20)
    report = demand_window_drift(tail, window_days=28)
    assert report["metric"] == "per_series_mean_daily_units"
    assert report["series_compared"] == 300
    assert report["classification"] == "significant"
    assert report["psi"] > 0.25


def test_demand_window_drift_stable_without_shift() -> None:
    tail = _tail_frame(n_series=300, days=56)
    report = demand_window_drift(tail, window_days=28)
    # Deterministic repeating pattern: recent vs prior windows match exactly.
    assert report["psi"] == pytest.approx(0.0, abs=1e-6)
    assert report["classification"] == "stable"


def test_demand_window_drift_requires_two_windows() -> None:
    tail = _tail_frame(n_series=50, days=40)
    with pytest.raises(ValueError, match="need at least"):
        demand_window_drift(tail, window_days=28)


def test_price_window_drift() -> None:
    stats = pd.DataFrame(
        {
            "first_price": np.linspace(1.0, 5.0, 200),
            "last_price": np.linspace(0.8, 4.0, 200),
            "price_changes": [1] * 200,
        }
    )
    report = price_window_drift(stats)
    assert report["series_compared"] == 200
    assert report["share_of_series_with_any_price_change"] == 1.0
    assert report["classification"] == "significant"


def test_drift_report_combines_both(tmp_path) -> None:
    tail = _tail_frame(n_series=150, days=56)
    stats = pd.DataFrame(
        {
            "first_price": np.linspace(1.0, 3.0, 150),
            "last_price": np.linspace(1.0, 3.0, 150),
            "price_changes": [0] * 150,
        }
    )
    report = drift_report(tail, stats, window_days=28)
    assert set(report) == {"demand", "price", "psi_thresholds"}
    assert report["price"]["psi"] == pytest.approx(0.0, abs=1e-6)


def test_score_predictions_per_horizon() -> None:
    frame = pd.DataFrame(
        {
            "actual_units": [5, 3, 10],
            "predicted_units": [6, 2, 10],
            "horizon": [1, 2, 1],
        }
    )
    report = score_predictions(frame)
    assert report["rows"] == 3
    # |5-6| + |3-2| + |10-10| = 2 over 18 -> WAPE 11.1111%
    assert report["overall"]["wape_percent"] == pytest.approx(11.1111, abs=1e-3)
    assert report["by_horizon"]["1"]["mae_units"] == 0.5
    assert report["by_horizon"]["2"]["mae_units"] == 1.0
    assert report["overall"]["mean_bias_units"] == 0.0


def test_score_predictions_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        score_predictions(pd.DataFrame(columns=["actual_units", "predicted_units"]))


def _fake_model_dir(base) -> Path:
    """Create a model version dir with a bundle file under ``base``."""

    version = Path(base) / "v1"
    version.mkdir(parents=True, exist_ok=True)
    (version / "bundle.joblib").write_bytes(b"stub")
    return Path(base)


def test_staleness_report_with_historical_dataset(tmp_path) -> None:
    models_dir = _fake_model_dir(tmp_path)
    report = staleness_report(
        models_dir,
        last_data_date="2016-04-24",
        model_version="v1",
    )
    assert report["dataset_age_days_vs_wall_clock"] > 3000
    assert "historical dataset" in report["dataset_staleness_note"]
    assert report["artifact"]["available"] is True
    assert report["artifact_age_days"] == 0


def test_staleness_report_without_artifact(tmp_path) -> None:
    report = staleness_report(
        tmp_path, last_data_date="2016-04-24", now=datetime(2026, 10, 20, tzinfo=UTC)
    )
    assert report["artifact"]["available"] is False
    assert report["artifact_age_days"] is None


def test_evaluation_report_reads_json(tmp_path) -> None:
    version_dir = tmp_path / "v1"
    version_dir.mkdir()
    payload = {"model": {"overall": {"wape_percent": 50.0}}, "baselines": {}}
    (version_dir / "evaluation.json").write_text(json.dumps(payload), encoding="utf-8")

    report = evaluation_report(tmp_path, "v1")
    assert report["model"]["overall"]["wape_percent"] == 50.0
    assert "replay_note" in report


def test_evaluation_report_missing_version(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        evaluation_report(tmp_path, "ghost")
