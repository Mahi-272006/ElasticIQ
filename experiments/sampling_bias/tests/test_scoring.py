"""Focused tests: scoring uses the project's WAPE definition and never
silently drops series or rows.

Run explicitly:

    .venv\\Scripts\\python.exe -m pytest experiments/sampling_bias/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import robustness
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.sampling_bias import scoring  # noqa: E402
from src.evaluation.metrics import mae, wape  # noqa: E402

DAYS = list(range(1886, 1914))  # 28 evaluation days, same window as production


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Two series x 28 days of hand-computable values."""
    records, truth = [], []
    values = {"s1": 2.0, "s2": 5.0}
    for series_id, base in values.items():
        for position, day in enumerate(DAYS):
            actual = base + position % 3
            predicted = actual + (1.0 if series_id == "s1" else -0.5)
            records.append(
                {"id": series_id, "day_idx": day, "predicted_units": predicted}
            )
            truth.append({"id": series_id, "day_idx": day, "actual": actual})
    predictions = pd.DataFrame(records)
    actuals = pd.DataFrame(truth)
    metadata = pd.DataFrame(
        {
            "id": ["s1", "s2"],
            "cat_id": ["FOODS", "HOBBIES"],
            "dept_id": ["FOODS_1", "HOBBIES_1"],
            "store_id": ["CA_1", "CA_2"],
            "state_id": ["CA", "CA"],
        }
    )
    return predictions, actuals, metadata, ["s1", "s2"]


def test_wape_uses_project_definition() -> None:
    predictions, actuals, metadata, cohort = _fixture()
    cell = scoring.score_cell(predictions, actuals, metadata, cohort)

    expected = round(
        wape(actuals["actual"], predictions["predicted_units"]), 4
    )
    assert cell["wape_percent"] == expected
    # and the project's own report function agrees to the digit
    assert cell["official"]["overall"]["wape_percent"] == expected
    assert cell["mae_units"] == round(
        mae(actuals["actual"], predictions["predicted_units"]), 4
    )
    # definition string points at the project implementation
    assert "src/evaluation/metrics.py" in scoring.WAPE_DEFINITION


def test_score_cell_integrity_is_complete_when_everything_is_predicted() -> None:
    predictions, actuals, metadata, cohort = _fixture()
    cell = scoring.score_cell(predictions, actuals, metadata, cohort)
    integrity = cell["integrity"]
    assert integrity["complete"] is True
    assert integrity["cohort_series"] == 2
    assert integrity["series_without_predictions"] == 0
    assert integrity["actual_rows_without_prediction"] == 0
    assert integrity["matched_rows"] == len(cohort) * len(DAYS)
    assert cell["rows"] == 2 * len(DAYS)
    assert cell["series"] == 2
    assert cell["total_actual_units"] == pytest.approx(actuals["actual"].sum())


def test_missing_series_are_reported_not_silently_dropped() -> None:
    predictions, actuals, metadata, cohort = _fixture()
    partial = predictions[predictions["id"] != "s2"]
    cell = scoring.score_cell(partial, actuals, metadata, cohort)
    integrity = cell["integrity"]
    assert integrity["complete"] is False
    assert integrity["series_without_predictions"] == 1
    assert integrity["series_without_predictions_preview"] == ["s2"]
    assert integrity["cohort_series_with_predictions"] == 1
    assert integrity["actual_rows_without_prediction"] == len(DAYS)
    # the rows actually present are still scored, and reported as such
    assert cell["rows"] == len(DAYS)
    assert integrity["expected_rows"] == 2 * len(DAYS)


def test_predictions_outside_cohort_are_reported() -> None:
    predictions, actuals, metadata, _ = _fixture()
    cell = scoring.score_cell(predictions, actuals, metadata, ["s1"])
    integrity = cell["integrity"]
    # the excluded rows are counted, not silently dropped
    assert integrity["predictions_for_series_outside_cohort"] == 1
    assert integrity["predictions_for_series_outside_cohort_preview"] == ["s2"]
    # and flagging them keeps the cell from being treated as clean
    assert integrity["complete"] is False
    assert cell["series"] == 1
    assert cell["rows"] == len(DAYS)


def test_nonfinite_predictions_fail_integrity() -> None:
    predictions, actuals, metadata, cohort = _fixture()
    broken = predictions.copy()
    broken.loc[0, "predicted_units"] = np.inf
    cell = scoring.score_cell(broken, actuals, metadata, cohort)
    assert cell["integrity"]["complete"] is False
    assert cell["integrity"]["nonfinite_predictions"] == 1


def test_split_predictions_requires_actual_column() -> None:
    predictions, _, _, _ = _fixture()
    with pytest.raises(ValueError, match="actual"):
        scoring.split_predictions_with_actuals(predictions)
    merged = predictions.merge(
        _fixture()[1], on=["id", "day_idx"], validate="one_to_one"
    )
    left, right = scoring.split_predictions_with_actuals(merged)
    assert "actual" not in left.columns
    assert list(right.columns) == ["id", "day_idx", "actual"]


def test_per_breakdown_wape_matches_project_wape() -> None:
    predictions, actuals, metadata, cohort = _fixture()
    cell = scoring.score_cell(predictions, actuals, metadata, cohort)

    for section, column in (("by_store", "store_id"), ("by_category", "cat_id")):
        breakdown = cell["breakdowns"][section]
        expected_keys = (
            {"CA_1", "CA_2"} if column == "store_id" else {"FOODS", "HOBBIES"}
        )
        assert set(breakdown) == expected_keys
        for key, entry in breakdown.items():
            indexed = metadata.set_index("id")
            series_ids = set(indexed[indexed[column] == key].index)
            sub_pred = predictions[predictions["id"].isin(series_ids)]
            sub_true = actuals[actuals["id"].isin(series_ids)]
            assert entry["wape_percent"] == round(
                wape(sub_true["actual"], sub_pred["predicted_units"]), 4
            ), f"{section}/{key} disagrees with project wape()"


def test_compare_cells_delta_winner_and_relative_change() -> None:
    predictions, actuals, metadata, cohort = _fixture()

    # arm B predicts the actuals exactly -> WAPE 0
    perfect = actuals.rename(columns={"actual": "predicted_units"})
    cell_a = scoring.score_cell(predictions, actuals, metadata, cohort)
    cell_b = scoring.score_cell(perfect, actuals, metadata, cohort)
    assert cell_b["wape_percent"] == 0.0
    assert cell_a["wape_percent"] > 0.0

    delta = scoring.compare_cells(cell_a, cell_b)
    assert delta["winner"] == "arm_b"
    assert delta["delta_pp"] == round(0.0 - cell_a["wape_percent"], 4)
    assert delta["relative_change_percent"] == pytest.approx(-100.0)

    # reversed: arm A has WAPE 0 -> relative change is reported as undefined
    # rather than faked with a division guard
    reverse = scoring.compare_cells(cell_b, cell_a)
    assert reverse["winner"] == "arm_a"
    assert reverse["relative_change_percent"] is None
    assert reverse["mae_delta_units"] == round(
        cell_a["mae_units"] - cell_b["mae_units"], 4
    )

    tie = scoring.compare_cells(cell_a, cell_a)
    assert tie["winner"] == "tie"
    assert tie["delta_pp"] == 0.0
    assert tie["cohorts_identical"] is True


def test_compare_cells_flags_different_cohorts() -> None:
    predictions, actuals, metadata, cohort = _fixture()
    cell = scoring.score_cell(predictions, actuals, metadata, cohort)
    smaller = scoring.score_cell(
        predictions[predictions["id"] == "s1"],
        actuals,
        metadata,
        ["s1"],
    )
    assert scoring.compare_cells(cell, smaller)["cohorts_identical"] is False


def test_zero_denominator_store_is_reported_undecided() -> None:
    predictions, actuals, metadata, cohort = _fixture()
    # give CA_2 a row with zero actuals only -> denominator zero for that store
    zero_actual = actuals.copy()
    zero_actual.loc[zero_actual["id"] == "s2", "actual"] = 0.0
    zero_pred = predictions.copy()
    zero_pred.loc[zero_pred["id"] == "s2", "predicted_units"] = 0.0
    cell_a = scoring.score_cell(predictions, zero_actual, metadata, cohort)
    cell_b = scoring.score_cell(zero_pred, zero_actual, metadata, cohort)

    delta = scoring.compare_cells(cell_a, cell_b)
    store = delta["per_store"]["CA_2"]
    assert store["denominator_nonzero"] is False
    assert store["winner"] is None
    # CA_1 is identical between the two cells: a decided tie, not a win
    assert delta["per_store"]["CA_1"]["winner"] == "tie"
    sweep = delta["per_store_sweep"]
    assert sweep["undecided_zero_denominator"] == 1
    assert sweep["arm_a_better_in"] == 0
    assert sweep["arm_b_better_in"] == 0
    assert sweep["ties"] == 1
