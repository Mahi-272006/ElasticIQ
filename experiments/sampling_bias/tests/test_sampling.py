"""Focused tests for the sampling-bias experiment's cohort construction.

Run explicitly (CI only collects ``tests/``):

    .venv\\Scripts\\python.exe -m pytest experiments/sampling_bias/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import robustness
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.sampling_bias import panel as panel_module  # noqa: E402
from experiments.sampling_bias import sampling  # noqa: E402

N_SERIES = 2000
SEED = 42


@pytest.fixture(scope="module")
def metadata() -> pd.DataFrame:
    frame = sampling.load_series_metadata()
    assert len(frame) == 30490, "expected the full catalog"
    return frame


# ------------------------------------------------------------ reproducibility ---


def test_seed_42_is_reproducible(metadata: pd.DataFrame) -> None:
    first = sampling.stratified_sample(metadata, N_SERIES, SEED)
    second = sampling.stratified_sample(metadata, N_SERIES, SEED)
    assert first[sampling.SERIES_ID].tolist() == second[sampling.SERIES_ID].tolist()
    # also reproducible from an independent copy of the metadata
    third = sampling.stratified_sample(metadata.copy(), N_SERIES, SEED)
    assert third[sampling.SERIES_ID].tolist() == first[sampling.SERIES_ID].tolist()


def test_different_seed_gives_different_draw(metadata: pd.DataFrame) -> None:
    base = sampling.stratified_sample(metadata, N_SERIES, SEED)
    other = sampling.stratified_sample(metadata, N_SERIES, SEED + 1)
    assert base[sampling.SERIES_ID].tolist() != other[sampling.SERIES_ID].tolist()


def test_sample_output_is_deterministically_sorted(metadata: pd.DataFrame) -> None:
    """Output order is (store, department, id) - stable across runs."""
    sample = sampling.stratified_sample(metadata, N_SERIES, SEED)
    keys = sample[["store_id", "dept_id", sampling.SERIES_ID]]
    assert keys.equals(keys.sort_values(["store_id", "dept_id", sampling.SERIES_ID],
                                        ignore_index=True))


# --------------------------------------------------------------- uniqueness -----


def test_exactly_2000_unique_series_no_duplicates(metadata: pd.DataFrame) -> None:
    sample = sampling.stratified_sample(metadata, N_SERIES, SEED)
    ids = sample[sampling.SERIES_ID]
    assert len(ids) == N_SERIES
    assert ids.nunique() == N_SERIES
    assert not ids.duplicated().any()
    assert not ids.isna().any()


def test_sample_ids_exist_in_dataset(metadata: pd.DataFrame) -> None:
    sample = sampling.stratified_sample(metadata, N_SERIES, SEED)
    unknown = set(sample[sampling.SERIES_ID]) - set(metadata[sampling.SERIES_ID])
    assert not unknown, f"sampled ids not in dataset: {sorted(unknown)[:5]}"


def test_head_selection_matches_production_first_n(metadata: pd.DataFrame) -> None:
    head = sampling.head_selection(metadata, N_SERIES)
    assert head == metadata[sampling.SERIES_ID].head(N_SERIES).tolist()
    assert len(head) == N_SERIES and len(set(head)) == N_SERIES


# --------------------------------------------------------------- metadata join ---


def test_manifest_join_matches_dataset_metadata(metadata: pd.DataFrame) -> None:
    """Every manifest row must reproduce the dataset's own metadata for its id."""
    cohort_a = sampling.head_selection(metadata, N_SERIES)
    cohort_b = sampling.stratified_sample(metadata, N_SERIES, SEED)[
        sampling.SERIES_ID
    ].tolist()
    manifest = sampling.build_manifest(
        metadata, {sampling.ARM_A: cohort_a, sampling.ARM_B: cohort_b}
    )

    assert len(manifest) == 2 * N_SERIES
    assert set(manifest["arm"]) == {sampling.ARM_A, sampling.ARM_B}
    for arm, cohort in ((sampling.ARM_A, cohort_a), (sampling.ARM_B, cohort_b)):
        rows = manifest.loc[manifest["arm"] == arm]
        # manifest rows are re-sorted deterministically; membership must match
        assert sorted(rows[sampling.SERIES_ID]) == sorted(cohort)
        assert rows[sampling.SERIES_ID].nunique() == N_SERIES

    joined = manifest.merge(
        metadata, on=sampling.SERIES_ID, suffixes=("", "_src"), validate="many_to_one"
    )
    assert len(joined) == 2 * N_SERIES, "join dropped or duplicated rows"
    for column in ("item_id", "dept_id", "cat_id", "store_id", "state_id"):
        assert (joined[column] == joined[f"{column}_src"]).all(), (
            f"{column} join mismatch"
        )
    # store <-> state consistency of the joined metadata
    assert (manifest["store_id"].str[:2] == manifest["state_id"]).all()


def test_manifest_is_deterministic(metadata: pd.DataFrame) -> None:
    cohort_a = sampling.head_selection(metadata, N_SERIES)
    cohort_b = sampling.stratified_sample(metadata, N_SERIES, SEED)[
        sampling.SERIES_ID
    ].tolist()
    cohorts = {sampling.ARM_A: cohort_a, sampling.ARM_B: cohort_b}
    first = sampling.build_manifest(metadata, cohorts)
    second = sampling.build_manifest(metadata, cohorts)
    pd.testing.assert_frame_equal(first, second)


# ------------------------------------------------------------------- coverage ---


def test_coverage_better_than_head_selection(metadata: pd.DataFrame) -> None:
    """Explicitly: the stratified arm covers strictly more of every dimension."""
    cohort_a = sampling.head_selection(metadata, N_SERIES)
    cohort_b = sampling.stratified_sample(metadata, N_SERIES, SEED)[
        sampling.SERIES_ID
    ].tolist()
    summary = sampling.coverage_summary(
        metadata, {sampling.ARM_A: cohort_a, sampling.ARM_B: cohort_b}
    )
    population, arm_a, arm_b = (
        summary["population"],
        summary[sampling.ARM_A],
        summary[sampling.ARM_B],
    )

    # head(2000) known shape: one store (CA_1), one state, six departments
    # (no FOODS_3 department), three categories.
    assert arm_a["store"] == 1 and arm_a["state"] == 1
    assert arm_a["department"] == 6
    assert arm_a["store_x_dept"] == 6

    # stratified sample covers the whole catalog on every dimension
    assert arm_b["store"] == population["store"] == 10
    assert arm_b["state"] == population["state"] == 3
    assert arm_b["department"] == population["department"] == 7
    assert arm_b["store_x_dept"] == population["store_x_dept"] == 70

    # strictly better than head selection on the dimensions head collapses
    for dimension in ("store", "state", "department", "store_x_dept"):
        assert arm_b[dimension] > arm_a[dimension], dimension


def test_coverage_counts_never_drop_rows(metadata: pd.DataFrame) -> None:
    cohort_a = sampling.head_selection(metadata, N_SERIES)
    cohort_b = sampling.stratified_sample(metadata, N_SERIES, SEED)[
        sampling.SERIES_ID
    ].tolist()
    counts = sampling.coverage_counts(
        metadata, {sampling.ARM_A: cohort_a, sampling.ARM_B: cohort_b}
    )
    for dimension in ("state", "store", "category", "department", "store_x_dept"):
        block = counts.loc[counts["dimension"] == dimension]
        assert block["population_count"].sum() == len(metadata)
        # nothing silently dropped: each arm's counts sum to exactly n
        assert block[sampling.ARM_A].sum() == N_SERIES
        assert block[sampling.ARM_B].sum() == N_SERIES


# ---------------------------------------------------------------- allocation ----


def test_allocation_is_proportional_and_sums_to_n() -> None:
    sizes = pd.Series({"a": 150, "b": 50, "c": 800})
    n = int(sizes.sum())
    allocation = sampling.allocate_stratified(sizes, n)
    assert int(allocation.sum()) == n
    assert (allocation >= 1).all()
    total = int(sizes.sum())
    exact = sizes / total * n
    # largest-remainder: each count stays within one unit of its exact share
    assert (allocation - exact).abs().lt(1.0 + 1e-9).all()


def test_allocation_covers_every_non_empty_stratum_even_when_tiny() -> None:
    sizes = pd.Series({"big": 900, **{f"s{index}": 2 for index in range(10)}})
    allocation = sampling.allocate_stratified(sizes, 50)
    assert set(allocation.index) == set(sizes.index)
    assert int(allocation.sum()) == 50
    assert (allocation >= 1).all(), "a non-empty stratum was left empty"


# ----------------------------------------------------------- chronology checks ---


def test_selection_never_carries_day_observations(metadata: pd.DataFrame) -> None:
    """Sampling operates on identifiers only: no d_N columns can be reordered."""
    sample = sampling.stratified_sample(metadata, N_SERIES, SEED)
    day_like = [
        column
        for column in sample.columns
        if column.startswith("d_") and column[2:].isdigit()
    ]
    assert day_like == [], f"sampler carried day columns: {day_like}"
    cohort_a = sampling.head_selection(metadata, N_SERIES)
    cohort_b = sample[sampling.SERIES_ID].tolist()
    manifest = sampling.build_manifest(
        metadata, {sampling.ARM_A: cohort_a, sampling.ARM_B: cohort_b}
    )
    day_like = [
        column
        for column in manifest.columns
        if column.startswith("d_") and column[2:].isdigit()
    ]
    assert day_like == [], f"manifest carried day columns: {day_like}"


def test_wide_to_panel_preserves_chronological_day_order() -> None:
    """The production melt keeps every series in ascending day_idx order,
    even when the wide day columns arrive out of order."""
    sales = pd.DataFrame(
        {
            "id": ["s2", "s1"],
            "item_id": ["i2", "i1"],
            "dept_id": ["FOODS_1", "FOODS_1"],
            "cat_id": ["FOODS", "FOODS"],
            "store_id": ["CA_1", "CA_1"],
            "state_id": ["CA", "CA"],
            # deliberately unsorted day columns: d_10 before d_2
            "d_10": [7, 8],
            "d_2": [3, 4],
            "d_1": [1, 2],
        }
    )
    calendar = pd.DataFrame(
        {
            "d": ["d_1", "d_2", "d_10"],
            "date": pd.to_datetime(["2016-01-01", "2016-01-02", "2016-01-10"]),
            "wm_yr_wk": [11101, 11101, 11102],
            "wday": [5, 6, 7],
            "month": [1, 1, 1],
            "is_event": [False, False, False],
            "snap_CA": [1, 0, 1],
            "snap_TX": [0, 0, 0],
            "snap_WI": [0, 0, 0],
            "day_idx": [1, 2, 10],
        }
    )
    prices = pd.DataFrame(
        {
            "store_id": ["CA_1", "CA_1"],
            "item_id": ["i1", "i2"],
            "wm_yr_wk": [11101, 11101],
            "sell_price": [1.5, 2.5],
        }
    )
    panel = panel_module.wide_to_panel(
        sales, ["d_10", "d_2", "d_1"], calendar, prices
    )

    for series_id, group in panel.groupby("id"):
        days = group["day_idx"].tolist()
        assert days == sorted(days), f"{series_id} not in chronological order: {days}"
        assert days == [1, 2, 10]

    # values follow their day, not their column position
    s1 = panel[panel["id"] == "s1"].set_index("day_idx")["units"]
    assert int(s1.loc[1]) == 2 and int(s1.loc[2]) == 4 and int(s1.loc[10]) == 8
    s2 = panel[panel["id"] == "s2"].set_index("day_idx")["units"]
    assert int(s2.loc[1]) == 1 and int(s2.loc[2]) == 3 and int(s2.loc[10]) == 7


# ------------------------------------------------------------ non-mutation ------


def test_sampler_does_not_modify_the_dataset(metadata: pd.DataFrame) -> None:
    before = metadata.copy(deep=True)
    digest_before = pd.util.hash_pandas_object(
        metadata[sampling.METADATA_COLUMNS].reset_index(drop=True)
    ).sum()
    sampling.stratified_sample(metadata, N_SERIES, SEED)
    sampling.head_selection(metadata, N_SERIES)
    digest_after = pd.util.hash_pandas_object(
        metadata[sampling.METADATA_COLUMNS].reset_index(drop=True)
    ).sum()
    assert digest_before == digest_after
    pd.testing.assert_frame_equal(before, metadata)


def test_strata_are_store_by_department(metadata: pd.DataFrame) -> None:
    sample = sampling.stratified_sample(metadata, N_SERIES, SEED)
    assert "stratum" in sample.columns
    assert (sample["stratum"] == sample["store_id"] + "|" + sample["dept_id"]).all()
    assert sample["stratum"].nunique() <= 70
