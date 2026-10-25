"""Cohort selection for the sampling-bias experiment (Arm A vs Arm B).

Arm A reproduces the production selection exactly: the first ``n`` series in
``sales_train_validation.csv`` file order - the same code path as
``src/data/pipeline.py`` (``sales.head(selected)``).

Arm B draws a reproducible stratified random sample of the same size:

* **strata**     = ``(store_id, dept_id)`` - 70 strata.  Department nests
  inside category, so proportional allocation across store x department also
  distributes stores and categories proportionally, while keeping the strata
  large enough that every one receives a non-zero integer allocation
  (smallest stratum = 149 series -> allocation >= 9 at n = 2000).
* **allocation** = proportional to stratum size using the largest-remainder
  rule, so the integer counts sum to exactly ``n``.  A guard additionally
  forces every non-empty stratum to receive at least one series when
  ``n >= n_strata``, taking the extra slots from the largest strata.
* **sampling**   = without replacement inside each stratum, with a single
  ``numpy.random.default_rng(seed)`` visited in sorted stratum order, so the
  result is byte-for-byte reproducible for a given seed.

Neither function mutates the metadata frame it receives, and neither touches
daily observations - selection operates purely on series identifiers, so
chronological order inside every series is untouched by construction.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.data.loader import get_csv_path

SERIES_ID = "id"
METADATA_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
STRATA_COLUMNS = ["store_id", "dept_id"]

DEFAULT_SAMPLE_SIZE = 2000
DEFAULT_SEED = 42

ARM_A = "A_head"
ARM_B = "B_stratified"

# Dimensions reported by the coverage comparison.  ``store_x_dept`` is derived
# because it is the stratum dimension itself (most granular reported view).
BASE_DIMENSIONS = {
    "state": "state_id",
    "store": "store_id",
    "category": "cat_id",
    "department": "dept_id",
}


def load_series_metadata(raw_dir=None) -> pd.DataFrame:
    """One row per product-store series with its metadata, in file order.

    Reads only the six identifier/metadata columns (never the 1,913 day
    columns), so this is cheap and allocates no daily observations.
    """

    path = get_csv_path("sales_train_validation.csv", data_dir=raw_dir)
    metadata = pd.read_csv(path, usecols=METADATA_COLUMNS)

    missing = [column for column in METADATA_COLUMNS if column not in metadata]
    if missing:
        raise ValueError(f"sales file is missing metadata columns: {missing}")
    if metadata[SERIES_ID].duplicated().any():
        raise ValueError("sales file contains duplicate series ids")
    return metadata


def head_selection(metadata: pd.DataFrame, n: int = DEFAULT_SAMPLE_SIZE) -> list[str]:
    """Arm A: the first ``n`` series in file order (production ``head(n)``)."""

    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if n > len(metadata):
        raise ValueError(f"Only {len(metadata)} eligible series, asked for {n}")
    return metadata[SERIES_ID].head(n).tolist()


def allocate_stratified(sizes: pd.Series, n: int) -> pd.Series:
    """Proportional integer allocation over ``sizes`` summing to exactly ``n``.

    ``sizes`` is a Series of non-negative stratum sizes indexed by stratum key.
    Uses the largest-remainder rule (ties keep the sorted index order), then
    guarantees >= 1 for every non-empty stratum when ``n >= n_strata``.
    """

    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if sizes.empty:
        raise ValueError("No strata supplied")
    if (sizes < 0).any():
        raise ValueError("Stratum sizes must be non-negative")

    total = int(sizes.sum())
    non_empty = sizes > 0
    if n > total:
        raise ValueError(f"Requested {n} series but only {total} are eligible")

    exact = sizes.astype(np.float64) / float(total) * n
    allocation = np.floor(exact).astype(int)

    remainder = int(n - allocation.sum())
    if remainder:
        # Stable sort on the fractional part: ties retain sorted index order.
        fractional = (exact - allocation).to_numpy()
        order = np.argsort(-fractional, kind="stable")
        for position in order[:remainder]:
            allocation.iloc[position] += 1

    # Small strata: never leave a non-empty stratum empty when we can avoid it.
    empty_mask = non_empty & (allocation == 0)
    zeros = list(np.flatnonzero(empty_mask.to_numpy()))
    if zeros and len(sizes) <= n:
        deficit = 0
        for position in zeros:
            allocation.iloc[position] = 1
            deficit += 1
        for _ in range(deficit):
            donor = int(np.argmax(allocation.to_numpy()))
            if allocation.iloc[donor] <= 1:
                raise RuntimeError(
                    "Cannot rebalance allocation without emptying a stratum"
                )
            allocation.iloc[donor] -= 1

    if int(allocation.sum()) != n:
        raise RuntimeError(
            f"Allocation sums to {int(allocation.sum())}, expected exactly {n}"
        )
    return allocation


def stratified_sample(
    metadata: pd.DataFrame,
    n: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Arm B: reproducible store x department stratified sample of size ``n``.

    Returns one row per selected series with its metadata plus the stratum it
    came from, its stratum size, and the stratum's allocation.  The frame is
    sorted by (store, department, id) so the output is deterministic.
    """

    required = set(METADATA_COLUMNS)
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata is missing columns: {sorted(missing)}")
    if metadata[SERIES_ID].duplicated().any():
        raise ValueError("metadata contains duplicate series ids")

    grouped = metadata.groupby(STRATA_COLUMNS, sort=True)[SERIES_ID]
    sizes = grouped.size()
    allocation = allocate_stratified(sizes, n)

    rng = np.random.default_rng(seed)
    chosen: list[str] = []
    for key in sizes.index:  # sorted order -> reproducible RNG consumption
        pool = np.sort(
            metadata.loc[
                (metadata["store_id"] == key[0]) & (metadata["dept_id"] == key[1]),
                SERIES_ID,
            ].to_numpy()
        )
        count = int(allocation.loc[key])
        if count:
            picked = rng.choice(pool, size=count, replace=False)
            chosen.extend(str(value) for value in picked)

    if len(chosen) != n:
        raise RuntimeError(f"Selected {len(chosen)} series, expected {n}")
    if len(set(chosen)) != n:
        raise RuntimeError("Stratified selection produced duplicate series ids")

    sample = metadata[metadata[SERIES_ID].isin(chosen)].copy()
    sample["stratum"] = (
        sample["store_id"].astype(str) + "|" + sample["dept_id"].astype(str)
    )
    sample["stratum_size"] = [
        int(sizes.loc[(store, dept)])
        for store, dept in zip(sample["store_id"], sample["dept_id"], strict=True)
    ]
    sample["stratum_allocation"] = [
        int(allocation.loc[(store, dept)])
        for store, dept in zip(sample["store_id"], sample["dept_id"], strict=True)
    ]
    sample = sample.sort_values(
        ["store_id", "dept_id", SERIES_ID], ignore_index=True
    )
    return sample


def build_manifest(
    metadata: pd.DataFrame,
    cohorts: dict[str, Sequence[str]],
) -> pd.DataFrame:
    """Long manifest with an ``arm`` column for every selected series."""

    frames = []
    for arm, ids in cohorts.items():
        ids = list(ids)
        if len(set(ids)) != len(ids):
            raise ValueError(f"Cohort {arm} contains duplicate ids")
        frame = metadata[metadata[SERIES_ID].isin(ids)].copy()
        found = len(frame)
        if found != len(ids):
            raise ValueError(
                f"Cohort {arm}: {len(ids)} ids requested but only {found} "
                "exist in the sales file"
            )
        frame["arm"] = arm
        frame["stratum"] = (
            frame["store_id"].astype(str) + "|" + frame["dept_id"].astype(str)
        )
        # How many of this arm landed in each stratum (allocation actually used).
        frame["stratum_allocation"] = frame.groupby("stratum")[SERIES_ID].transform(
            "size"
        )
        population_sizes = metadata.groupby(STRATA_COLUMNS)[SERIES_ID].size()
        frame["stratum_size"] = [
            int(population_sizes.loc[(store, dept)])
            for store, dept in zip(frame["store_id"], frame["dept_id"], strict=True)
        ]
        frames.append(frame)

    manifest = pd.concat(frames, ignore_index=True)
    columns = (
        ["arm"]
        + METADATA_COLUMNS
        + ["stratum", "stratum_size", "stratum_allocation"]
    )
    return manifest[columns].sort_values(
        ["arm", "store_id", "dept_id", SERIES_ID], ignore_index=True
    )


def _dimension_values(frame: pd.DataFrame) -> dict[str, pd.Series]:
    values = {
        name: frame[column].astype(str) for name, column in BASE_DIMENSIONS.items()
    }
    values["store_x_dept"] = (
        frame["store_id"].astype(str) + "|" + frame["dept_id"].astype(str)
    )
    return values


def coverage_counts(
    metadata: pd.DataFrame,
    cohorts: dict[str, Sequence[str]],
) -> pd.DataFrame:
    """Long-format coverage table: population vs each arm, by dimension/value."""

    population = metadata
    arms = {
        arm: metadata[metadata[SERIES_ID].isin(set(ids))]
        for arm, ids in cohorts.items()
    }

    pop_dimensions = _dimension_values(population)
    arm_dimensions = {arm: _dimension_values(frame) for arm, frame in arms.items()}

    rows: list[dict] = []
    for dimension in pop_dimensions:
        table = pop_dimensions[dimension].value_counts().rename("population_count")
        table = table.to_frame()
        for arm, dimensions in arm_dimensions.items():
            table[arm] = (
                dimensions[dimension]
                .value_counts()
                .reindex(table.index, fill_value=0)
            )

        for value, row in table.sort_index().iterrows():
            entry: dict = {"dimension": dimension, "value": value}
            entry.update({column: int(row[column]) for column in table.columns})
            rows.append(entry)

    return pd.DataFrame(rows)


def coverage_summary(
    metadata: pd.DataFrame,
    cohorts: dict[str, Sequence[str]],
) -> dict[str, dict[str, int]]:
    """Distinct level counts per dimension - the explicit coverage metric."""

    def _distinct(frame: pd.DataFrame) -> dict[str, int]:
        return {
            name: int(series.nunique())
            for name, series in _dimension_values(frame).items()
        }

    summary: dict[str, dict[str, int]] = {"population": _distinct(metadata)}
    for arm, ids in cohorts.items():
        summary[arm] = _distinct(
            metadata[metadata[SERIES_ID].isin(set(ids))]
        )
    return summary
