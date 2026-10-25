"""Build the Arm B processed panel using the production pipeline functions.

The only difference from the production data pipeline is *which* series are
kept: instead of ``sales.head(2000)`` we keep the stratified sample.  The
melt, calendar join, price join, and both files written here go through the
unchanged functions in ``src/data/pipeline.py`` (``melt_sales_long``,
``attach_calendar_fields``, ``attach_prices``, ``clean_calendar``), so the
resulting panel is built by exactly the production code path.

Memory discipline: the 1920-column wide CSV is streamed in chunks and rows
outside the cohort are dropped before the melt, so only ~2000 wide rows are
ever materialised (the raw file is never loaded whole).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import day_columns, get_csv_path, load_csv
from src.data.pipeline import (
    attach_calendar_fields,
    attach_prices,
    clean_calendar,
    melt_sales_long,
)
from src.data.storage import write_frame

DEFAULT_CHUNK_ROWS = 2048


def wide_to_panel(
    sales: pd.DataFrame,
    sales_day_columns: list[str],
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Production melt + calendar + price join for the given wide rows."""

    panel = melt_sales_long(sales, sales_day_columns)
    panel = attach_calendar_fields(panel, calendar)
    return attach_prices(panel, prices)


def load_selected_sales(
    selected_ids: Sequence[str],
    raw_dir: Path | None = None,
    chunksize: int = DEFAULT_CHUNK_ROWS,
) -> tuple[pd.DataFrame, list[str]]:
    """Stream the raw wide sales CSV, keeping only ``selected_ids`` rows.

    Returns ``(wide_frame, day_columns)``.  Raises if any requested series is
    absent from the file - silently training on fewer series would invalidate
    the controlled comparison.
    """

    selected = set(selected_ids)
    if not selected:
        raise ValueError("No series ids supplied")

    path = get_csv_path("sales_train_validation.csv", data_dir=raw_dir)
    header = pd.read_csv(path, nrows=0)
    sales_day_columns = day_columns(header.columns)
    if not sales_day_columns:
        raise ValueError("No d_N day columns found in sales file")

    day_dtypes = {column: np.int16 for column in sales_day_columns}
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, dtype=day_dtypes, chunksize=chunksize):
        hit = chunk[chunk["id"].isin(selected)]
        if len(hit):
            pieces.append(hit)

    if not pieces:
        raise ValueError("None of the requested series were found in the sales file")

    sales = pd.concat(pieces, ignore_index=True)
    missing = selected - set(sales["id"].unique())
    if missing:
        preview = sorted(missing)[:10]
        raise ValueError(
            f"{len(missing)} requested series are absent from the sales file "
            f"(e.g. {preview})"
        )
    return sales, sales_day_columns


def build_arm_panel(
    selected_ids: Sequence[str],
    out_dir: Path,
    raw_dir: Path | None = None,
    chunksize: int = DEFAULT_CHUNK_ROWS,
) -> dict:
    """Write ``sample_panel`` + ``calendar`` for Arm B into ``out_dir``.

    Nothing outside ``out_dir`` is written; the raw CSVs are read-only inputs.
    """

    started = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    calendar = clean_calendar(load_csv("calendar.csv", data_dir=raw_dir))
    sales, sales_day_columns = load_selected_sales(
        selected_ids, raw_dir=raw_dir, chunksize=chunksize
    )
    wide_rows, wide_series = len(sales), int(sales["id"].nunique())

    prices = load_csv(
        "sell_prices.csv",
        data_dir=raw_dir,
        dtype={"store_id": "category", "item_id": "category"},
    )
    panel = wide_to_panel(sales, sales_day_columns, calendar, prices)
    del sales, prices

    panel_path = write_frame(panel, out_dir / "sample_panel")
    calendar_path = write_frame(calendar, out_dir / "calendar")

    report = {
        "series": wide_series,
        "wide_rows": wide_rows,
        "day_columns": len(sales_day_columns),
        "panel_rows": int(len(panel)),
        "panel_day_min": int(panel["day_idx"].min()),
        "panel_day_max": int(panel["day_idx"].max()),
        "panel_columns": list(panel.columns),
        "expected_panel_rows": wide_series * len(sales_day_columns),
        "panel_is_dense": bool(
            len(panel) == wide_series * len(sales_day_columns)
        ),
        "sample_panel_path": str(panel_path),
        "calendar_path": str(calendar_path),
        "built_with": (
            "production melt_sales_long + attach_calendar_fields + attach_prices "
            "(src/data/pipeline.py), unmodified"
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    del panel
    return report


def compare_overlapping_series(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    overlap_ids: Sequence[str],
) -> dict:
    """Assert two panels hold identical rows for the shared series.

    Used to prove the Arm B panel differs from production *only* in which
    series it contains: for series present in both, every value (units,
    prices, calendar fields) must match exactly.
    """

    overlap = sorted(set(overlap_ids))
    if not overlap:
        return {"overlap_series": 0, "checked": False}

    columns = list(reference.columns)
    left = candidate[candidate["id"].isin(overlap)][columns].sort_values(
        ["id", "day_idx"], ignore_index=True
    )
    right = reference[reference["id"].isin(overlap)][columns].sort_values(
        ["id", "day_idx"], ignore_index=True
    )

    result = {
        "overlap_series": len(overlap),
        "checked": True,
        "candidate_rows": int(len(left)),
        "reference_rows": int(len(right)),
        "identical": False,
    }
    if len(left) != len(right):
        result["reason"] = "row counts differ"
        return result

    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False)
    except AssertionError as exc:  # pragma: no cover - failure path
        result["reason"] = str(exc).splitlines()[0]
        return result

    result["identical"] = True
    return result
