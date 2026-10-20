"""Storage helpers for processed datasets.

Processed frames are stored as Parquet when pyarrow is available (typed and
fast) and fall back to gzipped CSV otherwise, so the pipeline never hard-fails
on a missing optional dependency.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

try:  # pragma: no cover - availability depends on the environment
    import pyarrow  # noqa: F401

    PARQUET_AVAILABLE = True
except ImportError:  # pragma: no cover
    PARQUET_AVAILABLE = False

PARQUET_SUFFIX = ".parquet"
CSV_SUFFIX = ".csv.gz"


def write_frame(frame: pd.DataFrame, target: Path) -> Path:
    """Write a DataFrame to ``target`` (without extension) as parquet or CSV."""

    target.parent.mkdir(parents=True, exist_ok=True)

    if PARQUET_AVAILABLE:
        path = target.with_suffix(PARQUET_SUFFIX)
        frame.to_parquet(path, index=False)
    else:
        path = target.with_suffix(CSV_SUFFIX)
        frame.to_csv(path, index=False, compression="gzip")

    logger.info("Wrote %s rows to %s", f"{len(frame):,}", path)
    return path


def read_frame(path: Path) -> pd.DataFrame:
    """Read a processed frame written by :func:`write_frame`."""

    if path.suffix == PARQUET_SUFFIX:
        return pd.read_parquet(path)
    if path.name.endswith(CSV_SUFFIX):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported processed file: {path}")


def processed_path(directory: Path, name: str) -> Path | None:
    """Return the existing processed file for ``name`` in any supported format."""

    parquet = directory / f"{name}{PARQUET_SUFFIX}"
    if parquet.is_file():
        return parquet

    csv_gz = directory / f"{name}{CSV_SUFFIX}"
    if csv_gz.is_file():
        return csv_gz

    return None
