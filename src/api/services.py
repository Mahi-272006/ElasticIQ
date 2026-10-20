"""Application state: loads processed datasets and the model bundle once.

The FastAPI app holds a single :class:`AppState` created at startup. Frames
and the trained model bundle are loaded lazily on first access and then
cached for the process lifetime — requests never re-read Parquet files or
re-load joblib bundles.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_settings
from src.data.pipeline import load_processed
from src.forecasting.inference import forecast_from_history, latest_version, load_bundle

logger = logging.getLogger(__name__)

HISTORY_COLUMN = "series_history_tail"


def iso_date(value) -> str:
    """Render a timestamp-like value as a plain ISO date string."""

    return str(value)[:10]


class AppState:
    """Cached access to processed datasets and the trained model."""

    def __init__(
        self,
        processed_dir: Path | None = None,
        models_dir: Path | None = None,
    ) -> None:
        settings = get_settings()
        self.processed_dir = (
            Path(processed_dir) if processed_dir else settings.processed_data_dir
        )
        self.models_dir = Path(models_dir) if models_dir else settings.models_dir
        self._frames: dict[str, pd.DataFrame] = {}
        self._bundle: dict | None = None

    # ------------------------------------------------------------------ #
    # Cached frame access
    # ------------------------------------------------------------------ #

    def frame(self, name: str) -> pd.DataFrame:
        """Load and cache a processed frame by name."""

        if name not in self._frames:
            self._frames[name] = load_processed(name, self.processed_dir)
        return self._frames[name]

    @property
    def calendar(self) -> pd.DataFrame:
        return self.frame("calendar")

    @property
    def daily_total(self) -> pd.DataFrame:
        return self.frame("aggregates/daily_total_units")

    @property
    def daily_category(self) -> pd.DataFrame:
        return self.frame("aggregates/daily_category_units")

    @property
    def daily_store(self) -> pd.DataFrame:
        return self.frame("aggregates/daily_store_units")

    @property
    def products(self) -> pd.DataFrame:
        return self.frame("products")

    @property
    def price_stats(self) -> pd.DataFrame:
        return self.frame("price_stats")

    @property
    def history_tail(self) -> pd.DataFrame:
        return self.frame(HISTORY_COLUMN)

    @property
    def bundle(self) -> dict:
        """The trained model bundle (loaded once per process)."""

        if self._bundle is None:
            self._bundle = load_bundle(models_dir=self.models_dir)
        return self._bundle

    # ------------------------------------------------------------------ #
    # Domain helpers
    # ------------------------------------------------------------------ #

    def product_row(self, series_id: str) -> pd.Series | None:
        """Return the catalog row for one series, or None if unknown."""

        match = self.products[self.products["id"] == series_id]
        if match.empty:
            return None
        return match.iloc[0]

    def product_history(self, series_id: str) -> pd.DataFrame:
        """Observed daily units for one series (date, d, units).

        Sourced from the stored history tail (last observed days only).
        """

        tail = self.history_tail
        row = tail[tail["id"] == series_id]
        if row.empty:
            raise KeyError(f"Unknown series_id: {series_id}")

        day_cols = sorted(
            (col for col in tail.columns if col.startswith("d_")),
            key=lambda col: int(col[2:]),
        )
        date_map = self.calendar.set_index("d")["date"]

        history = pd.DataFrame(
            {
                "d": day_cols,
                "units": row[day_cols].iloc[0].to_numpy(dtype=float),
            }
        )
        history["date"] = history["d"].map(date_map).map(iso_date)
        return history[["date", "d", "units"]]

    def forecast_series(self, series_id: str, horizon: int) -> pd.DataFrame:
        """Run the trained model for one series (columns id..predicted_units)."""

        predictions = forecast_from_history(
            self.bundle,
            series_id,
            self.calendar,
            horizon,
            processed_dir=self.processed_dir,
        )
        return predictions

    def demand_stats(self, series_id: str, horizon: int) -> dict:
        """Forecast-driven demand statistics for inventory planning."""

        predictions = self.forecast_series(series_id, horizon)
        units = predictions["predicted_units"].to_numpy(dtype=float)
        return {
            "series_id": series_id,
            "horizon_days": horizon,
            "forecast_total_units": float(units.sum()),
            "daily_mean": float(units.mean()),
            "daily_std": float(units.std(ddof=1)) if len(units) > 1 else 0.0,
        }

    def last_data_date(self) -> str:
        """Last date present in the aggregates (dataset recency anchor)."""

        return str(self.daily_total["date"].max())

    def latest_model_version(self) -> str:
        return latest_version(self.models_dir)

    def list_model_versions(self) -> list[dict]:
        """Scan the models directory for versions with their metadata."""

        versions: list[dict] = []
        if not self.models_dir.is_dir():
            return versions

        for path in sorted(self.models_dir.iterdir()):
            if not path.is_dir():
                continue
            entry: dict = {"version": path.name}
            metadata_path = path / "metadata.json"
            if metadata_path.is_file():
                import json

                with metadata_path.open(encoding="utf-8") as handle:
                    entry["metadata"] = json.load(handle)
            entry["artifacts"] = sorted(
                file.name for file in path.iterdir() if file.is_file()
            )
            versions.append(entry)
        return versions

    # ------------------------------------------------------------------ #
    # Readiness (never raises; used by /health)
    # ------------------------------------------------------------------ #

    def readiness(self) -> dict:
        """Report which resources are available without raising."""

        report: dict = {"processed_dir": str(self.processed_dir)}
        required = {
            "calendar": "calendar",
            "daily_total": "aggregates/daily_total_units",
            "daily_category": "aggregates/daily_category_units",
            "daily_store": "aggregates/daily_store_units",
            "products": "products",
            "price_stats": "price_stats",
            "history_tail": HISTORY_COLUMN,
        }
        missing = [
            key for key, name in required.items() if not self._frame_exists(name)
        ]
        report["missing_datasets"] = missing
        report["data_ready"] = not missing

        try:
            report["model_version"] = self.latest_model_version()
            report["model_ready"] = True
        except FileNotFoundError as exc:
            report["model_ready"] = False
            report["model_error"] = str(exc)

        return report

    def _frame_exists(self, name: str) -> bool:
        from src.data.storage import processed_path

        return processed_path(self.processed_dir, name) is not None

    def zero_demand_share(self, series_id: str) -> float | None:
        """Share of zero-sales days in the stored history (intermittency)."""

        tail = self.history_tail
        row = tail[tail["id"] == series_id]
        if row.empty:
            return None
        day_cols = [col for col in tail.columns if col.startswith("d_")]
        values = row[day_cols].iloc[0].to_numpy(dtype=float)
        return float(np.mean(values == 0))
