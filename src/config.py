"""Central configuration for ElasticIQ.

All filesystem locations and tunable defaults live here so that no module
needs hard-coded paths or magic constants. Values can be overridden with
environment variables (see .env.example).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# ElasticIQ/ — two levels up from src/config.py.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable with a safe default."""

    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_tuple(name: str, default: str) -> tuple[str, ...]:
    """Read a comma-separated environment variable as a tuple."""

    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return tuple(item.strip() for item in default.split(",") if item.strip())
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    """Application settings resolved from environment variables."""

    project_root: Path = PROJECT_ROOT
    raw_data_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("ELASTICIQ_RAW_DATA_DIR") or (PROJECT_ROOT / "data" / "raw")
        )
    )
    processed_data_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "processed"
    )
    artifacts_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts")
    models_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "artifacts" / "models"
    )
    reports_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "artifacts" / "reports"
    )
    charts_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "artifacts" / "reports" / "charts"
    )

    random_seed: int = field(
        default_factory=lambda: _env_int("ELASTICIQ_RANDOM_SEED", 42)
    )
    sample_series: int = field(
        default_factory=lambda: _env_int("ELASTICIQ_SAMPLE_SERIES", 2000)
    )
    forecast_horizons: tuple[int, ...] = (7, 14, 28)

    database_url: str = field(
        default_factory=lambda: os.getenv(
            "ELASTICIQ_DATABASE_URL",
            f"sqlite:///{(PROJECT_ROOT / 'data' / 'elasticiq.db').as_posix()}",
        )
    )
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: _env_tuple(
            "ELASTICIQ_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        )
    )

    def ensure_directories(self) -> None:
        """Create derived directories that the application writes to."""

        for path in (
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_dir,
            self.reports_dir,
            self.charts_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""

    return Settings()
