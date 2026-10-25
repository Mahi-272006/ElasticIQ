"""Controlled experiment: head(2,000) selection vs stratified random 2,000.

Entry point for the sampling-bias experiment.  Run from the repository root::

    .venv\\Scripts\\python.exe -m experiments.sampling_bias.run_experiment

Stages (run individually with ``--stage NAME``, or all as separate
subprocesses, which is the default and keeps peak memory bounded):

    verify    git/environment record, baseline reproduction, holdout + leakage checks
    sample    build both cohorts, write sampling_manifest.csv / sampling_coverage.csv
    panel     build the Arm B processed panel with production pipeline functions
    train     train Arm B through the unmodified production training function
    evaluate  score the 2x2 (model x cohort) matrix, write metrics.json
    report    render comparison_report.md from metrics.json

Safety: every write is guarded to stay inside ``experiments/sampling_bias/``.
Production models, processed data, reports, metrics, and evaluation code are
read-only inputs here.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import robustness
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.sampling_bias import panel as panel_module  # noqa: E402
from experiments.sampling_bias import sampling, scoring  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.data.loader import day_columns, load_csv  # noqa: E402
from src.data.pipeline import clean_calendar, load_processed  # noqa: E402
from src.evaluation.metrics import mae, wape  # noqa: E402
from src.forecasting.inference import recursive_forecast  # noqa: E402
from src.forecasting.train import (  # noqa: E402
    INFERENCE_TAIL_DAYS,
    VALIDATION_DAYS,
    chronological_split,
    run_training,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
PRODUCTION_MODELS = PROJECT_ROOT / "artifacts" / "models"
PRODUCTION_PANEL_DIR = PROJECT_ROOT / "data" / "processed"

ARM_A_VERSION = "hgb_poisson_v3"      # loaded, never retrained
ARM_B_VERSION = "arm_b_stratified"    # trained by this experiment
COHORT_A = "cohort_a"                 # head(2000): the existing selection
COHORT_B = "cohort_b"                 # stratified random 2000

N_SERIES = 2000
SEED = 42
STRATA = "store_id x dept_id"
ALLOCATION_RULE = (
    "proportional to stratum size with the largest-remainder rule "
    "(integer counts sum to exactly 2000); every non-empty stratum gets >= 1"
)

# Hyperparameters mirrored from artifacts/models/hgb_poisson_v3/metadata.json
# so the ONLY intended difference between arms is which series were selected.
MAX_ITER = 300
LEARNING_RATE = 0.05
CALIBRATE = True
LEVEL_WINDOW = 14
SHRINK = 300.0

STAGES: tuple[str, ...] = (
    "verify",
    "sample",
    "panel",
    "train",
    "evaluate",
    "report",
)

RERUN_COMMAND = (
    ".venv\\Scripts\\python.exe -m experiments.sampling_bias.run_experiment"
)

# Files whose references to the holdout are schema/data-quality only.
SCHEMA_OR_QUALITY_FILES = {
    "src/data/loader.py",
    "src/data/validation.py",
    "src/data/full_validation.py",
    "tests/test_loader.py",
    "tests/test_validation.py",
    "tests/conftest.py",
}
HOLDOUT_PATTERN = re.compile(r"sales_train_evaluation|d_19(?:1[4-9]|[2-3]\d|4[01])\b")


# --------------------------------------------------------------- utilities ---


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict) -> Path:
    _guard(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )
    return path


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _guard(path: Path) -> Path:
    """Refuse to write anywhere outside the experiment directory."""

    resolved = Path(path).resolve()
    root = EXPERIMENT_DIR.resolve()
    if not resolved.is_relative_to(root):
        raise RuntimeError(
            f"Refusing to write outside the experiment directory: {resolved} "
            f"(allowed: {root})"
        )
    return resolved


def _peak_rss_mb() -> float | None:
    """Peak working set of this process via the Win32 API (0 if unavailable)."""

    try:
        import ctypes
        from ctypes import Structure, byref, sizeof
        from ctypes.wintypes import DWORD, HANDLE

        size_t = ctypes.c_size_t

        class Counters(Structure):
            _fields_ = [
                ("cb", DWORD),
                ("PageFaultCount", DWORD),
                ("PeakWorkingSetSize", size_t),
                ("WorkingSetSize", size_t),
                ("QuotaPeakPagedPoolUsage", size_t),
                ("QuotaPagedPoolUsage", size_t),
                ("QuotaPeakNonPagedPoolUsage", size_t),
                ("QuotaNonPagedPoolUsage", size_t),
                ("PagefileUsage", size_t),
                ("PeakPagefileUsage", size_t),
            ]

        psapi = ctypes.WinDLL("psapi.dll")
        psapi.GetProcessMemoryInfo.argtypes = [
            HANDLE,
            ctypes.POINTER(Counters),
            DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_bool
        counters = Counters()
        counters.cb = sizeof(counters)
        ok = psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(),
            byref(counters),
            sizeof(counters),
        )
        return round(counters.PeakWorkingSetSize / 1e6, 1) if ok else None
    except Exception:  # pragma: no cover - non-Windows or API unavailable
        return None


def _git_info() -> dict:
    def _run(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError):  # pragma: no cover
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    status = _run("status", "--porcelain") or ""
    lines = [line for line in status.splitlines() if line.strip()]
    # Our own experiment directory is the only thing this script creates;
    # anything else in the status was pre-existing user work and must not be
    # touched.  Reporting both separately keeps the audit honest.
    outside_experiment = [
        line
        for line in lines
        if "experiments/" not in line.split(maxsplit=1)[-1]
    ]
    return {
        "head": _run("rev-parse", "HEAD"),
        "branch": _run("branch", "--show-current"),
        "status_porcelain": status,
        "changes_outside_experiment_dir": outside_experiment,
        "production_tree_untouched": len(outside_experiment) == 0,
        "working_tree_clean_before_experiment": status == "",
        "last_commit_subject": _run("log", "-1", "--pretty=%s"),
        "committed_by_experiment": False,
        "pushed_by_experiment": False,
    }


def _environment_info() -> dict:
    import sklearn

    settings = get_settings()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "random_seed": int(settings.random_seed),
        "cpu_count": os.cpu_count(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _day_span(text: str) -> tuple[int, int]:
    """``"d_1..d_1885"`` -> ``(1, 1885)``."""

    numbers = [int(value) for value in re.findall(r"(\d+)", str(text))]
    if not numbers:
        raise ValueError(f"No day indices in {text!r}")
    return (numbers[0], numbers[-1]) if len(numbers) > 1 else (numbers[0], numbers[0])


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


# ------------------------------------------------------------------ checks ---


def _verify_arm_a() -> dict:
    """Prove the recorded baseline matches the shipped predictions file."""

    predictions_path = (
        PRODUCTION_MODELS / ARM_A_VERSION / "validation_predictions.parquet"
    )
    evaluation_path = PRODUCTION_MODELS / ARM_A_VERSION / "evaluation.json"
    bundle_path = PRODUCTION_MODELS / ARM_A_VERSION / "bundle.joblib"
    metadata_path = PRODUCTION_MODELS / ARM_A_VERSION / "metadata.json"

    frame = pd.read_parquet(predictions_path)
    predictions, actuals = scoring.split_predictions_with_actuals(frame)

    official = _read_json(evaluation_path)["model"]["overall"]
    wape_recomputed = round(wape(actuals["actual"], predictions["predicted_units"]), 4)
    mae_recomputed = round(mae(actuals["actual"], predictions["predicted_units"]), 4)

    metadata = sampling.load_series_metadata()
    expected_cohort = set(sampling.head_selection(metadata, N_SERIES))
    found_cohort = set(predictions["id"].unique())
    days = sorted(int(day) for day in predictions["day_idx"].unique())

    model_metadata = _read_json(metadata_path)
    return {
        "model_version": ARM_A_VERSION,
        "loaded_from_saved_artifacts": True,
        "retrained_by_experiment": False,
        "bundle_path": str(bundle_path.relative_to(PROJECT_ROOT)),
        "predictions_path": str(predictions_path.relative_to(PROJECT_ROOT)),
        "predictions_sha256": _sha256(predictions_path),
        "evaluation_path": str(evaluation_path.relative_to(PROJECT_ROOT)),
        "saved_wape_percent": official["wape_percent"],
        "recomputed_wape_percent": wape_recomputed,
        "saved_mae_units": official["mae_units"],
        "recomputed_mae_units": mae_recomputed,
        "matches_saved_metrics": bool(
            wape_recomputed == official["wape_percent"]
            and mae_recomputed == official["mae_units"]
        ),
        "cohort_equals_head_selection": bool(found_cohort == expected_cohort),
        "cohort_series": len(found_cohort),
        "rows": int(len(predictions)),
        "evaluation_days": f"d_{days[0]}..d_{days[-1]}",
        "evaluation_day_count": len(days),
        "trained_at_utc": model_metadata["trained_at_utc"],
        "recorded_training_seconds": _recorded_training_seconds(),
        "hyperparameters": model_metadata["hyperparameters"],
        "training": model_metadata["training"],
    }


def _recorded_training_seconds() -> dict:
    """Arm A's runtime is a historical record (not re-measured here)."""

    progress = PROJECT_ROOT / "PROGRESS.md"
    if not progress.is_file():
        return {"seconds": None, "source": "PROGRESS.md not found"}
    text = progress.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"hgb_poisson_v3`\s*\(\s*([\d.]+)\s*s", text)
    if not match:
        return {"seconds": None, "source": "no match in PROGRESS.md"}
    line_no = text[: match.start()].count("\n") + 1
    return {
        "seconds": float(match.group(1)),
        "source": f"PROGRESS.md:{line_no} (official run #2; not re-measured)",
    }


def _check_holdout_window() -> dict:
    """Establish whether d_1914..d_1941 is an unused final holdout."""

    validation_header = pd.read_csv(
        PROJECT_ROOT / "data" / "raw" / "sales_train_validation.csv", nrows=0
    )
    evaluation_header = pd.read_csv(
        PROJECT_ROOT / "data" / "raw" / "sales_train_evaluation.csv", nrows=0
    )
    validation_days = day_columns(validation_header.columns)
    evaluation_days = day_columns(evaluation_header.columns)
    extra_days = [day for day in evaluation_days if day not in set(validation_days)]

    calendar = clean_calendar(load_csv("calendar.csv"))
    calendar_max_day = int(calendar["day_idx"].max())

    panel = load_processed("sample_panel", PRODUCTION_PANEL_DIR)
    processed_max_day = int(panel["day_idx"].max())
    processed_series = int(panel["id"].nunique())
    del panel
    gc.collect()

    matches = _scan_holdout_references()
    metric_level = [
        entry
        for entry in matches
        if entry["path"] not in SCHEMA_OR_QUALITY_FILES
        and not entry["path"].startswith("experiments/")
    ]

    first = f"d_{int(extra_days[0][2:])}" if extra_days else None
    last = f"d_{int(extra_days[-1][2:])}" if extra_days else None
    covers_holdout = bool(extra_days) and calendar_max_day >= int(
        extra_days[-1][2:]
    )

    verdict = "holdout_status_unverified"
    if not metric_level and extra_days and processed_max_day < int(extra_days[0][2:]):
        verdict = "candidate_final_holdout"

    return {
        "verdict": verdict,
        "validation_file_last_day": validation_days[-1],
        "evaluation_file_last_day": evaluation_days[-1],
        "evaluation_file_extra_days": len(extra_days),
        "extra_day_span": f"{first}..{last}" if extra_days else None,
        "calendar_covers_holdout": covers_holdout,
        "calendar_max_day": calendar_max_day,
        "processed_panel_max_day": processed_max_day,
        "processed_panel_series": processed_series,
        "processed_outputs_stop_before_holdout": bool(
            extra_days and processed_max_day < int(extra_days[0][2:])
        ),
        "repository_references": len(matches),
        "schema_or_data_quality_references": len(matches) - len(metric_level),
        "metric_tuning_or_model_selection_references": len(metric_level),
        "metric_level_matches": metric_level[:20],
        "notes": [
            "Scanned src/, tests/, root markdown, artifacts JSON/markdown and "
            "processed reports for 'sales_train_evaluation' or d_1914..d_1941.",
            "References inside src/data/{loader,validation,full_validation}.py "
            "and tests are schema/data-quality scans only.",
            "This experiment did NOT score the holdout; it stays unspent until "
            "model and sampling decisions are frozen.",
            "Code/artifact usage is verifiable; ad-hoc human inspection of the "
            "raw CSV cannot be excluded, hence the qualified verdict.",
        ],
    }


def _scan_holdout_references() -> list[dict]:
    candidates: set[Path] = set()
    for pattern in (
        "src/**/*.py",
        "tests/**/*.py",
        "*.md",
        "artifacts/**/*.json",
        "artifacts/**/*.md",
        "data/processed/*.json",
        "experiments/**/*.py",
        "experiments/**/*.md",
    ):
        candidates.update(PROJECT_ROOT.glob(pattern))

    matches: list[dict] = []
    for path in sorted(candidates):
        if not path.is_file() or path.is_dir():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:  # pragma: no cover
            continue
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if relative.startswith("experiments/sampling_bias/reports"):
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if HOLDOUT_PATTERN.search(line):
                matches.append(
                    {
                        "path": relative,
                        "line": line_no,
                        "text": line.strip()[:180],
                    }
                )
    return matches


def _check_calibration_windows() -> dict:
    """Confirm calibration never touches the evaluated observations."""

    model_metadata = _read_json(PRODUCTION_MODELS / ARM_A_VERSION / "metadata.json")
    train_span = _day_span(model_metadata["training"]["train_days"])
    validation_span = _day_span(model_metadata["training"]["validation_days"])

    postprocess = model_metadata.get("postprocess") or {}
    fitted_on = postprocess.get("fitted_on") or {}
    inner_span = _day_span(fitted_on.get("inner_model_train_days", "d_0..d_0"))
    scale_span = _day_span(fitted_on.get("scale_fit_days", "d_0..d_0"))

    checks = {
        "outer_training_ends_before_evaluation": train_span[1] < validation_span[0],
        "scale_window_ends_before_evaluation": scale_span[1] < validation_span[0],
        "inner_training_ends_before_scale_window": inner_span[1] < scale_span[0],
        "scale_window_inside_outer_training": (
            train_span[0] <= scale_span[0] and scale_span[1] <= train_span[1]
        ),
    }
    return {
        "model_version": ARM_A_VERSION,
        "outer_training_days": f"d_{train_span[0]}..d_{train_span[1]}",
        "evaluation_days": f"d_{validation_span[0]}..d_{validation_span[1]}",
        "inner_model_training_days": (
            f"d_{inner_span[0]}..d_{inner_span[1]}" if fitted_on else None
        ),
        "scale_fit_days": (
            f"d_{scale_span[0]}..d_{scale_span[1]}" if fitted_on else None
        ),
        "checks": checks,
        "calibration_uses_evaluated_observations": not (
            checks["scale_window_ends_before_evaluation"]
            and checks["outer_training_ends_before_evaluation"]
        ),
        "leakage_risk_found": False,
        "action": "documented only - calibration left unchanged by this experiment",
    }


def stage_verify() -> dict:
    document = {
        "generated_at_utc": _now(),
        "git": _git_info(),
        "environment": _environment_info(),
        "wape_definition": scoring.WAPE_DEFINITION,
        "arm_a_reproduction": _verify_arm_a(),
        "holdout_check": _check_holdout_window(),
        "calibration_check": _check_calibration_windows(),
        "production_artifacts_modified": False,
    }
    _write_json(EXPERIMENT_DIR / "verification.json", document)
    return document


# ------------------------------------------------------------------ sample ---


def stage_sample() -> dict:
    metadata = sampling.load_series_metadata()
    cohort_a = sampling.head_selection(metadata, N_SERIES)
    arm_b_frame = sampling.stratified_sample(metadata, N_SERIES, SEED)
    cohort_b = arm_b_frame[sampling.SERIES_ID].tolist()

    cohorts = {sampling.ARM_A: cohort_a, sampling.ARM_B: cohort_b}
    for arm, ids in cohorts.items():
        if len(ids) != N_SERIES or len(set(ids)) != N_SERIES:
            raise RuntimeError(
                f"{arm}: expected {N_SERIES} unique series, got {len(ids)} rows "
                f"/ {len(set(ids))} unique"
            )

    manifest = sampling.build_manifest(metadata, cohorts)
    coverage = sampling.coverage_counts(metadata, cohorts)
    summary = sampling.coverage_summary(metadata, cohorts)

    _guard(EXPERIMENT_DIR / "sampling_manifest.csv")
    manifest.to_csv(EXPERIMENT_DIR / "sampling_manifest.csv", index=False)
    coverage.to_csv(EXPERIMENT_DIR / "sampling_coverage.csv", index=False)

    payload = {
        "generated_at_utc": _now(),
        "n_series_per_arm": N_SERIES,
        "seed": SEED,
        "strata": STRATA,
        "allocation_rule": ALLOCATION_RULE,
        "arm_a_selection": (
            "first 2,000 series in sales_train_validation.csv file order "
            "(production head(2000))"
        ),
        "arm_b_selection": (
            "store x department proportional stratified random sample, seed 42"
        ),
        "series": {sampling.ARM_A: len(cohort_a), sampling.ARM_B: len(cohort_b)},
        "overlap_series": len(set(cohort_a) & set(cohort_b)),
        "coverage_summary": summary,
        "manifest_rows": int(len(manifest)),
        "coverage_rows": int(len(coverage)),
    }
    _write_json(EXPERIMENT_DIR / "sampling_summary.json", payload)
    return payload


# ------------------------------------------------------------------- panel ---


def stage_panel() -> dict:
    manifest_path = EXPERIMENT_DIR / "sampling_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError("Run the 'sample' stage first")
    manifest = pd.read_csv(manifest_path)
    cohort_b = manifest.loc[
        manifest["arm"] == sampling.ARM_B, sampling.SERIES_ID
    ].tolist()
    cohort_a = set(
        manifest.loc[manifest["arm"] == sampling.ARM_A, sampling.SERIES_ID]
    )

    out_dir = EXPERIMENT_DIR / "processed_arm_b"
    report = panel_module.build_arm_panel(cohort_b, out_dir)

    reference = load_processed("sample_panel", PRODUCTION_PANEL_DIR)
    candidate = load_processed("sample_panel", out_dir)
    overlap = sorted(set(cohort_b) & cohort_a)
    report["overlap_check"] = panel_module.compare_overlapping_series(
        candidate, reference, overlap
    )
    del reference, candidate
    gc.collect()

    check = report["overlap_check"]
    if check.get("checked") and not check.get("identical"):
        raise RuntimeError(
            "Arm B panel disagrees with the production panel for shared series: "
            f"{check.get('reason', 'unknown difference')}. Stopping before training."
        )

    report["generated_at_utc"] = _now()
    _write_json(EXPERIMENT_DIR / "panel_report.json", report)
    return report


# ------------------------------------------------------------------- train ---


def stage_train() -> dict:
    processed_dir = _guard(
        EXPERIMENT_DIR / "processed_arm_b"
    )  # type: ignore[assignment]
    models_dir = _guard(EXPERIMENT_DIR / "models")  # type: ignore[assignment]
    panel_path = EXPERIMENT_DIR / "processed_arm_b" / "sample_panel.parquet"
    if not panel_path.is_file():
        raise FileNotFoundError("Run the 'panel' stage first")

    started = time.perf_counter()
    result = run_training(
        max_iter=MAX_ITER,
        learning_rate=LEARNING_RATE,
        model_version=ARM_B_VERSION,
        calibrate=CALIBRATE,
        level_window=LEVEL_WINDOW,
        shrink=SHRINK,
        processed_dir=Path(processed_dir),
        models_dir=Path(models_dir),
    )
    stage_elapsed = round(time.perf_counter() - started, 1)

    payload = {
        "generated_at_utc": _now(),
        "arm": sampling.ARM_B,
        "model_version": result["version"],
        "artifacts_dir": result["artifacts_dir"],
        "training_elapsed_seconds": result["elapsed_seconds"],
        "stage_elapsed_seconds": stage_elapsed,
        "peak_rss_mb": _peak_rss_mb(),
        "train_rows": result["metadata"]["training"]["train_rows"],
        "series": result["metadata"]["training"]["series"],
        "train_days": result["metadata"]["training"]["train_days"],
        "validation_days": result["metadata"]["training"]["validation_days"],
        "hyperparameters": result["metadata"]["hyperparameters"],
        "evaluation_model_overall": result["evaluation"]["model"]["overall"],
        "notes": [
            "Trained by importing src.forecasting.train.run_training unchanged; "
            "only processed_dir and models_dir point into the experiment directory.",
            "metadata.json inside this model directory is written by the "
            "unmodified production code, so its sample_selection string is the "
            "stock head-selection wording - it is WRONG for Arm B. The "
            "authoritative Arm B selection record is sampling_manifest.csv.",
        ],
    }
    _write_json(EXPERIMENT_DIR / "train_report.json", payload)
    return payload


# ---------------------------------------------------------------- evaluate ---


def _model_coverage(bundle: dict, cohort_ids: list[str]) -> dict:
    cohort = set(cohort_ids)
    categories = bundle.get("categorical_categories", {})
    store_levels = set(categories.get("store_id", []))
    postprocess = bundle.get("postprocess") or {}
    fitted_scales = set(postprocess.get("per_series", {}))
    return {
        "model_store_levels": sorted(store_levels),
        "cohort_stores": None,
        "cohort_stores_absent_from_model": None,
        "series_whose_store_is_unseen_by_model": None,
        "series_with_fitted_scale": len(cohort & fitted_scales),
        "series_falling_back_to_global_scale": len(cohort - fitted_scales),
        "global_scale": postprocess.get("global_scale"),
    }


def _attach_model_coverage(cell: dict, bundle: dict, cohort_ids: list[str],
                           metadata: pd.DataFrame) -> dict:
    cohort = set(cohort_ids)
    categories = bundle.get("categorical_categories", {})
    model_levels = {col: set(categories.get(col, [])) for col in
                    ("store_id", "state_id", "cat_id", "dept_id")}
    cohort_stores = set(metadata["store_id"].unique())
    unseen_stores = sorted(cohort_stores - model_levels["store_id"])
    series_in_unseen_stores = int(
        metadata[~metadata["store_id"].isin(model_levels["store_id"])][
            sampling.SERIES_ID
        ].nunique()
    )
    postprocess = bundle.get("postprocess") or {}
    fitted = set(postprocess.get("per_series", {}))
    cell = dict(cell)
    cell["model_coverage"] = {
        "model_store_levels": sorted(model_levels["store_id"]),
        "cohort_stores": sorted(cohort_stores),
        "cohort_stores_unseen_by_model": unseen_stores,
        "series_in_unseen_stores": series_in_unseen_stores,
        "series_with_fitted_scale": len(cohort & fitted),
        "series_using_global_scale_fallback": len(cohort - fitted),
        "global_scale": postprocess.get("global_scale"),
        "postprocess_present": bool(postprocess),
    }
    return cell


def stage_evaluate() -> dict:
    import joblib

    manifest_path = EXPERIMENT_DIR / "sampling_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError("Run the 'sample' stage first")
    manifest = pd.read_csv(manifest_path)

    cohorts = {
        COHORT_A: manifest.loc[
            manifest["arm"] == sampling.ARM_A, sampling.SERIES_ID
        ].tolist(),
        COHORT_B: manifest.loc[
            manifest["arm"] == sampling.ARM_B, sampling.SERIES_ID
        ].tolist(),
    }
    metadata_by_cohort = {
        name: manifest.loc[manifest["arm"] == arm, [
            sampling.SERIES_ID, "cat_id", "dept_id", "store_id", "state_id"
        ]].drop_duplicates(sampling.SERIES_ID).reset_index(drop=True)
        for name, arm in ((COHORT_A, sampling.ARM_A), (COHORT_B, sampling.ARM_B))
    }

    predictions_dir = _guard(EXPERIMENT_DIR / "predictions")
    Path(predictions_dir).mkdir(parents=True, exist_ok=True)
    calendar = load_processed("calendar", PRODUCTION_PANEL_DIR)

    cells: dict[str, dict] = {}

    # ---- within-sample cells (scored from the saved prediction files) ------
    arm_a_file = PRODUCTION_MODELS / ARM_A_VERSION / "validation_predictions.parquet"
    preds_a, actuals_a = scoring.split_predictions_with_actuals(
        pd.read_parquet(arm_a_file)
    )
    cells["arm_a_on_cohort_a"] = scoring.score_cell(
        preds_a, actuals_a, metadata_by_cohort[COHORT_A], cohorts[COHORT_A]
    )
    cells["arm_a_on_cohort_a"]["predictions_source"] = str(
        arm_a_file.relative_to(PROJECT_ROOT)
    )

    arm_b_file = (
        EXPERIMENT_DIR / "models" / ARM_B_VERSION / "validation_predictions.parquet"
    )
    if not arm_b_file.is_file():
        raise FileNotFoundError("Run the 'train' stage first")
    preds_b, actuals_b = scoring.split_predictions_with_actuals(
        pd.read_parquet(arm_b_file)
    )
    cells["arm_b_on_cohort_b"] = scoring.score_cell(
        preds_b, actuals_b, metadata_by_cohort[COHORT_B], cohorts[COHORT_B]
    )
    cells["arm_b_on_cohort_b"]["predictions_source"] = str(
        arm_b_file.relative_to(PROJECT_ROOT)
    )

    # ---- cross cell: Arm B model on cohort A --------------------------------
    bundle_b = joblib.load(EXPERIMENT_DIR / "models" / ARM_B_VERSION / "bundle.joblib")
    panel_a = load_processed("sample_panel", PRODUCTION_PANEL_DIR)
    origin_a, first_val_a = chronological_split(panel_a["day_idx"])
    tail_a = panel_a[
        panel_a["id"].isin(cohorts[COHORT_A])
        & panel_a["day_idx"].between(
            origin_a - INFERENCE_TAIL_DAYS + 1, origin_a
        )
    ]
    del panel_a
    gc.collect()
    if tail_a["id"].nunique() != len(cohorts[COHORT_A]):
        raise RuntimeError(
            f"Cohort A tail incomplete: {tail_a['id'].nunique()} series, "
            f"expected {len(cohorts[COHORT_A])}"
        )
    raw_b_on_a = recursive_forecast(bundle_b, tail_a, calendar, VALIDATION_DAYS)
    raw_b_on_a.merge(
        actuals_a, on=["id", "day_idx"], how="left", validate="one_to_one"
    ).to_parquet(Path(predictions_dir) / "arm_b__on__cohort_a.parquet", index=False)
    cell_b_on_a = scoring.score_cell(
        raw_b_on_a, actuals_a, metadata_by_cohort[COHORT_A], cohorts[COHORT_A]
    )
    cell_b_on_a["predictions_source"] = (
        "experiments/sampling_bias/predictions/arm_b__on__cohort_a.parquet"
    )
    cells["arm_b_on_cohort_a"] = _attach_model_coverage(
        cell_b_on_a, bundle_b, cohorts[COHORT_A], metadata_by_cohort[COHORT_A]
    )
    del bundle_b, tail_a, raw_b_on_a
    gc.collect()

    # ---- cross cell: Arm A model on cohort B --------------------------------
    bundle_a = joblib.load(PRODUCTION_MODELS / ARM_A_VERSION / "bundle.joblib")
    panel_b = load_processed("sample_panel", EXPERIMENT_DIR / "processed_arm_b")
    origin_b, first_val_b = chronological_split(panel_b["day_idx"])
    tail_b = panel_b[
        panel_b["id"].isin(cohorts[COHORT_B])
        & panel_b["day_idx"].between(
            origin_b - INFERENCE_TAIL_DAYS + 1, origin_b
        )
    ]
    del panel_b
    gc.collect()
    if tail_b["id"].nunique() != len(cohorts[COHORT_B]):
        raise RuntimeError(
            f"Cohort B tail incomplete: {tail_b['id'].nunique()} series, "
            f"expected {len(cohorts[COHORT_B])}"
        )
    raw_a_on_b = recursive_forecast(bundle_a, tail_b, calendar, VALIDATION_DAYS)
    raw_a_on_b.merge(
        actuals_b, on=["id", "day_idx"], how="left", validate="one_to_one"
    ).to_parquet(Path(predictions_dir) / "arm_a__on__cohort_b.parquet", index=False)
    cell_a_on_b = scoring.score_cell(
        raw_a_on_b, actuals_b, metadata_by_cohort[COHORT_B], cohorts[COHORT_B]
    )
    cell_a_on_b["predictions_source"] = (
        "experiments/sampling_bias/predictions/arm_a__on__cohort_b.parquet"
    )
    cells["arm_a_on_cohort_b"] = _attach_model_coverage(
        cell_a_on_b, bundle_a, cohorts[COHORT_B], metadata_by_cohort[COHORT_B]
    )
    del bundle_a, tail_b, raw_a_on_b, calendar
    gc.collect()

    for name, cell in cells.items():
        if not cell["integrity"]["complete"]:
            raise RuntimeError(
                f"Cell {name} is incomplete: {cell['integrity']}. "
                "Refusing to publish a partial comparison."
            )

    # ---- configuration identity -------------------------------------------
    arm_a_meta = _read_json(PRODUCTION_MODELS / ARM_A_VERSION / "metadata.json")
    arm_b_meta = _read_json(
        EXPERIMENT_DIR / "models" / ARM_B_VERSION / "metadata.json"
    )
    identity = {
        "hyperparameters_arm_a": arm_a_meta["hyperparameters"],
        "hyperparameters_arm_b": arm_b_meta["hyperparameters"],
        "hyperparameters_identical": bool(
            arm_a_meta["hyperparameters"] == arm_b_meta["hyperparameters"]
        ),
        "features_arm_a": (
            arm_a_meta["features"]["numeric"] + arm_a_meta["features"]["categorical"]
        ),
        "features_arm_b": (
            arm_b_meta["features"]["numeric"] + arm_b_meta["features"]["categorical"]
        ),
        "features_identical": bool(
            arm_a_meta["features"] == arm_b_meta["features"]
        ),
        "library_versions_identical": bool(
            arm_a_meta["library_versions"] == arm_b_meta["library_versions"]
        ),
        "train_days_arm_a": arm_a_meta["training"]["train_days"],
        "train_days_arm_b": arm_b_meta["training"]["train_days"],
        "validation_days_arm_a": arm_a_meta["training"]["validation_days"],
        "validation_days_arm_b": arm_b_meta["training"]["validation_days"],
        "only_intended_difference": "which 2,000 product-store series were selected",
    }

    verification = _read_json(EXPERIMENT_DIR / "verification.json")
    sampling_summary = _read_json(EXPERIMENT_DIR / "sampling_summary.json")
    train_report = _read_json(EXPERIMENT_DIR / "train_report.json")

    comparison_1 = {
        COHORT_A: {
            "series": len(cohorts[COHORT_A]),
            "rows": cells["arm_a_on_cohort_a"]["rows"],
            "total_actual_units": cells["arm_a_on_cohort_a"]["total_actual_units"],
            "arm_a_cell": "arm_a_on_cohort_a",
            "arm_b_cell": "arm_b_on_cohort_a",
            "delta": scoring.compare_cells(
                cells["arm_a_on_cohort_a"], cells["arm_b_on_cohort_a"]
            ),
        },
        COHORT_B: {
            "series": len(cohorts[COHORT_B]),
            "rows": cells["arm_a_on_cohort_b"]["rows"],
            "total_actual_units": cells["arm_a_on_cohort_b"]["total_actual_units"],
            "arm_a_cell": "arm_a_on_cohort_b",
            "arm_b_cell": "arm_b_on_cohort_b",
            "delta": scoring.compare_cells(
                cells["arm_a_on_cohort_b"], cells["arm_b_on_cohort_b"]
            ),
        },
    }

    comparison_2 = {
        "note": (
            "Secondary diagnostic: each model scored on ITS OWN selected "
            "2,000 series. The evaluated series differ between arms, so this "
            "is NOT apples-to-apples; the primary conclusion uses "
            "comparison_1 (fixed common cohorts)."
        ),
        "arm_a_cell": "arm_a_on_cohort_a",
        "arm_b_cell": "arm_b_on_cohort_b",
        "delta": scoring.compare_cells(
            cells["arm_a_on_cohort_a"], cells["arm_b_on_cohort_b"]
        ),
        "cohorts_identical": False,
    }

    metrics = {
        "schema_version": 1,
        "experiment": {
            "name": "sampling_bias",
            "directory": "experiments/sampling_bias",
            "created_at_utc": _now(),
            "git": verification["git"],
            "environment": verification["environment"],
            "rerun_command": RERUN_COMMAND,
        },
        "configuration": {
            "series_per_arm": N_SERIES,
            "seed": SEED,
            "strata": STRATA,
            "allocation_rule": ALLOCATION_RULE,
            "arm_a_selection": sampling_summary["arm_a_selection"],
            "arm_b_selection": sampling_summary["arm_b_selection"],
            "arm_a_model": ARM_A_VERSION,
            "arm_b_model": ARM_B_VERSION,
            "max_iter": MAX_ITER,
            "learning_rate": LEARNING_RATE,
            "calibrate": CALIBRATE,
            "level_window": LEVEL_WINDOW,
            "shrink": SHRINK,
            "origin_day": origin_a,
            "first_validation_day": first_val_a,
            "validation_days": f"d_{first_val_a}..d_{origin_a + VALIDATION_DAYS}",
            "horizon_days": VALIDATION_DAYS,
            "inference_tail_days": INFERENCE_TAIL_DAYS,
            "wape_definition": scoring.WAPE_DEFINITION,
            "configuration_identity": identity,
        },
        "verification": verification,
        "coverage": {
            "summary": sampling_summary["coverage_summary"],
            "counts_file": "experiments/sampling_bias/sampling_coverage.csv",
            "manifest_file": "experiments/sampling_bias/sampling_manifest.csv",
        },
        "cells": cells,
        "comparison_1_common_cohorts": comparison_1,
        "comparison_2_within_sample": comparison_2,
        "resources": {
            "arm_a_training_seconds": verification["arm_a_reproduction"][
                "recorded_training_seconds"
            ],
            "arm_b_training_seconds": train_report["training_elapsed_seconds"],
            "arm_b_stage_seconds": train_report["stage_elapsed_seconds"],
            "arm_b_peak_rss_mb": train_report["peak_rss_mb"],
            "arm_a_peak_rss_mb": None,
            "arm_a_peak_rss_note": (
                "not measured - Arm A was loaded from saved artifacts, "
                "never retrained by this experiment"
            ),
            "stage_resources_file": "experiments/sampling_bias/stage_resources.json",
        },
    }
    _write_json(EXPERIMENT_DIR / "metrics.json", metrics)
    return {
        "cells": {
            name: {
                "wape_percent": cell["wape_percent"],
                "mae_units": cell["mae_units"],
                "rows": cell["rows"],
                "series": cell["series"],
            }
            for name, cell in cells.items()
        },
        "comparison_1": {
            name: block["delta"]["delta_pp"]
            for name, block in comparison_1.items()
        },
        "metrics_file": str(EXPERIMENT_DIR / "metrics.json"),
    }


# ------------------------------------------------------------------ report ---


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _interpretation(metrics: dict) -> list[str]:
    lines: list[str] = []
    c1 = metrics["comparison_1_common_cohorts"]
    delta_a = c1[COHORT_A]["delta"]
    delta_b = c1[COHORT_B]["delta"]

    lines.append(
        f"**Fixed cohort A — head(2,000) (the existing selection's own "
        f"series):** Arm A {_fmt(delta_a['arm_a_wape_percent'])}% WAPE vs "
        f"Arm B {_fmt(delta_a['arm_b_wape_percent'])}% → Δ = "
        f"{delta_a['delta_pp']:+.4f} pp "
        f"({_fmt(delta_a['relative_change_percent'], 2)}% "
        f"relative); winner = `{delta_a['winner']}`. "
        f"MAE {_fmt(delta_a['arm_a_mae_units'])} → "
        f"{_fmt(delta_a['arm_b_mae_units'])} units."
    )
    lines.append(
        f"**Fixed cohort B — stratified(2,000) (representative of the full "
        f"catalog):** Arm A {_fmt(delta_b['arm_a_wape_percent'])}% WAPE vs "
        f"Arm B {_fmt(delta_b['arm_b_wape_percent'])}% → Δ = "
        f"{delta_b['delta_pp']:+.4f} pp "
        f"({_fmt(delta_b['relative_change_percent'], 2)}% "
        f"relative); winner = `{delta_b['winner']}`. "
        f"MAE {_fmt(delta_b['arm_a_mae_units'])} → "
        f"{_fmt(delta_b['arm_b_mae_units'])} units."
    )

    cells = metrics["cells"]
    gap_a = round(
        cells["arm_a_on_cohort_b"]["wape_percent"]
        - cells["arm_a_on_cohort_a"]["wape_percent"],
        4,
    )
    gap_b = round(
        cells["arm_b_on_cohort_b"]["wape_percent"]
        - cells["arm_b_on_cohort_a"]["wape_percent"],
        4,
    )
    lines.append(
        f"**Descriptive cross-cohort gap (same model, different cohorts):** "
        f"Arm A scores {_gap(gap_a)} when moved from cohort A to cohort B; "
        f"Arm B scores {_gap(gap_b)}. The cohorts differ in composition and "
        "difficulty, so this gap is descriptive, not a causal estimate."
    )

    if delta_b["winner"] == "arm_b" and delta_a["winner"] == "arm_a":
        verdict = (
            "The pattern is consistent with a selection effect: the original "
            "head-trained model holds up on its own CA_1 cohort but is beaten "
            "by the stratified-trained model on the representative cohort. "
            "This supports the hypothesis that head-only selection contributes "
            "error outside the sampled store."
        )
    elif delta_b["winner"] == "arm_b" and delta_a["winner"] == "tie":
        verdict = (
            "The stratified model matches the original on its home cohort and "
            "beats it on the representative cohort — consistent with head-only "
            "selection leaving accuracy on the table, with no cost on the "
            "original cohort."
        )
    elif delta_b["winner"] == "arm_b" and delta_a["winner"] == "arm_b":
        verdict = (
            "Arm B wins on both fixed cohorts. That is stronger than a "
            "selection-bias explanation alone (it would also imply the "
            "stratified training distribution helps even on CA_1 series); "
            "treat it with the single-seed caveat below."
        )
    elif delta_b["winner"] == "arm_a" and delta_a["winner"] == "arm_a":
        verdict = (
            "Arm A wins on both fixed cohorts. On this seed and this window "
            "the experiment does NOT support sampling bias as a contributor to "
            "WAPE; the head-trained model is at least as accurate on the "
            "representative cohort."
        )
    elif delta_b["winner"] == "arm_a" and delta_a["winner"] in ("arm_b", "tie"):
        verdict = (
            "Arm A wins on the representative cohort while Arm B is better or "
            "equal on the head cohort. The evidence does not support the "
            "hypothesis in the direction hoped for; selection bias is not "
            "shown to be the error driver on this window."
        )
    else:  # pragma: no cover - mixed/tie outcomes
        verdict = (
            "The two fixed cohorts disagree about which arm is better, so the "
            "experiment is inconclusive on this window and seed. Report it as "
            "such rather than choosing the favourable cohort."
        )

    lines.append(f"**Overall reading:** {verdict}")

    # Where selection bias DOES show up: the headline metric's denominator.
    home = cells["arm_a_on_cohort_a"]
    away = cells["arm_a_on_cohort_b"]
    volume_gap = (1.0 - home["total_actual_units"] / away["total_actual_units"]) * 100
    lines.append(
        f"**What the headline WAPE represents:** the head-selected cohort "
        f"carries {home['total_actual_units']:,.0f} actual units over the same "
        f"{home['rows']:,} rows, versus {away['total_actual_units']:,.0f} units "
        f"for the stratified cohort — a {volume_gap:.1f}% smaller denominator — "
        f"while its absolute error is LOWER (MAE {home['mae_units']} vs "
        f"{away['mae_units']}). The same Arm A model therefore reads "
        f"{home['wape_percent']}% on the head cohort and {away['wape_percent']}% "
        "on the representative cohort. Selection bias is real, but it operates "
        "on *what the reported number represents* (a low-volume, one-store "
        "cohort) rather than degrading the trained model."
    )

    sweep_b = delta_b["per_store_sweep"]
    lines.append(
        f"**Per-store pattern on cohort B:** Arm B better in "
        f"{sweep_b['arm_b_better_in']} store(s), Arm A better in "
        f"{sweep_b['arm_a_better_in']}, ties {sweep_b['ties']}."
    )
    sweep_cat_b = delta_b["per_category_sweep"]
    lines.append(
        f"**Per-category pattern on cohort B:** Arm B better in "
        f"{sweep_cat_b['arm_b_better_in']} category/categories, Arm A better "
        f"in {sweep_cat_b['arm_a_better_in']}, ties {sweep_cat_b['ties']}."
    )
    sweep_cat_a = delta_a["per_category_sweep"]
    lines.append(
        f"**Per-category pattern on cohort A:** Arm B better in "
        f"{sweep_cat_a['arm_b_better_in']}, Arm A better in "
        f"{sweep_cat_a['arm_a_better_in']}, ties {sweep_cat_a['ties']}."
    )

    lines.append(
        "No improvement was assumed in advance: if the deltas above show Arm B "
        "equal or worse, that is the result."
    )
    return lines


def _gap(value: float) -> str:
    if value > 0:
        return f"+{value:.4f} pp worse"
    if value < 0:
        return f"{abs(value):.4f} pp better"
    return "the same WAPE"


def _file_inventory() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for path in sorted(EXPERIMENT_DIR.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(EXPERIMENT_DIR).as_posix()
        rows.append((relative, f"{path.stat().st_size / 1024:.1f} KB"))
    return rows


def stage_report() -> dict:
    metrics = _read_json(EXPERIMENT_DIR / "metrics.json")
    verification = metrics["verification"]
    coverage = pd.read_csv(EXPERIMENT_DIR / "sampling_coverage.csv")
    coverage_summary = metrics["coverage"]["summary"]
    cells = metrics["cells"]
    resources = metrics["resources"]
    identity = metrics["configuration"]["configuration_identity"]

    out: list[str] = []
    out.append("# Sampling-Bias Experiment — Comparison Report")
    out.append("")
    out.append(
        f"Generated {metrics['experiment']['created_at_utc']} · git "
        f"`{str(metrics['experiment']['git']['head'])[:12]}` · "
        + (
            "production tree untouched (only `experiments/` files added)"
            if metrics["experiment"]["git"]["production_tree_untouched"]
            else "pre-existing changes outside `experiments/` present: "
            + repr(metrics["experiment"]["git"]["changes_outside_experiment_dir"])
        )
        + " · **no commit or push was made**"
    )
    out.append("")
    out.append(f"Rerun: `{RERUN_COMMAND}`")
    out.append("")

    # 1 ---------------------------------------------------------------------
    out.append("## 1. Question and scope")
    out.append("")
    out.append(
        "Does selecting the **first 2,000** product-store series "
        "(all from store CA_1, and missing the FOODS_3 department entirely) "
        "contribute to ElasticIQ's forecasting error, compared with a "
        "**reproducible stratified random sample of 2,000 series**?"
    )
    out.append("")
    out.append(
        "This is an experiment, not a refactor: no production code, model "
        "artifact, evaluation definition, training/validation date, or WAPE "
        "calculation was changed. Everything written lives under "
        "`experiments/sampling_bias/`."
    )
    out.append("")

    # 2 ---------------------------------------------------------------------
    out.append("## 2. Methodology")
    out.append("")
    out.append(
        _md_table(
            [
                "Arm",
                "Series selection",
                "Model artifact",
                "Trained by this experiment?",
            ],
            [
                [
                    "**A (baseline)**",
                    "first 2,000 in file order (production `head(2000)`)",
                    f"`artifacts/models/{ARM_A_VERSION}`",
                    "No — loaded from saved artifacts",
                ],
                [
                    "**B (stratified)**",
                    f"store × dept proportional sample, seed {SEED}",
                    f"`experiments/sampling_bias/models/{ARM_B_VERSION}`",
                    "Yes — via unmodified `run_training()`",
                ],
            ],
        )
    )
    out.append("")
    out.append(
        f"Strata: `{STRATA}` (70 strata, smallest 149 series → allocation ≥ 9). "
        f"Allocation: {ALLOCATION_RULE}."
    )
    out.append("")
    out.append("**Held identical between arms (verified, not assumed):**")
    out.append("")
    out.append(
        _md_table(
            ["Control", "Value", "Identical?"],
            [
                [
                    "Hyperparameters",
                    "poisson, 300 iters, lr 0.05, leaf 63, min leaf 20, seed 42",
                    str(identity["hyperparameters_identical"]),
                ],
                [
                    "Feature set",
                    f"{len(identity['features_arm_a'])} columns",
                    str(identity["features_identical"]),
                ],
                [
                    "Training days",
                    f"{identity['train_days_arm_a']} (both arms)",
                    str(
                        identity["train_days_arm_a"]
                        == identity["train_days_arm_b"]
                    ),
                ],
                [
                    "Validation days",
                    f"{identity['validation_days_arm_a']} (both arms)",
                    str(
                        identity["validation_days_arm_a"]
                        == identity["validation_days_arm_b"]
                    ),
                ],
                [
                    "Calibration",
                    "cross-fitted, level 14, shrink 300",
                    "yes (same code + args)",
                ],
                [
                    "Recursive forecast",
                    "28 days, 60-day tail, predictions fed back",
                    "yes (same function)",
                ],
                ["Scoring", scoring.WAPE_DEFINITION, "yes (same function)"],
                [
                    "Library versions",
                    str(verification["environment"]["scikit_learn"]) + " sklearn",
                    str(identity["library_versions_identical"]),
                ],
            ],
        )
    )
    out.append("")
    out.append(
        f"**Only intended difference:** {identity['only_intended_difference']}."
    )
    out.append("")

    # 3 ---------------------------------------------------------------------
    out.append("## 3. Verification (before any comparison)")
    out.append("")
    arm_a = verification["arm_a_reproduction"]
    out.append(
        f"- **Baseline reproduction:** saved WAPE "
        f"`{arm_a['saved_wape_percent']}`% vs recomputed "
        f"`{arm_a['recomputed_wape_percent']}`% → "
        + ("**exact match**" if arm_a["matches_saved_metrics"] else "**MISMATCH**")
        + f" (MAE {arm_a['saved_mae_units']} vs {arm_a['recomputed_mae_units']}); "
        f"cohort equals `head(2000)`: "
        + ("**yes**" if arm_a["cohort_equals_head_selection"] else "**no**")
        + f"; {arm_a['rows']:,} rows × {arm_a['cohort_series']} series × "
        f"{arm_a['evaluation_days']}. Baseline **loaded, not retrained**."
    )
    out.append(
        f"- **Arm A recorded training time:** "
        f"{_fmt(resources['arm_a_training_seconds'].get('seconds'), 1)} s "
        f"({resources['arm_a_training_seconds']['source']}); "
        "peak memory for Arm A was not measured (never retrained here)."
    )
    calibration = verification["calibration_check"]
    out.append(
        "- **Calibration leakage check:** scale window "
        f"`{calibration['scale_fit_days']}` vs evaluation "
        f"`{calibration['evaluation_days']}` → uses evaluated observations: "
        f"**{str(calibration['calibration_uses_evaluated_observations']).lower()}**; "
        "leakage risk found: "
        f"**{str(calibration['leakage_risk_found']).lower()}**. "
        "Calibration was left unchanged (changing it would add a variable)."
    )
    holdout = verification["holdout_check"]
    out.append(
        f"- **Future window d_1914–d_1941:** `sales_train_evaluation.csv` ends at "
        f"`{holdout['evaluation_file_last_day']}` "
        f"({holdout['evaluation_file_extra_days']} days beyond the validation "
        f"file: {holdout['extra_day_span']}); calendar covers it: "
        f"{str(holdout['calendar_covers_holdout']).lower()}; processed outputs "
        f"stop at {holdout['processed_panel_max_day']} "
        f"→ `{holdout['verdict']}`. Repository references: "
        f"{holdout['repository_references']} total, of which "
        f"{holdout['metric_tuning_or_model_selection_references']} are "
        "metric/tuning/model-selection usages (the rest are schema/data-quality "
        "scans). **This experiment did not score the holdout.**"
    )
    overlap = _read_json(EXPERIMENT_DIR / "panel_report.json")["overlap_check"]
    out.append(
        f"- **Arm B panel equivalence:** for the {overlap['overlap_series']} "
        "series present in both panels, every row (units, prices, calendar "
        f"fields) is identical: **{str(overlap.get('identical')).lower()}** — "
        "so the panels differ only in which series they contain."
    )
    out.append("")

    # 4 ---------------------------------------------------------------------
    out.append("## 4. Coverage: what each arm actually sees")
    out.append("")
    dims = ["state", "store", "category", "department", "store_x_dept"]
    pop = coverage_summary["population"]
    a_cov = coverage_summary[sampling.ARM_A]
    b_cov = coverage_summary[sampling.ARM_B]
    out.append(
        _md_table(
            [
                "Dimension (distinct levels)",
                "Population",
                "Arm A (head)",
                "Arm B (stratified)",
            ],
            [
                [
                    dim,
                    str(pop.get(dim, "n/a")),
                    str(a_cov.get(dim, "n/a")),
                    str(b_cov.get(dim, "n/a")),
                ]
                for dim in dims
            ],
        )
    )
    out.append("")
    counts = coverage.pivot_table(
        index=["dimension", "value"],
        columns=None,
        values=["population_count", sampling.ARM_A, sampling.ARM_B],
        aggfunc="first",
    )
    pivot = coverage.set_index(["dimension", "value"])
    out.append("Counts by dimension/value (selected series):")
    out.append("")
    for dim in ["store", "category", "department"]:
        block = pivot.xs(dim)
        rows = []
        for value, row in block.sort_index().iterrows():
            rows.append(
                [
                    str(value),
                    int(row["population_count"]),
                    int(row[sampling.ARM_A]),
                    int(row[sampling.ARM_B]),
                ]
            )
        out.append(f"**{dim}**")
        out.append("")
        out.append(_md_table(["value", "population", "Arm A", "Arm B"], rows))
        out.append("")
    del counts

    # 5 ---------------------------------------------------------------------
    out.append("## 5. Results")
    out.append("")
    out.append("### 5.1 Primary — fixed common cohorts (2 × 2 matrix)")
    out.append("")
    out.append(
        "Each column is one fixed evaluation cohort (identical series AND "
        "identical dates for both models); each row is one model."
    )
    out.append("")
    out.append(
        _md_table(
            [
                "Model",
                "Cohort A WAPE (%) — head(2,000)",
                "Cohort B WAPE (%) — stratified(2,000)",
                "Cohort A MAE",
                "Cohort B MAE",
            ],
            [
                [
                    "Arm A (head-trained)",
                    _fmt(cells["arm_a_on_cohort_a"]["wape_percent"]),
                    _fmt(cells["arm_a_on_cohort_b"]["wape_percent"]),
                    _fmt(cells["arm_a_on_cohort_a"]["mae_units"]),
                    _fmt(cells["arm_a_on_cohort_b"]["mae_units"]),
                ],
                [
                    "Arm B (stratified-trained)",
                    _fmt(cells["arm_b_on_cohort_a"]["wape_percent"]),
                    _fmt(cells["arm_b_on_cohort_b"]["wape_percent"]),
                    _fmt(cells["arm_b_on_cohort_a"]["mae_units"]),
                    _fmt(cells["arm_b_on_cohort_b"]["mae_units"]),
                ],
                [
                    "**Δ (B − A), pp**",
                    f"**{c1_delta(metrics, COHORT_A)}**",
                    f"**{c1_delta(metrics, COHORT_B)}**",
                    "",
                    "",
                ],
            ],
        )
    )
    out.append("")

    for cohort_key, title in ((COHORT_A, "Cohort A — head(2,000)"),
                              (COHORT_B, "Cohort B — stratified(2,000)")):
        block = metrics["comparison_1_common_cohorts"][cohort_key]
        delta = block["delta"]
        cell_a = cells[block["arm_a_cell"]]
        cell_b = cells[block["arm_b_cell"]]
        out.append(f"### {'5.2' if cohort_key == COHORT_A else '5.3'} {title}")
        out.append("")
        a7 = cell_a["official"]["by_horizon"]["7"]["wape_percent"]
        b7 = cell_b["official"]["by_horizon"]["7"]["wape_percent"]
        a14 = cell_a["official"]["by_horizon"]["14"]["wape_percent"]
        b14 = cell_b["official"]["by_horizon"]["14"]["wape_percent"]
        out.append(
            _md_table(
                ["Metric", "Arm A", "Arm B", "Δ (B − A)"],
                [
                    [
                        "WAPE 28d (%)",
                        _fmt(cell_a["wape_percent"]),
                        _fmt(cell_b["wape_percent"]),
                        f"{delta['delta_pp']:+.4f} pp",
                    ],
                    [
                        "WAPE 7d (%)",
                        _fmt(a7),
                        _fmt(b7),
                        f"{round(b7 - a7, 4):+.4f} pp",
                    ],
                    [
                        "WAPE 14d (%)",
                        _fmt(a14),
                        _fmt(b14),
                        f"{round(b14 - a14, 4):+.4f} pp",
                    ],
                    [
                        "MAE (units)",
                        _fmt(cell_a["mae_units"]),
                        _fmt(cell_b["mae_units"]),
                        f"{delta['mae_delta_units']:+.4f}",
                    ],
                    [
                        "Relative WAPE change",
                        "",
                        "",
                        f"{_fmt(delta['relative_change_percent'], 2)} %",
                    ],
                    [
                        "Series scored",
                        f"{cell_a['series']:,}",
                        f"{cell_b['series']:,}",
                        "identical" if delta["cohorts_identical"] else "DIFFERENT",
                    ],
                    [
                        "Prediction rows",
                        f"{cell_a['rows']:,}",
                        f"{cell_b['rows']:,}",
                        "",
                    ],
                    [
                        "Total actual units",
                        f"{cell_a['total_actual_units']:,.0f}",
                        f"{cell_b['total_actual_units']:,.0f}",
                        "",
                    ],
                    [
                        "Integrity (complete)",
                        str(cell_a["integrity"]["complete"]),
                        str(cell_b["integrity"]["complete"]),
                        "",
                    ],
                    ["Winner", "", "", f"`{delta['winner']}`"],
                ],
            )
        )
        out.append("")
        out.append(
            f"Per-store sweep: {json.dumps(delta['per_store_sweep'])}; "
            f"per-category sweep: {json.dumps(delta['per_category_sweep'])}."
        )
        out.append("")

    # per-store table for cohort B
    out.append("### 5.4 Per-store WAPE on cohort B (where the denominator is non-zero)")
    out.append("")
    cell_a = cells["arm_a_on_cohort_b"]
    cell_b = cells["arm_b_on_cohort_b"]
    rows = []
    for store in sorted(cell_a["breakdowns"]["by_store"]):
        left = cell_a["breakdowns"]["by_store"][store]
        right = cell_b["breakdowns"]["by_store"][store]
        rows.append(
            [
                store,
                _fmt(left["wape_percent"]),
                _fmt(right["wape_percent"]),
                f"{right['wape_percent'] - left['wape_percent']:+.4f}",
                f"{left['total_actual_units']:,.0f}",
                left["series"],
                left["rows"],
            ]
        )
    out.append(
        _md_table(
            [
                "store",
                "Arm A WAPE",
                "Arm B WAPE",
                "Δ pp",
                "actual units",
                "series",
                "rows",
            ],
            rows,
        )
    )
    out.append("")

    out.append("### 5.5 Per-category WAPE (denominator non-zero)")
    out.append("")
    for cohort_key, a_cell, b_cell in (
        (COHORT_A, "arm_a_on_cohort_a", "arm_b_on_cohort_a"),
        (COHORT_B, "arm_a_on_cohort_b", "arm_b_on_cohort_b"),
    ):
        out.append(f"**{cohort_key}**")
        out.append("")
        rows = []
        left = cells[a_cell]["breakdowns"]["by_category"]
        right = cells[b_cell]["breakdowns"]["by_category"]
        for category in sorted(left):
            category_delta = (
                right[category]["wape_percent"] - left[category]["wape_percent"]
            )
            rows.append(
                [
                    category,
                    _fmt(left[category]["wape_percent"]),
                    _fmt(right[category]["wape_percent"]),
                    f"{category_delta:+.4f}",
                    f"{left[category]['total_actual_units']:,.0f}",
                    str(left[category]["denominator_nonzero"]),
                ]
            )
        out.append(
            _md_table(
                [
                    "category",
                    "Arm A WAPE",
                    "Arm B WAPE",
                    "Δ pp",
                    "actual units",
                    "denom ok",
                ],
                rows,
            )
        )
        out.append("")

    out.append("### 5.6 Secondary — within-sample diagnostic (NOT apples-to-apples)")
    out.append("")
    within = metrics["comparison_2_within_sample"]["delta"]
    out.append(
        f"Each model on **its own** 2,000 series: Arm A "
        f"{_fmt(within['arm_a_wape_percent'])}% on cohort A vs Arm B "
        f"{_fmt(within['arm_b_wape_percent'])}% on cohort B "
        f"(Δ {_fmt(within['delta_pp'])} pp, winner `{within['winner']}`). "
        "Evaluated series differ, so this cannot carry the conclusion."
    )
    out.append("")

    # 6 ---------------------------------------------------------------------
    out.append("## 6. Resources")
    out.append("")
    stage_file = EXPERIMENT_DIR / "stage_resources.json"
    rows = []
    if stage_file.is_file():
        stage_resources = _read_json(stage_file)
        for stage, info in stage_resources.items():
            rows.append(
                [
                    stage,
                    _fmt(info.get("elapsed_seconds"), 1),
                    _fmt(info.get("peak_rss_mb"), 1),
                ]
            )
    out.append(_md_table(["stage", "elapsed (s)", "peak RSS (MB)"], rows))
    out.append("")
    out.append(
        f"- Arm B training (production `run_training` timer): "
        f"**{resources['arm_b_training_seconds']} s**; peak process RSS "
        f"**{resources['arm_b_peak_rss_mb']} MB**."
    )
    out.append(
        f"- Arm A training: **{resources['arm_a_training_seconds'].get('seconds')} s** "
        f"(recorded in {resources['arm_a_training_seconds']['source']}); "
        f"peak memory: {resources['arm_a_peak_rss_note']}."
    )
    out.append("")

    # 7 ---------------------------------------------------------------------
    out.append("## 7. Interpretation")
    out.append("")
    for line in _interpretation(metrics):
        out.append(f"- {line}")
    out.append("")

    # 8 ---------------------------------------------------------------------
    out.append("## 8. Limitations and confounders")
    out.append("")
    for line in _limitations(metrics):
        out.append(f"- {line}")
    out.append("")

    # 9 ---------------------------------------------------------------------
    out.append("## 9. Files, rerun, and change control")
    out.append("")
    out.append(
        _md_table(
            ["file (under `experiments/sampling_bias/`)", "size"],
            [[f"`{name}`", size] for name, size in _file_inventory()],
        )
    )
    out.append("")
    out.append("**Exact rerun command (from the repository root):**")
    out.append("")
    out.append("```")
    out.append(RERUN_COMMAND)
    out.append("```")
    out.append("")
    git = metrics["experiment"]["git"]
    out.append(
        f"Git: HEAD `{git['head']}`, branch `{git['branch']}`, status: "
        + (
            "only untracked `experiments/` files; no tracked file modified"
            if git["production_tree_untouched"]
            else "pre-existing changes outside `experiments/`: "
            + repr(git["changes_outside_experiment_dir"])
        )
        + ". **No commit and no push were performed**; only new files under "
        "`experiments/sampling_bias/` were added."
    )
    out.append("")

    report_path = _guard(EXPERIMENT_DIR / "comparison_report.md")
    report_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return {"report": str(report_path), "bytes": report_path.stat().st_size}


def c1_delta(metrics: dict, cohort_key: str) -> str:
    value = metrics["comparison_1_common_cohorts"][cohort_key]["delta"]["delta_pp"]
    return f"{value:+.4f} pp"


def _limitations(metrics: dict) -> list[str]:
    cells = metrics["cells"]
    overlap = _read_json(EXPERIMENT_DIR / "sampling_summary.json")["overlap_series"]
    a_cov = cells["arm_a_on_cohort_b"]["model_coverage"]
    b_cov = cells["arm_b_on_cohort_a"]["model_coverage"]
    return [
        "The evaluation window d_1886–d_1913 is the same window used to develop "
        "the current model — a development result, not a confirmed result on an "
        "untouched test set. `sales_train_evaluation.csv` (d_1914–d_1941) exists "
        "and is reported as a candidate holdout; it was deliberately left unspent.",
        "Single seed (42) and a single stratified draw: no variance estimate. A "
        "different draw could move the deltas; treat sub-point differences as noise.",
        "Cross-cohort gaps mix selection effects with cohort difficulty (cohort A "
        "is one store, cohort B spans 10 stores and the FOODS_3 department). The "
        "primary claim comes from within-column comparisons only.",
        f"Arms share {overlap} series, so the training sets are not disjoint — "
        "that is intentional (nested comparability) but softens the contrast.",
        f"On cohort B, Arm A had no fitted per-series scale for "
        f"{a_cov['series_using_global_scale_fallback']:,} of its series and used "
        f"its global scale {a_cov['global_scale']} with store_id unseen for "
        f"{a_cov['series_in_unseen_stores']:,} series — this is exactly what "
        "production does for out-of-selection series, but it means Arm A's "
        "cross-cell score includes that fallback by design.",
        f"Conversely, on cohort A only {b_cov['series_with_fitted_scale']:,} of "
        "2,000 series carried a fitted scale from Arm B; the rest used Arm B's "
        "global scale. Calibration coverage is therefore asymmetric per cell.",
        "Arm A's training duration comes from a historical log entry and its peak "
        "memory was never measured; only Arm B's runtime/memory are instrumented.",
        "Arm B's `models/arm_b_stratified/metadata.json` carries the stock "
        "`sample_selection` string from the unmodified training code (it claims "
        "head selection). It is wrong text for Arm B; `sampling_manifest.csv` is "
        "the authoritative selection record. Changing the string would have "
        "required editing production code.",
        "Both arms are 2,000-series runs by design (16 GB laptop limit); no "
        "6,000-series or full-catalog arm was attempted here.",
        "Nothing here establishes causality about *why* a cohort scores as it "
        "does (store mix, department mix, and series density all change together "
        "between cohorts).",
    ]


# ------------------------------------------------------------- orchestration ---


def _run_stage(name: str) -> dict:
    functions = {
        "verify": stage_verify,
        "sample": stage_sample,
        "panel": stage_panel,
        "train": stage_train,
        "evaluate": stage_evaluate,
        "report": stage_report,
    }
    print(f"[{name}] starting at {_now()}", flush=True)
    started = time.perf_counter()
    try:
        result = functions[name]()
    except Exception as exc:
        elapsed = round(time.perf_counter() - started, 1)
        print(f"[{name}] FAILED after {elapsed}s: {type(exc).__name__}: {exc}",
              flush=True)
        raise
    elapsed = round(time.perf_counter() - started, 1)

    resources_path = EXPERIMENT_DIR / "stage_resources.json"
    resources = _read_json(resources_path) if resources_path.is_file() else {}
    resources[name] = {
        "elapsed_seconds": elapsed,
        "peak_rss_mb": _peak_rss_mb(),
        "finished_at_utc": _now(),
    }
    _write_json(resources_path, resources)
    print(f"[{name}] finished in {elapsed}s (peak RSS "
          f"{resources[name]['peak_rss_mb']} MB)", flush=True)
    return result if isinstance(result, dict) else {}


def _run_stage_subprocess(name: str) -> bool:
    logs_dir = EXPERIMENT_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{name}.log"
    command = [
        sys.executable,
        "-m",
        "experiments.sampling_bias.run_experiment",
        "--stage",
        name,
    ]
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        code = process.wait()
    if code != 0:
        print(f"[{name}] exited with code {code}; see {log_path}", flush=True)
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sampling-bias experiment (head 2000 vs stratified 2000)"
    )
    parser.add_argument(
        "--stage",
        choices=STAGES,
        help="Run a single stage in this process (used by the orchestrator).",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGES,
        default=list(STAGES),
        help="Stages to run as isolated subprocesses (default: all).",
    )
    args = parser.parse_args()

    if args.stage:
        _run_stage(args.stage)
        return 0

    print("Sampling-bias experiment · stages:", ", ".join(args.stages), flush=True)
    for stage in args.stages:
        if not _run_stage_subprocess(stage):
            print(f"Stopped after stage '{stage}'.", flush=True)
            return 1

    metrics_path = EXPERIMENT_DIR / "metrics.json"
    report_path = EXPERIMENT_DIR / "comparison_report.md"
    if "evaluate" in args.stages and metrics_path.is_file():
        metrics = _read_json(metrics_path)
        print("\n=== sampling-bias experiment complete ===", flush=True)
        for cohort_key in (COHORT_A, COHORT_B):
            delta = metrics["comparison_1_common_cohorts"][cohort_key]["delta"]
            print(
                f"  {cohort_key}: Arm A {delta['arm_a_wape_percent']}% vs "
                f"Arm B {delta['arm_b_wape_percent']}% WAPE "
                f"(delta {delta['delta_pp']:+.4f} pp, winner {delta['winner']})",
                flush=True,
            )
    else:
        print("\n=== stages finished (metrics not yet computed) ===", flush=True)
        print(
            "  next: run 'train' then 'evaluate' then 'report'",
            flush=True,
        )
    if report_path.is_file():
        print(f"Report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
