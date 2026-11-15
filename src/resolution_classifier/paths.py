"""Locations for package data and pipeline artifacts.

The SQL mock and column contract sit next to this module, so they resolve
after a wheel install. Artifact directories stay at the repository root when
this tree is ``src/resolution_classifier``, and in the working directory otherwise.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
SQL_PATH = PACKAGE / "queries" / "pairs.sql"
CONTRACT_PATH = PACKAGE / "data_contract.yaml"


def project_root() -> Path:
    """Repository root in the source tree, otherwise the current directory."""
    candidate = PACKAGE.parent.parent
    if PACKAGE.parent.name == "src" and (candidate / "pyproject.toml").is_file():
        return candidate
    return Path.cwd()


ROOT = project_root()
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

RAW_PATH = DATA_DIR / "pairs.parquet"
FRAME_PATH = DATA_DIR / "model_frame.parquet"
SPLIT_PATH = DATA_DIR / "split.json"
BEST_PARAMS_PATH = DATA_DIR / "best_params.json"
MODEL_DIR = MODELS_DIR / "pairs_same_event"
MODEL_PATH = MODEL_DIR / "xgb_classifier.joblib"
METRICS_PATH = REPORTS_DIR / "eval_metrics.json"
