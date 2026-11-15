"""Pipeline coverage: contract checks, a small roundtrip, and a tune smoke test."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from resolution_classifier.data_contract import DataContract
from resolution_classifier.pair_encoder import FRAME_COLUMNS, KEYS, PAIR_SCORES
from resolution_classifier.training import FROZEN_BEST_PARAMS

ROOT = Path(__file__).resolve().parents[1]
_STAGES: dict[str, dict] = {}


def _stage(filename: str) -> dict:
    loaded = _STAGES.get(filename)
    if loaded is None:
        loaded = runpy.run_path(str(ROOT / "src" / "resolution_classifier" / filename))
        _STAGES[filename] = loaded
    return loaded


def _raw_pairs(n_per_class: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = n_per_class * 2
    labels = np.array([True] * n_per_class + [False] * n_per_class)
    base = np.where(labels, 0.7, 0.3)
    data: dict[str, object] = {
        "left_source": [f"src_l{i}" for i in range(n)],
        "left_id": [f"L{i}" for i in range(n)],
        "right_source": [f"src_r{i}" for i in range(n)],
        "right_id": [f"R{i}" for i in range(n)],
    }
    for column in PAIR_SCORES:
        data[column] = np.clip(base + rng.normal(0, 0.1, n), 0, 1)
    data["score"] = np.where(labels, 0.9, 0.1)
    data["label"] = labels
    return pd.DataFrame(data)


def test_contract_validates_engineered_frame():
    frame = _stage("02_engineer.py")["engineer"](_raw_pairs(6))
    contract = DataContract.load()
    assert [column.name for column in contract.columns] == list(frame.columns)
    assert list(frame.columns) == list(FRAME_COLUMNS)
    assert contract.target() == "label"
    assert contract.keys() == list(KEYS)
    assert set(frame["label"].unique()) <= {0, 1}


def test_contract_rejects_bad_label():
    frame = _stage("02_engineer.py")["engineer"](_raw_pairs(6))
    frame.loc[0, "label"] = 5
    with pytest.raises(ValueError, match="label"):
        DataContract.load().validate(frame)


def test_pipeline_roundtrip(tmp_path: Path):
    raw_path = tmp_path / "pairs.parquet"
    frame_path = tmp_path / "model_frame.parquet"
    split_path = tmp_path / "split.json"
    params_path = tmp_path / "best_params.json"
    model_path = tmp_path / "xgb_classifier.joblib"
    metrics_path = tmp_path / "metrics.json"

    _raw_pairs(40).to_parquet(raw_path, index=False)
    _stage("02_engineer.py")["run"](raw_path, frame_path)
    _stage("03_split.py")["run"](frame_path, split_path)
    _stage("05_tune.py")["run"](frame_path, split_path, params_path)  # frozen
    _stage("04_train.py")["run"](frame_path, split_path, model_path, params_path)
    metrics = _stage("06_eval.py")["run"](frame_path, split_path, model_path, metrics_path)

    bundle = joblib.load(model_path)
    assert set(bundle) == {"model", "encoder"}
    assert hasattr(bundle["encoder"], "feature_names_in_")
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved["n_test"] > 0
    assert 0.0 <= saved["accuracy"] <= 1.0
    assert "inv_weighted_f1" in metrics


def test_tune_writes_frozen_params(tmp_path: Path):
    out = tmp_path / "best_params.json"
    payload = _stage("05_tune.py")["run"](tmp_path / "f.parquet", tmp_path / "s.json", out)
    assert payload["source"] == "frozen"
    assert payload["params"] == dict(FROZEN_BEST_PARAMS)
    assert set(payload["params"]) == {
        "max_depth",
        "learning_rate",
        "n_estimators",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        "reg_alpha",
        "gamma",
    }


def test_tune_search_runs_one_trial(tmp_path: Path):
    raw_path = tmp_path / "pairs.parquet"
    frame_path = tmp_path / "model_frame.parquet"
    split_path = tmp_path / "split.json"
    _raw_pairs(30).to_parquet(raw_path, index=False)
    _stage("02_engineer.py")["run"](raw_path, frame_path)
    _stage("03_split.py")["run"](frame_path, split_path)
    params, value = _stage("05_tune.py")["search"](frame_path, split_path, n_trials=1)
    assert set(params) >= {"max_depth", "learning_rate", "n_estimators"}
    assert np.isfinite(value)
