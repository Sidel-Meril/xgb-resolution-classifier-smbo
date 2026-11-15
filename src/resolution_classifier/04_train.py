"""Stage 04: fit the booster, Platt-calibrate, and save the classifier.

Mirrors the "Best model" cell of the notebook. The encoder fits on all training
pairs. The booster (frozen best params) fits on a stratified booster slice; a
sigmoid (Platt) calibrator fits on the calibration slice without refitting the
trees. Saves ``{"model": calibrated, "encoder": encoder}`` to
``models/pairs_same_event/xgb_classifier.joblib`` so the bundle matches the one
``data-cleaning-smbo`` ships.

Usage::

    python src/resolution_classifier/04_train.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from resolution_classifier.pair_encoder import as_dense
from resolution_classifier.paths import BEST_PARAMS_PATH, FRAME_PATH, MODEL_PATH, SPLIT_PATH
from resolution_classifier.training import (
    CALIB_SIZE,
    FROZEN_BEST_PARAMS,
    SEED,
    THRESHOLD_HOLDOUT_SIZE,
    design_matrix,
    fit_base_xgb,
    platt_calibrate,
    stratified_split,
)


def _load_params(params_path: Path | None) -> dict:
    if params_path is None or not params_path.is_file():
        return dict(FROZEN_BEST_PARAMS)
    payload = json.loads(params_path.read_text(encoding="utf-8"))
    return dict(payload.get("params", payload))


def run(
    frame_path: Path = FRAME_PATH,
    split_path: Path = SPLIT_PATH,
    model_path: Path = MODEL_PATH,
    params_path: Path | None = BEST_PARAMS_PATH,
) -> Path:
    frame = pd.read_parquet(frame_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    best_params = _load_params(params_path)
    train_frame = frame.iloc[split["train_pos"]].reset_index(drop=True)
    test_frame = frame.iloc[split["test_pos"]].reset_index(drop=True)

    fit_frame, train_rest = stratified_split(
        train_frame, CALIB_SIZE + THRESHOLD_HOLDOUT_SIZE, SEED + 1
    )
    calib_frame, _threshold_frame = stratified_split(train_rest, 0.5, SEED + 2)
    encoder, x_train, _x_test = design_matrix(train_frame, test_frame)
    x_fit = as_dense(encoder.transform(fit_frame))
    x_calib = as_dense(encoder.transform(calib_frame))
    xgb_model = fit_base_xgb(best_params, x_fit, fit_frame["label"])
    model = platt_calibrate(xgb_model, x_calib, calib_frame["label"])

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "encoder": encoder}, model_path)
    print(
        f"booster={len(fit_frame)} platt={len(calib_frame)} "
        f"encoded_columns={x_train.shape[1]} -> {model_path}"
    )
    return model_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="04_train", description=__doc__)
    parser.add_argument("--frame", type=Path, default=FRAME_PATH)
    parser.add_argument("--split", type=Path, default=SPLIT_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--params", type=Path, default=BEST_PARAMS_PATH)
    args = parser.parse_args(argv)
    run(args.frame, args.split, args.model, args.params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
