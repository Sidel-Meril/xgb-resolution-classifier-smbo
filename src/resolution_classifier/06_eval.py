"""Stage 06: score the holdout pairs and write metrics.

Loads the saved classifier and its encoder, scores the test pairs with the
argmax (0.5) rule, and writes inverse-weighted F1, accuracy, balanced accuracy,
and macro F1 to ``reports/eval_metrics.json``.

Usage::

    python src/resolution_classifier/06_eval.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from resolution_classifier.pair_encoder import align_to_encoder, as_dense
from resolution_classifier.paths import FRAME_PATH, METRICS_PATH, MODEL_PATH, SPLIT_PATH
from resolution_classifier.training import inverse_weighted_f1, predict_pair_proba


def run(
    frame_path: Path = FRAME_PATH,
    split_path: Path = SPLIT_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> dict[str, float | int | None]:
    frame = pd.read_parquet(frame_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    bundle = joblib.load(model_path)
    model, encoder = bundle["model"], bundle["encoder"]
    test_frame = frame.iloc[split["test_pos"]].reset_index(drop=True)

    matrix = as_dense(encoder.transform(align_to_encoder(test_frame, encoder)))
    _classes, _proba, predicted = predict_pair_proba(model, matrix)
    y_true = test_frame["label"].astype(bool)
    predicted = predicted.astype(bool)

    metrics: dict[str, float | int | None] = {
        "inv_weighted_f1": inverse_weighted_f1(y_true, predicted),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(
            f1_score(y_true, predicted, labels=[False, True], average="macro", zero_division=0)
        ),
        "n_train": len(split["train_pos"]),
        "n_test": len(split["test_pos"]),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"inv_weighted_f1={metrics['inv_weighted_f1']:.3f} "
        f"accuracy={metrics['accuracy']:.3f} -> {metrics_path}"
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="06_eval", description=__doc__)
    parser.add_argument("--frame", type=Path, default=FRAME_PATH)
    parser.add_argument("--split", type=Path, default=SPLIT_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    args = parser.parse_args(argv)
    run(args.frame, args.split, args.model, args.metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
