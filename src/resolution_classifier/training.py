"""Shared fit/split/metric helpers for the same-event pair classifier.

Ported verbatim from ``notebooks/features/xgb_scoring.ipynb`` in
``data-cleaning-smbo`` so stages 03-06 reproduce the booster, the Platt
calibration, and the encoder that shipped ``models/pairs_same_event``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from resolution_classifier.pair_encoder import as_dense, make_preprocess

SEED = 42
N_FOLDS = 5
TEST_SIZE = 0.2
CALIB_SIZE = 0.15
THRESHOLD_HOLDOUT_SIZE = 0.15
N_TRIALS = 100

# Best Optuna trial from the executed notebook (cell 28). Tuning is long-running
# and is not re-run: these frozen params fit the shipped booster.
FROZEN_BEST_PARAMS: dict[str, float | int] = {
    "max_depth": 12,
    "learning_rate": 0.017543506723464546,
    "n_estimators": 854,
    "min_child_weight": 0.2481351286732511,
    "subsample": 0.6662030839083903,
    "colsample_bytree": 0.43153465847430644,
    "reg_lambda": 0.002668372944419076,
    "reg_alpha": 0.0012492570043185632,
    "gamma": 0.40521259301356766,
}


def stratified_split(
    frame: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratify on the true/false label. A one-pair class stays on the training side."""
    labels = frame["label"]
    rare = labels.value_counts()
    rare = rare[rare < 2].index
    held = frame.loc[labels.isin(rare)]
    rest = frame.loc[~labels.isin(rare)]
    if rest.empty:
        raise ValueError("Need at least one class with two pairs to stratify a test set.")
    train, test = train_test_split(
        rest,
        test_size=test_size,
        stratify=rest["label"],
        random_state=seed,
    )
    if not held.empty:
        train = pd.concat([train, held], axis=0)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def design_matrix(
    train_frame: pd.DataFrame,
    other_frame: pd.DataFrame,
) -> tuple[ColumnTransformer, np.ndarray, np.ndarray]:
    """Fit the encoder on precomputed feature rows and transform both sets."""
    preprocess, _skipped = make_preprocess(train_frame)
    encoder = clone(preprocess)
    train_matrix = as_dense(encoder.fit_transform(train_frame))
    other_matrix = as_dense(encoder.transform(other_frame))
    return encoder, train_matrix, other_matrix


def inverse_weighted_f1(y_true: object, y_pred: object) -> float:
    """Mean per-class F1, weighted by the inverse of the class size."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = pd.unique(y_true)
    scores = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    counts = np.array([(y_true == label).sum() for label in labels], dtype=float)
    weights = 1.0 / counts
    return float(np.average(scores, weights=weights))


def class_weights(labels: pd.Series) -> np.ndarray:
    """Balanced row weight, scaled so the average row weighs 1.

    False and True have the same total weight. The scale keeps ``min_child_weight``
    in the same units as a typical row.
    """
    counts = labels.value_counts()
    balanced = len(labels) / (labels.nunique() * labels.map(counts))
    return balanced.to_numpy(dtype=float)


def make_xgb(params: dict) -> XGBClassifier:
    """Binary classifier. predict_proba columns are P(False) and P(True)."""
    return XGBClassifier(
        **params,
        objective="binary:logistic",
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )


def fit_xgb(
    params: dict,
    x_fit: np.ndarray,
    fit_labels: pd.Series,
    x_other: np.ndarray,
) -> tuple[XGBClassifier, np.ndarray, np.ndarray, np.ndarray]:
    """Fit False as 0 and True as 1. Returned probabilities follow that order."""
    class_list = np.array([False, True])
    model = make_xgb(params)
    model.fit(
        x_fit,
        fit_labels.astype(bool).to_numpy().astype(int),
        sample_weight=class_weights(fit_labels),
    )
    probability = model.predict_proba(x_other)
    predicted = class_list[probability.argmax(axis=1)]
    return model, class_list, probability, predicted


def fit_base_xgb(params: dict, x_fit: np.ndarray, fit_labels: pd.Series) -> XGBClassifier:
    """Fit the booster on one matrix slice; class weights match ``fit_xgb``."""
    model = make_xgb(params)
    model.fit(
        x_fit,
        fit_labels.astype(bool).to_numpy().astype(int),
        sample_weight=class_weights(fit_labels),
    )
    return model


def platt_calibrate(
    base: XGBClassifier,
    x_calib: np.ndarray,
    calib_labels: pd.Series,
):
    """Sigmoid (Platt) scaling on held-out pairs; base trees are not refit."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    calibrated.fit(
        x_calib,
        calib_labels.astype(bool).to_numpy().astype(int),
    )
    return calibrated


def predict_pair_proba(model, x_other: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``predict_proba`` with False then True; hard label is the higher probability."""
    class_list = np.array([False, True])
    probability = np.asarray(model.predict_proba(x_other), dtype=float)
    predicted = class_list[probability.argmax(axis=1)]
    return class_list, probability, predicted


def build_test_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    metric_labels: list[bool] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summary micro/macro table and per-class precision, recall, and F1."""
    if metric_labels is None:
        metric_labels = [False, True]
    labels = y_true.astype(bool)
    predicted = np.asarray(y_pred, dtype=bool)
    summary = pd.DataFrame(
        {
            "metric": ["precision", "recall", "f1", "accuracy"],
            "micro": [
                precision_score(labels, predicted, average="micro", zero_division=0),
                recall_score(labels, predicted, average="micro", zero_division=0),
                f1_score(labels, predicted, average="micro", zero_division=0),
                accuracy_score(labels, predicted),
            ],
            "macro": [
                precision_score(
                    labels, predicted, labels=metric_labels, average="macro", zero_division=0
                ),
                recall_score(
                    labels, predicted, labels=metric_labels, average="macro", zero_division=0
                ),
                f1_score(labels, predicted, labels=metric_labels, average="macro", zero_division=0),
                balanced_accuracy_score(labels, predicted),
            ],
        }
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predicted,
        labels=metric_labels,
        zero_division=0,
    )
    per_class = pd.DataFrame(
        {
            "label": metric_labels,
            "n_examples": support.astype(int),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    return summary, per_class


def make_objective(train_frame: pd.DataFrame):
    """Build the Optuna objective: mean inverse-weighted F1 over stratified folds."""
    import optuna

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 6, 18),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        labels = train_frame["label"].astype(bool)
        fold_scores: list[float] = []
        for _train_idx, val_idx in splitter.split(train_frame, labels):
            fit_frame = train_frame.iloc[_train_idx].reset_index(drop=True)
            val_frame = train_frame.iloc[val_idx].reset_index(drop=True)
            _encoder, x_fit, x_val = design_matrix(fit_frame, val_frame)
            _model, _classes, _probability, predicted = fit_xgb(
                params, x_fit, fit_frame["label"], x_val
            )
            fold_scores.append(inverse_weighted_f1(val_frame["label"], predicted))
        return float(np.mean(fold_scores))

    return objective
