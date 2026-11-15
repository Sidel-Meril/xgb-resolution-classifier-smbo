"""Pair matrix for the same-event classifier.

Each side uses calendar intervals built from ``event_year`` / ``deadline_year``,
day-of-year start, and duration. Pair-level similarities and IoU go to
XGBoost, with indicators when a comparison is undefined. Tag tf-idf JSON on each article yields overlap, rarity, and soft-token
similarity columns. Aggregated geo location bundles yield distance and
agreement features.
``emb_score`` and the other pair scores take median imputation and min-max scaling.

Vendored verbatim from ``data_cleaning.resolve.pair_encoder`` (only the
``location_pair_features`` import is rewired to the vendored copy) so the pair
feature matrix and fitted encoder match the shipped classifier.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from resolution_classifier import location_pair_features as _location_pair_features

LOCATION_PAIR_FEATURES = _location_pair_features.LOCATION_PAIR_FEATURES
location_pair_features = _location_pair_features.location_pair_features

__all__ = (
    "DEADLINE_START_TAU_DAYS",
    "EVENT_START_TAU_DAYS",
    "DayOfYearEncoder",
    "FRAME_COLUMNS",
    "INTERVAL_DEFINED",
    "INTERVAL_SIMS",
    "KEYS",
    "LOCATION_PAIR_FEATURES",
    "PAIR_SCORES",
    "START_TAU_BY_KIND",
    "TFIDF_PAIR_FEATURES",
    "align_to_encoder",
    "as_dense",
    "calendar_ordinal",
    "interval_pair_features",
    "location_pair_features",
    "make_preprocess",
    "pair_model_frame",
    "tfidf_pair_features",
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

SIDES = ("left", "right")
EVENT_START_TAU_DAYS = 90.0
DEADLINE_START_TAU_DAYS = 360.0
START_TAU_BY_KIND = {
    "event": EVENT_START_TAU_DAYS,
    "deadline": DEADLINE_START_TAU_DAYS,
}
PAIR_SCORES = (
    "emb_score",
    "cross_score",
    "event_score",
    "deadline_score",
    "link_weight",
)
INTERVAL_SPECS = (
    ("event", "event_year", "doy_event_start", "doy_event_duration"),
    ("deadline", "deadline_year", "doy_deadline_start", "doy_deadline_duration"),
)
INTERVAL_SIMS = (
    "event_start_sim",
    "event_duration_sim",
    "event_interval_iou",
    "deadline_start_sim",
    "deadline_duration_sim",
    "deadline_interval_iou",
)
INTERVAL_DEFINED = (
    "event_interval_defined",
    "deadline_interval_defined",
)
TFIDF_PAIR_FEATURES = (
    "tfidf_shared_weight",
    "tfidf_weighted_overlap",
    "tfidf_weighted_jaccard",
    "tfidf_rare_shared_sum",
    "tfidf_max_shared_idf",
    "tfidf_unique_left_ratio",
    "tfidf_unique_right_ratio",
    "tfidf_shared_token_count",
    "tfidf_shared_concentration",
    "tfidf_soft_similarity",
)
SOFT_TOKEN_SIM_THRESHOLD = 85.0
KEYS = ("left_source", "left_id", "right_source", "right_id")
FRAME_COLUMNS = (
    *KEYS,
    *PAIR_SCORES,
    *INTERVAL_SIMS,
    *INTERVAL_DEFINED,
    *TFIDF_PAIR_FEATURES,
    *LOCATION_PAIR_FEATURES,
    "label",
)


def calendar_ordinal(year: object, doy: object) -> int | None:
    """Ordinal date for ``year`` and day-of-year ``doy``. Invalid cells are missing."""
    try:
        y = int(year)
        d = int(doy)
    except (TypeError, ValueError):
        return None
    if d < 1:
        return None
    try:
        return (date(y, 1, 1) + timedelta(days=d - 1)).toordinal()
    except (ValueError, OverflowError):
        return None


def side_interval(
    frame: pd.DataFrame, side: str, year_name: str, doy_name: str, duration_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Start, end, and validity mask for one side. Missing year, doy, or duration is invalid."""
    year_col = f"{side}_{year_name}"
    doy_col = f"{side}_{doy_name}"
    dur_col = f"{side}_{duration_name}"
    n = len(frame)
    start = np.full(n, np.nan, dtype=float)
    end = np.full(n, np.nan, dtype=float)
    valid = np.zeros(n, dtype=bool)
    if year_col not in frame.columns or doy_col not in frame.columns or dur_col not in frame.columns:
        return start, end, valid
    years = pd.to_numeric(frame[year_col], errors="coerce")
    doys = pd.to_numeric(frame[doy_col], errors="coerce")
    durations = pd.to_numeric(frame[dur_col], errors="coerce")
    for index in range(n):
        y, d, duration = years.iat[index], doys.iat[index], durations.iat[index]
        if pd.isna(y) or pd.isna(d) or pd.isna(duration):
            continue
        duration = int(duration)
        if duration < 0:
            continue
        ordinal = calendar_ordinal(y, d)
        if ordinal is None:
            continue
        start[index] = float(ordinal)
        end[index] = float(ordinal + duration)
        valid[index] = True
    return start, end, valid


def interval_pair_block(
    frame: pd.DataFrame,
    kind: str,
    year_name: str,
    doy_name: str,
    duration_name: str,
    tau_days: float,
) -> pd.DataFrame:
    """Start similarity, duration similarity, and IoU for one interval kind."""
    left_start, left_end, left_ok = side_interval(frame, "left", year_name, doy_name, duration_name)
    right_start, right_end, right_ok = side_interval(frame, "right", year_name, doy_name, duration_name)
    n = len(frame)
    pair_ok = left_ok & right_ok
    start_sim = np.full(n, np.nan, dtype=float)
    duration_sim = np.full(n, np.nan, dtype=float)
    iou = np.full(n, np.nan, dtype=float)
    tau = max(float(tau_days), 1e-9)
    left_dur = left_end - left_start
    right_dur = right_end - right_start
    for index in np.flatnonzero(pair_ok):
        start_sim[index] = np.exp(-abs(left_start[index] - right_start[index]) / tau)
        max_dur = max(left_dur[index], right_dur[index])
        duration_sim[index] = (
            min(left_dur[index], right_dur[index]) / max_dur if max_dur > 0 else 1.0
        )
        overlap_start = max(left_start[index], right_start[index])
        overlap_end = min(left_end[index], right_end[index])
        intersection = max(0.0, overlap_end - overlap_start)
        union = max(left_end[index], right_end[index]) - min(
            left_start[index], right_start[index]
        )
        if union > 0:
            iou[index] = intersection / union
        elif (
            left_start[index] == right_start[index]
            and left_end[index] == right_end[index]
        ):
            iou[index] = 1.0
    return pd.DataFrame(
        {
            f"{kind}_start_sim": start_sim,
            f"{kind}_duration_sim": duration_sim,
            f"{kind}_interval_iou": iou,
            f"{kind}_interval_defined": pair_ok.astype(float),
        },
        index=frame.index,
    )


def interval_pair_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """Six similarity columns and two defined flags for event and deadline intervals."""
    blocks = [
        interval_pair_block(
            pairs,
            kind,
            year_name,
            doy_name,
            duration_name,
            START_TAU_BY_KIND[kind],
        )
        for kind, year_name, doy_name, duration_name in INTERVAL_SPECS
    ]
    return pd.concat(blocks, axis=1)


def tf_idf_map(value: object) -> dict[str, float]:
    """Tag to weight from a stored tf-idf JSON cell. A missing cell is empty."""
    if isinstance(value, dict):
        parsed = value
    elif isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
    else:
        return {}
    if not isinstance(parsed, dict):
        return {}
    weights: dict[str, float] = {}
    for key, weight in parsed.items():
        if not isinstance(key, str):
            continue
        try:
            number = float(weight)
        except (TypeError, ValueError):
            continue
        if number > 0:
            weights[key] = number
    return weights


def soft_tfidf_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    """Token-soft overlap of two tf-idf maps. Exact tokens still match at weight 1."""
    if not left or not right:
        return 0.0
    total = 0.0
    for left_token, left_weight in left.items():
        best = 0.0
        for right_token, right_weight in right.items():
            if left_token == right_token:
                sim = 100.0
            else:
                sim = float(fuzz.ratio(left_token, right_token))
            if sim < SOFT_TOKEN_SIM_THRESHOLD:
                continue
            best = max(best, (sim / 100.0) * right_weight)
        total += left_weight * best
    denom = (sum(left.values()) * sum(right.values())) ** 0.5
    return total / denom if denom > 0 else 0.0


def tfidf_pair_metrics(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    """Overlap and rarity signals from two article tag tf-idf maps."""
    keys_left = set(left)
    keys_right = set(right)
    shared = keys_left & keys_right
    sum_left = sum(left.values())
    sum_right = sum(right.values())
    shared_mins = [min(left[token], right[token]) for token in shared]
    shared_weight = float(sum(shared_mins))
    union_tokens = keys_left | keys_right
    union_max = sum(
        max(left.get(token, 0.0), right.get(token, 0.0)) for token in union_tokens
    )
    unique_left = sum(left[token] for token in keys_left - keys_right)
    unique_right = sum(right[token] for token in keys_right - keys_left)
    rare_shared = [max(left[token], right[token]) for token in shared]
    overlap_denom = min(sum_left, sum_right)
    return {
        "tfidf_shared_weight": shared_weight,
        "tfidf_weighted_overlap": (
            shared_weight / overlap_denom if overlap_denom > 0 else 0.0
        ),
        "tfidf_weighted_jaccard": (
            shared_weight / union_max if union_max > 0 else 0.0
        ),
        "tfidf_rare_shared_sum": float(sum(rare_shared)),
        "tfidf_max_shared_idf": max(rare_shared) if rare_shared else 0.0,
        "tfidf_unique_left_ratio": unique_left / sum_left if sum_left > 0 else 0.0,
        "tfidf_unique_right_ratio": unique_right / sum_right if sum_right > 0 else 0.0,
        "tfidf_shared_token_count": float(len(shared)),
        "tfidf_shared_concentration": (
            max(shared_mins) / shared_weight if shared_weight > 0 else 0.0
        ),
        "tfidf_soft_similarity": soft_tfidf_similarity(left, right),
    }


def tfidf_pair_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Ten tf-idf comparison columns from ``left_tf_idf`` and ``right_tf_idf``."""
    if "left_tf_idf" not in frame.columns or "right_tf_idf" not in frame.columns:
        empty = pd.DataFrame(
            {column: np.nan for column in TFIDF_PAIR_FEATURES},
            index=frame.index,
        )
        return empty
    rows: list[dict[str, float]] = []
    for left_value, right_value in zip(
        frame["left_tf_idf"].to_numpy(),
        frame["right_tf_idf"].to_numpy(),
        strict=True,
    ):
        rows.append(tfidf_pair_metrics(tf_idf_map(left_value), tf_idf_map(right_value)))
    return pd.DataFrame(rows, index=frame.index)


class DayOfYearEncoder(BaseEstimator, TransformerMixin):
    """Sine and cosine of each day-of-year column. A missing day is zero."""

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.columns_ = [str(column) for column in frame.columns]
        return self

    def transform(self, X):
        frame = pd.DataFrame(X, columns=self.columns_)
        blocks = []
        for column in self.columns_:
            doy = pd.to_numeric(frame[column], errors="coerce")
            angle = 2 * np.pi * (doy - 1) / 365
            blocks.append(np.cos(angle).fillna(0.0).to_numpy(dtype=float))
            blocks.append(np.sin(angle).fillna(0.0).to_numpy(dtype=float))
        return np.column_stack(blocks)

    def get_feature_names_out(self, input_features=None):
        names = input_features if input_features is not None else self.columns_
        encoded = []
        for name in names:
            encoded.append(f"{name}_cos")
            encoded.append(f"{name}_sin")
        return np.asarray(encoded, dtype=object)


def as_dense(matrix: object) -> np.ndarray:
    """A dense float matrix. A sparse block is expanded."""
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=float)


def pair_model_frame(pairs: pd.DataFrame) -> pd.DataFrame:
    """One row per pair with interval comparisons and pair scores."""
    aligned = pairs.reset_index(drop=True)
    present = [column for column in FRAME_COLUMNS if column in aligned.columns]
    base = aligned.loc[:, present].copy()
    for column in PAIR_SCORES:
        if column in base.columns:
            base[column] = pd.to_numeric(base[column], errors="coerce")
    interval_columns = (*INTERVAL_SIMS, *INTERVAL_DEFINED)
    if all(column in aligned.columns for column in interval_columns):
        for column in interval_columns:
            base[column] = pd.to_numeric(aligned[column], errors="coerce")
    else:
        interval = interval_pair_features(aligned)
        for column in interval.columns:
            base[column] = interval[column].to_numpy()
    if all(column in base.columns for column in TFIDF_PAIR_FEATURES):
        for column in TFIDF_PAIR_FEATURES:
            base[column] = pd.to_numeric(base[column], errors="coerce")
    elif "left_tf_idf" in aligned.columns and "right_tf_idf" in aligned.columns:
        tfidf = tfidf_pair_features(aligned)
        for column in tfidf.columns:
            base[column] = tfidf[column].to_numpy()
    if all(column in aligned.columns for column in LOCATION_PAIR_FEATURES):
        for column in LOCATION_PAIR_FEATURES:
            base[column] = pd.to_numeric(aligned[column], errors="coerce")
    else:
        geo = location_pair_features(aligned)
        for column in geo.columns:
            base[column] = geo[column].to_numpy()
    for column in FRAME_COLUMNS:
        if column not in base.columns and column != "label":
            base[column] = np.nan
    if "label" in aligned.columns and "label" not in base.columns:
        base["label"] = aligned["label"].to_numpy()
    for column in (*TFIDF_PAIR_FEATURES, *LOCATION_PAIR_FEATURES):
        if column not in base.columns:
            base[column] = np.nan
    ordered = [column for column in FRAME_COLUMNS if column in base.columns]
    missing = [
        column
        for column in (*TFIDF_PAIR_FEATURES, *LOCATION_PAIR_FEATURES)
        if column not in ordered
    ]
    if missing:
        raise RuntimeError(f"pair_model_frame missing columns: {', '.join(missing)}")
    return base.loc[:, ordered]


def make_preprocess(model_frame: pd.DataFrame) -> tuple[ColumnTransformer, list[str]]:
    """Unfitted pair encoder. A numeric column with no observed value is left out."""

    def scaled(name: str, columns: tuple[str, ...]) -> tuple | None:
        kept = [
            column
            for column in columns
            if column in model_frame.columns and model_frame[column].notna().any()
        ]
        if not kept:
            return None
        return (
            name,
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", MinMaxScaler()),
                ]
            ),
            kept,
        )

    transformers: list[tuple] = []
    for name, columns in (
        ("pair_score", PAIR_SCORES),
        ("tfidf", TFIDF_PAIR_FEATURES),
        ("location", LOCATION_PAIR_FEATURES),
    ):
        block = scaled(name, columns)
        if block is not None:
            transformers.append(block)
    interval_cols = [
        column
        for column in (*INTERVAL_SIMS, *INTERVAL_DEFINED)
        if column in model_frame.columns
        and (model_frame[column].notna().any() or column.endswith("_defined"))
    ]
    if interval_cols:
        transformers.append(("interval", "passthrough", interval_cols))
    skipped = [
        column
        for column in (*PAIR_SCORES, *TFIDF_PAIR_FEATURES, *LOCATION_PAIR_FEATURES)
        if column not in model_frame.columns or bool(model_frame[column].isna().all())
    ]
    return ColumnTransformer(transformers=transformers, remainder="drop"), skipped


def align_to_encoder(pairs: pd.DataFrame, encoder: ColumnTransformer) -> pd.DataFrame:
    """Columns the fitted encoder saw. A missing tag is zero."""
    names = getattr(encoder, "feature_names_in_", None)
    if names is None:
        raise ValueError("The pair encoder has no fitted column names.")
    built = pair_model_frame(pairs)
    columns = [str(name) for name in names]
    aligned = pd.DataFrame(index=range(len(pairs)))
    for column in columns:
        if column in built.columns:
            aligned[column] = built[column].to_numpy()
            continue
        if column == "label":
            aligned[column] = False
            continue
        aligned[column] = np.nan
    return aligned
