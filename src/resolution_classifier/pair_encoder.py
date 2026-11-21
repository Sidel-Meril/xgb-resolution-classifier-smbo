"""Pair matrix for the same-event classifier.

Each side uses calendar intervals built from ``event_year`` / ``deadline_year``,
day-of-year start, and duration. Pair-level similarities and IoU go to
XGBoost, with indicators when a comparison is undefined. Title embeddings
``left_emb_*`` and ``right_emb_*`` stay on the extract. The model sees the
per-dimension distance ``abs(right_emb_N - left_emb_N)``. Aggregated geo
location bundles yield distance and agreement features.
``emb_score`` and the other pair scores take median imputation and min-max scaling.

Vendored from ``data_cleaning.resolve.pair_encoder``. The location import is
rewired to this package. ``EMBED_COLUMNS`` is inlined from
``data_cleaning.tags.02_title_embeddings`` (``intfloat/multilingual-e5-small``,
``emb_000`` through ``emb_383``).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from resolution_classifier import location_pair_features as _location_pair_features

# Title embedding width of intfloat/multilingual-e5-small.
EMBED_DIM = 384
EMBED_COLUMNS = tuple(f"emb_{index:03d}" for index in range(EMBED_DIM))

LOCATION_PAIR_FEATURES = _location_pair_features.LOCATION_PAIR_FEATURES
location_pair_features = _location_pair_features.location_pair_features

__all__ = (
    "DEADLINE_START_TAU_DAYS",
    "EMB_ABS_DIFF_COLUMNS",
    "EMBEDDING_PAIR_COLUMNS",
    "EVENT_START_TAU_DAYS",
    "DayOfYearEncoder",
    "FRAME_COLUMNS",
    "INTERVAL_DEFINED",
    "INTERVAL_SIMS",
    "KEYS",
    "LOCATION_PAIR_FEATURES",
    "PAIR_SCORES",
    "START_TAU_BY_KIND",
    "align_to_encoder",
    "as_dense",
    "calendar_ordinal",
    "embedding_abs_diff_features",
    "interval_pair_features",
    "location_pair_features",
    "make_preprocess",
    "pair_model_frame",
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
EMBEDDING_PAIR_COLUMNS = tuple(
    f"{side}_{column}" for side in SIDES for column in EMBED_COLUMNS
)
EMB_ABS_DIFF_COLUMNS = tuple(f"emb_abs_diff_{index:03d}" for index in range(EMBED_DIM))
KEYS = ("left_source", "left_id", "right_source", "right_id")
FRAME_COLUMNS = (
    *KEYS,
    *PAIR_SCORES,
    *INTERVAL_SIMS,
    *INTERVAL_DEFINED,
    *EMB_ABS_DIFF_COLUMNS,
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


def _side_embedding(frame: pd.DataFrame, side: str) -> np.ndarray:
    """One side's title embedding. A missing component is NaN."""
    n = len(frame)
    columns = []
    for stem in EMBED_COLUMNS:
        name = f"{side}_{stem}"
        if name in frame.columns:
            columns.append(pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float))
        else:
            columns.append(np.full(n, np.nan))
    return np.column_stack(columns)


def embedding_abs_diff_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-dimension distance ``abs(right_emb_N - left_emb_N)``."""
    distance = np.abs(_side_embedding(frame, "right") - _side_embedding(frame, "left"))
    return pd.DataFrame(distance, index=frame.index, columns=list(EMB_ABS_DIFF_COLUMNS))


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
    if all(column in aligned.columns for column in EMB_ABS_DIFF_COLUMNS):
        diffs = pd.DataFrame(
            {
                column: pd.to_numeric(aligned[column], errors="coerce").to_numpy()
                for column in EMB_ABS_DIFF_COLUMNS
            },
            index=base.index,
        )
    else:
        diffs = embedding_abs_diff_features(aligned)
    overlap = [column for column in diffs.columns if column in base.columns]
    if overlap:
        base = base.drop(columns=overlap)
    base = pd.concat([base, diffs], axis=1)
    if all(column in aligned.columns for column in LOCATION_PAIR_FEATURES):
        for column in LOCATION_PAIR_FEATURES:
            base[column] = pd.to_numeric(aligned[column], errors="coerce")
    else:
        geo = location_pair_features(aligned)
        for column in geo.columns:
            base[column] = geo[column].to_numpy()
    missing_columns = [
        column for column in FRAME_COLUMNS if column not in base.columns and column != "label"
    ]
    if missing_columns:
        filler = pd.DataFrame(np.nan, index=base.index, columns=missing_columns)
        base = pd.concat([base, filler], axis=1)
    if "label" in aligned.columns and "label" not in base.columns:
        base["label"] = aligned["label"].to_numpy()
    ordered = [column for column in FRAME_COLUMNS if column in base.columns]
    missing = [
        column
        for column in (*EMB_ABS_DIFF_COLUMNS, *LOCATION_PAIR_FEATURES)
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
        ("emb_abs_diff", EMB_ABS_DIFF_COLUMNS),
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
        for column in (*PAIR_SCORES, *EMB_ABS_DIFF_COLUMNS, *LOCATION_PAIR_FEATURES)
        if column not in model_frame.columns or bool(model_frame[column].isna().all())
    ]
    return ColumnTransformer(transformers=transformers, remainder="drop"), skipped


def align_to_encoder(pairs: pd.DataFrame, encoder: ColumnTransformer) -> pd.DataFrame:
    """Columns the fitted encoder saw. A missing column is empty."""
    names = getattr(encoder, "feature_names_in_", None)
    if names is None:
        raise ValueError("The pair encoder has no fitted column names.")
    built = pair_model_frame(pairs)
    columns = [str(name) for name in names]
    values = {
        column: (
            built[column].to_numpy()
            if column in built.columns
            else False
            if column == "label"
            else np.nan
        )
        for column in columns
    }
    return pd.DataFrame(values, index=range(len(pairs)))
