"""Column contract for the same-event pair modeling table.

Each YAML column carries dtype, nullability, roles, optional categories,
bounds, and the name of the transform that produced it. ``validate`` checks
a frame. ``model_matrix`` builds the numeric XGBoost matrix from feature
and categorical roles (the live pipeline encodes with the fitted
``ColumnTransformer`` instead; this method mirrors the template contract).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import attrs
import pandas as pd
import yaml

_YAML = Path(__file__).with_name("data_contract.yaml")


@attrs.define(frozen=True, slots=True)
class Column:
    name: str
    dtype: str
    nullable: bool = True
    role: tuple[str, ...] = ()
    categories: tuple[Any, ...] | None = None
    min_value: float | None = None
    max_value: float | None = None
    transform: str | None = None


@attrs.define(frozen=True, slots=True)
class DataContract:
    version: int
    columns: tuple[Column, ...]
    mock_sql: str = ""

    @classmethod
    def load(cls, path: Path | None = None) -> DataContract:
        raw = yaml.safe_load((path or _YAML).read_text(encoding="utf-8")) or {}
        columns = []
        for name, spec in (raw.get("columns") or {}).items():
            spec = spec or {}
            categories = spec.get("categories")
            columns.append(
                Column(
                    name=str(name),
                    dtype=str(spec.get("dtype", "float")),
                    nullable=bool(spec.get("nullable", True)),
                    role=tuple(spec.get("role") or ()),
                    categories=None if categories is None else tuple(categories),
                    min_value=None if spec.get("min") is None else float(spec["min"]),
                    max_value=None if spec.get("max") is None else float(spec["max"]),
                    transform=None if spec.get("transform") is None else str(spec["transform"]),
                )
            )
        return cls(
            version=int(raw.get("version", 0)),
            columns=tuple(columns),
            mock_sql=str(raw.get("mock_sql") or ""),
        )

    def names(self, role: str) -> list[str]:
        return [column.name for column in self.columns if role in column.role]

    def target(self) -> str:
        names = self.names("target")
        if len(names) != 1:
            raise ValueError("contract needs exactly one target column")
        return names[0]

    def keys(self) -> list[str]:
        return self.names("join_key")

    def validate(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [column.name for column in self.columns if column.name not in frame.columns]
        if missing:
            raise ValueError(f"missing columns: {missing}")
        ordered = [column.name for column in self.columns]
        if list(frame.columns) != ordered:
            frame = frame.loc[:, ordered]
        for column in self.columns:
            _check(frame[column.name], column)
        return frame

    def model_matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Numeric matrix. Categorical contract columns become fixed dummy sets."""
        specs = {column.name: column for column in self.columns}
        features = self.names("feature")
        missing = [name for name in features if name not in frame.columns]
        if missing:
            raise ValueError(f"missing features: {missing}")
        categorical = [name for name in self.names("categorical") if name in features]
        numeric_names = [name for name in features if name not in categorical]
        parts = [frame.loc[:, numeric_names].astype("float64")]
        for name in categorical:
            levels = [str(item) for item in (specs[name].categories or ())]
            values = frame[name].astype("string").fillna("__missing__")
            dummies = pd.get_dummies(values, prefix=name, dtype="float64")
            expected = [f"{name}_{level}" for level in levels]
            parts.append(dummies.reindex(columns=expected, fill_value=0.0))
        return pd.concat(parts, axis=1)


def _check(series: pd.Series, column: Column) -> None:
    """Raise if ``series`` breaks the column rule. Does not copy the frame."""
    if column.dtype == "string":
        if not column.nullable and bool(series.isna().any()):
            raise ValueError(f"{column.name}: nulls")
        if column.categories is not None:
            allowed = [str(item) for item in column.categories]
            values = series if isinstance(series.dtype, pd.StringDtype) else series.astype("string")
            bad = values.notna() & ~values.isin(allowed)
            if bool(bad.any()):
                raise ValueError(f"{column.name}: categories {series[bad].head(3).tolist()}")
        return

    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"{column.name}: non-numeric")
    if not column.nullable and bool(series.isna().any()):
        raise ValueError(f"{column.name}: nulls")
    if column.dtype == "int" and not pd.api.types.is_integer_dtype(series):
        fractional = series.notna() & ((series - series.round()).abs() > 1e-6)
        if bool(fractional.any()):
            raise ValueError(f"{column.name}: fractional")
    if column.min_value is not None and bool((series.notna() & (series < column.min_value)).any()):
        raise ValueError(f"{column.name}: min {column.min_value}")
    if column.max_value is not None and bool((series.notna() & (series > column.max_value)).any()):
        raise ValueError(f"{column.name}: max {column.max_value}")
    if column.categories is not None:
        bad = series.notna() & ~series.isin(list(column.categories))
        if bool(bad.any()):
            raise ValueError(f"{column.name}: categories {series[bad].head(3).tolist()}")
