"""Stage 01: load labeled candidate pairs from the warehouse.

Reads ``queries/pairs.sql`` and writes ``data/pairs.parquet``. One row is one
pair from ``staging.candidates_02_pairs_features`` joined to its label in
``staging.pairs_generator_01_same_event``. A ``score`` above 0.5 is True.

Usage::

    python src/resolution_classifier/01_get_data.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from resolution_classifier.db import fetch_dataframe, get_connection, has_database_config
from resolution_classifier.pair_encoder import PAIR_SCORES
from resolution_classifier.paths import RAW_PATH, SQL_PATH

DATE_COLUMNS = tuple(
    f"{side}_{stem}"
    for side in ("left", "right")
    for stem in (
        "doy_deadline_start",
        "doy_event_start",
        "doy_deadline_duration",
        "doy_event_duration",
        "event_year",
        "deadline_year",
    )
)


def load_pairs(sql_path: Path = SQL_PATH, conninfo: str | None = None) -> pd.DataFrame:
    """Return the labeled pair frame. ``label`` is ``score > 0.5``."""
    statement = sql_path.read_text(encoding="utf-8")
    with get_connection(conninfo) as connection:
        pairs = fetch_dataframe(connection, statement)
    for column in (*PAIR_SCORES, *DATE_COLUMNS):
        if column in pairs.columns:
            pairs[column] = pd.to_numeric(pairs[column], errors="coerce")
    pairs["score"] = pd.to_numeric(pairs["score"], errors="coerce")
    pairs = pairs.dropna(subset=["score"]).reset_index(drop=True)
    pairs["label"] = pairs["score"] > 0.5
    return pairs


def run(
    sql_path: Path = SQL_PATH,
    out_path: Path = RAW_PATH,
    conninfo: str | None = None,
) -> pd.DataFrame:
    if not has_database_config(conninfo):
        raise RuntimeError(
            "DATABASE_URL or PG* variables are required (copy .env.example -> .env)."
        )
    frame = load_pairs(sql_path, conninfo)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    true_n = int(frame["label"].sum())
    print(f"loaded {len(frame)} pairs ({true_n} True / {len(frame) - true_n} False) -> {out_path}")
    return frame


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="01_get_data", description=__doc__)
    parser.add_argument("--sql", type=Path, default=SQL_PATH)
    parser.add_argument("--out", type=Path, default=RAW_PATH)
    args = parser.parse_args(argv)
    run(args.sql, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
