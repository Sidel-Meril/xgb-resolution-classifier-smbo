"""Stage 02: build the pair modeling table and validate it.

``pair_encoder.pair_model_frame`` turns the raw pairs into interval,
embedding absolute-difference, and location features in the contract column
order. ``label`` is stored as 0/1.
Writes ``data/model_frame.parquet``.

Usage::

    python src/resolution_classifier/02_engineer.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from resolution_classifier.data_contract import DataContract
from resolution_classifier.pair_encoder import pair_model_frame
from resolution_classifier.paths import FRAME_PATH, RAW_PATH


def engineer(pairs: pd.DataFrame, contract: DataContract | None = None) -> pd.DataFrame:
    """Build the contracted pair modeling table from the raw pairs."""
    contract = contract or DataContract.load()
    frame = pair_model_frame(pairs)
    frame["label"] = frame["label"].astype(bool).astype("int64")
    return contract.validate(frame)


def run(raw_path: Path = RAW_PATH, out_path: Path = FRAME_PATH) -> pd.DataFrame:
    frame = engineer(pd.read_parquet(raw_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    print(f"engineered {len(frame)} pairs, {frame.shape[1]} columns -> {out_path}")
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="02_engineer", description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    parser.add_argument("--out", type=Path, default=FRAME_PATH)
    args = parser.parse_args(argv)
    run(args.raw, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
