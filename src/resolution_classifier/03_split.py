"""Stage 03: stratified train/test split over the pair modeling table.

The test set is 20% of the pairs, stratified by the True/False label; a class
with a single pair stays on the training side. Positions index
``data/model_frame.parquet`` and preserve the split's row order so the refit
matches the shipped fit. Writes ``data/split.json``.

Usage::

    python src/resolution_classifier/03_split.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from resolution_classifier.paths import FRAME_PATH, SPLIT_PATH
from resolution_classifier.training import SEED, TEST_SIZE, stratified_split


def run(
    frame_path: Path = FRAME_PATH,
    out_path: Path = SPLIT_PATH,
    *,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
) -> dict[str, list[int]]:
    frame = pd.read_parquet(frame_path)
    tagged = frame.copy()
    tagged["_pos"] = range(len(tagged))
    train, test = stratified_split(tagged, test_size, seed)
    payload = {
        "train_pos": [int(pos) for pos in train["_pos"]],
        "test_pos": [int(pos) for pos in test["_pos"]],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"split train={len(payload['train_pos'])} test={len(payload['test_pos'])} -> {out_path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="03_split", description=__doc__)
    parser.add_argument("--frame", type=Path, default=FRAME_PATH)
    parser.add_argument("--out", type=Path, default=SPLIT_PATH)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    args = parser.parse_args(argv)
    run(args.frame, args.out, test_size=args.test_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
