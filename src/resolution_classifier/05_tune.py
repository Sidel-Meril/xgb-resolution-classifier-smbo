"""Stage 05: Optuna search for the booster hyperparameters.

The objective is the mean inverse-weighted F1 over ``N_FOLDS`` stratified folds
on the training pairs (``training.make_objective``), with a ``TPESampler`` seeded
at ``SEED``. The full 100-trial search is long-running and is **not** re-run in
the pipeline: by default this stage writes the frozen best trial from the
notebook (``training.FROZEN_BEST_PARAMS``). Pass ``--run`` to search for real.

Writes ``data/best_params.json``.

Usage::

    python src/resolution_classifier/05_tune.py                 # frozen params
    python src/resolution_classifier/05_tune.py --run --n-trials 100
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd

from resolution_classifier.paths import BEST_PARAMS_PATH, FRAME_PATH, SPLIT_PATH
from resolution_classifier.training import (
    FROZEN_BEST_PARAMS,
    N_TRIALS,
    SEED,
    make_objective,
)


def _train_frame(frame_path: Path, split_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(frame_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    return frame.iloc[split["train_pos"]].reset_index(drop=True)


def search(frame_path: Path, split_path: Path, *, n_trials: int) -> tuple[dict, float]:
    """Run the Optuna study and return the best params and value."""
    import optuna
    from optuna.exceptions import ExperimentalWarning

    warnings.filterwarnings("ignore", category=ExperimentalWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    train_frame = _train_frame(frame_path, split_path)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(make_objective(train_frame), n_trials=n_trials)
    return dict(study.best_params), float(study.best_value)


def run(
    frame_path: Path = FRAME_PATH,
    split_path: Path = SPLIT_PATH,
    out_path: Path = BEST_PARAMS_PATH,
    *,
    run_search: bool = False,
    n_trials: int = N_TRIALS,
) -> dict:
    if run_search:
        params, best_value = search(frame_path, split_path, n_trials=n_trials)
        payload = {
            "params": params,
            "source": "optuna",
            "n_trials": n_trials,
            "best_value": best_value,
        }
    else:
        payload = {"params": dict(FROZEN_BEST_PARAMS), "source": "frozen", "n_trials": N_TRIALS}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"best params ({payload['source']}) -> {out_path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="05_tune", description=__doc__)
    parser.add_argument("--frame", type=Path, default=FRAME_PATH)
    parser.add_argument("--split", type=Path, default=SPLIT_PATH)
    parser.add_argument("--out", type=Path, default=BEST_PARAMS_PATH)
    parser.add_argument(
        "--run", action="store_true", help="run the Optuna search instead of frozen params"
    )
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    args = parser.parse_args(argv)
    run(args.frame, args.split, args.out, run_search=args.run, n_trials=args.n_trials)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
