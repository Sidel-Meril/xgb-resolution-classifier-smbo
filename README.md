# xgb-resolution-classifier-smbo

Same-event pair resolution classifier, scaffolded from
[ml-model-template-smbo](../ml-model-template-smbo). One training row is one
candidate pair from `staging.candidates_02_pairs_features`, labeled by `score`
on `staging.pairs_generator_01_same_event` (`score > 0.5` is True). The booster
is XGBoost with sigmoid (Platt) calibration; the saved bundle is
`{"model": CalibratedClassifierCV, "encoder": ColumnTransformer}`.

The training logic is ported verbatim from
`data-cleaning-smbo/notebooks/features/xgb_scoring.ipynb`, and the pair feature
builders (`pair_encoder.py`, `location_pair_features.py`, `geo/normalize.py`) are
vendored verbatim from `data-cleaning-smbo` so a refit reproduces the shipped fit.

## Pipeline

```text
01 get_data    queries/pairs.sql            -> data/pairs.parquet
02 engineer    pair_encoder features        -> data/model_frame.parquet
03 split       stratified 80/20             -> data/split.json
05 tune        Optuna (frozen by default)   -> data/best_params.json
04 train       XGBoost + Platt calibration  -> models/pairs_same_event/xgb_classifier.joblib
06 eval        inverse-weighted F1, accuracy-> reports/eval_metrics.json
```

Stage 01 reads the same Neon warehouse as `data-cleaning-smbo` (configure `PG*`
or `DATABASE_URL` in `.env`).

## Tuning is frozen

The 100-trial Optuna search (`training.make_objective`, seeded `TPESampler`) is
long-running and is **not** re-run. `05_tune.py` writes the best trial from the
notebook (`training.FROZEN_BEST_PARAMS`) by default. Pass `--run` to search for
real; the search is covered by a one-trial smoke test.

## Saved model

```python
from resolution_classifier import load_scorer

model, encoder = load_scorer()
```

`load_model()` / `load_scorer()` use a local
`models/pairs_same_event/xgb_classifier.joblib` when present. Otherwise the fit
is read from the DVC remote named in `.dvc/config` (`neon`, an `s3://` Neon
bucket) with `dvc.api.open` and cached by md5. The `s3://` remote needs
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` (see `.env.example`).

The shipped artifact is byte-for-byte the model `data-cleaning-smbo` hosts: it
is copied from there and pushed to DVC without re-running tuning or training.
`notebooks/parity_check.ipynb` proves the shipped model matches the
`data-cleaning-smbo` model by raw booster bytes, feature importances, and SHAP
values, and that the scaffolded pipeline reproduces the same fit.

Env overrides: `RESOLUTION_MODEL`, `RESOLUTION_REV`, `RESOLUTION_REPO`,
`RESOLUTION_CACHE`.

## Run

```text
uv sync
uv run python src/resolution_classifier/01_get_data.py
uv run python src/resolution_classifier/02_engineer.py
uv run python src/resolution_classifier/03_split.py
uv run python src/resolution_classifier/05_tune.py
uv run python src/resolution_classifier/04_train.py
uv run python src/resolution_classifier/06_eval.py
uv run pytest
```

`make dvc-push` publishes the model to the `neon` remote (reads `.env` for AWS
keys). `make dvc-pull` fetches it.
