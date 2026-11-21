# xgb-resolution-classifier-smbo

Same-event pair resolution classifier, scaffolded from
[ml-model-template-smbo](../ml-model-template-smbo). One training row is one
candidate pair from `staging.candidates_02_pairs_features`, labeled by `score`
on `staging.pairs_generator_01_same_event` (`score > 0.5` is True). The booster
is XGBoost with sigmoid (Platt) calibration; the saved bundle is
`{"model": CalibratedClassifierCV, "encoder": ColumnTransformer}`.

The training logic is ported from
`data-cleaning-smbo/notebooks/features/xgb_scoring.ipynb`. The pair feature
builders (`pair_encoder.py`, `location_pair_features.py`, `geo/normalize.py`)
are vendored from `data-cleaning-smbo`. Raw `left_emb_*` and `right_emb_*`
stay on the extract. The model uses `emb_abs_diff_000` through
`emb_abs_diff_383`, each `abs(right_emb_N - left_emb_N)`. Median imputation and
min-max scaling apply to pair scores, those distances, and the location features.

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
last executed notebook (`training.FROZEN_BEST_PARAMS`) by default. That trial
was fit on the previous tf-idf feature matrix. Pass `--run` to search on the
absolute-difference matrix; the search is covered by a one-trial smoke test.

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

The joblib currently in `models/pairs_same_event/` is the previous tf-idf fit
copied from `data-cleaning-smbo`. `notebooks/parity_check.ipynb` compares that
file to the notebook artifact. A refit through stages 01-06 trains on the
absolute-difference features; it will not match that previous joblib.

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
