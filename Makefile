.PHONY: help install get-data engineer split tune train eval all lint fmt test dvc-repro dvc-push dvc-pull

# Same-event pair resolution classifier. Stage 01 reads the Neon warehouse.
# The DVC remote "neon" is s3:// on Neon object storage; dvc-push/pull read .env
# so the s3:// remote can see AWS keys.
#
#   01_get_data   queries/pairs.sql           -> data/pairs.parquet
#   02_engineer   pair_encoder features        -> data/model_frame.parquet
#   03_split      stratified holdout           -> data/split.json
#   05_tune       Optuna (frozen by default)   -> data/best_params.json
#   04_train      XGBoost + Platt calibration  -> models/pairs_same_event/xgb_classifier.joblib
#   06_eval       inverse-weighted F1, accuracy-> reports/eval_metrics.json

help:
	@echo "Targets: install get-data engineer split tune train eval all test lint dvc-repro dvc-push dvc-pull"

install:
	uv sync

get-data:
	uv run python src/resolution_classifier/01_get_data.py

engineer:
	uv run python src/resolution_classifier/02_engineer.py

split:
	uv run python src/resolution_classifier/03_split.py

tune:
	uv run python src/resolution_classifier/05_tune.py

train:
	uv run python src/resolution_classifier/04_train.py

eval:
	uv run python src/resolution_classifier/06_eval.py

all: get-data engineer split tune train eval

dvc-repro:
	uv run dvc repro

dvc-push:
	uv run --env-file .env dvc push

dvc-pull:
	uv run --env-file .env dvc pull

lint:
	uv run ruff check src tests

fmt:
	uv run ruff format src tests

test:
	uv run pytest
