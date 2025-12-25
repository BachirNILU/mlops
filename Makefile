.RECIPEPREFIX := >
.PHONY: setup deps lint test serve-local mlflow-up mlflow-down mlflow-logs train predict-sample

ifeq ($(OS),Windows_NT)
PY_BOOTSTRAP := py -3.11
PY := .venv\Scripts\python.exe
else
PY_BOOTSTRAP := python3.11
PY := .venv/bin/python
endif

COMPOSE := docker compose -f infra/docker-compose.yml

setup: .venv deps
> $(PY) -m pre_commit install

.venv:
> $(PY_BOOTSTRAP) -m venv .venv

deps:
> $(PY) -m pip install --upgrade pip
> $(PY) -m pip install -r requirements-dev.txt

lint:
> $(PY) -m ruff check .
> $(PY) -m ruff format --check .

test:
> $(PY) -m pytest -q

serve-local:
> $(PY) -m uvicorn serving.app.main:app --host 127.0.0.1 --port 8000

mlflow-up:
> $(COMPOSE) up -d --build

mlflow-down:
> $(COMPOSE) down -v

mlflow-logs:
> $(COMPOSE) logs -f mlflow

train:
> $(PY) training/train.py

predict-sample:
> $(PY) training/predict_sample.py
