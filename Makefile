.RECIPEPREFIX := >
.PHONY: setup deps fmt lint test serve-local \
        mlflow-up mlflow-down mlflow-reset mlflow-logs \
        stack-up stack-down stack-reset stack-logs \
        train predict-sample \
        train-docker-build train-docker \
        api-docker-build api-docker-up api-docker-down api-docker-logs \
        demo-docker demo-test

ifeq ($(OS),Windows_NT)
PY_BOOTSTRAP := py -3.11
PY := .venv\Scripts\python.exe
CURL := curl.exe
else
PY_BOOTSTRAP := python3.11
PY := .venv/bin/python
CURL := curl
endif

COMPOSE := docker compose -f infra/docker-compose.yml

# Dockerized API is exposed on host port 8001 (container port 8000)
API_HOST ?= 127.0.0.1
API_PORT ?= 8001

setup: .venv deps
> $(PY) -m pre_commit install

.venv:
> $(PY_BOOTSTRAP) -m venv .venv

deps:
> $(PY) -m pip install --upgrade pip
> $(PY) -m pip install -r requirements-dev.txt

fmt:
> $(PY) -m ruff check --fix .
> $(PY) -m ruff format .

lint:
> $(PY) -m ruff check .

test:
> $(PY) -m pytest -q

serve-local:
> $(PY) -m uvicorn serving.app.main:app --host 127.0.0.1 --port 8000

# ---- MLflow stack (infra) ----
mlflow-up:
> $(COMPOSE) up -d --build

mlflow-down:
> $(COMPOSE) down

mlflow-reset:
> $(COMPOSE) down -v

mlflow-logs:
> $(COMPOSE) logs -f mlflow

# Friendlier aliases (same as mlflow-*)
stack-up: mlflow-up
stack-down: mlflow-down
stack-reset: mlflow-reset
stack-logs:
> $(COMPOSE) logs -f --tail 200

# ---- Local training (runs on your venv) ----
train:
> $(PY) training/train.py

predict-sample:
> $(PY) training/predict_sample.py

# ---- Docker training ----
train-docker-build:
> $(COMPOSE) --profile tools build trainer

train-docker:
> $(COMPOSE) --profile tools run --rm trainer

# ---- Docker serving ----
api-docker-build:
> $(COMPOSE) --profile app build api

api-docker-up:
> $(COMPOSE) --profile app up -d api

api-docker-down:
> $(COMPOSE) --profile app down

api-docker-logs:
> $(COMPOSE) logs -f api

# ---- One-command demo ----
# Brings up infra, rebuilds images (if needed), trains (registers model), then starts API
demo-docker: mlflow-up train-docker-build train-docker api-docker-build api-docker-up

# Smoke test the dockerized API (cross-platform curl)
demo-test:
> $(CURL) http://$(API_HOST):$(API_PORT)/health
> $(CURL) -X POST http://$(API_HOST):$(API_PORT)/predict -H "Content-Type: application/json" -d "{\"sepal_length\":5.1,\"sepal_width\":3.5,\"petal_length\":1.4,\"petal_width\":0.2}"
