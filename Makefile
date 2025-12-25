
---

#### `Makefile`
```makefile
SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

PYTHON ?= python3.11
VENV := .venv
BIN := $(VENV)/bin

.PHONY: help
help:
	@awk 'BEGIN {FS = ":.*##"; printf "\nTargets:\n"} /^[a-zA-Z0-9_-]+:.*##/ {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

$(VENV):
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip

.PHONY: setup
setup: $(VENV) ## Install dev dependencies and pre-commit hooks
	$(BIN)/pip install -r requirements-dev.txt
	$(BIN)/pre-commit install

.PHONY: lint
lint: ## Ruff lint
	$(BIN)/ruff check .

.PHONY: fmt
fmt: ## Ruff format (writes changes)
	$(BIN)/ruff format .

.PHONY: test
test: ## Run tests
	$(BIN)/pytest -q

.PHONY: serve-local
serve-local: $(VENV) ## Run API locally
	$(BIN)/uvicorn serving.app.main:app --host 127.0.0.1 --port 8000
