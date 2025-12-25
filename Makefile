.RECIPEPREFIX := >

# Use Python 3.11
# On Windows this uses the Python launcher if available.
PY_SYS ?= python3.11
ifeq ($(OS),Windows_NT)
PY_SYS ?= py -3.11
endif

VENV := .venv
VENV_PY := $(VENV)/bin/python
ifeq ($(OS),Windows_NT)
VENV_PY := $(VENV)\Scripts\python.exe
endif

.PHONY: help setup lint fmt test serve-local

help:
>@echo Targets:
>@echo "  make setup        Create venv, install deps, install pre-commit hooks"
>@echo "  make lint         Run ruff lint"
>@echo "  make fmt          Run ruff formatter"
>@echo "  make test         Run pytest"
>@echo "  make serve-local  Run FastAPI locally"

$(VENV):
>$(PY_SYS) -m venv $(VENV)
>$(VENV_PY) -m pip install --upgrade pip

setup: $(VENV)
>$(VENV_PY) -m pip install -r requirements-dev.txt
>$(VENV_PY) -m pre_commit install

lint: $(VENV)
>$(VENV_PY) -m ruff check .

fmt: $(VENV)
>$(VENV_PY) -m ruff format .

test: $(VENV)
>$(VENV_PY) -m pytest -q

serve-local: $(VENV)
>$(VENV_PY) -m uvicorn serving.app.main:app --host 127.0.0.1 --port 8000
