# tests/conftest.py
import os

# Make unit tests not depend on MLflow registry being up
os.environ.setdefault("ALLOW_MISSING_MODEL", "1")
