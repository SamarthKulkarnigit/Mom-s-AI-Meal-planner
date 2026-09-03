"""
Shared pytest configuration.

Runs every test against an isolated throwaway SQLite database so the
development database (backend/data/mealplanner.db) is never touched. The env
vars must be set before any backend module is imported, which happens here at
collection time (conftest is imported before the test modules in its folder).
"""

import os
import tempfile

_TMP_DIR = tempfile.mkdtemp(prefix="mealplanner_pytest_")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DIR}/test.db"
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-for-production")

# Ensure the schema exists for tests that use db.py directly without importing
# backend.main (which normally runs create_all at import time).
from backend.database import Base, engine  # noqa: E402
import backend.models  # noqa: E402, F401  (registers models on Base)
Base.metadata.create_all(bind=engine)
