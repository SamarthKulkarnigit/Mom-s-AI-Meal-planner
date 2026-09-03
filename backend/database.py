import os
import logging
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./backend/data/mealplanner.db")

if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# ---------------------------------------------------------------------------
# Idempotent schema migrations
# ---------------------------------------------------------------------------
# create_all() only creates missing tables; it never alters existing ones, so
# columns added to models after a database already exists (e.g. the local
# SQLite file) need a small guarded ALTER TABLE. Safe to call repeatedly and
# a no-op on fresh databases (the new column is created by create_all).

def _table_has_column(engine, table: str, column: str) -> bool:
    insp = inspect(engine)
    if not insp.has_table(table):
        return False
    return column in {c["name"] for c in insp.get_columns(table)}


def run_schema_migrations(target_engine=None):
    """
    Apply small idempotent schema changes (additive columns only).

    Pass target_engine to migrate an arbitrary database (used by tests);
    defaults to this module's configured engine. Never drops or rewrites data.

    Only alters tables that already exist (create_all creates the full, current
    schema for fresh databases), so this is a safe no-op on new installs.
    """
    eng = target_engine or engine
    try:
        insp = inspect(eng)
        if insp.has_table("schedule") and not _table_has_column(eng, "schedule", "reason"):
            with eng.begin() as conn:
                conn.execute(text("ALTER TABLE schedule ADD COLUMN reason TEXT"))
            logger.info("MIGRATION: added column schedule.reason")
    except Exception as exc:  # pragma: no cover - defensive; DB may be unreachable
        logger.warning("MIGRATION: could not verify/add schedule.reason: %s", exc)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
