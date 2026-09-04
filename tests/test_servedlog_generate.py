"""
tests/test_servedlog_generate.py

Regression tests for the local HTTP 500: backend/main.py generate_schedule()
builds family_context with `sl.dish.name`, but models.ServedLog previously had
no `dish` relationship (only dish_id), so any group WITH served-log history
raised `AttributeError: 'ServedLog' object has no attribute 'dish'` before the
Gemini try/except, producing HTTP 500.

Fix: models.ServedLog now declares `dish = relationship("Dish")` (ORM-only, no
schema change), matching ScheduleEntry/Rating/PollOption.

These tests fail on the pre-fix code (generate returns 500) and pass after.

Gemini is mocked everywhere — no real API calls are made.
"""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app, VALID_DAYS  # noqa: E402
from backend.database import SessionLocal  # noqa: E402
from backend import models as m  # noqa: E402


DISH_NAMES = [
    "Rajma Chawal",
    "Dal Makhani",
    "Palak Paneer",
    "Masala Dosa",
    "Idli Sambar",
    "Veg Pizza",
    "Pasta Arrabiata",
    "Chole Bhature",
    "Veg Hakka Noodles",
    "Pav Bhaji",
    "Dhokla",
    "Veg Burger",
    "Tacos",
    "Gulab Jamun",
]


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _create_family(client, creator):
    resp = client.post(
        "/create_family",
        json={"family_name": f"{creator} Family", "creator_name": creator, "password": "pw123456"},
    )
    assert resp.status_code == 200, resp.text
    code = resp.json()["group_code"]
    login = client.post("/login", data={"username": creator, "password": "pw123456"})
    assert login.status_code == 200, login.text
    return code, login.json()["access_token"]


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _add_dishes(code, names=None):
    s = SessionLocal()
    try:
        for n in (names or DISH_NAMES):
            s.add(m.Dish(group_code=code, name=n, source="Poll"))
        s.commit()
    finally:
        s.close()


def _add_served_logs(code, names, week=1):
    """Insert ServedLog rows (with valid dish_ids) exactly like the legacy data."""
    s = SessionLocal()
    try:
        dishes = {d.name: d.id for d in s.query(m.Dish).filter(m.Dish.group_code == code).all()}
        for i, name in enumerate(names):
            s.add(m.ServedLog(group_code=code, dish_id=dishes[name], day=VALID_DAYS[i], week=week))
        s.commit()
        return list(dishes.values())
    finally:
        s.close()


# --------------------------------------------------------------------------- #
# 1. ORM-level: sl.dish.name resolves through the relationship
# --------------------------------------------------------------------------- #

def test_servedlog_dish_relationship_resolves_name():
    """The relationship fix: a ServedLog row exposes .dish.name (was AttributeError)."""
    with TestClient(app) as client:
        code, _ = _create_family(client, "creator_rel_check")
        _add_dishes(code, names=["Masala Dosa"])
        _add_served_logs(code, ["Masala Dosa"], week=1)

        s = SessionLocal()
        try:
            sl = s.query(m.ServedLog).filter(m.ServedLog.group_code == code).one()
            # the fix: attribute access resolves the dish; pre-fix this raises
            assert sl.dish is not None
            assert sl.dish.name == "Masala Dosa"
        finally:
            s.close()


# --------------------------------------------------------------------------- #
# 2. End-to-end: generate works when ServedLog history exists (mocked Gemini)
# --------------------------------------------------------------------------- #

def test_generate_succeeds_with_servedlog_history(client):
    """
    The exact production failure: group WITH served_logs rows calls the weekly
    generate endpoint. Pre-fix this 500s with AttributeError; post-fix it
    returns a valid AI plan and passes the served dishes to Gemini as context.
    """
    code, token = _create_family(client, "creator_servedgen")
    _add_dishes(code)
    served_names = ["Masala Dosa", "Idli Sambar", "Veg Pizza", "Gulab Jamun"]
    _add_served_logs(code, served_names, week=1)

    captured = {}

    def _mock_llm(candidates, family_context):
        captured["family_context"] = family_context
        plan = []
        for i in range(7):
            plan.append({
                "day": VALID_DAYS[i],
                "dish_id": candidates[i]["dish_id"],
                "reason": "Grounded reason for the served-history regression test.",
            })
        return plan

    with patch("backend.llm_service.generate_weekly_plan", side_effect=_mock_llm):
        resp = client.post(f"/group/{code}/schedule/generate", headers=_auth(token))

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["week"] == 1
    assert data["ai_generated"] is True
    assert data["fallback_used"] is False
    assert len(data["schedule"]) == 7
    assert [e["day"] for e in data["schedule"]] == VALID_DAYS

    # the ServedLog dishes reached the LLM context as real dish names (the code
    # path that previously crashed with AttributeError)
    recent = captured["family_context"].get("recent_dishes", [])
    assert sorted(recent) == sorted(served_names)

    # exactly 7 rows persisted, one per day
    s = SessionLocal()
    try:
        rows = s.query(m.ScheduleEntry).filter(m.ScheduleEntry.group_code == code).all()
    finally:
        s.close()
    assert len(rows) == 7
    assert len({r.day for r in rows}) == 7


def test_generate_fallback_still_works_with_servedlog_history(client):
    """
    With ServedLog history present AND Gemini failing, the deterministic
    fallback still returns a 7-day plan (no 500, no interference from the
    served-log data in the recommender path).
    """
    code, token = _create_family(client, "creator_servedfb")
    _add_dishes(code)
    _add_served_logs(code, ["Masala Dosa", "Idli Sambar"], week=1)

    def _boom(candidates, family_context):
        raise RuntimeError("simulated Gemini API failure")

    with patch("backend.llm_service.generate_weekly_plan", side_effect=_boom):
        resp = client.post(f"/group/{code}/schedule/generate", headers=_auth(token))

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["ai_generated"] is False
    assert data["fallback_used"] is True
    assert len(data["schedule"]) == 7
    assert [e["day"] for e in data["schedule"]] == VALID_DAYS