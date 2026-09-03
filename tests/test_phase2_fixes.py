"""
tests/test_phase2_fixes.py

End-to-end (FastAPI TestClient + isolated SQLite) and regression tests for the
Phase 2 hardening work:

- AI generation succeeds and persists reasons
- Gemini failure falls back to the deterministic recommender
- Invalid / hallucinated Gemini dishes and duplicate days fall back
- Schedule regeneration replaces the same week without duplicate rows
- Week returned by the API matches the persisted week
- Persisted reasons survive reload (API + db.py schedule read)
- Group isolation (403) and unauthenticated access (401) on group endpoints
- /stats operates on the current week in Monday-Sunday order
- Daily feedback / rating rows persist through db.save_data
- API_URL is environment-driven (localhost only as dev fallback)
- Schema migration is idempotent

Gemini is always mocked — no real API calls are made.
"""

import os
import sys
import importlib
from unittest.mock import patch

import pandas as pd
import pytest
from sqlalchemy import create_engine, inspect, text

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app, VALID_DAYS  # noqa: E402
from backend.database import SessionLocal, run_schema_migrations  # noqa: E402
from backend import models as m  # noqa: E402
from backend.llm_service import LLMServiceError  # noqa: E402

import db as dbmod  # noqa: E402  (db.py helpers under test)


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

REASON_TEMPLATE = "High family rating with good weekly variety."


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

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


def _plan_from_candidates(candidates, holder=None, invalid_dish_id=None, duplicate_days=False):
    """Deterministic valid plan built from the actual candidate list Gemini receives."""
    plan = []
    for i in range(7):
        day = VALID_DAYS[i]
        if duplicate_days and i > 0:
            day = VALID_DAYS[i - 1]  # two entries claim the same day
        dish_id = candidates[i]["dish_id"] if i < len(candidates) else candidates[0]["dish_id"]
        if invalid_dish_id is not None and i == 0:
            dish_id = invalid_dish_id
        plan.append({
            "day": day,
            "dish_id": dish_id,
            "reason": f"{REASON_TEMPLATE} Chosen for balance.",
        })
    if holder is not None:
        holder["plan"] = plan
    return plan


def _generate(client, code, token, plan_fn):
    holder = {}

    def _side_effect(candidates, family_context):
        if plan_fn is not None:
            return plan_fn(candidates, family_context)
        return _plan_from_candidates(candidates, holder)

    with patch("backend.llm_service.generate_weekly_plan", side_effect=_side_effect):
        resp = client.post(f"/group/{code}/schedule/generate", headers=_auth(token))
    return resp, holder


def _rows_for(code, week=None):
    s = SessionLocal()
    try:
        q = s.query(m.ScheduleEntry).filter(m.ScheduleEntry.group_code == code)
        if week is not None:
            q = q.filter(m.ScheduleEntry.week == week)
        rows = q.all()
        return rows, sorted({r.week for r in rows})
    finally:
        s.close()


# --------------------------------------------------------------------------- #
# 1. AI generation succeeds; reasons persisted; week matches
# --------------------------------------------------------------------------- #

def test_ai_generation_succeeds_and_persists_reasons(client):
    code, token = _create_family(client, "creator_ai_ok")
    _add_dishes(code)

    resp, holder = _generate(client, code, token, plan_fn=None)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["week"] == 1
    assert data["ai_generated"] is True
    assert data["fallback_used"] is False
    assert data.get("fallback_notice") is None
    schedule = data["schedule"]
    assert len(schedule) == 7
    assert [e["day"] for e in schedule] == VALID_DAYS  # Monday-Sunday order
    assert all(e["reason"] for e in schedule)

    # week returned by the API matches the persisted week and db helper
    rows, weeks = _rows_for(code)
    assert len(rows) == 7
    assert weeks == [1]
    assert dbmod.get_current_planning_week(code) == data["week"]
    assert dbmod.get_plan_weeks(code) == [1]

    # reasons stored on the ScheduleEntry rows
    for entry in schedule:
        row = next(r for r in rows if r.day == entry["day"])
        assert row.reason == entry["reason"]

    # reasons survive a reload through the API
    reloaded = client.get(f"/group/{code}/schedule", headers=_auth(token))
    assert reloaded.status_code == 200
    reloaded_data = reloaded.json()
    assert reloaded_data["week"] == 1
    assert [(e["day"], e["reason"]) for e in reloaded_data["schedule"]] == [
        (e["day"], e["reason"]) for e in schedule
    ]


def test_regeneration_replaces_same_week_without_duplicates(client):
    code, token = _create_family(client, "creator_regen")
    _add_dishes(code)

    def _reversed(candidates, family_context):
        plan = []
        for i in range(7):
            plan.append({
                "day": VALID_DAYS[i],
                "dish_id": candidates[6 - i]["dish_id"],  # different assignment
                "reason": "Alternative grounded pick for variety.",
            })
        return plan

    resp1, _ = _generate(client, code, token, plan_fn=None)
    resp2, _ = _generate(client, code, token, plan_fn=_reversed)
    assert resp1.status_code == 200 and resp2.status_code == 200

    # same week regenerated in place — still exactly 7 rows for week 1
    assert resp1.json()["week"] == 1
    assert resp2.json()["week"] == 1
    rows, weeks = _rows_for(code)
    assert len(rows) == 7
    assert weeks == [1]

    dish_day = {(r.day, r.dish_id) for r in rows}
    assert len(dish_day) == 7  # no duplicate day/dish rows


# --------------------------------------------------------------------------- #
# 2. Gemini failures and invalid output fall back to the recommender
# --------------------------------------------------------------------------- #

def test_gemini_failure_falls_back_and_persists(client):
    code, token = _create_family(client, "creator_fb_err")
    _add_dishes(code)

    def _boom(candidates, family_context):
        raise LLMServiceError("simulated API quota exceeded")

    resp, _ = _generate(client, code, token, plan_fn=_boom)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["ai_generated"] is False
    assert data["fallback_used"] is True
    assert "fallback_notice" in data
    assert data["week"] == 1
    schedule = data["schedule"]
    assert len(schedule) == 7
    assert [e["day"] for e in schedule] == VALID_DAYS

    rows, weeks = _rows_for(code)
    assert len(rows) == 7
    assert weeks == [1]
    # fallback entries carry the deterministic grounded reason
    assert all(r.reason for r in rows)

    # fallback plan still reloads with reasons intact
    reloaded = client.get(f"/group/{code}/schedule", headers=_auth(token))
    assert reloaded.status_code == 200
    assert len(reloaded.json()["schedule"]) == 7


def test_invalid_gemini_dish_id_falls_back(client):
    code, token = _create_family(client, "creator_fb_invalid")
    _add_dishes(code)

    resp, _ = _generate(
        client, code, token,
        plan_fn=lambda c, fc: _plan_from_candidates(c, invalid_dish_id=999999),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["fallback_used"] is True
    # the persisted fallback never contains the hallucinated dish
    rows, _ = _rows_for(code)
    assert all(r.dish_id != 999999 for r in rows)
    assert len(rows) == 7


def test_duplicate_days_from_gemini_falls_back(client):
    code, token = _create_family(client, "creator_fb_dupday")
    _add_dishes(code)

    resp, _ = _generate(
        client, code, token,
        plan_fn=lambda c, fc: _plan_from_candidates(c, duplicate_days=True),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["fallback_used"] is True
    rows, _ = _rows_for(code)
    assert len(rows) == 7


# --------------------------------------------------------------------------- #
# 3. Auth: group isolation and unauthenticated access
# --------------------------------------------------------------------------- #

def test_unauthenticated_group_access_rejected(client):
    code, _ = _create_family(client, "creator_noauth")
    assert client.get(f"/group/{code}/members").status_code == 401
    assert client.get(f"/group/{code}/stats").status_code == 401
    assert client.get(f"/group/{code}/schedule").status_code == 401
    assert client.post(f"/group/{code}/schedule/generate").status_code == 401


def test_group_isolation_returns_403(client):
    code_a, token_a = _create_family(client, "creator_iso_a")
    code_b, token_b = _create_family(client, "creator_iso_b")

    # user A cannot read or mutate group B
    assert client.get(f"/group/{code_b}/members", headers=_auth(token_a)).status_code == 403
    assert client.get(f"/group/{code_b}/stats", headers=_auth(token_a)).status_code == 403
    assert client.get(f"/group/{code_b}/schedule", headers=_auth(token_a)).status_code == 403
    assert client.post(f"/group/{code_b}/schedule/generate", headers=_auth(token_a)).status_code == 403

    # user A can still access their own group (valid behavior preserved)
    assert client.get(f"/group/{code_a}/members", headers=_auth(token_a)).status_code == 200
    assert client.get(f"/group/{code_a}/stats", headers=_auth(token_a)).status_code == 200
    assert client.get(f"/group/{code_b}/members", headers=_auth(token_b)).status_code == 200


# --------------------------------------------------------------------------- #
# 4. /stats current week + ordering; week helpers
# --------------------------------------------------------------------------- #

def test_stats_uses_current_week_monday_to_sunday(client):
    code, token = _create_family(client, "creator_stats")
    _add_dishes(code)
    resp, _ = _generate(client, code, token, plan_fn=None)
    assert resp.status_code == 200

    stats = client.get(f"/group/{code}/stats", headers=_auth(token))
    assert stats.status_code == 200
    data = stats.json()
    assert data["week"] == 1
    assert [item["date"] for item in data["schedule"]] == VALID_DAYS
    assert len(data["schedule"]) == 7

    # simulate a later week existing — /stats and /schedule must show only the
    # latest week, never a mix of old rows
    names = list(DISH_NAMES)
    s = SessionLocal()
    try:
        dish_ids = [d.id for d in s.query(m.Dish).filter(m.Dish.group_code == code).order_by(m.Dish.id).all()]
        for i, day in enumerate(VALID_DAYS):
            s.add(m.ScheduleEntry(group_code=code, dish_id=dish_ids[i], week=3, day=day,
                                  reason="Legacy week three reason."))
        s.commit()
    finally:
        s.close()

    stats = client.get(f"/group/{code}/stats", headers=_auth(token))
    data = stats.json()
    assert data["week"] == 3
    assert [item["date"] for item in data["schedule"]] == VALID_DAYS
    assert len(data["schedule"]) == 7  # week 1 rows are not mixed in

    sch = client.get(f"/group/{code}/schedule", headers=_auth(token)).json()
    assert sch["week"] == 3
    assert len(sch["schedule"]) == 7
    assert all(e["reason"] == "Legacy week three reason." for e in sch["schedule"])


def test_week_helpers_fresh_group(client):
    code, _ = _create_family(client, "creator_weekhelpers")
    # nothing saved yet -> next logical week is 1, no plan weeks listed
    assert dbmod.get_current_planning_week(code) == 1
    assert dbmod.get_plan_weeks(code) == []


# --------------------------------------------------------------------------- #
# 5. Daily feedback / ratings persist through db.save_data
# --------------------------------------------------------------------------- #

def test_ratings_persist_through_save_data(client):
    code, _ = _create_family(client, "creator_feedback")
    _add_dishes(code, names=["Palak Paneer"])

    dish = None
    s = SessionLocal()
    try:
        dish = s.query(m.Dish).filter(m.Dish.group_code == code).first().name
    finally:
        s.close()

    df1 = pd.DataFrame([{
        "dish": dish, "user": "creator_feedback", "rating": 4.0,
        "week": 1, "day": "Monday", "comment": "Really tasty",
    }])
    dbmod.save_data(df1, f"ratings_{code}.csv")

    loaded = dbmod.load_data(f"ratings_{code}.csv")
    assert len(loaded) == 1
    assert float(loaded.iloc[0]["rating"]) == 4.0
    assert loaded.iloc[0]["comment"] == "Really tasty"

    # submitting feedback again updates the same row (unique group/dish/user)
    df2 = pd.DataFrame([{
        "dish": dish, "user": "creator_feedback", "rating": 5.0,
        "week": 2, "day": "Tuesday", "comment": "Even better second time",
    }])
    dbmod.save_data(df2, f"ratings_{code}.csv")

    loaded = dbmod.load_data(f"ratings_{code}.csv")
    assert len(loaded) == 1  # no duplicate rows
    # Re-rating updates the same row via the EMA policy (BETA = 0.5), so the
    # stored value converges toward the new feedback instead of replacing it:
    # 0.5*5.0 + 0.5*4.0 = 4.5. Week/day/comment still track the submission.
    assert float(loaded.iloc[0]["rating"]) == 4.5
    assert loaded.iloc[0]["comment"] == "Even better second time"
    assert int(loaded.iloc[0]["week"]) == 2
    assert loaded.iloc[0]["day"] == "Tuesday"


# --------------------------------------------------------------------------- #
# 6. Manual schedule save persists reasons through db.save_data
# --------------------------------------------------------------------------- #

def test_schedule_reason_persists_through_save_data(client):
    code, _ = _create_family(client, "creator_savedf")
    _add_dishes(code, names=["Masala Dosa", "Idli Sambar"])

    s = SessionLocal()
    try:
        ids = {d.name: d.id for d in s.query(m.Dish).filter(m.Dish.group_code == code).all()}
    finally:
        s.close()

    df = pd.DataFrame([
        {"Day": "Monday", "Dish": "Masala Dosa", "Reason": "Manual reason one.", "week": 1},
        {"Day": "Tuesday", "Dish": "Idli Sambar", "Reason": "Manual reason two.", "week": 1},
    ])
    dbmod.save_data(df, f"schedule_{code}_week1.csv")

    loaded = dbmod.load_data(f"schedule_{code}_week1.csv")
    assert len(loaded) == 2
    assert "Reason" in loaded.columns
    assert list(loaded["Day"]) == ["Monday", "Tuesday"]
    assert list(loaded["Reason"]) == ["Manual reason one.", "Manual reason two."]


# --------------------------------------------------------------------------- #
# 7. Not enough dishes -> clean 422 (no AI call, nothing persisted)
# --------------------------------------------------------------------------- #

def test_generate_without_enough_dishes_returns_422(client):
    code, token = _create_family(client, "creator_short")
    resp = client.post(f"/group/{code}/schedule/generate", headers=_auth(token))
    assert resp.status_code == 422
    rows, _ = _rows_for(code)
    assert rows == []


# --------------------------------------------------------------------------- #
# 8. API_URL is environment driven
# --------------------------------------------------------------------------- #

def test_api_url_is_environment_driven(monkeypatch):
    import api_client

    monkeypatch.setenv("API_URL", "https://meal-api.example.com")
    importlib.reload(api_client)
    assert api_client.API_URL == "https://meal-api.example.com"
    assert "127.0.0.1" not in api_client.API_URL

    # localhost remains the development fallback only
    monkeypatch.delenv("API_URL")
    importlib.reload(api_client)
    assert api_client.API_URL == "http://127.0.0.1:8000"


# --------------------------------------------------------------------------- #
# 9. Schema migration is idempotent and additive
# --------------------------------------------------------------------------- #

def test_schema_migration_adds_reason_idempotently(tmp_path):
    db_file = tmp_path / "old.db"
    eng = create_engine(f"sqlite:///{db_file}")

    with eng.begin() as conn:
        conn.execute(text(
            "CREATE TABLE schedule ("
            " id INTEGER PRIMARY KEY, group_code VARCHAR, dish_id INTEGER,"
            " week INTEGER, day VARCHAR)"
        ))
    cols = {c["name"] for c in inspect(eng).get_columns("schedule")}
    assert "reason" not in cols

    run_schema_migrations(target_engine=eng)  # first run adds the column
    cols = {c["name"] for c in inspect(eng).get_columns("schedule")}
    assert "reason" in cols

    run_schema_migrations(target_engine=eng)  # second run is a safe no-op
    cols = {c["name"] for c in inspect(eng).get_columns("schedule")}
    assert "reason" in cols
    assert "id" in cols  # nothing was dropped
