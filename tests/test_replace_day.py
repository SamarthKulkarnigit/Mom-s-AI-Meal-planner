"""
tests/test_replace_day.py

Focused tests for POST /group/{group_code}/schedule/replace — swapping ONE day
of a saved weekly plan without touching the other six days.

- A. Successful replacement changes exactly one day
- B. Other six days remain unchanged (dish + reason)
- C. A dish already used elsewhere in the week can never be selected
- D. Gemini receives grounded candidate evidence (used dishes excluded)
- E. Valid Gemini replacement passes
- F. Invalid Gemini dish_id triggers the deterministic fallback
- G. Malformed Gemini output triggers fallback
- H. Gemini/API failure triggers fallback
- I. Cross-group request returns 403
- J. Unauthenticated request returns 401
- K. No candidate available is handled cleanly (422)

Gemini is fully mocked — no real API calls.
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
    "Rajma Chawal", "Dal Makhani", "Palak Paneer", "Masala Dosa", "Idli Sambar",
    "Veg Pizza", "Pasta Arrabiata", "Chole Bhature", "Veg Hakka Noodles", "Pav Bhaji",
    "Dhokla", "Veg Burger", "Tacos", "Gulab Jamun",
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


def _plan_from_candidates(candidates):
    """Deterministic valid 7-day plan from the candidate list Gemini receives."""
    return [
        {"day": VALID_DAYS[i], "dish_id": candidates[i]["dish_id"], "reason": "High family rating with good weekly variety."}
        for i in range(7)
    ]


def _generate_plan(client, code, token):
    with patch("backend.llm_service.generate_weekly_plan", side_effect=_plan_from_candidates):
        resp = client.post(f"/group/{code}/schedule/generate", headers=_auth(token))
    assert resp.status_code == 200, resp.text
    return resp.json()


def _week_rows(code, week=1):
    s = SessionLocal()
    try:
        rows = (
            s.query(m.ScheduleEntry)
            .filter(m.ScheduleEntry.group_code == code, m.ScheduleEntry.week == week)
            .all()
        )
        return {e.day: {"dish_id": e.dish_id, "reason": e.reason} for e in rows}
    finally:
        s.close()


def _replace(client, code, token, week, day, side_effect=None, headers=None):
    def _default(candidates, day, current_dish_name, other_dish_names, family_context):
        return {"dish_id": candidates[0]["dish_id"], "reason": "Chosen because of its high family rating."}

    with patch("backend.llm_service.replace_day_plan", side_effect=side_effect or _default):
        resp = client.post(
            f"/group/{code}/schedule/replace",
            headers=headers if headers is not None else _auth(token),
            json={"week": week, "day": day},
        )
    return resp


# --------------------------------------------------------------------------- #
# A + E. Successful replacement changes exactly one day
# --------------------------------------------------------------------------- #

def test_replace_success_changes_only_the_requested_day(client):
    code, token = _create_family(client, "replacer")
    _add_dishes(code)
    _generate_plan(client, code, token)

    before = _week_rows(code)
    monday_before = before["Monday"]

    holder = {}
    with patch("backend.llm_service.replace_day_plan") as mock_replace:
        def _side_effect(candidates, day, current_dish_name, other_dish_names, family_context):
            holder["candidates"] = candidates
            assert day == "Monday"
            return {"dish_id": candidates[0]["dish_id"], "reason": "High family rating — a great pick for Monday."}

        mock_replace.side_effect = _side_effect
        resp = client.post(
            f"/group/{code}/schedule/replace",
            headers=_auth(token),
            json={"week": 1, "day": "Monday"},
        )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["week"] == 1
    assert data["day"] == "Monday"
    assert data["ai_generated"] is True
    assert data["fallback_used"] is False
    assert data["dish_id"] == holder["candidates"][0]["dish_id"]
    assert data["reason"] == "High family rating — a great pick for Monday."
    assert len(data["schedule"]) == 7
    assert [e["day"] for e in data["schedule"]] == VALID_DAYS

    after = _week_rows(code)
    # exactly the requested day changed
    assert after["Monday"]["dish_id"] != monday_before["dish_id"]
    assert after["Monday"]["reason"] == "High family rating — a great pick for Monday."
    # other six days: dish_id AND reason untouched
    for day in VALID_DAYS[1:]:
        assert after[day] == before[day], f"{day} changed: {before[day]} -> {after[day]}"


# --------------------------------------------------------------------------- #
# D. Gemini receives grounded candidate evidence (used dishes excluded)
# --------------------------------------------------------------------------- #

def test_replace_gemini_receives_grounded_evidence_without_used_dishes(client):
    code, token = _create_family(client, "evidencecheck")
    _add_dishes(code)
    _generate_plan(client, code, token)

    used = {e["dish_id"] for e in _week_rows(code).values()}

    holder = {}
    with patch("backend.llm_service.replace_day_plan") as mock_replace:
        def _side_effect(candidates, day, current_dish_name, other_dish_names, family_context):
            holder["candidates"] = candidates
            holder["other_dishes"] = other_dish_names
            return {"dish_id": candidates[0]["dish_id"], "reason": "Chosen because of its high family rating."}

        mock_replace.side_effect = _side_effect
        resp = client.post(
            f"/group/{code}/schedule/replace",
            headers=_auth(token),
            json={"week": 1, "day": "Wednesday"},
        )

    assert resp.status_code == 200, resp.text
    cands = holder["candidates"]
    assert len(cands) >= 1
    # every candidate carries the grounded structure
    for c in cands:
        assert {"dish_id", "dish_name", "recommendation_score", "evidence"}.issubset(c)
        assert isinstance(c["evidence"], list) and c["evidence"]
    # no dish already scheduled this week may be offered to Gemini
    cand_ids = {c["dish_id"] for c in cands}
    assert cand_ids.isdisjoint(used)
    # the six other scheduled dishes are named in the request context
    assert len(holder["other_dishes"]) == 6


# --------------------------------------------------------------------------- #
# C. A dish already used elsewhere in the week can never be selected
# --------------------------------------------------------------------------- #

def test_replace_rejects_dish_used_by_another_day_and_falls_back(client):
    code, token = _create_family(client, "dupecheck")
    _add_dishes(code)
    _generate_plan(client, code, token)

    before = _week_rows(code)
    tuesday_dish = before["Tuesday"]["dish_id"]

    def _bad_side_effect(candidates, day, current_dish_name, other_dish_names, family_context):
        # Gemini picks Tuesday's dish (already scheduled this week)
        return {"dish_id": tuesday_dish, "reason": "This dish looks great."}

    resp = _replace(client, code, token, 1, "Monday", side_effect=_bad_side_effect)

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["fallback_used"] is True  # invalid pick rejected -> deterministic fallback
    assert data["dish_id"] != tuesday_dish
    after = _week_rows(code)
    used_after = {after[d]["dish_id"] for d in VALID_DAYS}
    assert len(used_after) == 7  # still all unique
    for day in VALID_DAYS[1:]:
        assert after[day] == before[day]


# --------------------------------------------------------------------------- #
# F. Invalid Gemini dish_id triggers fallback
# --------------------------------------------------------------------------- #

def test_replace_invalid_dish_id_falls_back(client):
    code, token = _create_family(client, "invalidid")
    _add_dishes(code)
    _generate_plan(client, code, token)
    before = _week_rows(code)

    resp = _replace(
        client, code, token, 1, "Monday",
        side_effect=lambda c, day, cur, others, ctx: {"dish_id": 99999, "reason": "Seems tasty."},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["fallback_used"] is True
    assert data["dish_id"] != 99999
    assert data["dish_id"] not in {before[d]["dish_id"] for d in VALID_DAYS}
    # fallback reason is the deterministic grounded one
    assert data["reason"].startswith("Chosen by the recommendation engine")


# --------------------------------------------------------------------------- #
# G. Malformed Gemini output triggers fallback
# --------------------------------------------------------------------------- #

def test_replace_malformed_output_falls_back(client):
    code, token = _create_family(client, "malformed")
    _add_dishes(code)
    _generate_plan(client, code, token)

    # not a dict at all
    resp = _replace(client, code, token, 1, "Monday", side_effect=lambda *a, **k: [{"dish_id": 1}])
    assert resp.status_code == 200
    assert resp.json()["fallback_used"] is True

    # dict missing keys
    resp2 = _replace(client, code, token, 1, "Tuesday", side_effect=lambda *a, **k: {"dish_id": 101})
    assert resp2.status_code == 200
    assert resp2.json()["fallback_used"] is True

    # non-integer dish_id
    resp3 = _replace(client, code, token, 1, "Wednesday", side_effect=lambda *a, **k: {"dish_id": "abc", "reason": "whatever."})
    assert resp3.status_code == 200
    assert resp3.json()["fallback_used"] is True

    # too-short reason
    resp4 = _replace(client, code, token, 1, "Thursday", side_effect=lambda *a, **k: {"dish_id": 101, "reason": "ok"})
    assert resp4.status_code == 200
    assert resp4.json()["fallback_used"] is True


# --------------------------------------------------------------------------- #
# H. Gemini/API failure triggers fallback
# --------------------------------------------------------------------------- #

def test_replace_api_failure_falls_back(client):
    code, token = _create_family(client, "apifail")
    _add_dishes(code)
    _generate_plan(client, code, token)
    before = _week_rows(code)

    def _boom(candidates, day, current_dish_name, other_dish_names, family_context):
        raise RuntimeError("Gemini quota exceeded")

    resp = _replace(client, code, token, 1, "Monday", side_effect=_boom)
    assert resp.status_code == 200
    data = resp.json()
    assert data["fallback_used"] is True
    assert data["dish_id"] not in {before[d]["dish_id"] for d in VALID_DAYS}
    for day in VALID_DAYS[1:]:
        assert _week_rows(code)[day] == before[day]


# --------------------------------------------------------------------------- #
# I. Cross-group request returns 403
# --------------------------------------------------------------------------- #

def test_replace_cross_group_forbidden(client):
    code_a, token_a = _create_family(client, "replacer_a")
    code_b, token_b = _create_family(client, "replacer_b")
    _add_dishes(code_a)
    _generate_plan(client, code_a, token_a)

    resp = _replace(client, code_a, token_b, 1, "Monday")
    assert resp.status_code == 403
    assert "do not belong" in resp.json()["detail"].lower()


# --------------------------------------------------------------------------- #
# J. Unauthenticated request returns 401
# --------------------------------------------------------------------------- #

def test_replace_unauthenticated_rejected(client):
    code, token = _create_family(client, "anoncheck")
    _add_dishes(code)
    _generate_plan(client, code, token)

    resp = client.post(
        f"/group/{code}/schedule/replace",
        json={"week": 1, "day": "Monday"},
    )
    assert resp.status_code == 401


# --------------------------------------------------------------------------- #
# K. No candidate available is handled cleanly
# --------------------------------------------------------------------------- #

def test_replace_no_candidate_available_returns_422(client):
    # A family with exactly 7 dishes: every dish is already scheduled this week,
    # so after exclusion there is nothing left to replace with.
    code, token = _create_family(client, "nocands")
    s = SessionLocal()
    try:
        s.query(m.Dish).filter(m.Dish.group_code == code).delete()
        s.commit()
    finally:
        s.close()
    _add_dishes(code, names=DISH_NAMES[:7])
    _generate_plan(client, code, token)

    resp = _replace(client, code, token, 1, "Monday")
    assert resp.status_code == 422
    assert "No candidate" in resp.json()["detail"]


# --------------------------------------------------------------------------- #
# Validation edge cases
# --------------------------------------------------------------------------- #

def test_replace_missing_week_or_invalid_day(client):
    code, token = _create_family(client, "edgecheck")
    _add_dishes(code)
    _generate_plan(client, code, token)

    # week that does not exist
    resp = client.post(
        f"/group/{code}/schedule/replace",
        headers=_auth(token),
        json={"week": 42, "day": "Monday"},
    )
    assert resp.status_code == 404

    # invalid day name
    resp2 = client.post(
        f"/group/{code}/schedule/replace",
        headers=_auth(token),
        json={"week": 1, "day": "Funday"},
    )
    assert resp2.status_code == 422
