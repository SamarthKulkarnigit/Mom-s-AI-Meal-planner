"""Submission-gate adversarial tests for isolation and grounded output."""

from datetime import timedelta
from unittest.mock import patch

from fastapi.testclient import TestClient

import db
from backend import models as m
from backend.database import SessionLocal
from backend.main import app, create_access_token, _validate_llm_plan, VALID_DAYS
from backend.llm_service import _timeout_ms


def _create(client, name):
    response = client.post("/create_family", json={
        "family_name": f"{name} Family", "creator_name": name, "password": "pw123456",
    })
    code = response.json()["group_code"]
    token = client.post("/login", data={"username": name, "password": "pw123456"}).json()["access_token"]
    return code, token


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_tokens_and_cross_family_schedule_are_rejected():
    with TestClient(app) as client:
        code_a, token_a = _create(client, "ship_sec_a")
        code_b, _ = _create(client, "ship_sec_b")
        assert client.get(f"/group/{code_a}/schedule").status_code == 401
        assert client.get(f"/group/{code_a}/schedule", headers=_auth("bad.token")).status_code == 401
        expired = create_access_token({"sub": "ship_sec_a", "groupId": code_a}, timedelta(seconds=-1))
        assert client.get(f"/group/{code_a}/schedule", headers=_auth(expired)).status_code == 401
        assert client.get(f"/group/{code_b}/schedule", headers=_auth(token_a)).status_code == 403
        assert client.post(f"/group/{code_b}/schedule/generate", headers=_auth(token_a)).status_code == 403
        assert client.post(
            f"/group/{code_b}/schedule/replace", headers=_auth(token_a),
            json={"week": 1, "day": "Monday"},
        ).status_code == 403


def test_direct_mutations_reject_another_familys_identity_and_dish():
    with TestClient(app) as client:
        code_a, _ = _create(client, "ship_data_a")
        code_b, _ = _create(client, "ship_data_b")

    assert db.rate_dish(code_b, "Rajma Chawal", 5, user_name="ship_data_a") is False
    assert db.suggest_dish(code_b, "Private B Dish", "ship_data_b") is True
    assert db.vote_dish(code_b, "Private B Dish", "ship_data_a") is False
    # Daily feedback uses the same membership-checked rating path.
    assert db.rate_dish(
        code_b, "Rajma Chawal", 5, user_name="ship_data_a",
        week=1, day="Monday", comment="cross-family attempt",
    ) is False

    session = SessionLocal()
    try:
        assert session.query(m.Rating).filter(m.Rating.group_code == code_b).count() == 0
        assert session.query(m.PollVote).filter(m.PollVote.group_code == code_b).count() == 0
        assert session.query(m.Dish).filter(m.Dish.group_code == code_a, m.Dish.name == "Private B Dish").count() == 0
    finally:
        session.close()


def test_nonexistent_group_is_safe_404_for_matching_authenticated_user():
    with TestClient(app) as client:
        code, token = _create(client, "ship_missing_group")
        session = SessionLocal()
        try:
            session.query(m.Group).filter(m.Group.group_code == code).delete()
            session.commit()
        finally:
            session.close()
        assert client.get(f"/group/{code}/schedule", headers=_auth(token)).status_code == 404


def test_fabricated_and_excessive_reasons_are_rejected():
    candidate_ids = set(range(1, 8))
    good = [
        {"day": day, "dish_id": i + 1, "reason": "Adds variety to the weekly rotation."}
        for i, day in enumerate(VALID_DAYS)
    ]
    fabricated = [dict(entry) for entry in good]
    fabricated[0]["reason"] = "Mom rated this 4.9 and it cures diabetes."
    try:
        _validate_llm_plan(fabricated, candidate_ids, candidate_ids)
        assert False, "fabricated reason should fail"
    except ValueError as exc:
        assert "unsupported" in str(exc)

    excessive = [dict(entry) for entry in good]
    excessive[0]["reason"] = "variety " * 40
    try:
        _validate_llm_plan(excessive, candidate_ids, candidate_ids)
        assert False, "excessive reason should fail"
    except ValueError as exc:
        assert "too long" in str(exc)


def test_gemini_timeout_is_bounded(monkeypatch):
    monkeypatch.setenv("GEMINI_TIMEOUT_MS", "-50")
    assert _timeout_ms() == 1_000
    monkeypatch.setenv("GEMINI_TIMEOUT_MS", "999999")
    assert _timeout_ms() == 120_000
