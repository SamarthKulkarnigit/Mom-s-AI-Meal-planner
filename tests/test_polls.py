"""
tests/test_polls.py

Regression tests for the Polls "Approve / Cast Vote" flow:

- Votes on pending suggestions persist and count toward majority
- Re-voting does not double count
- An approved (majority) dish is promoted into the rotation and becomes
  visible to the recommendation engine
- Unapproved (pending) dishes stay hidden from the rotation/recommender
- Votes are scoped to the voter's own group
- The Streamlit Approve/Vote button actually persists a vote
- Voting for an existing rotation dish keeps working
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

APP_PATH = Path(__file__).resolve().parents[1] / "main.py"

import db  # noqa: E402
from backend.database import SessionLocal  # noqa: E402
from backend import models as m  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _make_group(creator, other_users=()):
    """API-style family: Group + registered User accounts (no Member rows)."""
    code = db.generate_group_code()
    s = SessionLocal()
    try:
        s.add(m.Group(group_code=code, family_name=f"{creator} Family", creator=creator))
        s.add(m.User(username=creator, hashed_password="x", group_code=code))
        for u in other_users:
            s.add(m.User(username=u, hashed_password="x", group_code=code))
        s.commit()
    finally:
        s.close()
    return code


def _count(model, **filters):
    s = SessionLocal()
    try:
        return s.query(model).filter_by(**filters).count()
    finally:
        s.close()


def _poll_votes(group_code):
    return _count(m.PollVote, group_code=group_code)


def _pending(group_code):
    return db.load_data(f"pending_{group_code}.csv")


def _rotation(group_code):
    return db.load_data(f"dishes_{group_code}.csv")


# --------------------------------------------------------------------------- #
# db-layer behaviour
# --------------------------------------------------------------------------- #

def test_vote_on_pending_suggestion_persists_and_counts():
    code = _make_group("poll_voter_alice", other_users=("poll_voter_bob",))
    assert db.suggest_dish(code, "Veg Biryani", "poll_voter_alice") is True

    # alice votes -> recorded, counts once even when re-voted
    assert db.vote_dish(code, "Veg Biryani", "poll_voter_alice") is True
    assert db.vote_dish(code, "Veg Biryani", "poll_voter_alice") is True
    assert _poll_votes(code) == 1
    standings = db.get_poll_results(code)
    assert len(standings) == 1
    assert standings.iloc[0]["dish"] == "Veg Biryani"
    assert int(standings.iloc[0]["votes"]) == 1

    # still pending (no majority yet) — dish row exists but is not in rotation
    assert len(_pending(code)) == 1
    dish = None
    s = SessionLocal()
    try:
        dish = s.query(m.Dish).filter(m.Dish.group_code == code, m.Dish.name == "Veg Biryani").first()
    finally:
        s.close()
    assert dish is not None and dish.source == "Pending"
    assert "Veg Biryani" not in _rotation(code)["dish"].tolist()
    assert "Veg Biryani" not in db.get_dishes(code)


def test_pending_dish_becomes_visible_to_recommender_after_majority():
    import ml_recommender

    code = _make_group("poll_approve_alice", other_users=("poll_approve_bob",))
    db.add_dish(code, "Rajma Chawal", source="Poll")  # one approved rotation dish

    assert db.suggest_dish(code, "Paneer Tikka", "poll_approve_alice") is True

    # one vote (1 of 2) -> pending, NOT a recommendation candidate
    db.vote_dish(code, "Paneer Tikka", "poll_approve_alice")
    candidates = ml_recommender.get_candidates_for_group(code, n_candidates=20)
    assert {c["dish_name"] for c in candidates} == {"Rajma Chawal"}

    # bob's vote reaches majority -> promoted into the rotation
    assert db.vote_dish(code, "Paneer Tikka", "poll_approve_bob") is True
    assert len(_pending(code)) == 0
    assert _poll_votes(code) == 0  # votes cleared after promotion

    assert "Paneer Tikka" in _rotation(code)["dish"].tolist()
    assert "Paneer Tikka" in db.get_dishes(code)
    candidates = ml_recommender.get_candidates_for_group(code, n_candidates=20)
    assert {c["dish_name"] for c in candidates} == {"Rajma Chawal", "Paneer Tikka"}

    s = SessionLocal()
    try:
        dish = s.query(m.Dish).filter(m.Dish.group_code == code, m.Dish.name == "Paneer Tikka").first()
        assert dish.source == "Poll"
    finally:
        s.close()


def test_single_member_family_approves_own_suggestion():
    code = _make_group("poll_sole_member")
    assert db.suggest_dish(code, "Veg Pizza", "poll_sole_member") is True
    assert db.vote_dish(code, "Veg Pizza", "poll_sole_member") is True
    assert len(_pending(code)) == 0
    assert "Veg Pizza" in db.get_dishes(code)


def test_vote_on_unknown_dish_returns_false_and_writes_nothing():
    code = _make_group("poll_unknown_dish")
    assert db.vote_dish(code, "Dish That Does Not Exist", "poll_unknown_dish") is False
    assert _poll_votes(code) == 0
    assert _count(m.Dish, group_code=code) == 0
    assert _count(m.PendingSuggestion, group_code=code) == 0


def test_votes_are_scoped_to_own_group():
    code_a = _make_group("poll_iso_a_user", other_users=("poll_iso_a2",))
    code_b = _make_group("poll_iso_b_user", other_users=("poll_iso_b2",))
    db.suggest_dish(code_a, "Pav Bhaji", "poll_iso_a_user")
    db.suggest_dish(code_b, "Misal Pav", "poll_iso_b_user")

    assert db.vote_dish(code_a, "Pav Bhaji", "poll_iso_a_user") is True

    # group A wrote only to A; group B untouched
    assert _poll_votes(code_a) == 1
    assert _poll_votes(code_b) == 0
    assert _count(m.Dish, group_code=code_b) == 0
    assert len(_pending(code_b)) == 1
    assert len(_pending(code_a)) == 1

    # group B voting its own dish leaves A alone
    assert db.vote_dish(code_b, "Misal Pav", "poll_iso_b_user") is True
    assert _poll_votes(code_a) == 1
    assert _poll_votes(code_b) == 1
    assert len(_pending(code_a)) == 1


def test_voting_existing_rotation_dish_still_works():
    code = _make_group("poll_rotation_voter", other_users=("poll_rotation_voter2",))
    db.add_dish(code, "Rajma Chawal", source="Poll")
    assert db.vote_dish(code, "Rajma Chawal", "poll_rotation_voter") is True
    assert _poll_votes(code) == 1
    # dish remains in the rotation and keeps its Poll source
    assert "Rajma Chawal" in db.get_dishes(code)
    s = SessionLocal()
    try:
        dish = s.query(m.Dish).filter(m.Dish.group_code == code, m.Dish.name == "Rajma Chawal").first()
        assert dish.source == "Poll"
    finally:
        s.close()


# --------------------------------------------------------------------------- #
# Streamlit UI button end-to-end
# --------------------------------------------------------------------------- #

def test_approve_vote_button_persists_vote(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import api_client

    code = _make_group("poll_ui_alice", other_users=("poll_ui_bob",))
    assert db.suggest_dish(code, "Veg Biryani", "poll_ui_alice") is True
    monkeypatch.setattr(api_client, "get_me", lambda: type("Resp", (), {
        "status_code": 200,
        "json": lambda self: {"username": "poll_ui_alice", "group_code": code},
    })())

    # Run the real app logged in, then navigate to the Polls page. The API is
    # not running, but api_client degrades gracefully to None (handled by UI).
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["logged_in"] = True
    at.session_state["group_code"] = code
    at.session_state["user_name"] = "poll_ui_alice"
    at.session_state["token"] = ""
    at.run()
    assert not at.exception, at.exception

    at.sidebar.radio[0].set_value("🗳 Polls").run()
    assert not at.exception, at.exception

    # find the Approve / Vote button for the pending dish
    vote_buttons = [b for b in at.button if "Approve / Vote" in b.label]
    assert len(vote_buttons) == 1, "expected one Approve/Vote button for the pending dish"
    vote_buttons[0].click().run()
    assert not at.exception, at.exception

    # vote persisted in the DB; still pending (1 of 2 votes)
    assert _poll_votes(code) == 1
    assert len(_pending(code)) == 1
    success = [s for s in at.success if "Vote registered" in s.value]
    assert success, "expected a success message after voting"
