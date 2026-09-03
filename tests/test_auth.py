"""
tests/test_auth.py

Authentication/security regression tests.

Backend (FastAPI + TestClient, isolated SQLite via conftest):
  1. successful registration (create + join)
  2. duplicate registration rejected with "Username already taken"
  3. successful login with the correct password issues a JWT
  4. login with an incorrect password is rejected
  5. login with a nonexistent username fails with the SAME generic error
  6. registration passwords are stored as bcrypt hashes, never plaintext
  7. a JWT is issued only after the correct password
  8. invalid / expired JWTs are rejected
  9. an authenticated user can access their own group's endpoints
 10. an authenticated user cannot access another group's endpoints (403)

Streamlit UI (AppTest + mocked api_client, no network):
  - The auth screen offers a real "Log In" option that calls /login only
  - Wrong password on Log In shows "Invalid username or password" and never
    "username already taken"; the user is not logged in
  - Logging in with an unknown username does NOT silently register an account
  - Join Family (registration) still rejects a duplicate username with
    "Username already taken" and hints at Log In
"""

import os
import sys
from datetime import timedelta

import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient  # noqa: E402

import bcrypt  # noqa: E402
import api_client  # noqa: E402

from backend.main import app, create_access_token  # noqa: E402
from backend.database import SessionLocal  # noqa: E402
from backend import models as m  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

class _Resp:
    """Minimal stand-in for requests.Response used by the UI tests."""

    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _create_family(client, creator, password="pw123456"):
    resp = client.post(
        "/create_family",
        json={"family_name": f"{creator} Family", "creator_name": creator, "password": password},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["group_code"]


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _user(username):
    s = SessionLocal()
    try:
        return s.query(m.User).filter(m.User.username == username).first()
    finally:
        s.close()


def _user_count(username):
    s = SessionLocal()
    try:
        return s.query(m.User).filter(m.User.username == username).count()
    finally:
        s.close()


# --------------------------------------------------------------------------- #
# 1. Registration
# --------------------------------------------------------------------------- #

def test_registration_creates_family_and_user(client):
    code = _create_family(client, "reg_creator")
    assert code

    # Join (registration) for a NEW member works too
    resp = client.post(
        "/join_family",
        json={"group_code": code, "username": "reg_joiner", "password": "memberpw1"},
    )
    assert resp.status_code == 200, resp.text

    for name in ("reg_creator", "reg_joiner"):
        u = _user(name)
        assert u is not None
        assert u.group_code == code


def test_duplicate_registration_rejected(client):
    code = _create_family(client, "dup_creator")

    # Same creator name via create_family
    resp = client.post(
        "/create_family",
        json={"family_name": "Other Family", "creator_name": "dup_creator", "password": "whatever1"},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Username already taken"

    # Existing username via join_family
    resp = client.post(
        "/join_family",
        json={"group_code": code, "username": "dup_creator", "password": "whatever1"},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Username already taken"


# --------------------------------------------------------------------------- #
# 3/4/5. Login — correct, incorrect, nonexistent
# --------------------------------------------------------------------------- #

def test_login_correct_password_succeeds(client):
    code = _create_family(client, "login_ok")
    resp = client.post("/login", data={"username": "login_ok", "password": "pw123456"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert body["group_code"] == code


def test_login_incorrect_password_rejected(client):
    _create_family(client, "login_badpw")
    resp = client.post("/login", data={"username": "login_badpw", "password": "not-the-password"})
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid username or password"
    assert "access_token" not in resp.json()


def test_login_nonexistent_username_same_generic_error(client):
    _create_family(client, "login_leakcheck")

    # Nonexistent username and wrong password must produce identical errors so
    # login never reveals whether a username exists.
    ghost = client.post("/login", data={"username": "no_such_user_xyz", "password": "anything1"})
    wrong = client.post("/login", data={"username": "login_leakcheck", "password": "anything1"})
    assert ghost.status_code == 401
    assert wrong.status_code == 401
    assert ghost.json()["detail"] == wrong.json()["detail"] == "Invalid username or password"

    # /login is never allowed to answer "username already taken"
    assert ghost.json()["detail"] != "Username already taken"


# --------------------------------------------------------------------------- #
# 6. Passwords are hashed, never plaintext
# --------------------------------------------------------------------------- #

def test_registration_password_is_hashed_not_plaintext(client):
    _create_family(client, "hash_check", password="super-secret-pw")
    u = _user("hash_check")
    assert u is not None
    assert u.hashed_password != "super-secret-pw"
    assert "super-secret-pw" not in u.hashed_password
    # valid bcrypt hash that verifies against the original password
    assert u.hashed_password.startswith("$2")
    assert bcrypt.checkpw(b"super-secret-pw", u.hashed_password.encode("utf-8"))


# --------------------------------------------------------------------------- #
# 7/8. JWTs
# --------------------------------------------------------------------------- #

def test_jwt_issued_only_after_correct_password(client):
    code = _create_family(client, "jwt_only")
    bad = client.post("/login", data={"username": "jwt_only", "password": "wrong-password"})
    assert bad.status_code == 401
    assert "access_token" not in bad.json()

    good = client.post("/login", data={"username": "jwt_only", "password": "pw123456"})
    assert good.status_code == 200
    assert good.json()["access_token"]


def test_invalid_and_expired_tokens_rejected(client):
    code = _create_family(client, "jwt_invalid")
    login = client.post("/login", data={"username": "jwt_invalid", "password": "pw123456"})
    token = login.json()["access_token"]

    # valid token works
    assert client.get("/me", headers=_auth(token)).status_code == 200

    # garbage token rejected
    assert client.get("/me", headers=_auth("not.a.real.token")).status_code == 401
    # forged-looking token with a bad signature rejected
    assert client.get("/me", headers=_auth("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJqd3RfaW52YWxpZCJ9.forged")).status_code == 401

    # expired token rejected
    expired = create_access_token(
        {"sub": "jwt_invalid", "groupId": code},
        expires_delta=timedelta(minutes=-5),
    )
    assert client.get("/me", headers=_auth(expired)).status_code == 401


# --------------------------------------------------------------------------- #
# 9/10. Group access
# --------------------------------------------------------------------------- #

def test_authenticated_user_accesses_own_group(client):
    code = _create_family(client, "own_group")
    token = client.post("/login", data={"username": "own_group", "password": "pw123456"}).json()["access_token"]

    assert client.get("/me", headers=_auth(token)).json()["group_code"] == code
    assert client.get(f"/group/{code}/stats", headers=_auth(token)).status_code == 200
    assert client.get(f"/group/{code}/members", headers=_auth(token)).status_code == 200
    assert client.get(f"/group/{code}/schedule", headers=_auth(token)).status_code == 200


def test_user_cannot_access_another_group(client):
    code_a = _create_family(client, "iso_a")
    code_b = _create_family(client, "iso_b")
    token_a = client.post("/login", data={"username": "iso_a", "password": "pw123456"}).json()["access_token"]

    # user from group A cannot reach B's endpoints
    for path in (f"/group/{code_b}/stats", f"/group/{code_b}/members", f"/group/{code_b}/schedule"):
        resp = client.get(path, headers=_auth(token_a))
        assert resp.status_code == 403, f"{path} -> {resp.status_code}"
    assert client.get(f"/group/{code_b}/stats", headers=_auth(token_a)).json()["detail"] == "You do not belong to this group"

    # and cannot generate a plan for B either
    resp = client.post(f"/group/{code_b}/schedule/generate", headers=_auth(token_a))
    assert resp.status_code == 403


# --------------------------------------------------------------------------- #
# Streamlit UI — Log In vs Join Family (registration)
# --------------------------------------------------------------------------- #

EMPTY_STATS = {"total_dishes": 0, "avg_rating": 0, "best_dish": "N/A", "fatigue_dish": "N/A", "week": 1, "schedule": []}


def _ss(at, key, default=None):
    """session_state access helper (AppTest's session_state has no .get)."""
    return at.session_state[key] if key in at.session_state else default


def _fake_ui(monkeypatch, login_resp=None, join_resp=None):
    """
    Fake every api_client call the auth screen and post-login Home page make.
    Patches stay active for the whole test (monkeypatch reverts at test end).
    """
    monkeypatch.setattr(api_client, "login", lambda *a, **k: login_resp if login_resp is not None else _Resp(401, {"detail": "Invalid username or password"}))
    monkeypatch.setattr(api_client, "join_family", lambda *a, **k: join_resp if join_resp is not None else _Resp(200, {"message": "Joined family successfully"}))
    monkeypatch.setattr(api_client, "get_members", lambda *a, **k: _Resp(200, []))
    monkeypatch.setattr(api_client, "get_home_stats", lambda *a, **k: _Resp(200, EMPTY_STATS))

    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("main.py", default_timeout=30)
    at.run()
    return at


def test_ui_offers_login_and_logs_in_with_correct_credentials(monkeypatch):
    code, token = "ABC123", "jwt-token-123"
    at = _fake_ui(monkeypatch, login_resp=_Resp(200, {"access_token": token, "group_code": code, "token_type": "bearer"}))
    assert not at.exception, at.exception

    # Log In tab is available
    radio = at.radio[0]
    assert any("Log In" in opt for opt in radio.options)
    radio.set_value("🔐 Log In").run()
    assert not at.exception, at.exception

    # fill credentials and submit
    at.text_input[0].set_value("mom")
    at.text_input[1].set_value("correct-password")
    login_buttons = [b for b in at.button if "Log In" in b.label]
    assert len(login_buttons) == 1
    login_buttons[0].click().run()
    assert not at.exception, at.exception

    assert at.session_state["logged_in"] is True
    assert at.session_state["token"] == token
    assert at.session_state["group_code"] == code
    assert at.session_state["user_name"] == "mom"


def test_ui_login_wrong_password_shows_generic_error_only(monkeypatch):
    at = _fake_ui(monkeypatch)  # login -> 401 by default
    assert not at.exception, at.exception

    at.radio[0].set_value("🔐 Log In").run()
    at.text_input[0].set_value("mom")
    at.text_input[1].set_value("wrong-password")
    [b for b in at.button if "Log In" in b.label][0].click().run()
    assert not at.exception, at.exception

    errors = [e.value for e in at.error]
    assert any("Invalid username or password" in e for e in errors)
    assert not any("Username already taken" in e for e in errors)
    assert _ss(at, "logged_in", False) is False
    assert not _ss(at, "token", "")


def test_ui_login_unknown_username_does_not_register(monkeypatch):
    # login -> 401 (unknown user); registration endpoints must NOT be touched
    before = _user_count("ghost_login_user")
    at = _fake_ui(monkeypatch)  # login 401 by default
    monkeypatch.setattr(api_client, "join_family", lambda *a, **k: pytest.fail("join_family must not be called for login"))
    at.radio[0].set_value("🔐 Log In").run()
    at.text_input[0].set_value("ghost_login_user")
    at.text_input[1].set_value("any-password")
    [b for b in at.button if "Log In" in b.label][0].click().run()
    assert not at.exception, at.exception

    errors = [e.value for e in at.error]
    assert any("Invalid username or password" in e for e in errors)
    assert _user_count("ghost_login_user") == before  # no silent registration
    assert _ss(at, "logged_in", False) is False


def test_ui_join_family_duplicate_username_rejected_with_hint(monkeypatch):
    at = _fake_ui(monkeypatch, join_resp=_Resp(400, {"detail": "Username already taken"}))
    assert not at.exception, at.exception

    # Join Family is the registration tab — pick it
    at.radio[0].set_value("🚪 Join Family (new member)").run()
    at.text_input[0].set_value("ABC123")   # family code
    at.text_input[1].set_value("mom")      # existing username
    at.text_input[2].set_value("some-password")
    [b for b in at.button if "Join Family" in b.label][0].click().run()
    assert not at.exception, at.exception

    errors = [e.value for e in at.error]
    assert any("Username already taken" in e for e in errors)
    assert _ss(at, "logged_in", False) is False
    assert not _ss(at, "token", "")


def test_ui_login_error_never_reports_username_taken_after_logout_reentry(monkeypatch):
    """A returning user who logs out and back in with a wrong password sees the
    generic error — never 'username already taken'."""
    # First login succeeds (returning account), the second (wrong password) fails
    calls = {"n": 0}

    def _stateful_login(username, password):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp(200, {"access_token": "tok1", "group_code": "ABC123", "token_type": "bearer"})
        return _Resp(401, {"detail": "Invalid username or password"})

    at = _fake_ui(monkeypatch)  # login -> 401 by default, overridden below
    monkeypatch.setattr(api_client, "login", _stateful_login)
    at.radio[0].set_value("🔐 Log In").run()
    at.text_input[0].set_value("mom")
    at.text_input[1].set_value("right-password")
    [b for b in at.button if "Log In" in b.label][0].click().run()
    assert not at.exception, at.exception
    assert at.session_state["logged_in"] is True

    # Log out
    logout_buttons = [b for b in at.sidebar.button if "Logout" in b.label]
    assert len(logout_buttons) == 1
    logout_buttons[0].click().run()
    assert not at.exception, at.exception
    assert at.session_state["logged_in"] is False

    # Try logging back in with the WRONG password
    at.radio[0].set_value("🔐 Log In").run()
    at.text_input[0].set_value("mom")
    at.text_input[1].set_value("wrong-password")
    [b for b in at.button if "Log In" in b.label][0].click().run()
    assert not at.exception, at.exception

    errors = [e.value for e in at.error]
    assert any("Invalid username or password" in e for e in errors)
    assert not any("Username already taken" in e for e in errors)
    assert at.session_state["logged_in"] is False
    assert _ss(at, "token", "") == ""
