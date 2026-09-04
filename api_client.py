import os
import streamlit as st
import requests

# Environment-driven backend URL. Localhost is only the development fallback;
# production deployments set API_URL to the deployed FastAPI service (Render).
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
CONNECT_TIMEOUT_SECONDS = 5

try:
    _gemini_timeout_ms = int(os.getenv("GEMINI_TIMEOUT_MS", "60000"))
except ValueError:
    _gemini_timeout_ms = 60000
READ_TIMEOUT_SECONDS = max(15, min(130, _gemini_timeout_ms // 1000 + 10))
REQUEST_TIMEOUT = (CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS)

def get_headers():
    headers = {}
    if "token" in st.session_state and st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    return headers

def _safe_get(url, **kwargs):
    try:
        kwargs.setdefault("timeout", REQUEST_TIMEOUT)
        return requests.get(url, **kwargs)
    except requests.exceptions.RequestException:
        return None

def _safe_post(url, **kwargs):
    try:
        kwargs.setdefault("timeout", REQUEST_TIMEOUT)
        return requests.post(url, **kwargs)
    except requests.exceptions.RequestException:
        return None

def create_family(family_name, creator_name, password):
    return _safe_post(
        f"{API_URL}/create_family",
        json={"family_name": family_name, "creator_name": creator_name, "password": password}
    )

def join_family(group_code, username, password):
    return _safe_post(
        f"{API_URL}/join_family",
        json={"group_code": group_code, "username": username, "password": password}
    )

def login(username, password):
    return _safe_post(
        f"{API_URL}/login",
        data={"username": username, "password": password}
    )

def get_me():
    return _safe_get(f"{API_URL}/me", headers=get_headers())

def get_members(group_code):
    return _safe_get(f"{API_URL}/group/{group_code}/members", headers=get_headers())

get_group_members = get_members  # alias

def get_home_stats(group_code):
    return _safe_get(f"{API_URL}/group/{group_code}/stats", headers=get_headers())

def get_ratings(group_code):
    return _safe_get(f"{API_URL}/ratings/{group_code}", headers=get_headers())

def get_dishes(group_code):
    return _safe_get(f"{API_URL}/dishes/{group_code}", headers=get_headers())

def get_schedule(group_code):
    """GET /group/{group_code}/schedule — current (latest) saved plan with reasons."""
    return _safe_get(f"{API_URL}/group/{group_code}/schedule", headers=get_headers())


def generate_schedule(group_code: str):
    """POST /group/{group_code}/schedule/generate — AI-assisted plan generation."""
    return _safe_post(
        f"{API_URL}/group/{group_code}/schedule/generate",
        headers=get_headers(),
    )


def replace_day(group_code: str, week: int, day: str):
    """POST /group/{group_code}/schedule/replace — replace ONE day of the saved plan."""
    return _safe_post(
        f"{API_URL}/group/{group_code}/schedule/replace",
        headers=get_headers(),
        json={"week": int(week), "day": day},
    )
