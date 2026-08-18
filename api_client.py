import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

def get_headers():
    headers = {}
    if "token" in st.session_state and st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    return headers

def _safe_get(url, **kwargs):
    try:
        return requests.get(url, **kwargs)
    except requests.exceptions.ConnectionError:
        return None

def _safe_post(url, **kwargs):
    try:
        return requests.post(url, **kwargs)
    except requests.exceptions.ConnectionError:
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
    return _safe_get(f"{API_URL}/schedule/{group_code}", headers=get_headers())
