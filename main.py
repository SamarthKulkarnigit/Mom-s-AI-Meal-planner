import streamlit as st
import api_client

from polls import poll_ui
from ratings import rate_dish, view_ratings
from schedule import meal_scheduler_ui
from analytics import analytics_ui
from daily_feedback import daily_feedback_ui

st.set_page_config(
    page_title="Meal Planner",
    page_icon="🍴",
    layout="wide"
)

# -----------------------------------
# CUSTOM STYLE INJECTION (PREMIUM DESIGN)
# -----------------------------------
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-top: 4px solid #ff4b4b;
        transition: transform 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Header card styling */
    .premium-header {
        background: linear-gradient(135deg, #ff4b4b 0%, #ff7675 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 20px rgba(255, 75, 75, 0.2);
    }
    
    /* Button transition */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: scale(1.02);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------
# APP TITLE
# -----------------------------------

st.title("🍴 Meal Planner App")

# -----------------------------------
# SESSION LOGIN
# -----------------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# -----------------------------------
# AUTH SCREEN
# -----------------------------------

def _complete_login(username: str, password: str, fallback_code: str = "") -> str:
    """
    Call the backend /login endpoint and, only on success, populate the JWT
    session state. Never logs in without a valid token.

    Returns "ok" (logged in), "bad" (invalid credentials), or "down"
    (backend unreachable).
    """
    resp = api_client.login(username, password)
    if resp is None:
        return "down"
    if resp.status_code == 200:
        data = resp.json()
        st.session_state.token = data.get("access_token", "")
        st.session_state.group_code = data.get("group_code") or fallback_code
        st.session_state.user_name = username
        st.session_state.logged_in = True
        return "ok"
    return "bad"


if not st.session_state.logged_in:

    st.subheader("👨‍👩‍👧‍👦 Welcome")

    auth_mode = st.radio(
        "Choose an option",
        [
            "🏠 Create Family",
            "🚪 Join Family (new member)",
            "🔐 Log In"
        ]
    )

    # -----------------------------------
    # CREATE FAMILY (register a new family)
    # -----------------------------------

    if auth_mode == "🏠 Create Family":
        family_name = st.text_input("Family Name")
        creator_name = st.text_input("Your Name")
        password = st.text_input("Password", type="password")
        if st.button("🏠 Create Family"):
            if family_name and creator_name and password:
                resp = api_client.create_family(family_name, creator_name, password)
                if resp is None:
                    st.error("Cannot reach the server. Please try again.")
                elif resp.status_code == 200:
                    code = resp.json()["group_code"]
                    if _complete_login(creator_name, password, fallback_code=code) == "ok":
                        st.success(f"✅ Family created! Your group code is: {code}")
                        st.rerun()
                    else:
                        st.success(f"✅ Family created! Your group code is: {code}")
                        st.info("Your account is ready — use 🔐 Log In to enter.")
                else:
                    st.error(resp.json().get("detail", "Failed to create family"))
            else:
                st.error("Please fill all fields.")

    # -----------------------------------
    # JOIN FAMILY (register a NEW account)
    # -----------------------------------

    elif auth_mode == "🚪 Join Family (new member)":
        st.caption("Joining creates a new account for your family. Already registered? Use 🔐 Log In.")
        join_code = st.text_input("Enter Family Code")
        join_user = st.text_input("Your Name")
        password = st.text_input("Password", type="password")
        if st.button("🚪 Join Family"):
            if join_code and join_user and password:
                resp = api_client.join_family(join_code, join_user, password)
                if resp is None:
                    st.error("Cannot reach the server. Please try again.")
                elif resp.status_code == 200:
                    if _complete_login(join_user, password, fallback_code=join_code) == "ok":
                        st.success("✅ Joined family successfully")
                        st.rerun()
                    else:
                        st.success("✅ Your account was created.")
                        st.info("Use 🔐 Log In to enter.")
                else:
                    detail = resp.json().get("detail", "Invalid family code or username taken")
                    if detail == "Username already taken":
                        st.error("Username already taken — if you already have an account, use 🔐 Log In instead.")
                    else:
                        st.error(detail)
            else:
                st.error("Please fill all fields.")

    # -----------------------------------
    # LOG IN (existing account only)
    # -----------------------------------

    else:
        login_user = st.text_input("Username")
        login_password = st.text_input("Password", type="password")
        if st.button("🔐 Log In"):
            if login_user and login_password:
                status = _complete_login(login_user, login_password)
                if status == "ok":
                    st.success(f"✅ Welcome back, {login_user}!")
                    st.rerun()
                elif status == "down":
                    st.error("Cannot reach the server. Please try again.")
                else:
                    # Generic on purpose: never reveals whether a username
                    # exists and never reports "username already taken".
                    st.error("Invalid username or password")
            else:
                st.error("Please fill in both username and password.")
    st.stop()

# Revalidate the Streamlit session before any page uses the direct SQLAlchemy
# compatibility layer. /me is authoritative for identity and family scope.
identity_resp = api_client.get_me()
if identity_resp is None:
    st.error("The backend is temporarily unavailable. Please try again shortly.")
    st.stop()
if identity_resp.status_code != 200:
    st.session_state.logged_in = False
    st.session_state.token = ""
    st.session_state.pop("user_name", None)
    st.session_state.pop("group_code", None)
    st.error("Your session has expired. Please log in again.")
    st.stop()

identity = identity_resp.json()
st.session_state.user_name = identity["username"]
st.session_state.group_code = identity["group_code"]

# -----------------------------------
# SESSION VARIABLES
# -----------------------------------

group_code = st.session_state.group_code
user_name = st.session_state.user_name

# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.title("🍴 Mom's AI Meal Planner")

st.sidebar.success(
    f"Logged in as {user_name}"
)

st.sidebar.caption(
    f"Group: {group_code}"
)

st.sidebar.markdown("---")

if st.sidebar.button("🚪 Logout"):

    st.session_state.logged_in = False
    st.session_state.token = ""
    st.session_state.pop("user_name", None)
    st.session_state.pop("group_code", None)

    st.rerun()

# -----------------------------------
# FAMILY ROSTER
# -----------------------------------

st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍👩‍👧‍👦 Family Roster")
members_resp = api_client.get_members(group_code)
if members_resp and members_resp.status_code == 200:
    members = members_resp.json()
    for m in members:
        st.sidebar.markdown(f"👤 **{m['username']}**")
else:
    st.sidebar.info("No members registered yet.")

# -----------------------------------
# NAVIGATION
# -----------------------------------

page = st.sidebar.radio(
    "📂 Navigation",
    [
        "🏠 Home",
        "🗳 Polls",
        "⭐ Rate Dishes",
        "📖 View Ratings",
        "🗓 Meal Scheduler",
        "💬 Daily Feedback",
        "📊 Analytics"
    ]
)

# -----------------------------------
# HOME PAGE
# -----------------------------------

if page == "🏠 Home":

    st.markdown(
        """
        <div class="premium-header">
            <h1 style="color: white; margin: 0; font-size: 2.5em;">🍴 Mom's AI Meal Planner</h1>
            <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0; font-size: 1.1em;">
                Smart, adaptive weekly meal recommendations powered by family feedback and temporal learning.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    import pandas as pd
    total_dishes = 0
    avg_rating = 0.0
    best_dish = "N/A"
    fatigue_dish = "N/A"
    schedule_df = None

    # Fetch stats from backend
    stats_resp = api_client.get_home_stats(group_code)
    if stats_resp and stats_resp.status_code == 200:
        stats = stats_resp.json()
        total_dishes = stats.get("total_dishes", 0)
        avg_rating = stats.get("avg_rating", 0.0)
        best_dish = stats.get("best_dish", "N/A")
        fatigue_dish = stats.get("fatigue_dish", "N/A")
        schedule_items = stats.get("schedule", [])
        if schedule_items:
            schedule_df = pd.DataFrame(schedule_items)

    # metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "🍽 Total Dishes",
        total_dishes
    )

    col2.metric(
        "⭐ Avg Rating",
        round(avg_rating, 2)
    )

    col3.metric(
        "🔥 Favorite Dish",
        best_dish
    )

    col4.metric(
        "📉 Fatigue Risk",
        fatigue_dish
    )

    st.markdown("---")

    st.subheader("📅 Current Weekly Plan")

    if (
        schedule_df is not None
        and not schedule_df.empty
    ):

        st.dataframe(
            schedule_df,
            width="stretch"
        )

    else:

        st.info(
            "Generate a weekly schedule to see recommendations."
        )

    st.markdown("---")

    st.info(
        "🤖 Recommendations adapt using family feedback and historical meal trends."
    )

# -----------------------------------
# POLLS
# -----------------------------------

elif page == "🗳 Polls":

    poll_ui(
        group_code,
        user_name
    )

# -----------------------------------
# RATE DISHES
# -----------------------------------

elif page == "⭐ Rate Dishes":

    rate_dish(
        group_code,
        user_name
    )

# -----------------------------------
# VIEW RATINGS
# -----------------------------------

elif page == "📖 View Ratings":

    view_ratings(
        group_code
    )

# -----------------------------------
# MEAL SCHEDULER
# -----------------------------------

elif page == "🗓 Meal Scheduler":

    meal_scheduler_ui(
        group_code
    )

# -----------------------------------
# DAILY FEEDBACK
# -----------------------------------

elif page == "💬 Daily Feedback":

    daily_feedback_ui(
        group_code,
        user_name
    )

# -----------------------------------
# ANALYTICS
# -----------------------------------

elif page == "📊 Analytics":

    analytics_ui(
        group_code
    )


