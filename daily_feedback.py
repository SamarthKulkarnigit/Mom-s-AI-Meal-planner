import streamlit as st
import pandas as pd
from datetime import datetime
import db

def daily_feedback_ui(group_code, user_name):
    st.subheader("💬 Daily Meal Feedback")

    # ----------------------------------------
    # DYNAMIC WEEK SELECTOR  (DB-backed)
    # ----------------------------------------
    active_week = db.get_current_active_week(group_code)

    # All weeks that have saved schedule rows (from the database, not files)
    available_weeks = db.get_plan_weeks(group_code)
    if not available_weeks:
        available_weeks = [1]

    default_index = 0
    if active_week in available_weeks:
        default_index = available_weeks.index(active_week)
    else:
        default_index = len(available_weeks) - 1

    current_week = st.selectbox(
        "📅 Select Week to View/Rate",
        available_weeks,
        index=default_index,
        key="feedback_week_selector"
    )

    ratings_df = db.load_data(f"ratings_{group_code}.csv")

    # ----------------------------------------
    # LOAD CURRENT WEEK SCHEDULE
    # ----------------------------------------
    schedule_file = f"schedule_{group_code}_week{current_week}.csv"
    schedule_df = db.load_data(schedule_file)

    if schedule_df is None or schedule_df.empty:
        st.warning(f"No weekly plan found for Week {current_week} yet.")
        return

    # ----------------------------------------
    # DETECT TODAY
    # ----------------------------------------
    today = datetime.today().strftime("%A")

    # normalize column names
    schedule_df.columns = [str(c).strip().lower() for c in schedule_df.columns]

    today_row = schedule_df[schedule_df["day"].astype(str).str.lower() == today.lower()]

    if today_row.empty:
        st.warning(f"No dish scheduled for {today} (Week {current_week}).")
        
        # Fallback: Let user select a day to rate instead
        st.markdown("### 🗓️ Or select any scheduled meal from this week:")
        selected_day = st.selectbox("Choose a day to rate", schedule_df["day"].astype(str).str.title().tolist())
        today_row = schedule_df[schedule_df["day"].astype(str).str.lower() == selected_day.lower()]
        if not today_row.empty:
            today = selected_day.title()
        else:
            return

    # ----------------------------------------
    # GET TODAY'S DISH
    # ----------------------------------------
    today_dish = today_row.iloc[0]["dish"]

    st.success(f"🍽️ Scheduled Meal for {today}: **{today_dish}**")

    st.markdown("---")

    # ----------------------------------------
    # RATING INPUT
    # ----------------------------------------
    rating = st.slider(
        "⭐ Rate today's meal",
        min_value=1.0,
        max_value=5.0,
        value=4.0,
        step=0.5
    )

    # ----------------------------------------
    # OPTIONAL COMMENT
    # ----------------------------------------
    comment = st.text_area("📝 Optional Feedback / Comments")
    already_rated = False

    if ratings_df is not None and not ratings_df.empty:
        ratings_df.columns = [str(c).strip().lower() for c in ratings_df.columns]
        
        # Ensure we have a helper numeric week column for type safety
        ratings_df["week_num"] = pd.to_numeric(ratings_df["week"], errors="coerce")
        existing = ratings_df[
            (ratings_df["user"].astype(str).str.strip().str.lower() == user_name.lower()) &
            (ratings_df["dish"].astype(str).str.strip().str.lower() == today_dish.lower()) &
            (ratings_df["week_num"] == int(current_week))
        ]

        if not existing.empty:
            already_rated = True
            st.info("💡 You already rated this meal. Submitting again will update your previous feedback.")

    # ----------------------------------------
    # SUBMIT
    # ----------------------------------------
    if st.button("✅ Submit Feedback"):
        new_row = pd.DataFrame([{
            "dish": today_dish,
            "user": user_name,
            "rating": rating,
            "week": int(current_week),
            "day": today,
            "comment": comment
        }])

        saved = db.rate_dish(
            group_code, today_dish, rating, user_name=user_name,
            week=int(current_week), day=today, comment=comment, overwrite=True,
        )
        if saved:
            st.success("🎉 Feedback submitted successfully!")
            st.balloons()
        else:
            st.error("Could not save feedback. Please refresh and try again.")
