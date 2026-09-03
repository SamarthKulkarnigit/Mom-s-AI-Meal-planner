# schedule.py
import streamlit as st
import pandas as pd
import db
import api_client


DAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def _schedule_file(group_code, week):
    # db.py compatibility shim name — persistence is fully SQLAlchemy-backed.
    return f"schedule_{group_code}_week{int(week)}.csv"


def _served_file(group_code):
    return f"served_log_{group_code}.csv"


def _mirror_to_served_log(group_code, week, plan_entries):
    """
    Mirror a saved week's Day/Dish rows into the served history (used by
    analytics and by the recommender for variety context). db.save_data is
    DB-backed and de-dupes on (group, dish, day, week).
    """
    if not plan_entries:
        return
    served_df = pd.DataFrame(
        [
            {"dish": e.get("dish_name", ""), "day": e["day"], "week": int(week)}
            for e in plan_entries
        ]
    )
    try:
        old_served = db.load_data(_served_file(group_code))
        if old_served is None or old_served.empty:
            combined = served_df.copy()
        else:
            combined = pd.concat([old_served, served_df], ignore_index=True)
        db.save_data(combined, _served_file(group_code))
    except Exception:
        pass  # non-critical; the plan itself was already persisted on the backend


def meal_scheduler_ui(group_code):

    st.subheader("🗓️ Weekly Meal Planner")

    # -----------------------------
    # CURRENT WEEK (single source of truth shared with the backend:
    # latest saved plan week, or week 1 when nothing is saved yet)
    # -----------------------------
    current_week = db.get_current_planning_week(group_code)

    existing_weeks = db.get_plan_weeks(group_code)
    has_saved_plan = bool(existing_weeks)

    if has_saved_plan:
        st.info(f"📅 Current Week: {current_week}")
    else:
        st.info("📅 No plan saved yet — generating will create Week 1.")

    # -----------------------------
    # LOAD DISHES (DB-backed)
    # -----------------------------
    polls_file = f"poll_{group_code}.csv"
    polls_df = db.load_data(polls_file)

    dishes_list = []

    if polls_df is not None and not polls_df.empty:
        dish_col = polls_df.columns[0]
        dishes_list = (
            polls_df[dish_col]
            .astype(str)
            .dropna()
            .unique()
            .tolist()
        )
    else:
        dishes_file = f"dishes_{group_code}.csv"
        ddf = db.load_data(dishes_file)
        if ddf is not None and not ddf.empty:
            first = ddf.columns[0]
            dishes_list = (
                ddf[first]
                .astype(str)
                .dropna()
                .unique()
                .tolist()
            )

    st.write("✅ Dishes available:", len(dishes_list))

    # -----------------------------
    # MANUAL SCHEDULER
    # -----------------------------
    st.markdown("### ✏️ Manually Assign Dishes")

    manual_schedule = {}
    options = ["--Select--"] + dishes_list

    for day in DAYS:
        choice = st.selectbox(
            f"{day}'s Meal",
            options,
            key=f"{group_code}_{current_week}_{day}",
        )
        if choice != "--Select--":
            manual_schedule[day] = choice

    # -----------------------------
    # SAVE MANUAL PLAN
    # -----------------------------
    if st.button("💾 Save Weekly Meal Plan (Manual)"):

        if manual_schedule:
            schedule_df = pd.DataFrame(
                list(manual_schedule.items()),
                columns=["Day", "Dish"],
            )
            schedule_df["week"] = current_week

            # Saves to the current week (replaces it in place if it exists).
            db.save_data(schedule_df, _schedule_file(group_code, current_week))

            # append to served history
            _mirror_to_served_log(group_code, current_week, [
                {"day": d, "dish_name": dish}
                for d, dish in manual_schedule.items()
            ])

            st.success(f"✅ Manual plan saved for week {current_week}")
        else:
            st.warning("Please assign at least one dish.")

    st.markdown("---")

    # -----------------------------
    # AI SCHEDULER  (Phase 2 — backend-driven)
    # -----------------------------
    st.markdown("### ✨ AI-Assisted Weekly Meal Plan")
    st.caption(
        "Powered by Gemini + the family recommendation engine. "
        "Falls back to the smart recommender automatically if AI is unavailable."
    )

    if st.button("✨ Generate This Week's Plan", use_container_width=True, type="primary"):
        with st.spinner("🤖 Generating your personalised weekly plan…"):
            try:
                resp = api_client.generate_schedule(group_code)
            except Exception as exc:
                st.error(f"Could not reach the backend: {exc}")
                resp = None

        if resp is None:
            st.error("❌ Backend is unavailable. Please make sure the API server is running.")
        elif resp.status_code == 403:
            st.error("❌ You are not authorised to generate a plan for this group.")
        elif resp.status_code == 422:
            detail = resp.json().get("detail", "Unknown error")
            st.warning(f"⚠️ {detail}")
        elif resp.status_code != 200:
            detail = ""
            try:
                detail = resp.json().get("detail", resp.text[:200])
            except Exception:
                detail = resp.text[:200]
            st.error(f"❌ Plan generation failed (HTTP {resp.status_code}): {detail}")
        else:
            data = resp.json()
            ai_generated = data.get("ai_generated", False)
            fallback_used = data.get("fallback_used", not ai_generated)
            week = data.get("week", "?")
            plan = data.get("schedule", data.get("plan", []))

            if fallback_used:
                notice = data.get(
                    "fallback_notice",
                    "AI planning is temporarily unavailable. We generated a recommendation-based plan instead.",
                )
                st.warning(f"⚠️ {notice}")
            else:
                st.success(f"✅ AI-generated plan for week {week} saved!")

            if plan:
                st.markdown(f"#### 📅 Week {week} — {'✨ AI-Curated' if ai_generated else '🤖 Recommendation-Based'} Plan")

                for entry in plan:
                    day = entry.get("day", "?")
                    dish = entry.get("dish_name", entry.get("dish_id", "?"))
                    reason = entry.get("reason", "") or ""

                    with st.container():
                        col_day, col_dish = st.columns([1, 3])
                        with col_day:
                            st.markdown(f"**{day.upper()}**")
                        with col_dish:
                            st.markdown(f"🍽️ **{dish}**")
                        if reason:
                            st.caption(f"💡 *{reason}*")
                    st.divider()

                # mirror the plan into served history for analytics/recommender
                _mirror_to_served_log(group_code, week, plan)

            else:
                st.warning("No plan entries returned from the backend.")

    # -----------------------------
    # VIEW SAVED PLANS  (DB-backed — no filesystem dependence)
    # -----------------------------
    st.markdown("### 📅 View Current / Past Plans")

    schedule_df = db.load_data(_schedule_file(group_code, current_week))
    if schedule_df is not None and not schedule_df.empty:
        st.markdown(f"**Saved plan for week {current_week}:**")
        st.dataframe(schedule_df, width="stretch")

        # -----------------------------
        # REPLACE A SINGLE DAY  (other six days stay untouched)
        # -----------------------------
        st.markdown("### 🔄 Replace a Single Day")
        st.caption(
            "Swap one meal without regenerating the week. "
            "The recommendation engine finds grounded candidates and AI explains the pick."
        )

        plan_days = schedule_df["Day"].astype(str).tolist() if "Day" in schedule_df.columns else []
        replace_result = None
        for day in plan_days:
            col_day, col_btn = st.columns([3, 1])
            with col_day:
                st.markdown(f"**{day}**")
            with col_btn:
                if st.button(
                    f"Replace {day}",
                    key=f"replace_{group_code}_{current_week}_{day}",
                    use_container_width=True,
                ):
                    with st.spinner(f"Replacing {day}…"):
                        try:
                            replace_result = api_client.replace_day(group_code, current_week, day)
                        except Exception as exc:
                            st.error(f"Could not reach the backend: {exc}")
                            replace_result = None

        if replace_result is not None:
            if replace_result.status_code == 200:
                data = replace_result.json()
                new_day = data.get("day", "?")
                new_dish = data.get("dish_name", "?")
                new_reason = data.get("reason", "") or ""
                if data.get("fallback_used"):
                    st.warning("⚠️ AI selection is temporarily unavailable — chose the top recommended dish instead.")
                else:
                    st.success(f"✅ {new_day} replaced with {new_dish}!")
                st.markdown(f"**{new_day}: {new_dish}**")
                if new_reason:
                    st.caption(f"💡 *{new_reason}*")
                # Refresh the displayed plan (only that day changed server-side).
                fresh = db.load_data(_schedule_file(group_code, current_week))
                if fresh is not None and not fresh.empty:
                    st.markdown(f"**Updated plan for week {current_week}:**")
                    st.dataframe(fresh, width="stretch")
            elif replace_result.status_code == 403:
                st.error("❌ You are not authorised to modify this group's plan.")
            elif replace_result.status_code == 422:
                detail = replace_result.json().get("detail", "Unknown error")
                st.warning(f"⚠️ {detail}")
            else:
                detail = ""
                try:
                    detail = replace_result.json().get("detail", replace_result.text[:200])
                except Exception:
                    detail = replace_result.text[:200]
                st.error(f"❌ Replace failed (HTTP {replace_result.status_code}): {detail}")
    else:
        st.info(f"No saved plan for week {current_week} yet.")

    existing = db.get_plan_weeks(group_code)
    if existing:
        st.markdown("**Past weeks available:** " + ", ".join(map(str, existing)))
        chosen = st.selectbox(
            "Open saved week:",
            ["--Select--"] + [str(w) for w in existing],
            key=f"open_week_{group_code}",
        )

        if chosen != "--Select--":
            df = db.load_data(_schedule_file(group_code, int(chosen)))
            if df is not None and not df.empty:
                st.dataframe(df, width="stretch")
            else:
                st.info("Saved week found but empty.")
