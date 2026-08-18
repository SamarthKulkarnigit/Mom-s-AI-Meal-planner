# schedule.py
import os
import time
import streamlit as st
import pandas as pd
import db
from pathlib import Path


DATA_DIR = Path("data")

def _get_schedule_filename(group_code, week):
    return f"schedule_{group_code}_week{int(week)}.csv"

def _list_existing_weeks(group_code):
    files = sorted([f for f in os.listdir("data") if f.startswith(f"schedule_{group_code}_week")])
    weeks = []
    for f in files:
        try:
            part = f.split("_week")[-1].split(".csv")[0]
            weeks.append(int(part))
        except Exception:
            pass
    return sorted(list(set(weeks)))


# Try to import the ML recommender (optional)
try:
    from ml_recommender import generate_weekly_plan_for_group
    _HAS_ML = True
except Exception:
    generate_weekly_plan_for_group = None
    _HAS_ML = False


def safe_rerun():
    """
    Try to force a Streamlit rerun in a few different ways.
    - Preferred: st.experimental_rerun() or st.rerun() if available.
    - Fallback: modify query params to trigger a rerun.
    - If all fails: show a message to the user.
    """
    try:
        # new stable API (if available)
        if hasattr(st, "rerun"):
            st.rerun()
            return
    except Exception:
        pass

    try:
        # older API
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
            return
    except Exception:
        pass

    try:
        # fallback: modify query params (causes Streamlit to re-run)
        st.experimental_set_query_params(_rerun=int(time.time()))
        return
    except Exception:
        # last resort: tell the user to refresh
        st.warning("Saved — please refresh the page to see updates.")


def meal_scheduler_ui(group_code):

    st.subheader("🗓️ Weekly Meal Planner")

    # -----------------------------
    # CENTRALIZED CURRENT PLANNING WEEK
    # -----------------------------
    current_week = db.get_current_planning_week(group_code)

    st.info(
        f"📅 Current Planning Week: {current_week}"
    )

    # -----------------------------
    # FILE PATHS
    # -----------------------------
    schedule_file = _get_schedule_filename(
        group_code,
        current_week
    )

    schedule_path = DATA_DIR / schedule_file

    served_file = f"served_log_{group_code}.csv"

    # -----------------------------
    # LOAD DISHES
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

    st.write(
        "✅ Dishes available:",
        len(dishes_list)
    )

    # -----------------------------
    # MANUAL SCHEDULER
    # -----------------------------
    st.markdown("### ✏️ Manually Assign Dishes")

    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    manual_schedule = {}

    options = ["--Select--"] + dishes_list

    for day in days:

        choice = st.selectbox(
            f"{day}'s Meal",
            options,
            key=f"{group_code}_{current_week}_{day}"
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
                columns=["Day", "Dish"]
            )

            schedule_df["week"] = current_week

            db.save_data(
                schedule_df,
                schedule_file
            )

            # also save as current/default schedule for homepage
            db.save_data(
                schedule_df,
                f"schedule_{group_code}.csv"
            )

            # append to served history
            old_served = db.load_data(served_file)

            if old_served is None or old_served.empty:
                combined = schedule_df.copy()

            else:
                combined = pd.concat(
                    [old_served, schedule_df],
                    ignore_index=True
                )

            db.save_data(
                combined,
                served_file
            )

            st.success(
                f"✅ Manual plan saved for week {current_week}"
            )

        else:
            st.warning(
                "Please assign at least one dish."
            )

    st.markdown("---")

    # -----------------------------
    # AI SCHEDULER
    # -----------------------------
    st.markdown(
        "### ✨ AI-Suggested Weekly Meal Plan"
    )

    if st.button("🤖 Generate AI Weekly Plan"):

        if (
            not _HAS_ML
            or generate_weekly_plan_for_group is None
        ):

            st.warning(
                "AI recommender unavailable."
            )

        else:

            try:
                plan_df, stats_df = generate_weekly_plan_for_group(group_code, return_stats=True)
                st.session_state.last_stats_df = stats_df
            except Exception as e:
                st.error(f"AI generation failed: {e}")
                plan_df = None
                stats_df = None

            if (
                plan_df is not None
                and not plan_df.empty
            ):

                if {
                    "Day",
                    "Dish"
                }.issubset(plan_df.columns):

                    # assign current week
                    plan_df["week"] = current_week

                    # save schedule
                    db.save_data(
                        plan_df,
                        schedule_file
                    )

                    # also save as current/default schedule for homepage
                    db.save_data(
                        plan_df,
                        f"schedule_{group_code}.csv"
                    )

                    # append to served history
                    old_served = db.load_data(
                        served_file
                    )

                    if (
                        old_served is None
                        or old_served.empty
                    ):

                        combined = plan_df.copy()

                    else:

                        combined = pd.concat(
                            [old_served, plan_df],
                            ignore_index=True
                        )

                    db.save_data(
                        combined,
                        served_file
                    )

                    st.success(
                        f"✅ AI-generated weekly plan saved to `{schedule_file}`"
                    )

                    st.dataframe(
                        plan_df,
                        width="stretch"
                    )

                    # -----------------------------
                    # AI RECOMMENDATION INSIGHTS
                    # -----------------------------
                    if stats_df is not None and not stats_df.empty:
                        with st.expander("🤖 AI Recommendation Insights & Explanations", expanded=True):
                            st.caption("Here is the underlying logic and scoring for why the AI selected these dishes for your weekly menu:")
                            
                            # Normalize columns in stats_df for matching
                            stats_df.columns = [str(c).strip().lower() for c in stats_df.columns]
                            
                            # Filter stats to only the chosen dishes
                            chosen_dishes = [str(d).strip().lower() for d in plan_df["Dish"].tolist()]
                            chosen_stats = stats_df[stats_df["dish"].astype(str).str.strip().str.lower().isin(chosen_dishes)].copy()
                            
                            explanations = []
                            for _, row in chosen_stats.iterrows():
                                dish_name = str(row["dish"]).title()
                                hybrid = row.get("hybrid_score", 0.0)
                                r_score = row.get("rating_score", 0.0)
                                p_score = row.get("popularity_score", 0.0)
                                cf_score = row.get("cf_score", 0.0)
                                
                                reasons = []
                                if r_score > 0.6:
                                    reasons.append("High average ratings from the family")
                                if p_score > 0.5:
                                    reasons.append("Highly voted in suggestions polls")
                                if cf_score > 0.5:
                                    reasons.append("Matches preferences of users with similar taste")
                                
                                if not reasons:
                                    reasons.append("Selected to prevent meal fatigue and maintain diversity")
                                    
                                explanations.append({
                                    "Dish Name": dish_name,
                                    "Recommendation Confidence": f"{hybrid * 100:.1f}%",
                                    "Primary Reason": " • ".join(reasons)
                                })
                            
                            if explanations:
                                st.dataframe(
                                    pd.DataFrame(explanations),
                                    width="stretch",
                                    hide_index=True
                                )

                else:

                    st.error(
                        "AI returned unexpected format."
                    )

            else:

                st.warning(
                    "Not enough data to generate recommendations. Please ensure you have dishes in rotation first."
                )

    st.markdown("---")

    # -----------------------------
    # VIEW SAVED PLANS
    # -----------------------------
    st.markdown(
        "### 📅 View Current / Past Plans"
    )

    if schedule_path.exists():

        st.markdown(
            f"**Saved plan for week {current_week}:**"
        )

        st.dataframe(
            db.load_data(schedule_file),
            width="stretch"
        )

    else:

        st.info(
            f"No saved plan for week {current_week} yet."
        )

    existing = _list_existing_weeks(group_code)

    if existing:

        st.markdown(
            "**Past weeks available:** "
            + ", ".join(map(str, existing))
        )

        chosen = st.selectbox(
            "Open saved week:",
            ["--Select--"] + [str(w) for w in existing],
            key=f"open_week_{group_code}"
        )

        if chosen != "--Select--":

            df = db.load_data(
                f"schedule_{group_code}_week{chosen}.csv"
            )

            if df is not None and not df.empty:

                st.dataframe(
                    df,
                    width="stretch"
                )

            else:

                st.info(
                    "Saved file found but empty."
                )