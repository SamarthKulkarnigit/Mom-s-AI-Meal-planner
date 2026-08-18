# analytics.py
import streamlit as st
import pandas as pd
import db

def analytics_ui(group_code):
    st.subheader("📊 Meal Analytics Dashboard")
    st.caption("Insights, trends, and preference evolution for your family group.")

    # Load and clean data
    ratings = db.load_data(f"ratings_{group_code}.csv")
    served = db.load_data(f"served_log_{group_code}.csv")

    if ratings.empty:
        st.warning("⚠️ No ratings data available yet. Submit daily feedback to populate analytics.")
        return

    # Normalize columns
    ratings.columns = [str(c).strip().lower() for c in ratings.columns]
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["dish", "rating"])
    ratings = ratings[ratings["dish"].astype(str).str.strip() != ""]

    if not served.empty:
        served.columns = [str(c).strip().lower() for c in served.columns]
        served = served.dropna(subset=["dish"])
        served = served[served["dish"].astype(str).str.strip() != ""]

    # ---------------------------------
    # PRE-COMPUTE METRICS & KPIs
    # ---------------------------------
    
    # 1. Favorite Dish
    avg_ratings = ratings.groupby("dish")["rating"].mean()
    best_dish = avg_ratings.idxmax() if not avg_ratings.empty else "N/A"
    best_score = avg_ratings.max() if not avg_ratings.empty else 0.0

    # 2. Best Week
    best_week = "N/A"
    best_week_score = 0.0
    if "week" in ratings.columns:
        ratings["week_numeric"] = pd.to_numeric(ratings["week"], errors="coerce").fillna(0).astype(int)
        weekly_scores = ratings.groupby("week_numeric")["rating"].mean()
        if not weekly_scores.empty:
            best_week = f"Week {weekly_scores.idxmax()}"
            best_week_score = weekly_scores.max()

    # 3. Fatigue Detection
    fatigue_dish = "None"
    max_drop = 0.0
    fatigue_data = []
    if "week" in ratings.columns:
        grouped = ratings.groupby("dish")
        for dish, g in grouped:
            if g["week"].nunique() < 2:
                continue
            weekly_avg = g.groupby("week")["rating"].mean().sort_index()
            first_rating = weekly_avg.iloc[0]
            last_rating = weekly_avg.iloc[-1]
            drop = first_rating - last_rating
            
            fatigue_data.append({
                "dish": dish,
                "initial_rating": round(first_rating, 2),
                "latest_rating": round(last_rating, 2),
                "rating_drop": round(drop, 2)
            })
        
        fatigue_df = pd.DataFrame(fatigue_data)
        if not fatigue_df.empty:
            fatigue_df = fatigue_df.sort_values("rating_drop", ascending=False)
            highest_fatigue = fatigue_df.iloc[0]
            if highest_fatigue["rating_drop"] > 0:
                fatigue_dish = highest_fatigue["dish"]
                max_drop = highest_fatigue["rating_drop"]

    # 4. Most Improved
    improved_dish = "None"
    max_improvement = 0.0
    improvement_data = []
    if "week" in ratings.columns:
        grouped = ratings.groupby("dish")
        for dish, g in grouped:
            if g["week"].nunique() < 2:
                continue
            weekly_avg = g.groupby("week")["rating"].mean().sort_index()
            first_rating = weekly_avg.iloc[0]
            last_rating = weekly_avg.iloc[-1]
            improvement = last_rating - first_rating
            
            improvement_data.append({
                "dish": dish,
                "initial_rating": round(first_rating, 2),
                "latest_rating": round(last_rating, 2),
                "improvement": round(improvement, 2)
            })
        
        improve_df = pd.DataFrame(improvement_data)
        if not improve_df.empty:
            improve_df = improve_df.sort_values("improvement", ascending=False)
            highest_improve = improve_df.iloc[0]
            if highest_improve["improvement"] > 0:
                improved_dish = highest_improve["dish"]
                max_improvement = highest_improve["improvement"]

    # Render top-level metric cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔥 Family Favorite", f"{best_dish}", f"⭐ {best_score:.2f}")
    col2.metric("🏆 Best Week", f"{best_week}", f"⭐ {best_week_score:.2f}" if best_week != "N/A" else "No Data")
    col3.metric("📉 Fatigue Risk", f"{fatigue_dish}", f"-{max_drop:.2f} rating drop" if fatigue_dish != "None" else "Healthy rotation")
    col4.metric("📈 Most Improved", f"{improved_dish}", f"+{max_improvement:.2f} rating gain" if improved_dish != "None" else "Stable preferences")

    st.markdown("---")

    # ---------------------------------
    # DASHBOARD TABS
    # ---------------------------------
    tab_ratings, tab_rotation, tab_fatigue = st.tabs([
        "⭐ Ratings & Satisfaction Trends", 
        "🍽️ Meal Rotation & Evolution", 
        "📉 Fatigue & Improvement Analysis"
    ])

    with tab_ratings:
        st.markdown("### ⭐ Meal Satisfaction Overview")
        
        subcol1, subcol2 = st.columns(2)
        
        with subcol1:
            st.markdown("#### Top 10 Rated Dishes")
            top_rated = ratings.groupby("dish")["rating"].mean().sort_values(ascending=False).head(10)
            if not top_rated.empty:
                st.bar_chart(top_rated, use_container_width=True)
                st.dataframe(top_rated.reset_index().rename(columns={"rating": "Average Rating"}), width="stretch", hide_index=True)
            else:
                st.info("Not enough ratings data yet.")
                
        with subcol2:
            st.markdown("#### Lowest 10 Rated Dishes")
            low_rated = ratings.groupby("dish")["rating"].mean().sort_values(ascending=True).head(10)
            if not low_rated.empty:
                st.bar_chart(low_rated, use_container_width=True)
                st.dataframe(low_rated.reset_index().rename(columns={"rating": "Average Rating"}), width="stretch", hide_index=True)
            else:
                st.info("No ratings data yet.")

        st.markdown("---")
        st.markdown("### 📈 Weekly Satisfaction Trend")
        if "week" in ratings.columns and ratings["week"].nunique() > 1:
            weekly_avg = ratings.groupby("week_numeric")["rating"].mean().sort_index()
            st.line_chart(weekly_avg, use_container_width=True)
        else:
            st.info("ℹ️ Weekly Satisfaction Trend will populate once ratings span multiple weeks.")

    with tab_rotation:
        st.markdown("### 🍽️ Most Frequently Served Dishes")
        if not served.empty and "dish" in served.columns:
            served_counts = served["dish"].value_counts().head(10)
            if not served_counts.empty:
                st.bar_chart(served_counts, use_container_width=True)
                st.dataframe(served_counts.reset_index().rename(columns={"count": "Times Served"}), width="stretch", hide_index=True)
        else:
            st.info("ℹ️ Rotation data will show once schedules are marked as served.")

        st.markdown("---")
        st.markdown("### 🤖 Recommendation Evolution Over Time")
        if not served.empty and "week" in served.columns and "dish" in served.columns:
            # Evolution of dish serving count grouped by week
            evolution = served.groupby(["week", "dish"]).size().reset_index(name="count")
            evolution["week"] = pd.to_numeric(evolution["week"], errors="coerce")
            
            # Pivot table to make it easy to draw multiple lines
            pivot_evo = evolution.pivot(index="week", columns="dish", values="count").fillna(0)
            st.line_chart(pivot_evo, use_container_width=True)
        else:
            st.info("ℹ️ Evolution data requires historical weekly schedules.")

    with tab_fatigue:
        st.markdown("### 📉 Dish Fatigue Detection")
        st.caption("Fatigue is detected when a dish's rating drops over time as a result of oversaturation.")
        if "week" in ratings.columns and fatigue_data:
            fatigue_df = pd.DataFrame(fatigue_data).sort_values("rating_drop", ascending=False)
            if not fatigue_df.empty:
                st.bar_chart(data=fatigue_df.head(10), x="dish", y="rating_drop", use_container_width=True)
                st.dataframe(fatigue_df, width="stretch", hide_index=True)
        else:
            st.info("ℹ️ Fatigue analysis requires ratings spanning multiple weeks for comparison.")

        st.markdown("---")
        st.markdown("### 📈 Most Improved Dishes")
        st.caption("Dishes that have experienced the largest rating increases over time.")
        if "week" in ratings.columns and improvement_data:
            improve_df = pd.DataFrame(improvement_data).sort_values("improvement", ascending=False)
            if not improve_df.empty:
                st.bar_chart(data=improve_df.head(10), x="dish", y="improvement", use_container_width=True)
                st.dataframe(improve_df, width="stretch", hide_index=True)
        else:
            st.info("ℹ️ Improvement analysis requires ratings spanning multiple weeks.")

    st.success("✅ Analytics dashboard loaded successfully.")