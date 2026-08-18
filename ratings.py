import streamlit as st
import db
import pandas as pd

def rate_dish(group_code: str, user_name: str):
    st.subheader("⭐ Rate a Family Dish")
    st.caption("Submit or update your personal rating for any dish in the family rotation.")

    # Load list of dishes available (rotation)
    dishes = db.get_dishes(group_code)
    if not dishes:
        st.warning("⚠️ No dishes in rotation yet. Suggest new dishes in Polls or wait for poll approvals.")
        return

    dish = st.selectbox("Choose a dish to rate", ["--Select--"] + sorted(dishes))
    if dish == "--Select--":
        return

    # Use a nice card for rating input
    with st.container():
        st.markdown(
            f"""
            <div style="background-color: rgba(151, 151, 151, 0.05); padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                <h4 style="margin: 0; color: #ff9f43;">🍽️ {dish}</h4>
            </div>
            """, 
            unsafe_allow_html=True
        )

        rating = st.slider("Rate this dish", min_value=1.0, max_value=5.0, value=4.0, step=0.5)
        week_input = st.text_input("Week number (optional, integer)", value="", placeholder="e.g. 1, 2, 3")
        week = int(week_input) if week_input.strip().isdigit() else None

        if st.button("💾 Submit Rating", use_container_width=True):
            db.rate_dish(group_code, dish, rating, user_name=user_name, week=week, overwrite=True)
            st.success(f"🎉 Rated **{dish}** with {rating} stars!")
            st.balloons()

def view_ratings(group_code: str):
    st.subheader("📖 Dish Ratings Overview")
    st.caption("Average family satisfaction score for each dish in the rotation.")
    
    avg_df = db.get_average_ratings(group_code)
    if avg_df.empty:
        st.info("ℹ️ No ratings yet. Go to 'Rate Dishes' or 'Daily Feedback' to start rating.")
        return
        
    # display with stars approx (rounded)
    avg_df.columns = [str(c).strip().lower() for c in avg_df.columns]
    avg_df["stars"] = avg_df["average_rating"].round().astype(int).apply(lambda x: "⭐" * max(1, min(5, x)))
    
    # Sort and rename columns for display
    display_df = avg_df.sort_values("average_rating", ascending=False).reset_index(drop=True)
    display_df = display_df.rename(columns={
        "dish": "Dish Name",
        "average_rating": "Average Rating",
        "stars": "Star Rating"
    })
    
    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True
    )