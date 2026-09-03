import streamlit as st
import db

def poll_ui(group_code: str, user_name: str):
    st.subheader("🗳️ Dish Polls & Suggestions")
    st.caption("Suggest new dishes to add to the family rotation. Suggestions are auto-promoted once they reach majority approval.")

    # ------------------------------------------------
    # SUGGEST NEW DISH
    # ------------------------------------------------
    with st.expander("➕ Suggest a New Dish to the Family", expanded=False):
        new_dish = st.text_input("What dish would you like to suggest?", placeholder="e.g. Vegetable Biryani, Tacos, Paneer Tikka")
        
        if st.button("🚀 Suggest Dish", use_container_width=True):
            if not new_dish.strip():
                st.error("Please enter a dish name.")
            else:
                added = db.suggest_dish(group_code, new_dish.strip(), user_name)
                if added:
                    st.success(f"✅ '{new_dish.strip()}' has been suggested! Ask family members to vote.")
                    st.rerun()
                else:
                    st.warning("⚠️ This dish is already pending approval in polls.")

    st.markdown("---")

    # ------------------------------------------------
    # PENDING DISHES
    # ------------------------------------------------
    st.markdown("### 🕒 Pending Suggestions & Voting")
    
    pending = db.get_pending_suggestions(group_code)
    members_count = db.get_group_members_count(group_code)
    majority_needed = (members_count // 2) + 1 if members_count > 0 else 1

    if pending is None or pending.empty:
        st.info("ℹ️ No pending suggestions at the moment. Feel free to suggest one above!")
    else:
        st.markdown(f"💡 *Majority approval threshold: **{majority_needed}** votes (Family size: {members_count} members)*")
        
        # Display as cards in columns or rows
        for idx, row in pending.iterrows():
            dish = str(row.get("dish", "")).strip()
            if not dish:
                continue

            suggester = row.get("suggester", "someone")

            # Fetch latest live vote count
            results_df = db.get_poll_results(group_code)
            votes = 0
            if results_df is not None and not results_df.empty:
                results_df.columns = [str(c).strip().lower() for c in results_df.columns]
                match = results_df[results_df["dish"].astype(str).str.strip().str.lower() == dish.lower()]
                if not match.empty:
                    votes = int(match.iloc[0]["votes"])

            # Card structure
            with st.container():
                st.markdown(
                    f"""
                    <div style="background-color: rgba(151, 151, 151, 0.05); padding: 15px; border-radius: 10px; margin-bottom: 15px; border-left: 5px solid #ff4b4b;">
                        <h4 style="margin: 0; color: #ff4b4b;">🍽️ {dish}</h4>
                        <span style="font-size: 0.85em; color: grey;">Suggested by <b>{suggester}</b></span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Progress bar
                progress_val = min(1.0, votes / majority_needed) if majority_needed > 0 else 0.0
                st.progress(progress_val, text=f"{votes} of {majority_needed} votes received")

                col_btn, col_metric = st.columns([1, 4])
                
                with col_btn:
                    if st.button(
                        "👍 Approve / Vote", 
                        key=f"vote_{dish}_{group_code}_{user_name}",
                        use_container_width=True
                    ):
                        try:
                            ok = db.vote_dish(group_code, dish, user_name)
                        except Exception as e:
                            ok = False
                            st.error(f"Voting failed: {e}")
                        if ok:
                            st.success(f"Vote registered for {dish}!")
                        else:
                            st.error(f"Could not register vote for {dish}. Please try again.")
                        st.rerun()
                            
                with col_metric:
                    st.caption(f"Status: {'✅ Promoted!' if votes >= majority_needed else '⏳ Pending votes...'}")
                
                st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")

    # ------------------------------------------------
    # POLL RESULTS
    # ------------------------------------------------
    st.markdown("### 📊 Active Vote Standings")
    results = db.get_poll_results(group_code)

    if results is None or results.empty:
        st.info("No votes cast yet.")
    else:
        try:
            results.columns = [str(c).strip().lower() for c in results.columns]
            if "votes" in results.columns:
                results = results.sort_values("votes", ascending=False)
            
            st.dataframe(
                results.rename(columns={"dish": "Dish Name", "votes": "Total Votes"}),
                width="stretch",
                hide_index=True
            )
        except Exception as e:
            st.error(f"Could not display poll standings: {e}")