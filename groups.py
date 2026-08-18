import streamlit as st
from db import save_group


def create_group_ui():
    st.subheader("Create a new group")
    head_name = st.text_input("Your Name")
    group_name = st.text_input("Group Name")

    if st.button("Create Group"):
        if head_name and group_name:
            group_code = save_group(group_name, head_name)
            st.success(f"Group created! Share this code: **{group_code}**")
        else:
            st.error("Please fill all fields.")
