import streamlit as st
from db import group_exists, add_member


def join_group_ui():
    st.subheader("Join an existing group")
    join_code = st.text_input("Enter Group Code").upper()

    if join_code:
        if group_exists(join_code):
            name = st.text_input("Your Name")
            likes = st.text_area("Likes")
            dislikes = st.text_area("Dislikes")

            if st.button("Submit Preferences"):
                if name:
                    add_member(join_code, name, likes, dislikes)
                    st.success("Preferences saved!")
                else:
                    st.error("Please enter your name.")
        else:
            st.error("Invalid group code.")
