import streamlit as st

# Remove hamburger and 
hide_st_style = """
            <style>
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """

def inject_style():
    st.markdown(hide_st_style, unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;">Invitrace Assistance Version 0.0.1</p>', unsafe_allow_html=True)