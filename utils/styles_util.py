import streamlit as st

# Remove hamburger and 
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """

def inject_style():
    st.markdown(hide_st_style, unsafe_allow_html=True)