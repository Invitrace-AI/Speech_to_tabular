import streamlit as st

# Before
"""
if 'num' not in st.session_state:
    st.session_state['num'] = 3
"""
st.slider('chosse num',10,20,14,key='num')

#num_var = st.slider('chosse num',10,20,14)

if st.session_state['num']:
    st.text(st.session_state['num'])
