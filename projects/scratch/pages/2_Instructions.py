import streamlit as st

with open("gstt_elastic_directory.md","r",encoding="utf-8") as file:
    content = file.read()

st.markdown(content)
