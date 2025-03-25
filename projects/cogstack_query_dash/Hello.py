import streamlit as st
from csdash import dbaccess

config = dbaccess.load_config("utils/config_dash.yaml")

st.set_page_config(
    page_title="Hello Cogstack",
)

st.title(config["intro"])
st.sidebar.success("Select different page.")
st.markdown(config["hellotext"])
st.markdown(config["learnings"])
st.markdown("## Note")
st.write("the following keywords are removed from summary stats")
st.write(config["escapekwlist"])
