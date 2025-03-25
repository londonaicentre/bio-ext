import streamlit as st
from dotenv import load_dotenv
import os 
import pandas as pd
from csdash import dbaccess, transforms

# app name declaration
st.set_page_config(
    page_title="Hello Cogstack",
)
# import functions and cache them
load_config = st.cache_data(dbaccess.load_config)
connect_cogstack = st.cache_resource(dbaccess.connect_cogstack)
global_overview = st.cache_data(transforms.global_overview)
list_indexes= st.cache_data(dbaccess.list_indexes)

# load config files and dot env and save to session
config = load_config("utils/config_dash.yaml")
load_dotenv()
user = os.environ.get("ELASTIC_API_ID")
if "config" not in st.session_state:
    st.session_state["config"] = config

# connect to db and save to session 
es = connect_cogstack()
if "es" not in st.session_state:
    st.session_state["es"] = es
kwremovelist = config["escapekwlist"]

# list the indexes and use them 
csindexes = list_indexes(_es=es)
if 'csindexes' not in st.session_state:
    st.session_state["csindexes"] = csindexes

cogstack_brief = global_overview(_es = es, indexlist=csindexes)
if "cogstack_brief" not in st.session_state:
    st.session_state["cogstack_brief"] = cogstack_brief
    
# STREAMLIT app UI beings

# title properties
st.title(config["intro"])
st.sidebar.success("Select different page.")
st.markdown(config["hellotext"])
st.markdown(f"_Now successfully logged in for user:_{user}")

# expanders
with st.expander("## Learnings and Bugs"):
    st.markdown(config["learnings"])

with st.expander("## Fields not in EDA"):
    st.write("the following keywords are removed from summary stats")
    st.write(config["escapekwlist"])

with st.expander("## Available Indexes"):
    st.write("these are index descriptions, click on Indexes tab to explore them individually.")
    st.markdown("These are the available indexes, documents in each and their sizes.")
    st.write(cogstack_brief)
    st.table(cogstack_brief.dtypes)


