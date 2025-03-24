import streamlit as st
import pandas as pd
from csdash import dbaccess

#

# Load existing state objects
if "indexlist" in st.session_state:
    cogstack_indexes = st.session_state["indexlist"]
else:
    st.write("cogstack_indexes not available, run page Indexes first")

if "es" in st.session_state:
    es = st.session_state["es"]
    st.write("es is available.")
else:
    st.write("es is not available.")

# Select which index to do searches on
indexoption = st.selectbox(
    "Select cogstack index to search:",
    cogstack_indexes,
    index=None,
    placeholder="Select index ...",
)

st.write("You selected:", indexoption)

# Query text
st.write("Here, input your query string as JSON")
txt = st.text_area("query", "here is your query")

# Visualise query
st.markdown("### JSON visualisation of the query")
st.json(txt)


# Visualise result
st.markdown("### RESULT")
df = dbaccess.fetch_sample_data(es=es, query=txt, index_name=indexoption)
st.write("### SEARCH OUTPUT")
st.write(f" the size of output: {df.shape}")
st.write("the data(note size default to 10):")
st.write(df)
