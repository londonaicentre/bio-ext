import streamlit as st
import pandas as pd

def fetch_sample_data(query, index_name):
    """ this function  need es object from elastic search
    once initiated, use this function to send queries to server. 

    Args:
        query (json): search string as json object 
        index_name (str): index names from the elastic seach 

    Returns:
        dataframe: pandas which has flattened the json
    """
    response = es.search(index = index_name, body = query)
    _results = response["hits"]["hits"]
    _df = pd.json_normalize(_results)
    return _df

if "indexlist" in st.session_state:
    cogstack_indexes = st.session_state["indexlist"]
else: 
    st.write("cogstack_indexes not available")

if "es" in st.session_state:
    es = st.session_state["es"]
    st.write("es is available.")
else:
    st.write("es is not available.")

option = st.selectbox(
    "Select index to search?",
    cogstack_indexes,
    index=None,
    placeholder="Select index ...",
)

st.write("You selected:", option)

txt = st.text_area(
    "query","boss"
)

st.write("### Json of the query")
st.json(txt)
st.write(txt)

st.write("### result")
df = fetch_sample_data(query=txt,index_name=option)
st.write("### Search output")
st.write(df.shape)
st.write("the data:")
st.write(df)
