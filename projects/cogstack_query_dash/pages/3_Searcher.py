import streamlit as st
from csdash import dbaccess, transforms

# load
load_config = st.cache_data(dbaccess.load_config)
check_load_states = transforms.check_load_states
config = load_config("utils/config_dash.yaml")

# Load existing state objects
csindexes = check_load_states("csindexes")
es = check_load_states("es")

# Select which index to do searches on
indexoption = st.selectbox(
    "Select cogstack index to search:",
    csindexes,
    index=None,
    placeholder="Select index ...",
)

st.write("You selected:", indexoption)

# Query text
st.write("Here, input your query string as JSON")
txt = st.text_area("query", config["examplequery"])

# Visualise query
st.markdown("### JSON visualisation of the query")
st.json(txt, expanded=False)


# Visualise result
st.markdown(f"### RESULT {indexoption}")
df = dbaccess.fetch_query(es=es, query=txt, index_name=indexoption)
st.write(f"### SEARCH OUTPUT FROM {indexoption}")
st.write(f" the size of output: {df.shape}")
st.write("the data(note size default to 10):")
st.dataframe(df)
