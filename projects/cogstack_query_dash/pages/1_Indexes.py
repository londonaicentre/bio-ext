import streamlit as st
from dotenv import load_dotenv
import os
from csdash import dbaccess, transforms
import pandas as pd

# ENVIRON VARS, LOAD CUSTOM UTILS AND APPLY DECORATORS
## DOT ENV
load_dotenv()
user = os.environ.get("ELASTIC_API_ID")
## APPLYING DECORATORS
load_config = st.cache_data(dbaccess.load_config)
connect_cogstack = st.cache_resource(dbaccess.connect_cogstack)
list_and_fetch_data = st.cache_data(dbaccess.list_and_fetch_data)
global_overview = st.cache_data(transforms.global_overview)
characterisedf = st.cache_data(transforms.characterisedf)
mappingtypes = st.cache_data(dbaccess.get_mapping_types)
get_top_10kw = st.cache_data(dbaccess.get_top_10kw)
get_date_ranges = st.cache_data(dbaccess.get_date_ranges)
get_num_stats = st.cache_data(dbaccess.get_num_stats)

# custom functions

def display_dataframe_overview(index,_es,kwremovelist):
    overview_df = cogstack_brief[cogstack_brief["index"] == index ].to_dict("records")[0]
    headerstring = f"{overview_df["index"]} , docs= {overview_df["docs.count"]},size = {overview_df["store.size"]}, no_cols={overview_df["fields_counts"]}"
    with st.expander(headerstring):
        _df = all_df[i]
        mapping = _es.indices.get_mapping(index = i)
        st.markdown("#### Columns")
        st.write(mapping[i]["mappings"])
        characterisedf(_data=_df)
        columntypes = mappingtypes(_es=es,indexname=i)
        st.markdown("### Columns grouped by datatypes ")
        st.write(columntypes)
        process_columns(index,columntypes,kwremovelist)
    return 

def process_columns(index,columntypes,kwremovelist):
    try: 
        kw = columntypes["keyword"]
        cleankw = [i for i in kw if i not in kwremovelist]
        datefields = columntypes["date"]
        numcols = list(
                    set(list(columntypes["scaled_float"]) + list(columntypes["integer"]))
                )
        #keywords
        if kw:
                st.write("{i} has no keywords")
        else: 
            st.markdown("### Top levels and counts")
            st.write(kw)
            top10 = get_top_10kw(_es=es,indexname=i,fieldlist=cleankw)
            st.write(top10)
        # for dates
        if datefields:
            st.write("{i} has no datecolumns")
        else: 
            dateranges = get_date_ranges(_es=es,indexname=i,fieldlist=datefields)
            st.markdown("### date ranges")
            st.write(dateranges)
        #for numeric columns
        if numcols: 
            st.markdown("### Numeric column summaries")
            numsumm = get_num_stats(_es=es, indexname=i, fieldlist=numcols)
            st.write(numsumm)
        else:
            st.write("no numeric columns here")
    except Exception as e:
        st.write(f"{e} error for index {i}")

# CONNECT COGSTACK, INSTANTIATE OBJECTS
config = load_config("utils/config_dash.yaml")
es = connect_cogstack()
kwremovelist = config["escapekwlist"]

# STREAMLIT APP UI 

st.title(config["start2"])
st.markdown(config["instructions"])
# allow user to select sample size.
samplesize = st.number_input("Select docs size, default 100, max 10k",max_value=10000, value=100)
config["all_cols_query"]["size"] = samplesize

# saving es object in state
st.write(f"Successfully logged into Elastic Session for user: {user}")
if "es" not in st.session_state:
    st.session_state["es"] = es

# collect sample data and save index into state
all_indexes, all_df = list_and_fetch_data(_es=es,query=config["all_cols_query"])
if 'indexlist' not in st.session_state:
    st.session_state['indexlist'] = all_indexes

st.markdown("These are the available indexes, documents in each and their sizes.")
cogstack_brief = global_overview(_es = es, indexlist=all_indexes)

st.write(cogstack_brief)
st.write(cogstack_brief.dtypes)
st.markdown("## Now this looks into each individual dataframes")

# this for loop create the expander objects in streamlit each with dataframe name 
for i in all_df.keys():
        display_dataframe_overview(index=i,_es=es,kwremovelist=kwremovelist)
        process_columns(index=i,)
        
st.write("END OF THE PAGE")


