import streamlit as st
import json
import os
from csdash import dbaccess, transforms
import pandas as pd

# TODO: refactor session state checkers
# TODO: make a small lambda that pretty prints headers
#
# ENVIRON 
# Load existing state objects
if "csindexes" in st.session_state:
    csindexes = st.session_state["csindexes"]
else:
    st.write("cogstack_indexes not available")

if "es" in st.session_state:
    es = st.session_state["es"]
    st.write("es is available.")
else:
    st.write("es is not available.")

if "config" in st.session_state:
    config = st.session_state["config"]
    st.write("config is available.")
else:
    st.write("config is not available.")

if "cogstack_brief" in st.session_state:
    cogstack_brief = st.session_state["cogstack_brief"]
    st.write("cogstack brief is here.")
else:
    st.write("cogstack_brief is not available")

kwremovelist = config["escapekwlist"]

## APPLYING DECORATORS
fetch_sampledata = st.cache_data(dbaccess.fetch_sampledata)
#characterisedf = st.cache_data(transforms.characterisedf)
mappingtypes = st.cache_data(dbaccess.get_mapping_types)
get_top_10kw = st.cache_data(dbaccess.get_top_10kw)
get_date_ranges = st.cache_data(dbaccess.get_date_ranges)
get_num_stats = st.cache_data(dbaccess.get_num_stats)

# custom functions

def display_dataframe_overview(index,_es,kwremovelist):
    overview_df = cogstack_brief[cogstack_brief["index"] == index ].to_dict("records")[0]
    headerstring = f"{overview_df["index"]} , docs= {overview_df["docs.count"]},size = {overview_df["store.size"]}, no_cols={overview_df["fields_counts"]}"
    with st.expander(headerstring):
        _df = all_df[index]
        print(f"{index}")
        mapping = _es.indices.get_mapping(index = index)
        st.markdown(f"#### Fields in {index}")
        st.json(mapping[index]["mappings"],expanded=False)
        st.markdown("#### Fields grouped by datatypes ")
        columntypes = mappingtypes(_es=es,indexname=index)
        st.json(columntypes,expanded=False)
        process_columns(index,columntypes,kwremovelist)
        print(f"passing {index} to {_df} to characteriser")
        transforms.characterisedf(_data=_df)
    return 

def process_columns(index,columntypes,kwremovelist):
    try: 
        kw = columntypes["keyword"]
        cleankw = [field for field in kw if field not in kwremovelist]
        datefields = columntypes["date"]
        st.write('mark')
        numcols = list(
                    set(list(columntypes["scaled_float"]) + list(columntypes["integer"]))
                )
        #keywords
        if not kw:
                st.write("{index} has no keywords")
        else: 
            st.markdown("#### Top categories and counts in fields")
            top10 = get_top_10kw(_es=es,indexname=index,fieldlist=cleankw)
            st.json(json.dumps(top10),expanded=False)
        # for dates
        if not datefields:
            st.write("{index} has no datecolumns")
        else: 
            dateranges = get_date_ranges(_es=es,indexname=index,fieldlist=datefields)
            st.markdown("#### Date Ranges")
            st.json(dateranges,expanded=False)
        #for numeric columns
        if not numcols: 
            st.write("No numeric columns in {index}")
        else:
            st.markdown("#### Numeric column summaries")
            numsumm = get_num_stats(_es=es, indexname=index, fieldlist=numcols)
            st.json(json.dumps(numsumm),expanded=False)
    except Exception as e:
        st.write(f"{e} error for index {index}")
    return 


# STREAMLIT APP UI 

st.title(config["start2"])
st.markdown(config["instructions"])
# allow user to select sample size.
samplesize = st.number_input("Select docs size, default 100, max 10k",max_value=10000, value=100)
config["all_cols_query"]["size"] = samplesize

# collect sample data and save index into state
all_df = fetch_sampledata(_es=es, indexlist = csindexes , query=config["all_cols_query"])

st.markdown("## Now this looks into each individual dataframes")

# this for loop create the expander objects in streamlit each with dataframe name 
for i in all_df.keys():
        display_dataframe_overview(index=i,_es=es,kwremovelist=kwremovelist)
        
st.write("END OF THE PAGE")

