import streamlit as st
import json
import pandas as pd
from dotenv import load_dotenv
from elasticsearch import helpers
from bioext.elastic_utils import ElasticsearchSession, GsttProxyNode
import yaml
import os
import nltk
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# CUSTOM FUNCTIONS
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

@st.cache_data
def list_and_fetch_data(_es,query):
    """iterate over each cogstack index and extract data.

    Args:
        es (elastic search session object): need to be initiated. 

    Returns:
        cogstack_indexes: index name as list 
        results: dict object which holds dataframes
    """
    cogstack_indexes = list(es.indices.get_mapping().keys())
    results = {}
    for index in cogstack_indexes:
        _data = fetch_sample_data(query,index_name = index)
        results[index] = _data
        print(f"{index} is appended to dict.")
    return cogstack_indexes, results
    
@st.cache_data
def load_config(path):
    """
    simple function to load config.yaml returning dict.
    """
    with open(path,"r") as file:
        yamlvalues = yaml.safe_load(file)
    return yamlvalues

@st.cache_resource
def connect_cogstack():
    """simple function to establish elastic search session using bio-ext
    """
    session = ElasticsearchSession(conn_mode="API",proxy = GsttProxyNode)
    return session.es

@st.cache_data
def global_overview(_es,indexlist):
    """"
    function to help create a general over view of 
    """
    docs = es.cat.indices(format="json",h=["index","docs.count"])
    sizes = es.cat.indices(format="json",h=["index","store.size"])
    _dfc = pd.DataFrame(docs)
    _dfs = pd.DataFrame(sizes)
    _df = pd.merge(_dfc,_dfs, on = "index", how = "left")
    
    _fieldcounts = []
    for i in indexlist:
        _mapping = es.indices.get_mapping(index = i)
        _fieldcounts.append({"index":i, "fields_counts":len(_mapping[i]["mappings"]["properties"].keys())})
    _dff = pd.DataFrame(_fieldcounts)
    _df = pd.merge(_df,_dff, on = "index",how = "left")
    _df["docs.count"] = _df["docs.count"].astype("int64")
    return _df 

@st.cache_data
def characterisedf(_data):
    """
    given data, it does basic EDA 
    
    """
    x = _data.shape
    y = _data.columns
    st.write(f"Dimensions of this dataframe: {x}")
    st.subheader("Example data")
    st.write(_data)
    
    if "_source.document_Content" in list(y):
        _data["_source.document_Content"] = _data["_source.document_Content"].astype(str)
        _data = _data.assign(word_count = _data["_source.document_Content"].apply(lambda x: len(x.split())))
        st.write("Word count statisics for column document Content")
        st.write(_data["word_count"].describe())
        st.write("top 10 words")
        combined_text = "".join(_data["_source.document_Content"].tolist())
        words = combined_text.split()
        #words_cleaned = remove_stopwords(text=words)
        words_counts = Counter(words)
        top_10 = words_counts.most_common(10)
        w,c = zip(*top_10)
        st.write(w)
        st.write(c)
    
@st.cache_data    
def remove_stopwords(text):
    """
    remove stop words using nltk but currently not functional yet
    """
    #nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))
    _tokens = word_tokenize(text)
    _filtered = [w for w in _tokens if not w in stop_words]
    return _filtered 

@st.cache_data
def get_summary(_es,index):
    """to gain 5 number summary of the index ones 
    """
    mapping = _es.indices.get_mapping(index=index)
    properties = mapping[index]["mappings"]["properties"] 
    
    return properties

# LOAD variables
load_dotenv()
user = os.environ.get("ELASTIC_API_ID")
config = load_config("pages/config_dash.yaml")

# Streamlit 

st.title(config["start2"])
st.markdown(config["instructions"])
samplesize = st.number_input("Select docs size, default 10, max 10k",max_value=10000, value=100)
config["all_cols_query"]["size"] = samplesize

es = connect_cogstack()
st.write(f"Successfully logged into Elastic Session for user: {user}")
if "es" not in st.session_state:
    st.session_state["es"] = es

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
    overview_df = cogstack_brief[cogstack_brief["index"] == i ].to_dict("records")[0]
    headerstring = f"{overview_df["index"]} , docs= {overview_df["docs.count"]},size = {overview_df["store.size"]}, no_cols={overview_df["fields_counts"]}"
    with st.expander(headerstring):
        _df = all_df[i]
        mapping = es.indices.get_mapping(index = i)
        st.markdown("#### Columns")
        st.write(mapping[i]["mappings"])
        characterisedf(_data=_df)
        #resultsum = get_summary(_es=es,index=i)
        #st.write(resultsum)
        
st.write("ZZZ")




