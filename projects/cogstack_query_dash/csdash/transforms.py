import streamlit as st
import pandas as pd
from collections import Counter


def global_overview(_es, indexlist):
    """
    function to help create a general over view of entire cogstack
    """
    docs = _es.cat.indices(format="json", h=["index", "docs.count"])
    sizes = _es.cat.indices(format="json", h=["index", "store.size"])
    _dfc = pd.DataFrame(docs)
    _dfs = pd.DataFrame(sizes)
    _df = pd.merge(_dfc, _dfs, on="index", how="left")

    _fieldcounts = []
    for i in indexlist:
        _mapping = _es.indices.get_mapping(index=i)
        _fieldcounts.append(
            {
                "index": i,
                "fields_counts": len(_mapping[i]["mappings"]["properties"].keys()),
            }
        )
    _dff = pd.DataFrame(_fieldcounts)
    _df = pd.merge(_df, _dff, on="index", how="left")
    _df["docs.count"] = _df["docs.count"].astype("int64")
    return _df


def characterisedf(_data):
    """
    given data as pandas dataframe, it does basic EDA

    """
    x = _data.shape
    y = _data.columns
    st.write(f"Dimensions of this dataframe: {x}")
    st.subheader("Example data")
    st.write(_data)

    if "_source.document_Content" in list(y):
        _data["_source.document_Content"] = _data["_source.document_Content"].astype(
            str
        )
        _data = _data.assign(
            word_count=_data["_source.document_Content"].apply(lambda x: len(x.split()))
        )
        st.write("Word count statisics for column document Content")
        st.write(_data["word_count"].describe())
        st.write("top 10 words")
        combined_text = "".join(_data["_source.document_Content"].tolist())
        words = combined_text.split()
        # words_cleaned = remove_stopwords(text=words)
        words_counts = Counter(words)
        top_10 = words_counts.most_common(10)
        w, c = zip(*top_10)
        st.write(w)
        st.write(c)


def get_summary(_es, index):
    """to return fields available in index"""
    mapping = _es.indices.get_mapping(index=index)
    properties = mapping[index]["mappings"]["properties"]

    return properties


def check_load_states(string):
    """
    check the presence of a session state object from streamlit and if not present, inform the user.

    Args:
        string (stream lit state object): pass as string

    Returns:
        stream lit state: will return and assign.
    """
    if string in st.session_state:
        response = st.session_state[string]
    else:
        st.write(f"{string}not available")

    return response
