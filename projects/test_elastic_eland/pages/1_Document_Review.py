import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from bioext.elastic_utils import ElasticsearchSession, GsttProxyNode

st.title("Document Review")

if st.session_state.get("query") is None:
    st.warning("No query defined.")
    st.stop()

refresh_btn = st.button("Randomise")

if refresh_btn:
    st.rerun()

load_dotenv()

es_session = ElasticsearchSession(proxy=GsttProxyNode, conn_mode="API")

query_object = {
    "size": 10,
    "_source": ["_id", "document_Content"],
    "query": {
        "function_score": {
            "query": st.session_state.query,
            "random_score": {},
        },
    },
}

res = es_session.es.search(index=st.session_state.selected_index, body=query_object)

df = pd.DataFrame(res["hits"]["hits"])
df["document_Content"] = df["_source"].str["document_Content"]

st.dataframe(df[["_id", "document_Content"]])
