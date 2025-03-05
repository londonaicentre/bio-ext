import streamlit as st
import eland
import numpy as np
import json
from dotenv import load_dotenv
from bioext.elastic_utils import ElasticsearchSession, GsttProxyNode

st.title("Document Review")

if st.session_state.get("query") is None:
    st.warning("No query defined.")
    st.stop()

st.json(st.session_state["query"])

load_dotenv()

es_session = ElasticsearchSession(proxy=GsttProxyNode, conn_mode="API")

df = (
    eland.DataFrame(
        es_session.es,
        st.session_state.selected_index,
        columns=["document_Content"],
    )
    .es_query(st.session_state.get("query"))
    .sample(10)
)

st.dataframe(df)
