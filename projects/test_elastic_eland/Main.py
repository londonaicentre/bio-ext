import streamlit as st
import eland
from bioext.elastic_utils import ElasticsearchSession, GsttProxyNode
from dotenv import load_dotenv
import numpy as np
import json
import plotly.express as ex
import utils

DEFAULT_QUERY_STRING = """
{"bool": {"must": [{"wildcard": {"document_Content": "*brca*"}}]}}
"""

load_dotenv()

es_session = ElasticsearchSession(proxy=GsttProxyNode, conn_mode="API")

indices = list(es_session.list_indices().keys())

if "selected_index" not in st.session_state:
    st.session_state.selected_index = None


def update_index():
    st.session_state.selected_index = indices[0] = st.session_state.index_widget


index_select_widget = st.selectbox(
    "Elastic Index",
    indices,
    index=indices.index(st.session_state.selected_index)
    if st.session_state.selected_index in indices
    else None,
    on_change=update_index,
    key="index_widget",
    placeholder="Select index",
)

st.title("Rapid Query Feasibility Search")

input_query = st.text_area("Query as JSON", value=DEFAULT_QUERY_STRING)

print(st.session_state.selected_index)

submit_btn = st.button(
    "Submit", disabled=True if st.session_state.selected_index is None else False
)

if submit_btn:
    query_object = json.loads(input_query)
    st.session_state["query"] = query_object

    documents_count= utils.get_number_of_results(query=query_object, index=index_select_widget, session=es_session.es)['count']

    st.write(f"Returned **{documents_count}** documents")

    if documents_count == 0:
        st.warning("No results found.")
        st.stop()

    # Plot Event Activity Date
    st.subheader("Result Frequency by Date")
    activity_time = utils.aggregate_by_date(
        query_object, index_select_widget, "activity_Date", es_session.es
    )
    activity_time_plot = ex.line(activity_time, x="date", y="count")
    st.plotly_chart(activity_time_plot)

    # NHS Number Data Frequency
    st.subheader("Count Frequency by Patient (with NHS Number)")
    nhs_number_frequency = utils.aggregate_by_nhs_numbers(
        query_object, index_select_widget, es_session.es
    )
    nhs_number_frequency_plot = ex.line(
        nhs_number_frequency, x="Count Frequency", y="Density", log_x=True
    ).update_layout(
        xaxis_title="Count per Patient"
    )
    st.plotly_chart(nhs_number_frequency_plot)

    # Age Density
    st.subheader("Patient age at Event")
    age_at_event = utils.aggregate_by_event_age(
        query_object, index_select_widget, es_session.es
    )
    age_at_event_plot = ex.histogram(age_at_event, x="age", y="count")
    st.plotly_chart(age_at_event_plot)

    
