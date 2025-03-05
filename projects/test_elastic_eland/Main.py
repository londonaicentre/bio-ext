import streamlit as st
import eland
from bioext.elastic_utils import ElasticsearchSession, GsttProxyNode
from dotenv import load_dotenv
import numpy as np
import json
import plotly.express as ex

DEFAULT_QUERY_STRING = """
{"query": {"bool": {"must": [{"wildcard": {"document_Content": "*brca*"}}]}}}
"""
QUERY_COLUMNS = ["document_CreatedWhen", "patient_NhsNumber", "patient_Age"]

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
    index=indices.index(st.session_state.selected_index) if st.session_state.selected_index in indices else None,
    on_change=update_index,
    key="index_widget",
    placeholder="Select index",
)

st.title("Rapid Query Feasibility Search")

input_query = st.text_area("Query as JSON", value=DEFAULT_QUERY_STRING)

print(st.session_state.selected_index)

submit_btn = st.button("Submit", disabled=True if st.session_state.selected_index is None else False)

if submit_btn:
    st.json(input_query)
    query_object = json.loads(input_query)

    df = eland.DataFrame(
        es_session.es,
        st.session_state.selected_index,
        columns=QUERY_COLUMNS,
    )

    res = df.es_query(
        query_object,
    )

    st.write(f"Got: {len(res):,} results.")

    res = res.to_pandas(show_progress=True)

    to_plot = (
        res["document_CreatedWhen"]
        .dt.date.value_counts()
        .rename_axis("date")
        .reset_index(name="count")
        .sort_values("date")
    )

    line_chart_date = ex.line(to_plot, x="date", y="count")

    st.plotly_chart(line_chart_date)

    age_distrib = (
        res.patient_Age.apply(np.floor)
        .value_counts()
        .rename_axis("age_at_event")
        .reset_index(name="count")
        .sort_values("age_at_event")
    )

    st.dataframe(to_plot)

    age_distrib_plot = ex.histogram(age_distrib, x="age_at_event", y="count")

    st.plotly_chart(age_distrib_plot)

    nhs_number_count = (
        res.patient_NhsNumber.value_counts()
        .reset_index(name="count")
        .value_counts()
        .reset_index(name="count_agg")
    )

    nhs_number_count_plot = ex.histogram(
        nhs_number_count, x="count", y="count_agg"
    ).update_layout(xaxis_title="Number of NHS Counts", yaxis_title="Density")

    st.plotly_chart(nhs_number_count_plot)

    st.session_state["query"] = query_object

    st.link_button("Review Documents", "./Document_Review")
