import streamlit as st
from pathlib import Path


def serve_path(primary_path, secondary_path):
    """small function to serve a correct elastic directory.md

    Args:
        primary_path (str): path to distant but correct elastic directory
        secondary_path (str): path to local in utils

    Returns:
        path: path to either of these
    """
    primary = Path(primary_path)
    secondary = Path(secondary_path)
    if primary.exists():
        return primary
    else:
        return secondary


with open(
    serve_path(
        primary_path="../../../sde_aic_internal_docs/nlp/gstt_elastic_directory.md",
        secondary_path="utils/gstt_elastic_directory.md",
    ),
    "r",
    encoding="utf-8",
) as file:
    content = file.read()

st.markdown(content)
