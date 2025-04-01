import streamlit as st
from pathlib import Path

def serve_path(primary_path,secondary_path):
    primary = Path(primary_path)
    
    if primary.exists():
        return primary
    else:
        return secondary

with open(serve_path(primary_path="../../../sde_aic_internal_docs/nlp/gstt_elastic_directory.md", secondary_path="utils/gstt_elastic_directory.md"), "r", encoding="utf-8") as file:
    content = file.read()

st.markdown(content)
