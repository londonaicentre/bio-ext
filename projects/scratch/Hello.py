import streamlit as st
import yaml

with open("pages/config_dash.yaml","r") as file:
    config = yaml.safe_load(file)

st.set_page_config(
    page_title="Hello Cogstack",
)

st.title(config["intro"])
st.sidebar.success("Select different page.")
st.markdown(config["hellotext"])
st.markdown("""
            
            ### Learnings / Bugs 
            * For example, Elastic Search has a data type "text" vs "keyword". Keywords are almost like a category but not quite. It is not broken down
            any further and analysed by Elastic Search. Elastic Search breaks down **text**. Therefore, you can use fuzzy matching for text column but not for keyword columns.
            See more here [Text vs Keyword by Elastic.co](https://www.elastic.co/blog/strings-are-dead-long-live-strings)
            
            * Elasticsearch API for large searches need scroll function
            * Streamlit converts pandas to pyarrow. there is an error regarding that. i will fix later.
            * Elastic search column names are not columns but called "fields".
            * future features [use nltk or spacy --spacy has some issues with version conflict. nltk has difficulty downloading onto streamlit environment to remove stop words]
            * append column data types and if keywrod or categorical, append min  max, value counts etc
            """)