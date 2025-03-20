import pickle
import streamlit as st
from collections import Counter

with open("data/results.pkl", "rb") as handle:
    unserialized_dict = pickle.load(handle)
    


def characterisedf(data):
    x = data.shape
    y = data.columns
    st.write(f"Dimensions of this dataframe: {x}")
    st.subheader("Columns names")
    st.write(y)
    st.subheader("all rows")
    st.write(data)
    
    if "_source.document_Content" in list(y):
        data["_source.document_Content"] = data["_source.document_Content"].astype(str)
        data = data.assign(word_count = data["_source.document_Content"].apply(lambda x: len(x.split())))
        st.write("Word count statisics for column document Content")
        st.write(data["word_count"].describe())
        st.write("top 10 words")
        combined_text = "".join(data["_source.document_Content"].tolist())
        words = combined_text.split()
        #words_cleaned = pipeline.clean(words)
        words_counts = Counter(words)
        top_10 = words_counts.most_common(10)
        w,c = zip(*top_10)
        st.write(w)
        st.write(c)
        
        

for i in unserialized_dict.keys():
    with st.expander(i):
        st.header(i)
        _df = unserialized_dict[i]
        characterisedf(data=_df)