import pandas as pd
import time
import altair as alt
import hdbscan
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import numpy as np
import umap
import pickle


def timeit(func):
    """simple timer function"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function {func.__name__} took {elapsed_time:.4f} seconds to execute.")
        return result

    return wrapper


@timeit
def reduce_dimensions(np_array):
    """
    given an aray
    initialise a UMAP object and then using a cosine similarity under take dimension reduction. 
    """

    reducer = umap.UMAP(n_components=2, metric="cosine", n_neighbors=80, min_dist=0.1)
    return reducer.fit_transform(np_array)


@timeit
def cluster_points(np_array, min_cluster_size=4, max_cluster_size=50):
    """ clustering function using PCA and HDBSCAn
    """

    pca = PCA(n_components=50)
    np_array = pca.fit_transform(np_array)

    hdb = hdbscan.HDBSCAN(
        min_samples=10,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
    ).fit(np_array)

    return np.where(hdb.labels == -1, "outlier", "cluster_" + hdb.labels_.astype(str))

@timeit
def make_encodings(x,textcol,model):
    """ensure column names are correct, otherwise it wont work
    custom function to make embeddings.

    Args:
        x (pandas datframe): pandas dataframe with text in the correct coolumn
        textcol (column name as string): supply column name where text lives as string

    Returns:
        : pandas dataframe with new column 
    """
    x[embeddings] = x[textcol].apply(lambda x: model.encode(x))
    return x

@timeit
def create_dimension_reduced_embeddings(df,textcol,model):
    """
    function that runs all of the above except clustering, generate embedding,
    dimension reduce the embeddings and return to dataframe. 
    """
    df = make_encodings(x=df,textcol=textcol,model =model)
    embeddings= df["embeddings"].tolist()
    dimension_reduced_embeddings = reduce_dimensions(embeddings)
    df["axis1"] = dimension_reduced_embeddings[:,0]
    df["axis2"] = dimension_reduced_embeddings[:,1]
    
    return df 

# Read source data 
df = pd.read_pickle("test.pkl")

# download sentence transformers model
model = SentenceTransformer(
    "sentence-transformers/distiluse-base-multilingual-cased-v1"
)

# create embeddings and dimension reduce them 
df =create_dimension_reduced_embeddings(df=df,textcol = "text", model = model)

# make an altair interactive plot
# needs text column 
chart = (
    alt.Chart(df)
    .mark_point()
    .encode(
        x="axis1",
        y="axis2",
        tooltip=["axis1","axis2","text"]
    )
)


