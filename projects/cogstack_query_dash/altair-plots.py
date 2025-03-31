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

    reducer = umap.UMAP(n_components=2, metric="cosine", n_neighbors=80, min_dist=0.1)
    return reducer.fit_transform(np_array)


@timeit
def cluster_points(np_array, min_cluster_size=4, max_cluster_size=50):

    pca = PCA(n_components=50)
    np_array = pca.fit_transform(np_array)

    hdb = hdbscan.HDBSCAN(
        min_samples=10,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
    ).fit(np_array)

    return np.where(hdb.labels == -1, "outlier", "cluster_" + hdb.labels_.astype(str))


df = pd.read_pickle("../../../misc/data/imdbunsup.pkl")

model = SentenceTransformer(
    "sentence-transformers/distiluse-base-multilingual-cased-v1"
)


@timeit
def func(x):
    return model.encode(x)


df["embeddings"] = df["text"].apply(lambda x: func(x))

df.to_pickle("imdbunsup_embedding.pkl")


df = pd.read_pickle("imdbunsup_embedding.pkl")
df2 = df.sample(frac=0.1)
embeddings_array = df2["embeddings"].to_numpy()
embeddings_array = embeddings_array[0]
hdb_labels = cluster_points(embeddings_array.reshape(-1, 1))

embeddings_2d = reduce_dimensions(embeddings_array.reshape(-1, 1))
# 5 seconds to execute


ls = df2["embeddings"].tolist()
u = reduce_dimensions(ls)
# 7 seconds

df2["axis1"] = u[:, 0]
df2["axis2"] = u[:, 1]

df2.to_pickle("transformed.pkl")
