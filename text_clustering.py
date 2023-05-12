from numpy.random.mtrand import RandomState
# import the dataset from sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import euclidean_distances

# viz libs
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import csv

N_CLUSTERS = 4


def tf_idf_vectorization(df):
    # initialize the vectorizer
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.95)
    # fit_transform applies TF-IDF to clean texts - we save the array of vectors in X
    X = vectorizer.fit_transform(df['News'].astype('U').values)

    return X


def lsa_func(X_tfidf):
    lsa = make_pipeline(TruncatedSVD(n_components=100), Normalizer(copy=False))
    X_lsa = lsa.fit_transform(X_tfidf)

    return X_lsa


def text_clustering(df, X):
    # initialize kmeans with 4 centroids
    kmeans = KMeans(n_clusters=N_CLUSTERS, init="k-means++", max_iter=500)
    # fit the model
    kmeans.fit(X)
    # store cluster labels in a variable
    clusters = kmeans.labels_

    # distance between clusters
    stats(kmeans)

    # dimensional_reduction
    # initialize PCA with 2 components
    pca = PCA(n_components=2, random_state=42)
    # pass our X to the pca and store the reduced vectors into pca_vecs
    pca_vecs = pca.fit_transform(X)
    # save our two dimensions into x0 and x1
    x0 = pca_vecs[:, 0]
    x1 = pca_vecs[:, 1]

    # assign clusters and PCA vectors to columns in the original dataframe
    df['cluster'] = clusters
    df['x0'] = x0
    df['x1'] = x1

    # map clusters to appropriate labels
    cluster_map = {0: "1", 1: "2", 2: "3", 3: "4"}
    # apply mapping
    df['cluster'] = df['cluster'].map(cluster_map)

    return df


def vizualice(df):
    # set image size
    plt.figure(figsize=(12, 7))
    # set a title
    plt.title("TF-IDF + KMeans Rest-Mex",
              fontdict={"fontsize": 18})
    # set axes names
    plt.xlabel("X0", fontdict={"fontsize": 16})
    plt.ylabel("X1", fontdict={"fontsize": 16})
    # create scatter plot with seaborn, where hue is the class used to group the data
    sns.scatterplot(data=df, x='x0', y='x1', hue='cluster', palette="viridis")
    plt.show()


def stats(kmeans):
    dists = euclidean_distances(kmeans.cluster_centers_)

    tri_dists = dists[np.triu_indices(N_CLUSTERS, 1)]
    max_dist, avg_dist, min_dist = tri_dists.max(), tri_dists.mean(), tri_dists.min()
    print(f"max dist {max_dist}")
    print(f"avg dist {avg_dist}")
    print(f"min dist {min_dist}")


if __name__ == "__main__":
    df = pd.read_csv('/content/dataset_cleaned.csv')
    # df = pd.read_csv('./datasets/dataset_translated.csv')

    # df['News'] = df['News'].apply(
    #    lambda x: text_preprocess.preprocess(x, remove_stopwords=True))

    X = tf_idf_vectorization(df)
    X_lsa = lsa_func(X)
    df_clustered = text_clustering(df, X_lsa)
    vizualice(df_clustered)

    print(df_clustered.head())

    df_clustered['task'] = 'thematic'
    df_clustered = df_clustered[['task', 'ID', 'cluster']]
    df_clustered.to_csv(
        '/content/dataset_classified.txt', header=None, index=None, sep='\t', quoting=csv.QUOTE_ALL)
