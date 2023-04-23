# import the dataset from sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# viz libs
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
# import text_preprocess


def tf_idf_vectorization(df):
    # initialize the vectorizer
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)
    # fit_transform applies TF-IDF to clean texts - we save the array of vectors in X
    X = vectorizer.fit_transform(df['News'].apply(lambda x: np.str_(x)))

    return X


def text_clustering(df, X):
    # initialize kmeans with 3 centroids
    kmeans = KMeans(n_clusters=5, random_state=42)
    # fit the model
    kmeans.fit(X)
    # store cluster labels in a variable
    clusters = kmeans.labels_

    # dimensional_reduction
    # initialize PCA with 2 components
    pca = PCA(n_components=2, random_state=42)
    # pass our X to the pca and store the reduced vectors into pca_vecs
    pca_vecs = pca.fit_transform(X.toarray())
    # save our two dimensions into x0 and x1
    x0 = pca_vecs[:, 0]
    x1 = pca_vecs[:, 1]

    # assign clusters and PCA vectors to columns in the original dataframe
    df['cluster'] = clusters
    df['x0'] = x0
    df['x1'] = x1

    # map clusters to appropriate labels
    cluster_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
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


if __name__ == "__main__":
    df = pd.read_csv('/content/dataset_cleaned.csv')
    # df = pd.read_csv('./datasets/dataset_translated.csv')

    # df['News'] = df['News'].apply(
    #    lambda x: text_preprocess.preprocess(x, remove_stopwords=True))

    X = tf_idf_vectorization(df)
    df_clustered = text_clustering(df, X)
    vizualice(df_clustered)

    df_clustered.to_csv('./datasets/dataset_clustered.csv', index=False)
