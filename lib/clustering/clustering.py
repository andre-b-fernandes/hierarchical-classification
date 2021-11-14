from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict

from lib.data.data import CATEGORY_COL
from lib.metrics import PREDICTION_COL

RAW_PREDICTION = "Raw_Prediction"

def train(matrix: np.ndarray, n_categories: int) -> KMeans:
    """
    A function which fits a k means clustering scikit learn object
    with the embeddings matrix provided in the params and
    according with n_categories setting the number of clusters.
    Args:
        matrix: np.ndarray A numpy array representing the embeddings
        matrix 
        n_categories: int The number of clusters to use.
    Returns:
        sklearn.cluster.KMeans A scikit learn kmeans object. 
    """
    k_means = KMeans(
        n_clusters=n_categories,
        verbose=1
    )
    return k_means.fit(matrix)

def build_histogram(df: pd.DataFrame) -> defaultdict:
    """
    A function which builds a histogram for the number of
    matches a category has for each cluster. A cluster
    might get matched with multiple categories, this way
    we can evaluate the highest category match per cluster and
    later on associate each cluster with a category.
    Args:
        df pd.DataFrame A pandas dataframe with computed results
        and a prediction column for each product
    Returns:
        defaultdict A default dictionary whose default value is 
        another defaultdictionary whose default value is int(0)
    """
    histogram = defaultdict(
        lambda: defaultdict(int)
    )

    for _, row in tqdm(df.iterrows(), desc="Building label histogram....", total=len(df)):
        prediction = row[RAW_PREDICTION]
        label = row[CATEGORY_COL]
        histogram[prediction][label] += 1 
    
    return histogram

def build_label_mapper(df: pd.DataFrame) -> dict:
    """ A function which maps a cluster label prediction
    to a category based on a computed histogram, taking into
    account a provided results pandas dataframe.
    Args:
        df pd.DataFrame A pandas dataframe with computed results
        and a prediction column for each product
    Returns:
        dict A dictionary which maps a cluster label to a category name.
    """
    histogram = build_histogram(df=df)
    mapper = dict()
    for prediction, count in tqdm(histogram.items(), desc="Building label mapper...", total=len(histogram)):
        label = max(count, key=lambda key: histogram[prediction][key])
        mapper[prediction] = label

    return mapper

def predict(k_means: KMeans, df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """
    A function which uses a k means scikit lean object to return a new
    pandas dataframe with a prediction column with the labels outputed
    by that object using an embeddings matrix.
    Args:
        k_means: scikit_learn.cluster.KMeans A scikit learn kmeans object which will
        be used to compute the clusters.
        df: pd.DataFrame A pandas dataframe where we will put the predictions.
        embeddings: np.ndarray The embeddings for which we want to
        compute the respective cluster.
    """
    df[RAW_PREDICTION] = k_means.predict(embeddings)
    cluster_mapper = build_label_mapper(df=df)
    df[PREDICTION_COL] = df.apply(lambda row: cluster_mapper[row[RAW_PREDICTION]], axis=1)
    df.drop(columns=[RAW_PREDICTION], axis=0)
    return df
