import os
import numpy as np
import pandas as pd
import tensorflow_hub as hub
from typing import Any, Tuple, List

TSFLOW_UNIV_ENC_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
UNPARSED_TRAIN_FILE_PATH = "data/amazon/train_40k.csv"
PARSED_TRAIN_FILE_PATH = "data/amazon/train_parsed_40k.csv"
UNPARSED_VALIDATION_FILE_PATH = "data/amazon/val_10k.csv"
PARSED_VALIDATION_FILE_PATH = "data/amazon/val_parsed_10k.csv"
PROD_ID_COL = "productId"
TEXT_COL = "Text"
TITLE_COL = "Title"
TARGETS_COL = "Targets"
CATEGORIES = ['Cat1', 'Cat2', 'Cat3']
CATEGORY_COL = "CATEGORY"
UNKNOWN_CAT = 'unknown'

def extract_cat_from_row(row: pd.Series) -> str:
    """
    A function which extracts the most specific
    category from a row. Raises a ValueError when
    no specific category is found.
    Args:
        row pd.Series A pandas dataframe row
    Returns:
        str The category name
    """
    categories = row[reversed(CATEGORIES)]
    for category in categories:
        if category != UNKNOWN_CAT:
            return category        

    raise ValueError(
        f"None of {categories} is",
        f"different than {UNKNOWN_CAT}"
    )

def create_embedding_columns(n_cols: int) -> List[str]:
    """
    A function which returns a list of embedding column names
    from 0 up until n_cols.
    Args:
        n_cols: int The number of embedding columns
    Returns:
        List[str] A list of embedding column names.
    """
    return [f"embed_{i}" for i in range(n_cols)]

def load_universal_encoder() -> Any:
    """
    A function which loads the google universal encoder
    from TensorHub. It will cache it after it runs the first
    time.
    Returns:
        A Google Universal Encoder instance.
    """
    os.environ['TFHUB_DOWNLOAD_PROGRESS'] = "1"
    encoder = hub.load(TSFLOW_UNIV_ENC_URL)
    return encoder

def read_unp_file(path: str) -> pd.DataFrame:
    """
    A function which reads an unparsed csv data file whose
    path is given in the parameters and returns a pandas
    dataframe representing that csv file. An unparsed file
    is a dataset whose embeddings haven't been calculated.
    Args:
        path: str A string representing a local path pointing
        to a dataset file.
    Returns:
        pd.DataFrame A pandas dataframe.
    """
    df = pd.read_csv(path, sep=",")
    cols_to_keep = {PROD_ID_COL, TITLE_COL, TEXT_COL}.union(CATEGORIES)
    df.drop(columns=set(df.columns)-cols_to_keep, axis=1, inplace=True)
    df.set_index(PROD_ID_COL, inplace=True)
    targets = df.apply(lambda row: f"{row[TITLE_COL]} {row[TEXT_COL]}", axis=1).values
    df[TARGETS_COL] = targets
    df.drop(columns=[TITLE_COL, TEXT_COL], axis=1, inplace=True)
    return df

def parse_file(in_path: str, out_path: str) -> pd.DataFrame:
    """
    A function which parses a dataset file whose path is given
    in `in_path` and ouputs a csv dataset into disk returning a
    pandas dataframe representing that dataset, which is filled
    with the embeddings calculated via the google tensorflow hub
    encoder.
    Args:
        in_path: str A file input path, to be read into a pandas
        dataframe and parsed to form embeddings.
        out_path: str A file output path where the parsed file will
        be persisted.
    Returns
        pd.DataFrame A pandas dataframe representing the parsed
        dataset file.
    """
    train_df = read_unp_file(path=in_path)
    encoder = load_universal_encoder()
    targets = train_df[TARGETS_COL].values

    embeddings = encoder(targets)
    vec_size = embeddings.shape[1]

    parsed_train_df = pd.DataFrame(
        data=embeddings.numpy(),
        columns=create_embedding_columns(n_cols=vec_size),
        index=train_df.index
    )

    parsed_train_df[CATEGORY_COL] = train_df.apply(lambda row: extract_cat_from_row(row), axis=1)
    parsed_train_df.to_csv(out_path, sep=",")
    return parsed_train_df

def read_parsed_train_file() -> pd.DataFrame:
    """
    A function which reads the parsed training dataset file.
    Returns:
        pd.DataFrame A pandas dataframe representing a parsed
        training dataset file with embedding vectors.
    """
    return pd.read_csv(PARSED_TRAIN_FILE_PATH, sep=",").set_index(PROD_ID_COL)

def read_parsed_validation_file() -> pd.DataFrame:
    """
    A function which reads the parsed csv validation
    dataset file path.
    Returns:
        pd.DataFrame A pandas dataframe representing a parsed
        training dataset file with embedding vectors.
    """
    return pd.read_csv(PARSED_VALIDATION_FILE_PATH, sep=",").set_index(PROD_ID_COL)

def load_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, int]:
    """
    A function which takes a parsed pandas dataframe which
    contains embeddings and returns the dataframe without those embeddings,
    indexed by product_id and with the category columns, the embeddings as 
    a numpy array ordered according to the pandas dataframe and the number
    of unique different categories.
    Args:
        df pd.DataFrame A parsed pandas with embeddings.
    Returns:
        Tuple[pd.DataFrame, np.ndarray, int] A tuple of 3 elements,
        a pandas dataframe with categories and indexed by product_id,
        the embeddings as a numpy arrays and the number of different
        categories in the dataset.
    """
    _, cols = df.shape
    vec_size = cols - 1 # 1 category column
    cols = create_embedding_columns(n_cols=vec_size)
    embeddings = df[cols].values
    df.drop(columns=cols, axis=1, inplace=True)
    n_categories = len(pd.unique(df[CATEGORY_COL].values.ravel()))
    return df, embeddings, n_categories

def load_training_data() -> Tuple[pd.DataFrame, np.ndarray, int]:
    """
    A function which loads the training data.
    Returns:
        Returns:
        Tuple[pd.DataFrame, np.ndarray, int] A tuple of 3 elements,
        a pandas dataframe with categories and indexed by product_id,
        the embeddings as a numpy arrays and the number of different
        categories in the dataset.
    """
    df = read_parsed_train_file()
    return load_data(df=df)

def load_validation_data()-> Tuple[pd.DataFrame, np.ndarray, int]:
    """
    A function which loads the validation data.
    Returns:
        Returns:
        Tuple[pd.DataFrame, np.ndarray, int] A tuple of 3 elements,
        a pandas dataframe with categories and indexed by product_id,
        the embeddings as a numpy arrays and the number of different
        categories in the dataset.
    """
    df = read_parsed_validation_file()
    return load_data(df=df)

def parse_validation_file() -> pd.DataFrame:
    """
    A function which parses the validation file returning
    a pandas dataframe representing the validation dataset.
    Returns:
        pd.DataFrame A pandas dataframe of the validation
        csv dataset
    """
    return parse_file(
        in_path=UNPARSED_VALIDATION_FILE_PATH,
        out_path=PARSED_VALIDATION_FILE_PATH
    )

def parse_train_file() -> pd.DataFrame:
    """
    A function which parses the validation file returning
    a pandas dataframe representing the training dataset.
    Returns:
        pd.DataFrame A pandas dataframe of the training
        csv dataset
    """
    return parse_file(
        in_path=UNPARSED_TRAIN_FILE_PATH,
        out_path=PARSED_TRAIN_FILE_PATH
    )
