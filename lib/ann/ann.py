from typing import Tuple
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.python.data import Dataset
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

from lib.metrics.metrics import PREDICTION_COL


class ANN(Model):
    """ An artificial neural network model using the tensorflow subclassing API.
    It takes into account the number of available categories
    which we can predict. 
    """
    def __init__(self, n_categories: int):
        """
        RNN's constructor.
        Args:
            n_categories: int The number of categories.
        """
        super().__init__()
        self.dense = Dense(units=1000, activation="relu")
        self.out_layer = Dense(units=n_categories, activation="softmax")
        self.compile(
            optimizer=Adam(),
            loss=CategoricalCrossentropy(from_logits=True),
            metrics=[CategoricalAccuracy()]
        )

    def call(self, inputs):
        """
        A function which is executed during training at each
        iteration.
        Args:
            inputs: A tensor which will be provided as an input
            which will be an embedding vector.
        """
        x = self.dense(inputs)
        return self.out_layer(x)


def train(embeddings: np.ndarray, categories: np.ndarray) -> Tuple[ANN, dict]:
    """
    A function which trains a ANN model returning an instance of the ANN class
    with a fitted object. It trains it with the provided embeddings and 
    categories which are processed and turned into one-hot-encodings.
    Args:
        embeddings: np.ndarray A numpy array with embedding vectos
        categories: np.ndarray A numpy array made of category names.
    Returns:
        Tuple[ANN, dict] A tuple object with a ANN instance and dictionary which
        maps an integer encoding of a category to its name.  
    """
    label_encoder = LabelEncoder()
    encoded_cats = label_encoder.fit_transform(categories)
    one_hot_cats = to_categorical(encoded_cats, num_classes=len(set(categories)))
    dataset = Dataset.from_tensor_slices((embeddings, one_hot_cats))
    dataset = dataset.batch(64).prefetch(64)
    ann = ANN(n_categories=len(set(categories)))
    ann.fit(dataset, epochs=20)
    return ann, dict(zip(encoded_cats, categories))

def predict(ann: ANN, embeddings: np.ndarray, df: pd.DataFrame, categories: dict):
    """
    A function which uses an ANN instance to predict categories through the provided
    embeddings,and a category mapper dictionary.
    Args:
        ann: Ann A ANN instance used to predict categories.
        embeddings: np.ndarray A numpy array with embedding vectos
        df: pd.DataFrame A pandas dataframe where the final results will be put.
        categories: dict A dictionary which maps an integer encoding of a category to its name.
    Returns:
        pd.DataFrame A Pandas dataframe with category predictions.
    """
    predictions = ann.predict(embeddings)
    indices = np.argmax(predictions, axis=1)
    mapped_predictions = list(map(lambda index: categories[index], indices))
    df[PREDICTION_COL] = mapped_predictions
    return df