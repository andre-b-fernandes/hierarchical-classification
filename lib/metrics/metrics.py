import pandas as pd
from lib.data.data import CATEGORY_COL


PREDICTION_COL = "Prediction"

def hit_ratio(df: pd.DataFrame) -> float:
    """
    A function which calculates the hit ratio of a results dataframe.
    Args:
        df pd.DataFrame A pandas dataframe with computed results
        and a prediction column for each product
    Returns:
        float The hit ratio, that is, the number of times it matched
        the correct category out of the total length in percentage.
    """
    counter = len(df[df[PREDICTION_COL] == df[CATEGORY_COL]])
    ratio = round((counter / len(df)), 4) * 100
    return ratio
    