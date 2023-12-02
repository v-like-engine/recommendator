from math import sqrt

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.utils.config import user_col, movie_col, rating_col


def evaluate_predictions(pred_df: pd.DataFrame, targets: pd.DataFrame):
    """
    Calculate and return metrics of how close predicted ratings to the target ones.
    The lower values are obtained in both MAE and RMSE, the better.
    MAE is mean absolute error
    RMSE is root mean squared error
    Please consider looking up in the knowledge bases at the internet for their formulas
    :param pred_df: dataframe matrix of predicted ratings; user_id should be row indices, movie_id are column names
    :param targets: test set dataframe in format [user_id | movie_id | rating | timestamp]
    :return: dictionary of metrics calculated on test set
    """
    predictions = []
    for _, row in targets.iterrows():
        user = int(row[user_col])
        item = int(row[movie_col])
        if item in pred_df.columns:
            predictions.append(pred_df.iloc[user - 1][item])
        else:
            predictions.append(np.mean(pred_df.iloc[user - 1]))
    mae = mean_absolute_error(targets[rating_col], predictions)
    rmse = sqrt(mean_squared_error(targets[rating_col], predictions))
    return {'mae': mae, 'rmse': rmse}
