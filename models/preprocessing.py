import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import torch
from models.utils.config import *


class MovieLensDataset:
    def __init__(self, user_item_matrix: pd.DataFrame, mask=False, impute=False, impute_strategy='mean'):
        self.data = user_item_matrix
        self.mask_tile = None
        if mask:
            self.mask_transform()
        elif impute:
            if impute_strategy:
                self.impute_nan_by_simple_imputer(impute_strategy)

    def impute_nan_by_simple_imputer(self, strategy='median'):
        """
        transform data by imputing NaN values with the given strategy using sklearn SimpleImputer.
        In-place transformation.
        :param strategy: strategy parameter of SimpleImputer; supports mean, median, most_frequent, constant
        :return: None (in-place transformation)
        """
        imputer = SimpleImputer(strategy=strategy, fill_value=0)
        user_item_matrix_imputed = imputer.fit_transform(self.data.T).T
        self.data = pd.DataFrame(user_item_matrix_imputed, index=self.data.index, columns=self.data.columns)

    def mask_transform(self):
        """
        We need to remove NaN values in out utility matrix before applying SVD.
        So we use mask array from numpy, then fill it with mean values.
        Finally, we remove the average per item from all the entries, to obtain 0.0 at previously NaN values,
        and other values clearly indicating rating (positive or negative).

        Does not transform in-place, but returns the matrix
        :return: imputed utility matrix, per-item average (to add on the last steps)
        """
        utility_matrix = self.data.to_numpy().astype(float)
        mask = np.isnan(utility_matrix)
        masked_arr = np.ma.masked_array(utility_matrix, mask)
        item_means = np.mean(masked_arr, axis=0)
        utility_matrix = masked_arr.filled(item_means)
        x = np.tile(item_means, (utility_matrix.shape[0], 1))
        utility_matrix = utility_matrix - x
        self.mask_tile = x
        return utility_matrix, self.mask_tile

    def head(self, k):
        """
        Equivalent to pandas.DataFrame.head
        :param k: number of rows to show
        :return: data head with k rows
        """
        return self.data.head(k)

    def tail(self, k):
        """
        Equivalent to pandas.DataFrame.tail
        :param k: number of rows to show
        :return: data tail with k rows
        """
        return self.data.tail(k)

    def __len__(self):
        """
        Method to return number of rows
        :return: number of rows of data
        """
        num_rows = self.data.shape[0]
        return num_rows
