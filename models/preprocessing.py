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


def extract_from_pandas(data_pd):
    users, movies = [], []
    for i in range(data_pd.shape[0]):
        users.append(data_pd.iloc[i][user_col])
        movies.append(data_pd.iloc[i][movie_col])
    predictions = data_pd.rating.values
    return users, movies, predictions


def load_data_cil(path=dataset_path, file=data_file, frac=0.1):
    """
    Data preprocessing for Dataset creation
    :param path:
    :param file:
    :param frac:
    :return:
    """
    data_pd = pd.read_csv(path + file, sep='\t', names=names)
    users, movies, predictions = extract_from_pandas(data_pd)
    data = pd.DataFrame.from_dict(
        {user_col: users, movie_col: movies, "rating": predictions}
    )

    indices_u, indices_m = np.unique(data[user_col]), np.unique(data[movie_col])
    n_u = indices_u.size
    n_m = indices_m.size
    n_r = data.shape[0]
    indices_u = list(indices_u)
    indices_m = list(indices_m)

    udict = {}
    for i, u in enumerate(np.unique(data[user_col]).tolist()):
        udict[u] = i
    mdict = {}
    for i, m in enumerate(np.unique(data[movie_col]).tolist()):
        mdict[m] = i

    idx = np.arange(n_r)
    np.random.shuffle(idx)

    train_r = np.zeros((n_m, n_u), dtype="float32")
    test_r = np.zeros((n_m, n_u), dtype="float32")

    for i in range(n_r):
        u_id = data.loc[idx[i]][user_col]
        m_id = data.loc[idx[i]][movie_col]
        r = data.loc[idx[i]]["rating"]

        if i < int(frac * n_r):
            test_r[indices_m.index(m_id), indices_u.index(u_id)] = r
        else:
            train_r[indices_m.index(m_id), indices_u.index(u_id)] = r

    # masks indicating non-zero entries
    train_m = np.greater(train_r, 1e-12).astype("float32")
    test_m = np.greater(test_r, 1e-12).astype("float32")

    print("data matrix loaded")
    print("num of users: {}".format(n_u))
    print("num of movies: {}".format(n_m))
    print("num of training ratings: {}".format(n_r - int(frac * n_r)))
    print("num of test ratings: {}".format(int(frac * n_r)))

    return n_m, n_u, train_r, train_m, test_r, test_m


class CILDataset(torch.utils.data.Dataset):
    """
    pytorch Dataset for further casting to DataLoader
    """
    def __init__(self, data_path, file):
        self.data = load_data_cil(data_path, file)

    def __len__(self):
        return 1

    def __getitem__(self, _):
        return self.data


class CILDataLoader(torch.utils.data.DataLoader):
    """
    pytorch DataLoader creation for torch models
    """
    def __init__(self, file="u.data", data_path="data/raw/ml-100k", num_workers=8):
        super().__init__(
            CILDataset(data_path, file), batch_size=None, num_workers=num_workers
        )
