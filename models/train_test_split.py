import pandas as pd

from models.utils.config import user_col, movie_col


def train_test_split_latest(data, test_ratio=0.2, users_col=user_col, movies_col=movie_col):
    users = data[users_col].unique()
    movies = data[movies_col].unique()
    test = pd.DataFrame(columns=data.columns)
    train = pd.DataFrame(columns=data.columns)
    test_ratio = test_ratio
    for u in users:
        temp = data[data[users_col] == u]
        n = len(temp)
        test_size = int(test_ratio*n)
        temp = temp.sort_values('timestamp').reset_index()

        test = pd.concat([test, temp.iloc[n-1-test_size :]])
        train = pd.concat([train, temp.iloc[: n-2-test_size]])
    return train, test, users, movies
