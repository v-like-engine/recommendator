import pandas as pd

from models.preprocessing import MovieLensDataset
from models.recommender_svd.evaluate import find_best_k
from models.utils.config import *

if __name__ == '__main__':
    baseline = True
    if baseline:
        train, test = (pd.read_csv(dataset_path + u1_base_file, sep='\t', names=names),
                       pd.read_csv(dataset_path + u1_test_file, sep='\t', names=names))
        user_item_matrix_train = train.pivot(index=user_col, columns=movie_col, values=rating_col)
        dataset = MovieLensDataset(user_item_matrix_train)
        print(find_best_k(dataset, test, [2, 15, 16, 17, 18]))
    else:
        pass
