import numpy as np
import pandas as pd
from scipy.linalg import sqrtm

from models.preprocessing import MovieLensDataset


class RecommenderSVD:
    def __init__(self, data: MovieLensDataset):
        self.data = data

    def svd_factorize(self, k):
        M, x = self.data.mask_transform()
        U, s, V = np.linalg.svd(M, full_matrices=False)
        U, s, V = U[:, 0:k], sqrtm(np.diag(s)[0:k, 0:k]), V[0:k, :]
        factorized = np.dot(np.dot(U, s), np.dot(s, V))
        factorized = factorized + x
        return factorized

    def predict_ratings(self, n_components):
        predicted_ratings = self.svd_factorize(n_components)
        return pd.DataFrame(predicted_ratings, index=self.data.data.index, columns=self.data.data.columns)

    def generate_recommendations(self, user_id, k):
        predicted_ratings = self.predict_ratings(k)
        user_ratings = predicted_ratings.loc[user_id]
        user_ratings = user_ratings.sort_values(ascending=False)
        return user_ratings.head(k)
