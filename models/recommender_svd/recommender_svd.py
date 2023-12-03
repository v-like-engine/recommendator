import numpy as np
import pandas as pd
from scipy.linalg import sqrtm

from models.preprocessing import MovieLensData


class RecommenderSVD:
    """
    Recommender class that uses SVD
    """
    def __init__(self, data: MovieLensData):
        self.data = data

    def svd_factorize(self, k):
        """
        Main function to apply SVD
        :param k: n_components hyperparameter for SVD
        :return: predicted ratings
        """
        M, x = self.data.mask_transform()
        U, s, V = np.linalg.svd(M, full_matrices=False)
        U, s, V = U[:, 0:k], sqrtm(np.diag(s)[0:k, 0:k]), V[0:k, :]
        factorized = np.dot(np.dot(U, s), np.dot(s, V))
        factorized = factorized + x
        return factorized

    def predict_ratings(self, n_components):
        """
        Easy-to-go function to obtain predicted ratings
        :param n_components: hyperparameter for SVD
        :return: predicted ratings pandas.DataFrame
        """
        predicted_ratings = self.svd_factorize(n_components)
        return pd.DataFrame(predicted_ratings, index=self.data.data.index, columns=self.data.data.columns)

    def generate_recommendations(self, user_id, k):
        """
        Function to generate top k recommendations for user user_id using the SVD
        by predicting on dataset
        :param user_id: id of user
        :param k: number of recommendations
        :return: pd.DataFrame top k predictions (movie_id | predicted rating)
        """
        predicted_ratings = self.predict_ratings(k)
        user_ratings = predicted_ratings.loc[user_id]
        user_ratings = user_ratings.sort_values(ascending=False)
        return user_ratings.head(k)
