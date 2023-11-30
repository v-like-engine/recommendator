from benchmark.evaluate import evaluate_predictions
from models.recommender_svd.recommender_svd import RecommenderSVD


def find_best_k(dataset, targets, k_pool=(2, 3, 5, 8, 16, 32, 50), print_best=True):
    results = []
    for f in k_pool:
        predicted_ratings = RecommenderSVD(dataset).predict_ratings(f)
        results.append([f, evaluate_predictions(predicted_ratings, targets)])
    results.sort(key=lambda x: x[1]['rmse'])
    if print_best:
        print(f'Best number of parameters for SVD is {results[0][0]} with RMSE being {results[0][1]["rmse"]}',
              f'and MAE equal to {results[0][1]["mae"]}')
    return results[0][0], results[0][1]


def evaluate_model(dataset, targets, k=5):
    predicted_ratings = RecommenderSVD(dataset).predict_ratings(k)
    return evaluate_predictions(predicted_ratings, targets)
