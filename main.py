import pandas as pd
import argparse

from models.glocal_k.evaluate import evaluate_model, generate_recommendations
from models.glocal_k.train_loop import train_glocal_k
from models.preprocessing import MovieLensData, MovieLensDataLoader
from models.recommender_svd.evaluate import evaluate_model as evaluate_model_svd
from models.recommender_svd.recommender_svd import RecommenderSVD
from models.utils.config import *
from models.utils.utils import set_seed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Your script description')

    def t_or_f(arg):
        provided = str(arg).upper()
        if 'TRUE'.startswith(provided):
            return True
        elif 'FALSE'.startswith(provided):
            return False


    # Add the arguments
    parser.add_argument('--seed', type=int, default=None, help='The seed for random number generation')
    parser.add_argument('--baseline', type=t_or_f, default=True, help='Whether to use the baseline model')
    parser.add_argument('--glocal', type=t_or_f, default=True, help='Whether to use the glocal model')
    parser.add_argument('--train_glocal', type=t_or_f, default=False, help='Whether to train the glocal model')
    parser.add_argument('--evaluate', type=t_or_f, default=True, help='Whether to evaluate the model')
    parser.add_argument('--recommend', type=t_or_f, default=True, help='Whether to generate recommendations')
    parser.add_argument('--user_id', type=int, default=1, help='The user id for which to generate recommendations')
    parser.add_argument('--k_recommendations', type=int, default=5, help='The number of recommendations to generate')
    parser.add_argument('--n_components', type=int, default=16, help='The number of components for the glocal model')
    parser.add_argument('--weights', type=str, default='finetuning-epoch=79-fine_train_rmse=0.8498-fine_test_rmse=0.9031.ckpt', help='The filename of GLocal-K weights to use')

    # Parse the arguments
    args = parser.parse_args()
    print(args.baseline, args.glocal)
    set_seed(args.seed if args.seed is not None else seed)
    test = pd.read_csv(dataset_path + u1_test_file, sep='\t', names=names)
    if args.baseline:
        train = pd.read_csv(dataset_path + u1_base_file, sep='\t', names=names)
        user_item_matrix_train = train.pivot(index=user_col, columns=movie_col, values=rating_col)
        dataset = MovieLensData(user_item_matrix_train)
        if args.evaluate:
            print(f'SVD baseline approach with n_components {args.n_components} evaluation:')
            print(evaluate_model_svd(dataset, test, 16))
        if args.recommend:
            print(f'SVD baseline approach {args.k_recommendations} recommendations for user {args.user_id}:')
            print(RecommenderSVD(dataset).generate_recommendations(args.user_id, args.k_recommendations))
    if args.glocal:
        train = pd.read_csv(dataset_path + data_file, sep='\t', names=names)
        user_item_matrix_train = train.pivot(index=user_col, columns=movie_col, values=rating_col)
        dataset = MovieLensData(user_item_matrix_train).data.to_numpy().T
        best_checkpoint_filename = args.weights
        if args.train_glocal:
            best_checkpoint_filename = train_glocal_k().split('/')[-1].split('\\')[-1]
        dataloader = MovieLensDataLoader(file="u.data", num_workers=NUM_WORKERS)
        _, _, train_r, _, _, _ = next(iter(dataloader))
        if args.evaluate:
            print(evaluate_model(train_r, test, best_checkpoint_filename))
        if args.recommend:
            print(generate_recommendations(args.user_id, args.k_recommendations, train_r, best_checkpoint_filename))
