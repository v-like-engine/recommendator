"""
Configs and paths to use in the models.
"""

project_path = 'V:/Users/Vladislav Honour/recommendator'

'''
Please change the field above to the path/to/project on your device. Needed for model weights loading.


Fields below are used for the data loading
'''

names = ['user_id', 'movie_id', 'rating', 'timestamp']
user_col = names[0]
movie_col = names[1]
rating_col = names[2]

dataset_path = 'data/raw/ml-100k/'
data_file = 'u.data'
u1_base_file = 'u1.base'
u1_test_file = 'u1.test'


'''
GLocal-K model parameters
'''

experiment_dir = "/data/interim/experiment"
checkpoint_dir = "/data/interim/experiment/checkpoints"

# glocal config
NUM_WORKERS = 2
n_hid = 1000
n_dim = 5
n_layers = 3
gk_size = 5
lambda_2 = 20.  # l2 regularisation
lambda_s = 0.006
iter_p = 5  # optimization
iter_f = 5
epoch_p = 30
epoch_f = 80
dot_scale = 1.0  # scaled dot product
lr_pre = 0.1
lr_fine = 1.0
optimizer = 'lbfgs'  # lbfgs, adam
lr_scheduler = 'none'
weight_decay = 0.

'''
the variable below sets the seed
'''
seed = 42
