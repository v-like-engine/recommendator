## Problem

Provide a reliable recommendation system to recommend movies to users. MovieLens-100k dataset is available for training.
Define evaluation metrics and create a model to solve the problem

## Solution

Two approaches were utilized in the present project. Let us start from evaluation technique, then we will proceed to approaches.

### Evaluation

RMSE (root mean squared error) and MAE (mean absolute error) are considered as evaluation metrics. The less - the better.
Metrics measure how predicted ratings are close to the target ones.

### Baseline

Simple SVD used as a baseline approach, resulting in RMSE = 0.9933 on test set (u1.test)

### Advanced model

Global-Local Kernel ([paper](https://arxiv.org/pdf/2108.12184.pdf)) is used as a model to outperform the baseline.
Model with the best weights obtained (available on [hugging face](https://huggingface.co/v-like/glocal-k-weights/tree/main)) result in RMSE = 0.8577

To load weights, please download all the weights added to hugging face by the commit named
> Upload most promising locally-trained model checkpoint. RMSE on u1.test: 0.86467

and put them to the directory (relative to project root)
> /data/interim/experiment/checkpoints/

To make everything work, change the value of project_path variable at (relative to project root)
> /models/utils/config.py

If any errors due to the path are raised, please consider training model with the pre-defined parameter,
it must give similar evaluation scores (maybe shifted a little bit due to the random weights)

## How to launch

### Quick start
To launch the models, please run
> pip install -r requirements.txt

in command line to install all the libraries used in the project.

Then, run
> python main.py --seed 42 --baseline True --glocal True --train_glocal False --evaluate True --recommend True --user_id 1 --k_recommendations 5 --n_components 16

to launch the models with recommended parameters and obtain evaluation score and 5 recommendations for user 1 from both models (with the weights from hugging face, no training)

Different weights can be assigned via --weights argument (provide only filename of weights .ckpt file, include extension)

### Arguments

> --seed

sets manual seed. If omitted, sets seed to the one from config.py

> --baseline
> --glocal

show whether we want to run baseline or not, run glocal-k or not (correspondingly; use True or False in any capitalizing case)

> --train_glocal

if True, trains Glocal-K from scratch, uses the config.py hyperparameters. If False, load glocal from weights. Omitted if --glocal is False

> --evaluate --recommend

determine printed results. if set to True, perform evaluation using metrics and recommendations, correspondingly

> --user_id --k_recommendations 

parameters for recommendations. user_id is id of users to recommend for, k is number of recommendations. Omitted if --recommend is False

>  --n_components

parameter n_components of SVD.

> --weights

filename of weights to be used while loading the GLocal-K model. Omitted if --glocal is False or --train_glocal is True

Default parameters are shown in the Quick start section example run