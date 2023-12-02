from models.glocal_k.networks import GLocalKFine, GLocalKPre
from models.preprocessing import MovieLensDataLoader

import pytorch_lightning as pl
import time
from pathlib import Path
from shutil import copy
from pytorch_lightning.loggers import TensorBoardLogger

from models.utils.config import *


def train_glocal_k():
    """
    Function to train GLocal-K model.

    Partially taken from pytorch adaptation of official GLocal-Kernel code.
    Source of the adaptation can be accessed by any of the links in
    data/external/'GLocal-K implementation torch-adapted source.txt'
    Source of the original tensorflow-based implementation can be accessed by the link in
    data/external/'GLocal-K official tensorflow-based implementation.txt'

    Trains Global-Local Kernel model and saves checkpoints.
    Launch parameters and paths are set in models/utils/config.py.
    Initially set parameters are those recommended by the authors of source implementation, adjusted for the needs
    of current project.

    Evaluation results: model trained on the pre-set parameters (initial from config.py) reached RMSE = 0.86467
    on training set, with MAE being equal to 0.67775

    NOTE! For correct saving and loading procedures of the weights, change the project_path constant in config.py
    to the global path to your project root. E.g. 'C:/projects/recommendator'.
    Some checkpoints may fail to load due to the project path inconsistency. In that case, try training on the
    pre-set parameters with correct path constants on your machine. Results will be nearly reproduced
    (note the relation with random seed that is set manually).

    :return: whole global path to the best checkpoint obtained during training. Checkpoint has all the needed
    description and validation metric values in the filename
    """
    exp_dir = project_path + '/' + experiment_dir
    model_pre = f"nhid-{n_hid}-ndim--{n_dim}-layers-{n_layers}-lambda2-{lambda_2}-lambdas-{lambda_s}-iterp-{iter_p}-iterf-{iter_f}-gk-{gk_size}-epochp-{epoch_p}-epochf-{epoch_f}-dots-{dot_scale}_"
    model_dir = Path(exp_dir, f"{model_pre}/{time.time():.0f}")
    model_dir.mkdir(exist_ok=True, parents=True)
    model_dir.joinpath("results").mkdir()
    model_dir = model_dir.as_posix()

    print(f"Model configuration set for training: {model_pre}")
    movielens_dataloader = MovieLensDataLoader(file="u.data", num_workers=NUM_WORKERS)
    n_m, n_u, train_r, train_m, test_r, test_m = next(iter(movielens_dataloader))
    logger = TensorBoardLogger(save_dir=exp_dir, log_graph=True)

    glocal_k_pre = GLocalKPre(
        n_hid,
        n_dim,
        n_layers,
        lambda_2,
        lambda_s,
        iter_p,
        n_u,
        lr=lr_pre,
        optim=optimizer,
    )
    pretraining_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=f"{exp_dir}/checkpoints",
        filename="pretraining-{epoch}-{pre_train_rmse:.4f}-{pre_test_rmse:.4f}",
        monitor="pre_test_rmse",
        save_top_k=2,
        mode="min",
        save_last=True,
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    pretraining_trainer = pl.Trainer(
        callbacks=[pretraining_checkpoint, lr_monitor],
        max_epochs=epoch_p,
        log_every_n_steps=1,
        logger=logger,
    )
    pretraining_trainer.fit(glocal_k_pre, movielens_dataloader, movielens_dataloader)
    pre_ckpt = f"{exp_dir}/checkpoints/pre_last.ckpt"
    copy(pretraining_checkpoint.last_model_path, pre_ckpt)
    pre_ckpt = f"{exp_dir}/checkpoints/pre_best.ckpt"
    copy(pretraining_checkpoint.best_model_path, pre_ckpt)

    glocal_k_fine = GLocalKFine(
        gk_size,
        iter_f,
        dot_scale,
        n_m,
        pre_ckpt,
        lr=lr_fine,
        optim=optimizer,
    )
    finetuning_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=f"{exp_dir}/checkpoints",
        filename="finetuning-{epoch}-{fine_train_rmse:.4f}-{fine_test_rmse:.4f}",
        monitor="fine_test_rmse",
        save_top_k=2,
        mode="min",
        save_last=True,
    )
    finetuning_trainer = pl.Trainer(
        callbacks=[finetuning_checkpoint, lr_monitor],
        max_epochs=epoch_f,
        log_every_n_steps=1,
        logger=logger,
    )
    finetuning_trainer.fit(glocal_k_fine, movielens_dataloader, movielens_dataloader)
    return finetuning_checkpoint.best_model_path
