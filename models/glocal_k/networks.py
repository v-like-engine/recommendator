import torch
from typing import Optional
import optuna
import pytorch_lightning as pl

from models.glocal_k.kernels import GlobalKernel, global_conv, LocalKernel


class GLocalKPre(pl.LightningModule):
    """
    Network part for Fine-tuning stage, uses pre-trained GLocalKPre with saved weights, and utilizes GlobalKernel
    from kernels.py

    Partially taken from pytorch adaptation of official GLocal-Kernel code.
    Source of the adaptation can be accessed by any of the links in
    data/external/'GLocal-K implementation torch-adapted source.txt'
    Source of the original tensorflow-based implementation can be accessed by the link in
    data/external/'GLocal-K official tensorflow-based implementation.txt'
    """
    def __init__(self, n_hid, n_dim, n_layers, lambda_2, lambda_s, iter_p, n_u, lr: float = 0.1,
                 trial: Optional[optuna.trial.Trial] = None, optim: Optional[str] = "lbfgs",
                 scheduler: Optional[str] = "none", **kwargs,):
        super().__init__()
        self.save_hyperparameters()

        self.iter_p = iter_p
        self.local_kernel = LocalKernel(n_layers, n_u, n_hid, n_dim, torch.sigmoid, lambda_s, lambda_2)
        self.lr = lr
        self.trial = trial
        self.optim = optim
        self.scheduler = scheduler

    def forward(self, x):
        """
        Usual forward function of torch module, returns predictions made on x
        :param x: data
        :return: predictions by module
        """
        return self.local_kernel(x)

    def training_step(self, batch, batch_idx):
        _, _, train_r, train_m, _, _ = batch

        pred_p, reg_losses = self(train_r)
        diff = train_m * (train_r - pred_p)
        loss_p = torch.sum(diff**2) / 2 + reg_losses
        return loss_p

    def validation_step(self, batch, batch_idx):
        _, _, train_r, train_m, test_r, test_m = batch

        pred_p, _ = self(train_r)

        error_train = (train_m * (torch.clip(pred_p, 1.0, 5.0) - train_r) ** 2).sum() / torch.sum(train_m)
        train_rmse = torch.sqrt(error_train)

        error = (test_m * (torch.clip(pred_p, 1.0, 5.0) - test_r) ** 2).sum() / torch.sum(test_m)
        test_rmse = torch.sqrt(error)

        self.log("pre_train_rmse", train_rmse)
        self.log("pre_test_rmse", test_rmse)
        if self.trial is not None:
            self.trial.report(test_rmse.item(), step=self.global_step)

    def configure_optimizers(self):
        """
        Different optimizers are available originally.
        However, we have stuck to lbfgs since it has shown the best performance
        :return: optimizer and scheduler in a dictionary format
        """
        if self.optim == "adam":
            optimizer = torch.optim.AdamW(self.local_kernel.parameters(), lr=self.lr)
        elif self.optim == "lbfgs":
            optimizer = torch.optim.LBFGS(
                self.local_kernel.parameters(),
                max_iter=self.iter_p,
                history_size=10,
                lr=self.lr,
            )
        elif self.optim == "sgd":
            optimizer = torch.optim.SGD(self.local_kernel.parameters(), lr=self.lr)
        else:
            raise ValueError(
                "Only adam, lbfgs, and sgd options are possible for optimizer."
            )

        if self.scheduler == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, gamma=0.995
            )
        elif self.scheduler == "reducelronplateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=4, factor=0.5, min_lr=1e-3
            )
        elif self.scheduler == "none":
            return optimizer
        else:
            raise ValueError(f"Unkown lr scheduler: {self.scheduler}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "pre_test_rmse"},
        }


class GLocalKFine(pl.LightningModule):
    """
    Network part for Fine-tuning stage, uses pre-trained GLocalKPre with saved weights, and utilizes GlobalKernel
    from kernels.py

    Partially taken from pytorch adaptation of official GLocal-Kernel code.
    Source of the adaptation can be accessed by any of the links in
    data/external/'GLocal-K implementation torch-adapted source.txt'
    Source of the original tensorflow-based implementation can be accessed by the link in
    data/external/'GLocal-K official tensorflow-based implementation.txt'
    """
    def __init__(self, gk_size, iter_f, dot_scale, n_m, local_kernel_checkpoint, lr: float = 0.1,
                 trial: Optional[optuna.trial.Trial] = None, optim: Optional[str] = "lbfgs",
                 scheduler: Optional[str] = "none", *args, **kwargs,):
        super().__init__()
        self.save_hyperparameters()

        self.iter_f = iter_f

        self.local_kernel = GLocalKPre.load_from_checkpoint(local_kernel_checkpoint)
        self.local_kernel.mode = "train"

        self.global_kernel = GlobalKernel(n_m, gk_size, dot_scale)
        self.lr = lr
        self.trial = trial
        self.optim = optim
        self.scheduler = scheduler

    def forward(self, x):
        """
        Usual forward function of torch module, returns predictions made on x
        :param x: data
        :return: predictions by module
        """
        pred, _ = self.local_kernel(global_conv(x, self.global_kernel(self.local_kernel(x)[0])))
        return pred

    def training_step(self, batch, batch_idx):
        _, _, train_r, train_m, _, _ = batch

        pred_f, reg_losses = self.local_kernel(global_conv(train_r, self.global_kernel(self.local_kernel(train_r)[0])))

        diff = train_m * (train_r - pred_f)
        loss_f = torch.sum(diff**2) / 2 + reg_losses

        return loss_f

    def validation_step(self, batch, batch_idx):
        _, _, train_r, train_m, test_r, test_m = batch

        pred_f = self(train_r)

        error_train = (train_m * (torch.clip(pred_f, 1.0, 5.0) - train_r) ** 2).sum() / torch.sum(train_m)
        train_rmse = torch.sqrt(error_train)

        error = (test_m * (torch.clip(pred_f, 1.0, 5.0) - test_r) ** 2).sum() / torch.sum(test_m)
        test_rmse = torch.sqrt(error)

        self.log("train_rmse", train_rmse)
        self.log("test_rmse", test_rmse)
        self.log("fine_train_rmse", train_rmse)
        self.log("fine_test_rmse", test_rmse)
        if self.trial is not None:
            self.trial.report(test_rmse.item(), step=self.global_step)

    def configure_optimizers(self):
        """
        Different optimizers are available originally.
        However, we have stuck to lbfgs since it has shown the best performance
        :return: optimizer and scheduler in a dictionary format
        """
        if self.optim == "adam":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        elif self.optim == "lbfgs":
            optimizer = torch.optim.LBFGS(
                self.parameters(), max_iter=self.iter_f, history_size=10, lr=self.lr
            )
        elif self.optim == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError(
                "Only adam, lbfgs, and sgd options are possible for optimizer."
            )
        if self.scheduler == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, gamma=0.995
            )
        elif self.scheduler == "reducelronplateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=4, factor=0.5, min_lr=1e-3
            )
        elif self.scheduler == "none":
            return optimizer
        else:
            raise ValueError(f"Unkown lr scheduler: {self.scheduler}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "test_rmse"},
        }
