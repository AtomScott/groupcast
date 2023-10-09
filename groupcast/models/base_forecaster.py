import argparse

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from ..transforms import Resample


class BaseForecaster(pl.LightningModule):
    def __init__(
        self,
        optimizer_params={"lr": 0.001},
        scheduler_params={"T_0": 10, "eta_min": 1e-6},
        roll_out_steps=10,
        original_fps=25,
        inference_fps=25,
    ):
        super().__init__()
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params
        self.downsample = Resample(original_fps, inference_fps)
        self.upsample = Resample(inference_fps, original_fps)
        self.roll_out_steps = roll_out_steps
        self.original_fps = original_fps
        self.inference_fps = inference_fps

        self.mse_loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        x = self.transform(x)
        y_pred = self.model(x)
        y_pred = self.inverse_transform(y_pred)
        return y_pred

    def compute_base_metrics(self, y_pred, y_gt):
        return self.loss_fn(y_pred, y_gt)
    
    def log_losses(self):
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        pass
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.roll_out(x, n_steps=y.shape[1], y_gt=y) # train with teacher forcing
        loss = self.loss_fn(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.roll_out(x, n_steps=y.shape[1], y_gt=y) # validate with teacher forcing
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.roll_out(x, n_steps=y.shape[1]) # test without teacher forcing
        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss, logger=True)

    def configure_optimizers(self):
        """Defines the optimizer and, if necessary, the learning rate scheduler.
        By default, uses Adam. Can be overridden by specific forecasters."""
        optimizer = AdamW(self.parameters(), **self.optimizer_params)
        scheduler = CosineAnnealingWarmRestarts(optimizer, **self.scheduler_params)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def roll_out(self, x, n_steps=None, y_gt=None):
        next_positions = []

        for i in range(n_steps):
            next_position = self.model(x)

            if y_gt is not None:
                x = torch.cat([x[:, 1:, :], y_gt[:, i, :].unsqueeze(1)], dim=1)
            else:
                x = torch.cat([x[:, 1:, :], next_position.unsqueeze(1)], dim=1)
            next_positions.append(next_position)

        return torch.stack(next_positions, dim=1)

    def transform(self, x):
        """
        Optionally, transform input data to a different space.
        Useful if preprocessing involves normalization or other transformations.
        """
        x = self.downsample(x)
        return x

    def inverse_transform(self, y_pred):
        """
        Optionally, convert normalized/transformed predictions back to the original space.
        Useful if preprocessing involves normalization or other transformations.
        """
        # By default, do nothing and return predictions as is.
        y_pred = self.upsample(y_pred)
        return y_pred

    def compute_metrics(self, predictions, ground_truth):
        """
        Compute forecasting-specific metrics.
        Can be overridden by specific forecasters to add more metrics.
        """
        return self.loss_fn(predictions, ground_truth)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Add model-specific arguments to the CLI. Useful when using hyperparameter search or tuning tools."""
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
        # Add other model-specific arguments here
        return parser
