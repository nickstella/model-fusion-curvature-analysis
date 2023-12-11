from typing import Dict

import lightning
import torch
import torch.nn as nn

from model_fusion.models import ModelType


class BaseModel(lightning.LightningModule):
    """Base model for all models in this project"""
    def __init__(self, model_type: ModelType, model_hparams: Dict, loss_module: nn.Module = nn.CrossEntropyLoss(), lr=1e-3, *args, **kwargs):
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.lr = lr
        self.loss_module = loss_module
        self.save_hyperparameters()
        self.model = model_type.get_model(**model_hparams)

    def forward(self, x):
        """Forward pass, returns logits"""
        x = self.model(x)
        return x

    def f_step(self, batch, batch_idx, train=True, *args, **kwargs):
        """Forward step, returns loss and logits"""
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_module(y_hat, y)

        metrics = {'loss': loss}
        self.log_metrics(metrics, train)
        return loss, y_hat

    def training_step(self, batch, batch_idx, *args, **kwargs):
        """Training step, returns loss"""
        return self.f_step(batch, batch_idx, train=True, *args, **kwargs)[0]

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        """Validation step, returns loss"""
        return self.f_step(batch, batch_idx, train=False, *args, **kwargs)[0]

    def test_step(self, batch, batch_idx, *args, **kwargs):
        """Test step, returns loss"""
        return self.f_step(batch, batch_idx, train=False, *args, **kwargs)[0]

    def configure_optimizers(self):
        """Configure optimizers"""
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def log_metrics(self, metrics, train, prog_bar=True):
        prefix = "train_" if train else "val_"
        metrics_prefixed = {prefix + str(key): val for key, val in metrics.items()}
        self.log_dict(metrics_prefixed, prog_bar=prog_bar)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        # OPTIONAL
        # model specific args
        return parent_parser
