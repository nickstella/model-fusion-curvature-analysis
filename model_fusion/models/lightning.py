from typing import Dict

import lightning
from torchmetrics import Accuracy
import torch
import torch.nn as nn

from model_fusion.models import ModelType


class BaseModel(lightning.LightningModule):
    """Base model for all models in this project"""
    def __init__(self, model_type: ModelType, model_hparams: Dict, loss_module: nn.Module = nn.CrossEntropyLoss,
                 *args, **kwargs):
        super().__init__()
        self.lightning_params = kwargs
        self.loss_module = loss_module()
        self.train_losses = []
        self.accuracy = Accuracy(task='multiclass', num_classes=model_hparams['num_classes'])

        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()

        self.model_type = model_type
        self.model_hparams = model_hparams
        self.model = model_type.get_model(**model_hparams)

    def forward(self, x):
        """Forward pass, returns logits"""
        x = self.model(x)
        return x

    def f_step(self, batch, batch_idx, train=True, log_metrics=True, *args, **kwargs):
        """Forward step, returns loss and logits"""
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_module(y_hat, y)

        if log_metrics:
            accuracy = self.accuracy(y_hat, y)
            metrics = {'loss': loss, 'accuracy': accuracy}
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
        optimizer_name = self.lightning_params.get('optimizer', 'adam')
        if optimizer_name == 'adam':
            optimizer = self.configure_adam()
        elif optimizer_name == 'sgd':
            optimizer = self.configure_sgd()
        else:
            raise ValueError(f'Optimizer {optimizer} not supported')

        if 'lr_scheduler' not in self.lightning_params:
            return optimizer
        if self.lightning_params['lr_scheduler'] == 'step':
            scheduler = self.configure_step_lr(optimizer)
        elif self.lightning_params['lr_scheduler'] == 'multistep':
            scheduler = self.configure_multistep_lr(optimizer)
        elif self.lightning_params['lr_scheduler'] == 'plateau':
            scheduler = self.configure_plateau_lr(optimizer)
        else:
            raise ValueError(f'LR scheduler {self.lightning_params["lr_scheduler"]} not supported')
        return [optimizer], [scheduler]

    def configure_adam(self):
        lr = self.lightning_params.get('lr', 1e-3)
        weight_decay = self.lightning_params.get('weight_decay', 0)

        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def configure_sgd(self):
        lr = self.lightning_params.get('lr', 1e-3)
        momentum = self.lightning_params.get('momentum', 0)
        weight_decay = self.lightning_params.get('weight_decay', 0)
        nesterov = self.lightning_params.get('nesterov', False)

        return torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)

    def configure_step_lr(self, optimizer):
        step_size = self.lightning_params.get('step_size', None)
        if step_size is None:
            raise ValueError('step_size size must be specified for StepLR')
        lr_decay_factor = self.lightning_params.get('lr_decay_factor', 0.1)

        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=lr_decay_factor)

    def configure_multistep_lr(self, optimizer):
        lr_decay_epochs = self.lightning_params.get('lr_decay_epochs', None)
        if lr_decay_epochs is None:
            raise ValueError('lr_decay_epochs must be specified for MultiStepLR')
        lr_decay_factor = self.lightning_params.get('lr_decay_factor', 0.1)

        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_epochs, gamma=lr_decay_factor)

    def configure_plateau_lr(self, optimizer):
        lr_decay_factor = self.lightning_params.get('lr_decay_factor', 0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_decay_factor, patience=7, verbose=True)
        lr_monitor_metric = self.lightning_params.get('lr_monitor_metric', 'val_loss')
        return {
            'scheduler': scheduler,
            'monitor': lr_monitor_metric
        }

    def log_metrics(self, metrics, train, prog_bar=True):
        if train:
            self.train_losses.append(metrics['loss'])
        prefix = "train_" if train else "val_"
        metrics_prefixed = {prefix + str(key): val for key, val in metrics.items()}
        self.log_dict(metrics_prefixed, prog_bar=prog_bar)

    def on_train_epoch_end(self) -> None:
        metrics = {'avg_train_loss': torch.stack(self.train_losses).mean()}
        self.log_dict(metrics, prog_bar=True)
        self.train_losses = []

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        # OPTIONAL
        # model specific args
        return parent_parser
