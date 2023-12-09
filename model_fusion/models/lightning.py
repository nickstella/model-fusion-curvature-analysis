from typing import Dict, Callable

import lightning
import torch.nn as nn

from model_fusion.models import ModelType


class BaseModel(lightning.LightningModule):
    """An abstract base class for all models."""
    def __init__(self, model_type: ModelType, model_hparams: Dict, loss_module: nn.Module = nn.CrossEntropyLoss()):
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = model_type.get_model(**model_hparams)
        # Create loss module
        self.loss_module = loss_module
        # Example input for visualizing the graph in Tensorboard

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        # validation_step defines the train loop. It is independent of forward
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        # test_step defines the train loop. It is independent of forward
        raise NotImplementedError

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        raise NotImplementedError

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        # OPTIONAL
        # model specific args
        return parent_parser
