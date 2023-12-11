from typing import Dict, List
import lightning as L
from lightning import Callback, Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb

from model_fusion.config import WANDB_PROJECT_NAME
from model_fusion.datasets import DataModuleType
from model_fusion.models import ModelType
from model_fusion.models.lightning import BaseModel


def setup_training(experiment_name: str,
                   model_type: ModelType, model_hparams: Dict,
                   datamodule_type: DataModuleType, datamodule_hparams: dict,
                   max_epochs: int,
                   wandb_tags: List[str], *args, **kwargs):
    # Create the model
    model = BaseModel(model_type=model_type, model_hparams=model_hparams)
    # Create the datamodule
    datamodule = datamodule_type.get_data_module(**datamodule_hparams)
    # Create the logger
    logger_config = model_hparams | datamodule_hparams | {'max_epochs': max_epochs, 'model_type': model_type, 'datamodule_type': datamodule_type}
    logger = get_wandb_logger(experiment_name, logger_config, wandb_tags)
    # Create the trainer
    trainer = L.Trainer(max_epochs=max_epochs, logger=logger, *args, **kwargs)
    return model, datamodule, trainer


def get_wandb_logger(experiment_name: str, hparams: dict, wandb_tags: List[str] = []):
    wandb_config = {
        # set the wandb project where this run will be logged
        'project': WANDB_PROJECT_NAME,
        # name of the run on wandb
        'name': experiment_name,
        # track hyperparameters and run metadata
        'config': hparams,
        'tags': wandb_tags
    }
    return WandbLogger(log_model="all", **wandb_config)
