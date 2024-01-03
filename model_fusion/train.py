from typing import Dict, List
import lightning as L
from lightning import Callback, Trainer
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
from model_fusion.config import WANDB_PROJECT_NAME
from model_fusion.datasets import DataModuleType
from model_fusion.models import ModelType
from model_fusion.models.lightning import BaseModel


def setup_training(experiment_name: str,
                   model_type: ModelType, model_hparams: Dict, lightning_params: Dict,
                   datamodule_type: DataModuleType, datamodule_hparams: dict,
                   min_epochs: int,
                   max_epochs: int,
                   wandb_tags: List[str],
                   early_stopping: bool = True,
                   *args, **kwargs):
    # Create the model
    model = BaseModel(model_type=model_type, model_hparams=model_hparams, **lightning_params)
    # Create the datamodule
    datamodule = datamodule_type.get_data_module(**datamodule_hparams)
    # Create the logger
    logger_config = {'model_hparams': model_hparams} | {'datamodule_hparams': datamodule_hparams} | {'lightning_params': lightning_params} | {'min_epochs': min_epochs, 'max_epochs': max_epochs, 'model_type': model_type, 'datamodule_type': datamodule_type, 'early_stopping': early_stopping}
    logger = get_wandb_logger(experiment_name, logger_config, wandb_tags)
    # Callbacks for the trainer
    callbacks = []
    # Add early stopping callback
    if early_stopping:
        monitor = kwargs.pop('monitor', 'val_loss')
        patience = kwargs.pop('patience', 15)
        callbacks.append(EarlyStopping(monitor=monitor, patience=patience))

    checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode="max")
    callbacks.append(checkpoint_callback)
    # Create the trainer
    trainer = L.Trainer(min_epochs=min_epochs, max_epochs=max_epochs, logger=logger, callbacks=callbacks, deterministic='warn', *args, **kwargs)
    return model, datamodule, trainer

def setup_testing(experiment_name: str,
                  model_type: ModelType, model_hparams: dict,
                  datamodule_type: DataModuleType, datamodule_hparams: dict,
                  wandb_tags: List[str],
                  *args, **kwargs):

    # Create the datamodule
    datamodule = datamodule_type.get_data_module(**datamodule_hparams)
    # Create the logger
    logger_config = model_hparams | datamodule_hparams | {'model_type': model_type, 'datamodule_type': datamodule_type}
    logger = get_wandb_logger(experiment_name, logger_config, wandb_tags)
    # Create the trainer
    trainer = L.Trainer(max_epochs=0, logger=logger, *args, **kwargs)
    return datamodule, trainer


def get_wandb_logger(experiment_name: str, hparams: dict, wandb_tags: List[str] = []):
    wandb_config = {
        # set the wandb project where this run will be logged
        'project': WANDB_PROJECT_NAME,
        # name of the run on wandb
        'name': experiment_name,
        # 'checkpoint_name': experiment_name, #TODO change naming convention for checkpoints
        # track hyperparameters and run metadata
        'config': hparams,
        'tags': wandb_tags
    }
    return WandbLogger(log_model="all", **wandb_config)
