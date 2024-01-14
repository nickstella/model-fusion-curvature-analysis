import model_fusion.parameters as parameters
import numpy as np
import torch.nn as nn
from pathlib import Path
import wandb
from model_fusion.train import setup_testing
from model_fusion.datasets import DataModuleType
from model_fusion.models import ModelType
from model_fusion.models.lightning import BaseModel
from model_fusion.config import CHECKPOINT_DIR
from model_fusion.ot_fusion import compute_activations, wasserstein_ensemble
from lightning import seed_everything

def run_otfusion(
        batch_size: int,
        datamodule_type: DataModuleType,
        datamodule_hparams: dict,
        model_type: ModelType,
        model_hparams: dict,
        modelA: BaseModel,
        modelB: BaseModel,
        wandb_tag: str,
        is_vgg: bool = False
    ):
    seed_everything(42, workers=True)

    print("------- Setting up parameters -------")
    args = parameters.get_parameters()

    print(datamodule_hparams)

    if model_type == ModelType.VGG11:
        args.handle_skips = False
        args.acts_num_samples = 75

    print("The parameters are: \n", args)

    models = [modelA, modelB]

    print("------- OT model fusion -------")
    activations = compute_activations.get_model_activations(args, models, datamodule_type)
    otfused_model, aligned_base_model = wasserstein_ensemble.get_otfused_model(args, models, activations, datamodule_type, datamodule_hparams)

    print("------- Evaluating ot fusion model -------")
    experiment_name = f"{model_type.value}_{datamodule_type.value}_batch_size_{batch_size}_{wandb_tag}"
    wandb_tags = [f"{model_type.value}", f"{datamodule_type.value}", f"Batch size {batch_size}", f"{wandb_tag}"]

    datamodule, trainer = setup_testing(experiment_name, model_type, model_hparams, datamodule_type, datamodule_hparams, wandb_tags)

    datamodule.prepare_data()
    datamodule.setup('test')

    trainer.test(otfused_model, dataloaders=datamodule.test_dataloader())

    checkpoint_callback = trainer.checkpoint_callback
    checkpoint_callback.on_validation_end(trainer, otfused_model)

    wandb.finish()

    return otfused_model, aligned_base_model

if __name__ == '__main__':
    run_otfusion()
