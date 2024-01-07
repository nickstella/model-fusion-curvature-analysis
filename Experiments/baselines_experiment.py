import model_fusion.parameters as parameters
import numpy as np
import torch.nn as nn
from pathlib import Path
import wandb
from model_fusion.train import setup_training, setup_testing
from model_fusion.datasets import DataModuleType
from model_fusion.models import ModelType
from model_fusion.models.lightning import BaseModel
from model_fusion.config import BASE_DATA_DIR, CHECKPOINT_DIR
from model_fusion.ensembling import prediction_ensembling, vanilla_averaging

def run_baselines(
        datamodule_type: DataModuleType, 
        datamodule_hparams: dict, 
        model_type: ModelType, 
        model_hparams: dict,
        modelA: BaseModel,
        modelB: BaseModel,
        wandb_tag: str
    ):

    models = [modelA, modelB]
    
    for model in models:
        model.eval()

    # run baselines
    print("------- Prediction based ensembling -------")
    prediction_ensembling_model = prediction_ensembling.get_prediction_ensemble(models)

    print("------- Naive ensembling of weights -------")
    vanilla_averaging_model = vanilla_averaging.get_weight_averaged_model(models)

    print("------- Evaluating baselines -------")
    batch_size = 1024
    datamodule_hparams = {'batch_size': batch_size, 'data_dir': BASE_DATA_DIR}

    experiment_name = f"{model_type.value}_{datamodule_type.value}_{wandb_tag}"
    wandb_tags = [f"{model_type.value}", f"{datamodule_type.value}", f"{wandb_tag}"]
    
    datamodule, trainer = setup_testing(experiment_name, model_type, model_hparams, datamodule_type, datamodule_hparams, wandb_tags)

    datamodule.prepare_data()
    datamodule.setup('test')

    print("------- Evaluating base models -------")
    for model in models:
        trainer.test(model, dataloaders=datamodule.test_dataloader())

    print("------- Evaluating prediction ensembling -------")
    prediction_ensembling.evaluate_prediction_ensemble(prediction_ensembling_model, datamodule.test_dataloader(), loss_module=nn.CrossEntropyLoss())
    
    print("------- Evaluating vanilla averaging -------")
    trainer.test(vanilla_averaging_model, dataloaders=datamodule.test_dataloader())

    wandb.finish()

    return vanilla_averaging_model
    

if __name__ == '__main__':
    run_baselines()
