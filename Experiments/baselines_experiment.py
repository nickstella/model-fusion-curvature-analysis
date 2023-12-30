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
        batch_size: int, 
        max_epochs: int, 
        datamodule_type: DataModuleType, 
        datamodule_hparams: dict, 
        model_type: ModelType, 
        model_hparams: dict,
        checkpointA: str,
        checkpointB: str,
        wandb_tag: str
    ):

    print("------- Setting up parameters -------")
    args = parameters.get_parameters()
    print("The parameters are: \n", args)

    print("------- Loading models -------")
    run = wandb.init()
    
    artifact = run.use_artifact(checkpointA, type='model')
    artifact_dir = artifact.download(root=CHECKPOINT_DIR)
    modelA = BaseModel.load_from_checkpoint(Path(artifact_dir)/"model.ckpt")

    artifact = run.use_artifact(checkpointB, type='model')
    artifact_dir = artifact.download(root=CHECKPOINT_DIR)
    modelB = BaseModel.load_from_checkpoint(Path(artifact_dir)/"model.ckpt")

    models = [modelA, modelB]
    
    print("Done loading all the models")

    # set seed for numpy based calculations
    NUMPY_SEED = 100
    np.random.seed(NUMPY_SEED)

    for model in models:
        model.eval()

    # run baselines
    print("------- Prediction based ensembling -------")
    prediction_ensembling_model = prediction_ensembling.get_prediction_ensemble(models)

    print("------- Naive ensembling of weights -------")
    vanilla_averaging_model = vanilla_averaging.get_weight_averaged_model(models)

    print("------- Evaluating models -------")
    experiment_name = f"{model_type.value}_{datamodule_type.value}_batch_size_{batch_size}_{wandb_tag}"
    wandb_tags = [f"{model_type.value}", f"{datamodule_type.value}", f"Batch size {batch_size}", f"{wandb_tag}"]
    
    datamodule, trainer = setup_testing(experiment_name, model_type, model_hparams, datamodule_type, datamodule_hparams, wandb_tags)

    datamodule.prepare_data()
    datamodule.setup('test')

    print("------- evaluating base models -------")
    for model in models:
        trainer.test(model, dataloaders=datamodule.test_dataloader())

    print("------- evaluating prediction ensembling -------")
    prediction_ensembling.evaluate_prediction_ensemble(prediction_ensembling_model, datamodule.test_dataloader(), loss_module=nn.CrossEntropyLoss())
    
    print("------- evaluating vanilla averaging -------")
    trainer.test(vanilla_averaging_model, dataloaders=datamodule.test_dataloader())

    wandb.finish()
    

if __name__ == '__main__':
    run_baselines()
