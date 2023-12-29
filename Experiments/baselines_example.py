import model_fusion.parameters as parameters
import torch
import numpy as np

from pathlib import Path
import wandb

from model_fusion.train import setup_training
from model_fusion.datasets import DataModuleType
from model_fusion.models import ModelType
from model_fusion.models.lightning import BaseModel
from model_fusion.config import BASE_DATA_DIR, CHECKPOINT_DIR

from model_fusion.ensembling import prediction_ensembling, vanilla_averaging

def run_baselines():

    print("------- Setting up parameters -------")
    args = parameters.get_parameters()
    print("The parameters are: \n", args)


    print("------- Loading models -------")

    batch_size = 32
    max_epochs = 1
    datamodule_type = DataModuleType.CIFAR10
    datamodule_hparams = {'batch_size': batch_size, 'data_dir': BASE_DATA_DIR}

    model_type = ModelType.RESNET18
    model_hparams = {'num_classes': 10, 'num_channels': 3, 'bias': False}

    wandb_tags = ['RESNET-18', 'CIFAR_10', f"Batch size {batch_size}", "prediction ensembling"]
    _, datamodule, trainer = setup_training(f'RESNET-18 CIFAR-10 B32', model_type, model_hparams, datamodule_type, datamodule_hparams, max_epochs=max_epochs, wandb_tags=wandb_tags)

    run = wandb.init()
    
    artifact = run.use_artifact('model-fusion/Model Fusion/model-8907soe2:v0', type='model')
    artifact_dir = artifact.download(root=CHECKPOINT_DIR)
    modelA = BaseModel.load_from_checkpoint(Path(artifact_dir)/"model.ckpt")

    artifact = run.use_artifact('model-fusion/Model Fusion/model-8907soe2:v0', type='model')
    artifact_dir = artifact.download(root=CHECKPOINT_DIR)
    modelB = BaseModel.load_from_checkpoint(Path(artifact_dir)/"model.ckpt")
                    
    datamodule.prepare_data()
    test_loader = datamodule.test_dataloader()

    models = [modelA, modelB]
    datamodule.setup('test')

    for model in models:
        trainer.test(model, dataloaders=datamodule.test_dataloader())

    wandb.finish()
    print("Done loading all the models")


    # set seed for numpy based calculations
    NUMPY_SEED = 100
    np.random.seed(NUMPY_SEED)

    # run baselines
    print("------- Prediction based ensembling -------")
    prediction_acc = prediction_ensembling.ensemble(args, models, test_loader)

    print("------- Naive ensembling of weights -------")
    vanilla_averaging.ensemble(args, models)

if __name__ == '__main__':

    run_baselines()
