import model_fusion.parameters as parameters
import torch
import numpy as np
# from models.data import get_dataloader
# import models.routines as routines
# import baselines.prediction_ensemble
# import baselines.vanilla_avg
import model_fusion.ot_fusion.wasserstein_ensemble as wasserstein_ensemble
import model_fusion.ot_fusion.compute_activations as compute_activations
# import os
# import utils
# import numpy as np
# import sys
# import torch
# import models.train as cifar_train
# from tensorboardX import SummaryWriter

from pathlib import Path
import wandb

from model_fusion.train import setup_training
from model_fusion.datasets import DataModuleType
from model_fusion.models import ModelType
from model_fusion.models.lightning import BaseModel
from model_fusion.config import BASE_DATA_DIR, CHECKPOINT_DIR

if __name__ == '__main__':

    print("------- Setting up parameters -------")
    args = parameters.get_parameters()
    print("The parameters are: \n", args)

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # TODO loading configuration
    # config, second_config = utils._get_config(args)
    # args.config = config
    # args.second_config = second_config

    
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
    datamodule.setup('test')
    test_loader = datamodule.test_dataloader()

    models = [modelA, modelB]
    datamodule.setup('test')

    # for model in models:
    #     trainer.test(model, dataloaders=datamodule.test_dataloader())

    wandb.finish()
    print("Done loading all the models")


# set seed for numpy based calculations
NUMPY_SEED = 100
np.random.seed(NUMPY_SEED)

print("------- OT model fusion -------")
activations = compute_activations.get_model_activations(args, models, config=None)
geometric_model = wasserstein_ensemble.geometric_ensembling(args, models, test_loader, activations)
