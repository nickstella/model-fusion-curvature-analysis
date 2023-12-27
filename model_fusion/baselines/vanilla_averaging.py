import torch
from pathlib import Path
import wandb

from model_fusion.train import setup_training
from model_fusion.datasets import DataModuleType
from model_fusion.models import ModelType
from model_fusion.models.lightning import BaseModel
from model_fusion.config import BASE_DATA_DIR, CHECKPOINT_DIR

def get_avg_parameters(networks, proportion=None):
    avg_weights = []
    for param_group in zip(*[net.parameters() for net in networks]):

        if proportion is not None:
            weighted_param_group = [param * proportion[i] for i, param in enumerate(param_group)]
            avg_param = torch.sum(torch.stack(weighted_param_group), dim=0)
        
        else:
            print("shape of stacked params is ", torch.stack(param_group).shape) # (2, 400, 784)
            avg_param = torch.mean(torch.stack(param_group), dim=0)
        
        avg_weights.append(avg_param)
    
    return avg_weights

def ensemble(args, networks, test_loader):

    # TODO change harcoded values
    
    # define network
    batch_size = 32
    max_epochs = 1
    datamodule_type = DataModuleType.CIFAR10
    datamodule_hparams = {'batch_size': batch_size, 'data_dir': BASE_DATA_DIR}

    model_type = ModelType.RESNET18
    model_hparams = {'num_classes': 10, 'num_channels': 3, 'bias': False}

    wandb_tags = ['RESNET-18', 'CIFAR_10', f"Batch size {batch_size}", "vanilla averaging"]

    model, datamodule, trainer = setup_training(f'RESNET-18 CIFAR-10 B32', model_type, model_hparams, datamodule_type, datamodule_hparams, max_epochs=max_epochs, wandb_tags=wandb_tags)

    # set the weights of the ensembled network
    proportion = [(1-args.ensemble_step), args.ensemble_step]
    avg_weights = get_avg_parameters(networks, proportion)
    
    for avg_param, (name, _) in zip(avg_weights, model.named_parameters()):
        model.state_dict()[name].copy_(avg_param.data)

    datamodule.prepare_data()
    datamodule.setup('test')
    trainer.test(model, dataloaders=datamodule.test_dataloader())

    wandb.finish()

    