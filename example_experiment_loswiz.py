import wandb

from model_fusion.train import setup_training
from model_fusion.datasets import DataModuleType
from model_fusion.models import ModelType
import lmc_utils as lmc
import torch


def run_experiment():
    data_module_type = DataModuleType.CIFAR10
    data_module_hparams1 = {'batch_size': 16}
    data_module_hparams2 = {'batch_size': 32}


    model_type = ModelType.RESNET18
    model_hparams = {'num_classes': 10, 'num_channels': 1}

    wandb_tags = ['example']

    model1, datamodule1, trainer1 = setup_training('example_experiment', model_type, model_hparams, data_module_type, data_module_hparams1, max_epochs=1, wandb_tags=wandb_tags)
    #model2, datamodule2, trainer2 = setup_training('example_experiment', model_type, model_hparams, data_module_type, data_module_hparams2, max_epochs=1, wandb_tags=wandb_tags)

    trainer1.fit(model1, datamodule=datamodule1)
    #trainer2.fit(model2, datamodule=datamodule2)


    save_path = r'C:\Users\filos\OneDrive\Desktop\ETH\model-fusion\saved_models\model.t7'

    # Save the model using torch.save
    torch.save(model1, save_path)

    

    wandb.finish()


if __name__ == '__main__':
    run_experiment()