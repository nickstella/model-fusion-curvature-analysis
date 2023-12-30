import wandb
import torch
from model_fusion.train import setup_training
from model_fusion.datasets import DataModuleType
from model_fusion.models import ModelType
from model_fusion.config import BASE_DATA_DIR


def train_resnet18_cifar10(min_epochs=200, max_epochs=300, batch_size=256):
    datamodule_type = DataModuleType.CIFAR10
    datamodule_hparams = {'batch_size': batch_size, 'data_dir': BASE_DATA_DIR}

    model_type = ModelType.RESNET18
    model_hparams = {'num_classes': 10, 'num_channels': 3, 'bias': False}
    lightning_params = {'optimizer': 'sgd', 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 0.0001, 'lr_scheduler': 'multistep', 'lr_decay_factor': 0.1, 'lr_decay_epochs': [150, 250]}
    wandb_tags = ['RESNET-18', 'CIFAR_10', f"Batch size {batch_size}"]

    model, datamodule, trainer = setup_training(f'RESNET-18 CIFAR-10 B{batch_size}', model_type, model_hparams, lightning_params, datamodule_type, datamodule_hparams, min_epochs=min_epochs, max_epochs=max_epochs, wandb_tags=wandb_tags)

    datamodule.prepare_data()

    datamodule.setup('fit')
    trainer.fit(model, train_dataloaders=datamodule.train_dataloader(), val_dataloaders=datamodule.val_dataloader())

    datamodule.setup('test')
    trainer.test(model, dataloaders=datamodule.test_dataloader())

    wandb.finish()


if __name__ == '__main__':
    train_resnet18_cifar10()
