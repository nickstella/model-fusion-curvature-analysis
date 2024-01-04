import wandb

from model_fusion.train import setup_training
from model_fusion.datasets import DataModuleType
from model_fusion.models import ModelType
from model_fusion.config import BASE_DATA_DIR
import torch


def run_experiment(min_epochs=3, max_epochs=5, batch_size=32):
    datamodule_type = DataModuleType.CIFAR10
    datamodule_hparams = {'batch_size': batch_size, 'data_dir': BASE_DATA_DIR}

    model_type = ModelType.VGG11
    model_hparams = {'num_classes': 10, 'num_channels': 3, 'bias': False}
    lightning_params = {'optimizer': 'sgd', 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 0.0001, 'lr_scheduler': 'multistep', 'lr_decay_factor': 0.1, 'lr_decay_epochs': [150, 250]}
    wandb_tags = ['Loss', 'visualization', 'example']

    model, datamodule, trainer = setup_training('Saving plot for losviz', model_type, model_hparams, lightning_params, datamodule_type, datamodule_hparams, min_epochs=min_epochs, max_epochs=max_epochs, wandb_tags=wandb_tags)

    datamodule.prepare_data()

    datamodule.setup('fit')
    trainer.fit(model, train_dataloaders=datamodule.train_dataloader(), val_dataloaders=datamodule.val_dataloader())

    datamodule.setup('test')
    trainer.test(model, dataloaders=datamodule.test_dataloader())

    wandb.finish()

    save_path = r'C:\Users\filos\OneDrive\Desktop\ETH\loss-landscape\sidak\cifar10_vgg11_bs32.t7'

    torch.save(model.state_dict(), save_path)


    wandb.finish()


if __name__ == '__main__':
    run_experiment()






