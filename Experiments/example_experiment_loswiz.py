import wandb

from model_fusion.train import setup_training
from model_fusion.datasets import DataModuleType
from model_fusion.models import ModelType
from model_fusion.config import BASE_DATA_DIR
import torch


def run_experiment(min_epochs=1, max_epochs=3, batch_size=32):
    datamodule_type = DataModuleType.CIFAR10
    datamodule_hparams = {'batch_size': batch_size, 'data_dir': BASE_DATA_DIR}

    model_type = ModelType.VGG11
    model_hparams = {'num_classes': 10, 'num_channels': 3, 'bias': False}
    lr = 0.1 * (batch_size / 32) * 0.025
    lightning_params = {'optimizer': 'sgd', 'lr': lr, 'momentum': 0.9, 'weight_decay': 0.0001, 'lr_scheduler': 'plateau', 'lr_decay_factor': 0.1, 'lr_monitor_metric': 'val_loss'}
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






