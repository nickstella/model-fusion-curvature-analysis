import wandb

from model_fusion.train import setup_training
from model_fusion.datasets import DataModuleType
from model_fusion.models import ModelType
import lmc_utils as lmc
from model_fusion.config import BASE_DATA_DIR


def run_experiment():
    datamodule_type = DataModuleType.MNIST
    datamodule_hparams1 = {'batch_size': 16, 'data_dir': BASE_DATA_DIR}
    datamodule_hparams2 = {'batch_size': 32, 'data_dir': BASE_DATA_DIR}

    model_type = ModelType.RESNET18
    model_hparams = {'num_classes': 10, 'num_channels': 1, 'bias': False}
    lr = 0.01
    lightning_params = {'optimizer': 'sgd', 'lr': lr, 'momentum': 0.9, 'weight_decay': 0.0001, 'lr_scheduler': 'plateau', 'lr_decay_factor': 0.1, 'lr_monitor_metric': 'val_loss'}
    min_epochs=0
    max_epochs=1


    wandb_tags = ['example']

    model1, datamodule1, trainer1 = setup_training('example_experiment', model_type, model_hparams, lightning_params, datamodule_type, datamodule_hparams1, min_epochs=min_epochs, max_epochs=max_epochs, wandb_tags=wandb_tags)
    model2, datamodule2, trainer2 = setup_training('example_experiment', model_type, model_hparams, lightning_params, datamodule_type, datamodule_hparams2, min_epochs=min_epochs, max_epochs=max_epochs, wandb_tags=wandb_tags)

    datamodule1.prepare_data()
    datamodule1.prepare_data()

    datamodule1.setup('fit')
    datamodule2.setup('fit')

    trainer1.fit(model1, train_dataloaders=datamodule1.train_dataloader(), val_dataloaders=datamodule1.val_dataloader())
    trainer2.fit(model2, train_dataloaders=datamodule2.train_dataloader(), val_dataloaders=datamodule2.val_dataloader())

    loss_model1,loss_model2,barrier, alpha_max = lmc.compute_losses_and_barrier(model1,model2, datamodule2,granularity=5)
    print(f"Loss model 1: {loss_model1:.5f}, Loss model 2: {loss_model2:.5f}, Alpha argmax: {alpha_max:.5f}")
    print(f"Barrier: {barrier:.5f}")

    wandb.finish()


if __name__ == '__main__':
    run_experiment()