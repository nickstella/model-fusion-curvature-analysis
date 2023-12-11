import wandb

from model_fusion.train import setup_training
from model_fusion.datasets import DataModuleType
from model_fusion.models import ModelType


def run_experiment():
    data_module_type = DataModuleType.MNIST
    data_module_hparams = {'batch_size': 32}

    model_type = ModelType.RESNET18
    model_hparams = {'num_classes': 10, 'num_channels': 1}

    wandb_tags = ['example']

    model, datamodule, trainer = setup_training('example_experiment', model_type, model_hparams, data_module_type, data_module_hparams, max_epochs=1, wandb_tags=wandb_tags)

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

    wandb.finish()


if __name__ == '__main__':
    run_experiment()
