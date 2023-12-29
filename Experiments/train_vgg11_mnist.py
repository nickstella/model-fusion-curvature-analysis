import wandb

from model_fusion.train import setup_training
from model_fusion.datasets import DataModuleType
from model_fusion.models import ModelType
from model_fusion.config import BASE_DATA_DIR


def train_vgg11_mnist(max_epochs=1, batch_size=32):
    datamodule_type = DataModuleType.MNIST
    # MNIST images are 28x28, so we need to resize it to 32x32 to match the input size of VGG-11 due to it having 5 maxpool layers
    datamodule_hparams = {'batch_size': batch_size, 'data_dir': BASE_DATA_DIR, 'resize': (32, 32)}

    model_type = ModelType.VGG11
    model_hparams = {'num_classes': 10, 'num_channels': 1, 'bias': False}

    wandb_tags = ['VGG-11', 'MNIST', f"Batch size {batch_size}"]

    model, datamodule, trainer = setup_training(f'VGG-11 MNIST B{batch_size}', model_type, model_hparams, datamodule_type, datamodule_hparams, max_epochs=max_epochs, wandb_tags=wandb_tags)

    datamodule.prepare_data()

    datamodule.setup('fit')
    trainer.fit(model, train_dataloaders=datamodule.train_dataloader(), val_dataloaders=datamodule.val_dataloader())

    datamodule.setup('test')
    trainer.test(model, dataloaders=datamodule.test_dataloader())

    wandb.finish()


if __name__ == '__main__':
    train_vgg11_mnist()
