import wandb
from lightning.pytorch import seed_everything

from model_fusion.train import setup_training
from model_fusion.datasets import DataModuleType
from model_fusion.models import ModelType
from model_fusion.config import BASE_DATA_DIR


def train_vgg11_cifar10(min_epochs=50, max_epochs=200, batch_size=32, model_seed=42, data_seed=42, learning_rate=0.1, data_augmentation=True):
    seed_everything(model_seed, workers=True)

    datamodule_type = DataModuleType.CIFAR10
    datamodule_hparams = {'batch_size': batch_size, 'data_dir': BASE_DATA_DIR, 'seed': data_seed, 'data_augmentation': data_augmentation}

    model_type = ModelType.VGG11
    model_hparams = {'num_classes': 10, 'num_channels': 3, 'bias': False}

    lightning_params = {'optimizer': 'sgd', 'lr': learning_rate, 'momentum': 0.9, 'weight_decay': 0.0001, 'lr_scheduler': 'plateau', 'lr_decay_factor': 0.1, 'lr_monitor_metric': 'val_loss', 'model_seed': model_seed, 'data_seed': data_seed}

    wandb_tags = ['VGG-11', 'CIFAR_10', f"Batch size {batch_size}"]

    model, datamodule, trainer = setup_training(f'VGG-11 CIFAR-10 B{batch_size}', model_type, model_hparams, lightning_params, datamodule_type, datamodule_hparams, min_epochs=min_epochs, max_epochs=max_epochs, wandb_tags=wandb_tags)

    datamodule.prepare_data()

    datamodule.setup('fit')
    trainer.fit(model, train_dataloaders=datamodule.train_dataloader(), val_dataloaders=datamodule.val_dataloader())

    datamodule.setup('test')
    trainer.test(model, dataloaders=datamodule.test_dataloader())

    wandb.finish()


if __name__ == '__main__':
    train_vgg11_cifar10()
