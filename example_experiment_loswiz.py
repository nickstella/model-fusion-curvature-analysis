import wandb

from model_fusion.train import setup_training
from model_fusion.datasets import DataModuleType
from model_fusion.models import ModelType
<<<<<<< Updated upstream
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

    
=======
from model_fusion.config import BASE_DATA_DIR
import torch


def train_resnet18_cifar10(min_epochs=1, max_epochs=2, batch_size=32):
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

    save_path = r'C:\Users\filos\OneDrive\Desktop\ETH\model-fusion\saved_models\cifar10_resnet18_bs32.t7'

    torch.save(model.state_dict(), save_path)

>>>>>>> Stashed changes

    wandb.finish()


if __name__ == '__main__':
<<<<<<< Updated upstream
    run_experiment()
=======
    train_resnet18_cifar10()






>>>>>>> Stashed changes
