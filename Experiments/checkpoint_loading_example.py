from pathlib import Path
import wandb

from model_fusion.train import setup_training
from model_fusion.datasets import DataModuleType
from model_fusion.models import ModelType
from model_fusion.models.lightning import BaseModel
from model_fusion.config import BASE_DATA_DIR, CHECKPOINT_DIR

batch_size = 32
max_epochs = 1
datamodule_type = DataModuleType.CIFAR10
datamodule_hparams = {'batch_size': batch_size, 'data_dir': BASE_DATA_DIR}

model_type = ModelType.RESNET18
model_hparams = {'num_classes': 10, 'num_channels': 3, 'bias': False}

wandb_tags = ['RESNET-18', 'CIFAR_10', f"Batch size {batch_size}", "prediction ensembling"]
model, datamodule, trainer = setup_training(f'RESNET-18 CIFAR-10 B32', model_type, model_hparams, datamodule_type, datamodule_hparams, max_epochs=max_epochs, wandb_tags=wandb_tags)

for name, param in model.named_parameters():
    print(f"Parameter name: {name}, Shape: {param.shape}")

run = wandb.init()
artifact = run.use_artifact('model-fusion/Model Fusion/RESNET-18_CIFAR-10_B32:v0', type='model')
artifact_dir = artifact.download(root=CHECKPOINT_DIR)

model2 = BaseModel.load_from_checkpoint(Path(artifact_dir)/"model.ckpt")
                      
datamodule.prepare_data()

# datamodule.setup('fit')
# trainer.fit(model, train_dataloaders=datamodule.train_dataloader(), val_dataloaders=datamodule.val_dataloader())

datamodule.setup('test')
trainer.test(model, dataloaders=datamodule.test_dataloader())

wandb.finish()
