import wandb
from pathlib import Path
import wandb
from model_fusion.datasets import DataModuleType
from model_fusion.models import ModelType
from model_fusion.models.lightning import BaseModel
from model_fusion.config import BASE_DATA_DIR, CHECKPOINT_DIR
import model_fusion.lmc_utils as lmc
from pyhessian import hessian
import torch
import torch.nn as nn

def run_pyhessian(
        datamodule_type: DataModuleType, 
        model: BaseModel
    ):

    datamodule_hparams = {'batch_size': 51, 'data_dir': BASE_DATA_DIR}
    datamodule = datamodule_type.get_data_module(**datamodule_hparams)
    datamodule.prepare_data()
    datamodule.setup('fit')
    
    data = []
    for batch in datamodule.train_dataloader():
        x, y = batch
        data.append((x.cuda(), y.cuda()))

    criterion = nn.CrossEntropyLoss()
    model.eval()
    model = model.cuda()
    
    hessian_comp = hessian(model, criterion, data=data, cuda=True)
    
    return hessian_comp

    

if __name__ == '__main__':
    run_pyhessian()