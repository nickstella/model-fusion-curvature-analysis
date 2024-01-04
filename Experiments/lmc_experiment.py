import wandb
from pathlib import Path
import wandb
from model_fusion.datasets import DataModuleType
from model_fusion.models import ModelType
from model_fusion.models.lightning import BaseModel
from model_fusion.config import BASE_DATA_DIR, CHECKPOINT_DIR
import model_fusion.lmc_utils as lmc

def run_lmc(
        datamodule_type: DataModuleType, 
        modelA: BaseModel,
        modelB: BaseModel,
        granularity: int = 20,
    ):

    datamodule_hparams = {'batch_size': 1024, 'data_dir': BASE_DATA_DIR}
    datamodule = datamodule_type.get_data_module(**datamodule_hparams)
    datamodule.prepare_data()
    datamodule.setup('fit')

    loss_model1,loss_model2,barrier, alpha_max = lmc.compute_losses_and_barrier(modelA, modelB, datamodule, granularity)
    print(f"Loss model 1: {loss_model1:.5f}, Loss model 2: {loss_model2:.5f}, Alpha argmax: {alpha_max:.5f}")
    print(f"Barrier: {barrier:.5f}")

if __name__ == '__main__':
    run_lmc()