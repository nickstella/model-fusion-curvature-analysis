from model_fusion.datasets import DataModuleType
from model_fusion.models.lightning import BaseModel
from model_fusion.config import BASE_DATA_DIR, CHECKPOINT_DIR
from pyhessian import hessian
from model_fusion.plot_density import get_esd_plot
import torch.nn as nn
import numpy as np

def run_pyhessian(
        datamodule_type: DataModuleType, 
        model: BaseModel, 
        num_batches : int = 10,
        compute_top_eigenvalues: bool = True,
        compute_trace: bool = True,
        compute_density: bool = False,
        figure_name: str = 'example.pdf'
    ):

    datamodule_hparams = {'batch_size': 32, 'data_dir': BASE_DATA_DIR}
    datamodule = datamodule_type.get_data_module(**datamodule_hparams)
    datamodule.prepare_data()
    datamodule.setup('test')
    dataloader = datamodule.test_dataloader()
    
    hessian_dataloader = []
    for i, (inputs, labels) in enumerate(dataloader):
        hessian_dataloader.append((inputs.to(model.device), labels.to(model.device)))
        if i ==  num_batches:
            break

    criterion = nn.CrossEntropyLoss()

    model = model.cuda()
    model.eval()
    
    hessian_comp = hessian(model, criterion, dataloader=hessian_dataloader, cuda=True)
    
    if compute_top_eigenvalues:
        top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=1)
        print("The top Hessian eigenvalue of this model is %.4f"%top_eigenvalues[-1])

    if compute_trace:
        trace = hessian_comp.trace()
        print('\n***Trace: ', np.mean(trace))

    if compute_density:
        density_eigen, density_weight = hessian_comp.density()
        get_esd_plot(density_eigen, density_weight,figure_name)
    

if __name__ == '__main__':
    run_pyhessian()