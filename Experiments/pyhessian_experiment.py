from model_fusion.datasets import DataModuleType
from model_fusion.models.lightning import BaseModel
from model_fusion.config import BASE_DATA_DIR, CHECKPOINT_DIR
from pyhessian import hessian
from pyhessian import density_plot
import torch.nn as nn
import numpy as np

def run_pyhessian(
        datamodule_type: DataModuleType, 
        model: BaseModel
    ):

    datamodule_hparams = {'batch_size': 512, 'data_dir': BASE_DATA_DIR}
    datamodule = datamodule_type.get_data_module(**datamodule_hparams)
    datamodule.prepare_data()
    datamodule.setup('fit')
    
    hessian_dataloader = []
    for i, (inputs, labels) in enumerate(datamodule.train_dataloader()):
        hessian_dataloader.append((inputs.cuda(), labels.cuda()))
        if i ==  5:
            break

    criterion = nn.CrossEntropyLoss()

    model = model.cuda()
    model.eval()
    
    hessian_comp = hessian(model,
                           criterion,
                           dataloader=hessian_dataloader,
                           cuda=True)
    
    top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
    print("The top Hessian eigenvalue of this model is %.4f"%top_eigenvalues[-1])

    trace = hessian_comp.trace()

    print('\n***Trace: ', np.mean(trace))

    density_eigen, density_weight = hessian_comp.density()

    density_plot.get_esd_plot(density_eigen, density_weight)
    

if __name__ == '__main__':
    run_pyhessian()