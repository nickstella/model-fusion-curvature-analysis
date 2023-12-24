import wandb

from model_fusion.train import setup_training
from model_fusion.datasets import DataModuleType
from model_fusion.models import ModelType
import torch


def get_network_parameters(model):
    """
    Get the parameters of a PyTorch network.

    Args:
        model (torch.nn.Module): The input neural network.

    Returns:
        A list of parameter tensors.
    """
    params = []
    for param in model.parameters():
        params.append(param.detach())
    return params

def set_network_parameters(model, params):
    """
    Set the parameters of a PyTorch network.

    Args:
        model (torch.nn.Module): The input neural network.
        params (list): A list of parameter tensors.

    Returns:
        None
    """
    for model_param, input_param in zip(model.parameters(), params):
        model_param.data.copy_(input_param)



data_module_type = DataModuleType.MNIST


#define two different batch size to discrimante among parents
data_module_hparams1 = {'batch_size': 32}
data_module_hparams2 = {'batch_size': 64}

model_type = ModelType.RESNET18
model_hparams = {'num_classes': 10, 'num_channels': 1}

wandb_tags = ['example']

model1, datamodule1, trainer1 = setup_training('example_experiment', model_type, model_hparams, data_module_type, data_module_hparams1, max_epochs=1, wandb_tags=wandb_tags)
model2, datamodule2, trainer2 = setup_training('example_experiment', model_type, model_hparams, data_module_type, data_module_hparams2, max_epochs=1, wandb_tags=wandb_tags)


#fit the models and retreive the parameters
trainer1.fit(model1, datamodule=datamodule1)
params1 = get_network_parameters(model1)

print("params1: ", params1[0])

trainer2.fit(model2, datamodule=datamodule2)
params2 = get_network_parameters(model2)

print("params2: ", params2[0])


# Create a convex combination of the parameters, here we simply do it once. Later we will let alpha vary
alpha = 0.5  
combined_params = []
for param1, param2 in zip(params1, params2):
    combined_param = alpha * param1 + (1 - alpha) * param2
    combined_params.append(torch.nn.Parameter(combined_param))


#check
print("combined params",combined_params[0])
#indeed they correspond to the average here

wandb.finish()

