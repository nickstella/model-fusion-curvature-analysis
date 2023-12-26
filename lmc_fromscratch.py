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

def combine_parameters(params1,params2,alpha):
    """
    Combine two sets of parameters with a convex combination.
    """
    combined_params = []
    for param1, param2 in zip(params1, params2):
        combined_param = alpha * param1 + (1 - alpha) * param2
        combined_params.append(torch.nn.Parameter(combined_param))
    return combined_params

def update_network_parameters(model, new_params):
    """
    Update the parameters of a PyTorch network with new parameters.

    Args:
        model (torch.nn.Module): The input neural network.
        new_params (list): A list of parameter tensors to replace the original parameters.

    Returns:
        None
    """
    # Make sure the number of new parameters matches the number of original parameters
    assert len(list(model.parameters())) == len(new_params), "Number of parameters does not match."

    # Update the model parameters with new_params
    for param, new_param in zip(model.parameters(), new_params):
        param.data.copy_(new_param)



data_module_type = DataModuleType.MNIST

#define two different batch size to discrimante among parents
data_module_hparams1 = {'batch_size': 32}
data_module_hparams2 = {'batch_size': 64}

model_type = ModelType.RESNET18
model_hparams = {'num_classes': 10, 'num_channels': 1}

wandb_tags = ['example']

model1, datamodule1, trainer1 = setup_training('example_experiment', model_type, model_hparams, data_module_type, data_module_hparams1, max_epochs=1, wandb_tags=wandb_tags)
model2, datamodule2, trainer2 = setup_training('example_experiment', model_type, model_hparams, data_module_type, data_module_hparams2, max_epochs=1, wandb_tags=wandb_tags)


#fit the parent models and retreive the parameters
trainer1.fit(model1, datamodule=datamodule1)
params1 = get_network_parameters(model1)

trainer2.fit(model2, datamodule=datamodule2)
params2 = get_network_parameters(model2)

print("PARENT 1 RESULTS:")
trainer1.test(model1, datamodule=datamodule1)
print("PARENT 2 RESULTS:")
trainer2.test(model2, datamodule=datamodule2)


# Computing the paramters on the linear path between the two parents. Here we do it once, later we will do it in a loop over alpha
alpha = 0.5  
combined_params = combine_parameters(params1,params2,alpha)

# Create a new model and set its parameters to the combined parameters
update_network_parameters(model1, combined_params)

print("COMBINED RESULTS:")
trainer1.test(model1, datamodule=datamodule1)
model1.eval()

# Initialize a variable to accumulate the training loss
total_loss = 0.0

# Iterate over the training data and compute predictions
for batch in datamodule1.train_dataloader():
    inputs, targets = batch

    # Forward pass
    predictions = model1(inputs)

    # Compute the loss
    loss = model1.loss_module(predictions, targets)
    total_loss += loss.item()

# Calculate the average training loss
average_loss = total_loss / len(datamodule1.train_dataloader())
print(f"Average training loss: {average_loss}")



wandb.finish()

