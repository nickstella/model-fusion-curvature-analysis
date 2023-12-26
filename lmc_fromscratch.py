import wandb

from model_fusion.train import setup_training
from model_fusion.datasets import DataModuleType
from model_fusion.models import ModelType
import torch
import numpy as np
import copy


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


def combine_parameters(params1, params2, alpha):
    """
    Combine two sets of parameters with a convex combination.
    """
    combined_params = []
    for param1, param2 in zip(params1, params2):
        combined_param = alpha * param1.clone() + (1 - alpha) * param2.clone()
        combined_params.append(combined_param)
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

def compute_loss(model,datamodule):


    model.eval()
    # Initialize a variable to accumulate the training loss
    total_loss = 0.0

    # Iterate over the training data and compute predictions
    for batch in datamodule.train_dataloader():
       inputs, targets = batch

       # Forward pass
       predictions = model(inputs)
       # Compute the loss
       loss = model.loss_module(predictions, targets)
       total_loss += loss.item()

    # Calculate the average training loss
    average_loss = total_loss / len(datamodule.train_dataloader())
    return average_loss


def compute_max_and_avg_loss(model1, model2):
    """
    Computes the maximum and average loss on the linear path among 2 parent networks
    """
    losses = []
    params1 = get_network_parameters(model1)
    params2 = get_network_parameters(model2)

    print("params 1", params1[0][0][0])
    print("params 2", params2[0][0][0])

    # Initialize the fused model as a copy of model1 
    fused_model = copy.deepcopy(model1)

    alphas = np.linspace(0, 1, 10)

    # Iterate over the linear path and compute the maximum loss
    for alpha in alphas:

        # Compute the parameters on the linear path
        combined_params = combine_parameters(params1, params2, alpha)

        # Update the model parameters with the combined parameters
        update_network_parameters(fused_model, combined_params)

        # Compute the loss on the linear path
        loss = compute_loss(fused_model, datamodule1)
        losses.append(loss)
        print(f"Alpha: {alpha:.2f}, Loss: {loss:.2f}")

    # Compute the maximum and average loss
    max_loss = np.max(losses)
    average_loss = np.mean(losses)

    return max_loss, average_loss


data_module_type = DataModuleType.MNIST

#define two different batch size to discrimante among parents
data_module_hparams1 = {'batch_size': 32}
data_module_hparams2 = {'batch_size': 64}

model_type = ModelType.RESNET18
model_hparams = {'num_classes': 10, 'num_channels': 1}

wandb_tags = ['example']

model1, datamodule1, trainer1 = setup_training('example_experiment', model_type, model_hparams, data_module_type, data_module_hparams1, max_epochs=1, wandb_tags=wandb_tags)
model2, datamodule2, trainer2 = setup_training('example_experiment', model_type, model_hparams, data_module_type, data_module_hparams2, max_epochs=1, wandb_tags=wandb_tags)

#fit the parent models 
trainer1.fit(model1, datamodule=datamodule1)
trainer2.fit(model2, datamodule=datamodule2)

#compute the max and average loss on the linear path
max_loss, avg_loss = compute_max_and_avg_loss(model1, model2)
print(f"Maximum loss: {max_loss:.2f}, Average loss: {avg_loss:.2f}")


wandb.finish()

