import wandb
from model_fusion.train import setup_training
from model_fusion.datasets import DataModuleType
from model_fusion.models import ModelType
import torch
import numpy as np
import copy
import tqdm


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

    Args:
        params1 (list): A list of parameter tensors.
        params2 (list): A list of parameter tensors.
        alpha (float): The convex combination weight.

    Returns:
        A list of parameter tensors.
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

    for param, new_param in zip(model.parameters(), new_params):
        assert torch.all(torch.eq(param.data, new_param)), "Parameters are not updated correctly."

def compute_loss(model,datamodule):
    """
    Compute the average train loss of a PyTorch network given its datamodule.

    Args:
        model (torch.nn.Module): The input neural network.
        datamodule (torch.utils.data.DataLoader): The input datamodule.
    
    Returns:
        average_loss (float): The average training loss.
    """
    # Initialize a variable to accumulate the training loss
    total_loss = 0.0

    # Iterate over the training data and compute loss
    for batch in datamodule.train_dataloader():
       loss, y_hat = model.f_step(batch, 0, train=True, log_metrics=False)
       total_loss += loss.item()

    # Calculate the average training loss
    average_loss = total_loss / len(datamodule.train_dataloader())
    return average_loss


def compute_max_and_avg_loss(model1, model2, datamodule, granularity = 20):
    """
    Computes the maximum and average loss on the linear path among 2 parent networks by defining a linspace of size granularity

    Args:
        model1 (torch.nn.Module): The first input neural network.
        model2 (torch.nn.Module): The second input neural network.
        datamodule1 or 2 (torch.utils.data.DataLoader): The input datamodule of the network with the larger batch size.
        granularity (int): The number of points on the linear path.

    Returns:
        max_loss (float): The maximum loss on the linear path.
        average_loss (float): The average loss on the linear path.
    """
    losses = []
    params1 = get_network_parameters(model1)
    params2 = get_network_parameters(model2)

    # Initialize the fused model as a copy of model1 
    fused_model = copy.deepcopy(model1)

    alphas = np.linspace(0, 1, granularity)


    # Iterate over the linear path and compute the maximum loss
    for alpha in alphas:

        # Compute the parameters on the linear path
        combined_params = combine_parameters(params1, params2, alpha)

        # Update the model parameters with the combined parameters
        update_network_parameters(fused_model, combined_params)

        # Compute the loss on the linear path

        with torch.no_grad():
            loss = compute_loss(fused_model, datamodule)
            losses.append(loss)
            print(f"Alpha: {alpha:.2f}, Train average loss: {loss:.5f}")

    # Compute the maximum and average loss
    max_loss = np.max(losses)
    average_loss = np.mean(losses)

    return max_loss, average_loss