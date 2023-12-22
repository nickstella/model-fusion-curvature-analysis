import model_fusion.train as train
from model_fusion.config import WANDB_PROJECT_NAME
from model_fusion.datasets import DataModuleType
from model_fusion.models import ModelType
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from model_fusion.models.lightning import BaseModel
import wandb
import lightning as L

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

def compute_loss(network, loss_metric, dataloader):
    network.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs, targets

            outputs = network(inputs)
            loss = loss_metric(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)

def compute_max_and_avg_loss(model1, model2, loss_metric, dataloader):
    """
    Computes the maximum and average loss on the linear path among 2 network weights

    Args:
        model1, model2 (torch.nn.Module): The 2 neural network.
        loss_metric (function): The loss function to use.

    Returns:
        A tuple of 2 floats: the maximum and average loss on the linear path.
    """

    weights1 = get_network_parameters(model1)
    weights2 = get_network_parameters(model2)

    #weights1 = model1.state_dict()
    #weights2 = model2.state_dict()

    losses = []
    alphas = np.linspace(0, 1, 10000)

    for alpha in alphas:
        weights_comb = alpha * weights1 + (1 - alpha) * weights2
        model_comb = set_network_parameters(model1, weights_comb)
        loss = compute_loss(model_comb, loss_metric, dataloader)

        losses.append(loss)

    max_loss = np.max(losses)
    avg_loss = np.mean(losses)

    return max_loss, avg_loss


# Train the model 1
trained_model1, data_module1, trainer1 = train.setup_training('cacca',
                   ModelType.RESNET18, {'num_classes': 10, 'num_channels': 1},
                DataModuleType.MNIST, {'batch_size': 32},
                   1,
                   ['example'])
# Train the model
trainer1.fit(trained_model1, data_module1)
# Access trained weights
#trained_weights1 = trained_model1.state_dict()

# Train the model 2
trained_model2, data_module2, trainer2 = train.setup_training('cacca',
                   ModelType.RESNET18, {'num_classes': 10, 'num_channels': 1},
                DataModuleType.MNIST, {'batch_size': 16},
                   2,
                   ['example'])
# Train the model
trainer2.fit(trained_model2, data_module2)
# Access trained weights
#trained_weights2 = trained_model2.state_dict()

dataloader = DataLoader(MNIST('data', train=True, download=True), batch_size=32, shuffle=True)
maxloss, avgloss = compute_max_and_avg_loss(trained_model1, trained_model2, loss_metric = torch.nn.CrossEntropyLoss(), dataloader = dataloader)
print("Max loss: " + str(maxloss))
print("\nAvg loss: " + str(avgloss))
