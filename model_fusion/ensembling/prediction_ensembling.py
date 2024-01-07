import logging
from typing import Callable, List

import torch
from torch import nn


def get_prediction_ensemble(models: List[nn.Module], weights: torch.Tensor = None) -> Callable:
    """Returns the ensembled model as a lambda function"""
    if len(models) == 0:
        logging.error("No models provided")
        return
    
    if weights is None:
        weights = torch.ones(len(models))
    
    ensembled_model = lambda x: (
        torch.stack([weight * model(x) for weight, model in zip(weights, models)])
        ).sum(dim=0) / weights.sum()

    return ensembled_model

def evaluate_prediction_ensemble(ensembled_model, test_loader, loss_module): 
    
    test_loss = 0
    correct = 0
    total = 0

    for data, target in test_loader:
        output = ensembled_model(data)
        test_loss += loss_module(output, target).item()
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        total += target.size(0)

    test_loss = test_loss / len(test_loader)
    accuracy = correct / total * 100
    print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss, accuracy))
