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
