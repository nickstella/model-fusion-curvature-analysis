import copy
import logging
from typing import List

import torch
from torch import nn


def get_weight_averaged_model(models: List[nn.Module], weights: torch.Tensor = None) -> nn.Module:
    if len(models) == 0:
        logging.error("No models provided")
        return

    ensembled_model = copy.deepcopy(models[0])

    if weights is None:
        weights = torch.ones(len(models))

    parameter_dicts = [dict(model.named_parameters()) for model in models]

    # set the weights of the ensembled network
    for param_name, _ in ensembled_model.named_parameters():
        avg_param = torch.stack(
            [weight * model_parameters[param_name] for weight, model_parameters in zip(weights, parameter_dicts)]
            ).sum(dim=0) / weights.sum()
        ensembled_model.state_dict()[param_name].copy_(avg_param.data)

    return ensembled_model
