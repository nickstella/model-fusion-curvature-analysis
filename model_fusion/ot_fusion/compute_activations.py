'''
Source: https://github.com/sidak/otfusion
'''

from typing import List
from torch import nn
from model_fusion.datasets import DataModuleType
from model_fusion.config import BASE_DATA_DIR
import copy
import logging


def get_model_activations(args, models: List[nn.Module], datamodule_type: DataModuleType):

    if args.activation_histograms and args.act_num_samples > 0:
        
        # create unit batch size dataloader
        batch_size = 1
        datamodule_hparams = {'batch_size': batch_size, 'data_dir': BASE_DATA_DIR}

        datamodule = datamodule_type.get_data_module(**datamodule_hparams)
        datamodule.prepare_data()
        datamodule.setup('fit')
        train_loader = datamodule.train_dataloader()

        activations = compute_activations_across_models(models, train_loader, args.act_num_samples)


    else:
        logging.error("No activation computated")


    return activations

def compute_activations_across_models(base_models, train_loader, num_samples):

    models = []
    for base_model in base_models:
        models.append(copy.deepcopy(base_model))

    # hook that computes the mean activations across data samples
    def get_activation(activation, name):
        
        def hook(model, input, output):
            if name not in activation:
                activation[name] = output.detach()
            else:
                activation[name] = (num_samples_processed * activation[name] + output.detach()) / (num_samples_processed + 1)
        return hook

    activations = {}

    print("Computing activations")

    for idx, model in enumerate(models):

        # Initialize the activation dictionary for each model
        activations[idx] = {}

        # Set forward hooks for all layers inside a model
        for name, layer in model.named_modules():
            if name == '':
                # print("excluded")
                continue
            layer.register_forward_hook(get_activation(activations[idx], name))
            # print("set forward hook for layer named: ", name)

        # Set the model in train mode
        model.train()

    # Run the same data samples ('num_samples' many) across all the models
    num_samples_processed = 0
    for data, target in train_loader:
        
        for model in models:
            model(data)
        
        num_samples_processed += 1
        
        if num_samples_processed == num_samples:
            break
    
    print(f"Activations computed across {num_samples_processed} samples out of {len(train_loader)}")
    
    return activations

