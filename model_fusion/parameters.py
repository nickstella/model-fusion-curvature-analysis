# from utils import dotdict, get_timestamp_other, mkdir
import json
import copy
import os

class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# commenting this now
def get_base_args():
    parameters = {
        # default parameters NOT MODIFY
        'eval_aligned': True, # evaluate aligned parent model 0 aligned wrt to parent model 1
        'num_models': 2,
        'width_ratio': 1, # ratio of the widths of the hidden layers between the two models
        'handle_skips': True, # handle shortcut skips in resnet which decrease dimension          #I PUT FALSE FOR VGG
        'exact': True, # compute exact optimal transport (True = emd, False = sinkhorn)
        'activation_seed': 21, # seed for computing activations
        'activation_histograms': True,
        'ground_metric': 'euclidean', # euclidean, cosine
        'ground_metric_normalize': 'none', # log, max, none, median, mean
        'same_model': False, # if the two models are the same
        
        # OT alignment settings 
        'geom_ensemble_type': 'acts', # wts, acts - fusion based on weights (wts) or activations (acts).                 
        'act_num_samples': 200, # number of samples to compute activation stats                                           #I PUT 75 FOR VGG
        'skip_last_layer': False, # skip the last layer of the model
        'skip_last_layer_type': 'average', # second, average
        
        # OT alignment tunable parameters (default values are ok)
        'softmax_temperature': 1, # softmax temperature for activation weights
        'past_correction': True, # use the current weights aligned by multiplying with past transport map
        'correction': True, # scaling correction for OT (for when acts are not properly normalized)
        'normalize_acts': False, # normalize the vector of activations
        'normalize_wts': False, # normalize the vector of weights
        'activation_normalize': False, # normalize activations before computing stats
        'center_acts': False, # subtract mean only across the samples for use in act based alignment
        'prelu_acts': False, # do activation based alignment based on pre-relu acts
        'pool_acts': False, # do activation based alignment based on pooling acts
        'pool_relu': False, # do relu first before pooling acts
        'importance': None, # l1, l2, l11, l12 - importance measure to use for building probab mass
        'proper_marginals': False, # consider the marginals of transport map properly (connected with importance)
        'not_squared': True, # dont square the ground metric
        'dist_normalize': False, # normalize distances by act num samples (connected with ground_metric_eff)
        'clip_gm': False, # to clip ground metric
        'clip_min': 0, # Value for clip-min for gm
        'clip_max': 5, # Value for clip-max for gm
        'tmap_stats': False, # print tmap stats
        'ensemble_step': 0.5, # rate of adjustment towards the second model
        'ground_metric_eff': False, # memory efficient calculation of ground metric
        'reg': 1e-2, # regularization parameter for OT
    }
    return dotdict(parameters)

def get_parameters(config_file=None):

    base_args = get_base_args()

    if config_file is not None:
        args = copy.deepcopy(base_args)
        with open(config_file, 'r') as f:
            file_params = dotdict(json.load(f))
            for param, value in file_params.items():
                setattr(args, param, value)
    else:
        args = base_args

    return args
