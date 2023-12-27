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
        # general parameters
        'gpu_id': -1,
        'debug': False,
        'config_file': None,
        'same_model': -1, # index of the same model to average with itself
        'load_models': '', # path/name of directory from where to load the models
        'eval_aligned': False, # evaluate aligned parent model 0 aligned wrt to parent model 1
        
        'tensorboard': False,
        'tensorboard_root': './tensorboard',
        
        # dataset parameters
        'to_download':True, # set to True if MNIST/dataset hasn't been downloaded,
        'disable_bias': True, # no bias at all in fc or conv layers,
        'dataset': 'cifar10', # mnist, cifar10, cifar100
        'cifar_style_data': True, # use data loader in cifar style
        'no_random_trainloaders': False, # get train loaders without any random transforms to ensure consistency
        
        # model parameters
        'num_models': 2,
        'model_name': 'resnet18_nobias_nobn', # vgg11_nobias, vgg11_half_nobias, vgg11_doub_nobias, vgg11_quad_nobias, resnet18_nobias, resnet18_nobias_nobn
        'second_model_name': None, # name of second model
        'width_ratio': 1, # ratio of the widths of the hidden layers between the two models
        'handle_skips': True, # handle shortcut skips in resnet which decrease dimension

        # training parameters
        'n_epochs': 1,
        'enable_dropout': False,
        'batch_size_train': 32,
        'batch_size_test': 32,
        'learning_rate': 1e-3,
        'momentum': 0.5,
        'log_interval': 100,

         # finetuning parameters
        'retrain_epochs': 0, # number of epochs to retrain all the models & their avgs
        'skip_retrain': -1, # which of the original models to skip retraining (-1 stands for skipping retraing of both parents)
        'retrain_seed': -1, # seed for retraining
        'retrain_lr_decay': -1, # amount by which to reduce the initial lr while retraining the model avgs
        'retrain_lr_decay_factor': None, # lr decay factor when the LR is gradually decreased by Step LR
        'retrain_lr_decay_epochs': None, # epochs at which retrain lr decay factor should be applied. underscore separated!
        'retrain_avg_only': False, # retraining models post OT fusion and vanilla averaging (but not base models)
        'retrain_geometric_only': False, # retraining the model post OT fusion only
        'reinit_trainloaders': False, # reinit train loader when starting retraining of each model
        'deterministic': False, # do retrain in deterministic mode

        # OT alignment parameters
        'geom_ensemble_type': 'acts', # wts, acts - fusion based on weights (wts) or activations (acts).
        'exact': True, # compute exact optimal transport (True = emd, False = sinkhorn)
        'unbalanced': False, # use unbalanced OT (for sinkhorn)
        'normalize_acts': False, # normalize the vector of activations
        'normalize_wts': False, # normalize the vector of weights
        'past_correction': True, # use the current weights aligned by multiplying with past transport map
        'activation_seed': 42, # seed for computing activations
        'update_acts': False, # update acts during the alignment of model0
        'activation_histograms': True,
        'act_num_samples': 200, # number of samples to compute activation stats 
        'softmax_temperature': 1, # softmax temperature for activation weights
        'activation_mode': None, # mean, std, meanstd, raw - mode that chooses how the importance of a neuron is calculated.
        'activation_normalize': False, # normalize activations before computing stats
        'center_acts': False, # subtract mean only across the samples for use in act based alignment
        'skip_last_layer': False,
        'skip_last_layer_type': 'average', # second, average
        'reg': 1e-2, # regularization strength for sinkhorn
        'reg_m': 1e-3, # regularization strength for marginals in unbalanced sinkhorn
        'weight_stats': False, # log neuron-wise weight vector stats.
        'sinkhorn_type': 'normal', # normal, stabilized, epsilon
        'gromov': False, # use gromov wasserstein distance and barycenters
        'gromov_loss': 'square_loss', # choice of loss function for gromov wasserstein computations
        'prelu_acts': False, # do activation based alignment based on pre-relu acts
        'pool_acts': False, # do activation based alignment based on pooling acts
        'pool_relu': False, # do relu first before pooling acts
        'print_distances': False, # print OT distances for every layer
        'importance': None, # l1, l2, l11, l12 - importance measure to use for building probab mass
        'proper_marginals': False, # consider the marginals of transport map properly (connected with importance)
        'correction': False, # scaling correction for OT (for when acts are not properly normalized)


        # metric parameters
        'ground_metric': 'euclidean', # euclidean, cosine
        'ground_metric_normalize': 'log', # log, max, none, median, mean
        'not_squared': False, # dont square the ground metric
        'ground_metric_eff': False, # memory efficient calculation of ground metric
        'dist_normalize': False, # normalize distances by act num samples (connected with ground_metric_eff)
        'clip_gm': True, # to clip ground metric
        'clip_min': 0, # Value for clip-min for gm
        'clip_max': 5, # Value for clip-max for gm
        'tmap_stats': False, # print tmap stats
        'ensemble_step': 0.5, # rate of adjustment towards the second model
        'ground_metric_eff': False, # memory efficient calculation of ground metric
        
    }
    return dotdict(parameters)

# def dump_parameters(args):
#     print("dumping parameters at ", args.config_dir)
#     # mkdir(args.config_dir)
#     with open(os.path.join(args.config_dir, args.exp_name + ".json"), 'w') as outfile:
#         if not (type(args) is dict or type(args) is utils.dotdict):
#             json.dump(vars(args), outfile, sort_keys=True, indent=4)
#         else:
#             json.dump(args, outfile, sort_keys=True, indent=4)


# TODO adjust with base args + config 
def get_parameters():

    base_args = get_base_args()

    # Here we utilize config files to setup the parameters
    # if base_args.config_file:
    #     args = copy.deepcopy(base_args)
    #     with open(os.path.join(base_args.config_dir, base_args.config_file), 'r') as f:
    #         file_params = dotdict(json.load(f))
    #         for param, value in file_params.items():
    #             if not hasattr(args, param):
    #                 # If it doesn't contain, then set from config
    #                 setattr(args, param, value)
    #             elif getattr(args, param) == parser.get_default(param):
    #                 # Or when it has , but is the default, then override from config
    #                 setattr(args, param, value)
    
    # else:
        # these remain unmodified from what was default or passed in via CLI
    args = base_args

    # Setup a timestamp for the experiment and save it in args
    # if hasattr(args, 'timestamp'):
    #     # the config file contained the timestamp parameter from the last experiment
    #     # (which say is being reproduced) so save it as well
    #     args.previous_timestamp = args.timestamp
    # args.timestamp = get_timestamp_other()

    # Set rootdir and other dump directories for the experiment
    # args.rootdir = os.getcwd()
    # if args.sweep_name is not None:
    #     args.baseroot = args.rootdir
    #     args.rootdir = os.path.join(args.rootdir, args.sweep_name)
    # else:
    #     args.baseroot = args.rootdir

    # args.config_dir = os.path.join(args.rootdir, 'configurations')
    # args.result_dir = os.path.join(args.rootdir, 'results')
    # args.exp_name = "exp_" + args.timestamp
    # args.csv_dir = os.path.join(args.rootdir, 'csv')
    # utils.mkdir(args.config_dir)
    # utils.mkdir(args.result_dir)
    # utils.mkdir(args.csv_dir)
    # if not hasattr(args, 'save_result_file') or args.save_result_file is None:
    #     args.save_result_file = 'default.csv'

    # Dump these parameters for reproducibility.
    # These should be inside a different directory than the results,
    # because then you have to open each directory separately to see what it contained!
    # dump_parameters(args)
    return args
