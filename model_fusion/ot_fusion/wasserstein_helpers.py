import ot
import torch
import numpy as np
from model_fusion.train import setup_training
from model_fusion.datasets import DataModuleType
from model_fusion.models import ModelType
from model_fusion.config import BASE_DATA_DIR
import wandb
import math
import model_fusion.ot_fusion.compute_activations as compute_activations

def cost_matrix(x, y, p=2):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
    return c

def get_histogram(args, idx, cardinality, layer_name, activations=None, return_numpy = True, float64=False):
    if activations is None:
        # returns a uniform measure
        if not args.unbalanced:
            print("returns a uniform measure of cardinality: ", cardinality)
            return np.ones(cardinality)/cardinality
        else:
            return np.ones(cardinality)
        
    # ASK softmax temperature
    else:
        # return softmax over the activations raised to a temperature
        # layer_name is like 'fc1.weight', while activations only contains 'fc1'
        print(activations[idx].keys())
        unnormalized_weights = activations[idx][layer_name.split('.')[0]]
        print("For layer {},  shape of unnormalized weights is ".format(layer_name), unnormalized_weights.shape)
        unnormalized_weights = unnormalized_weights.squeeze()
        assert unnormalized_weights.shape[0] == cardinality

        if return_numpy:
            if float64:
                return torch.softmax(unnormalized_weights / args.softmax_temperature, dim=0).data.cpu().numpy().astype(
                    np.float64)
            else:
                return torch.softmax(unnormalized_weights / args.softmax_temperature, dim=0).data.cpu().numpy()
        else:
            return torch.softmax(unnormalized_weights / args.softmax_temperature, dim=0)
        
def print_stats(arr, nick=""):
    print(nick)
    print("summary stats are: \n max: {}, mean: {}, min: {}, median: {}, std: {} \n".format(
        arr.max(), arr.mean(), arr.min(), np.median(arr), arr.std()
    ))

def get_activation_distance_stats(activations_0, activations_1, layer_name=""):
    if layer_name != "":
        print("In layer {}: getting activation distance statistics".format(layer_name))
    M = cost_matrix(activations_0, activations_1) ** (1/2)
    mean_dists =  torch.mean(M, dim=-1)
    max_dists = torch.max(M, dim=-1)[0]
    min_dists = torch.min(M, dim=-1)[0]
    std_dists = torch.std(M, dim=-1)

    print("Statistics of the distance from neurons of layer 1 (averaged across nodes of layer 0): \n")
    print("Max : {}, Mean : {}, Min : {}, Std: {}".format(torch.mean(max_dists), torch.mean(mean_dists), torch.mean(min_dists), torch.mean(std_dists)))

def update_model(args, model, new_params, test=False, test_loader=None, reversed=False, idx=-1):

    updated_model = ModelType.RESNET18

    layer_idx = 0
    model_state_dict = model.state_dict()

    print("len of model_state_dict is ", len(model_state_dict.items()))
    print("len of new_params is ", len(new_params))

    for key, value in model_state_dict.items():
        print("updated parameters for layer ", key)
        model_state_dict[key] = new_params[layer_idx]
        layer_idx += 1
        if layer_idx == len(new_params):
            break


    updated_model.load_state_dict(model_state_dict)

    # TODO change harcoded values + check if correct
    if test:

        batch_size = 32
        max_epochs = 1
        datamodule_type = DataModuleType.CIFAR10
        datamodule_hparams = {'batch_size': batch_size, 'data_dir': BASE_DATA_DIR}

        model_type = ModelType.RESNET18
        model_hparams = {'num_classes': 10, 'num_channels': 3, 'bias': False}

        wandb_tags = ['RESNET-18', 'CIFAR_10', f"Batch size {batch_size}", "vanilla averaging"]

        _, datamodule, trainer = setup_training(f'RESNET-18 CIFAR-10 B32', model_type, model_hparams, datamodule_type, datamodule_hparams, max_epochs=max_epochs, wandb_tags=wandb_tags)

        datamodule.prepare_data()
        datamodule.setup('test')
        trainer.test(model, dataloaders=datamodule.test_dataloader())

        wandb.finish()
    
    else:
         print("Not testing the updated model")
         final_acc = None

    return updated_model, final_acc

def check_activation_sizes(args, acts0, acts1):
    if args.width_ratio == 1:
        return acts0.shape == acts1.shape
    else:
        return acts0.shape[-1]/acts1.shape[-1] == args.width_ratio

def process_activations(args, activations, layer0_name, layer1_name):
    activations_0 = activations[0][layer0_name.replace('.' + layer0_name.split('.')[-1], '')].squeeze(1)
    activations_1 = activations[1][layer1_name.replace('.' + layer1_name.split('.')[-1], '')].squeeze(1)

    # assert activations_0.shape == activations_1.shape
    check_activation_sizes(args, activations_0, activations_1)

    if args.same_model != -1:
        # sanity check when averaging the same model (with value being the model index)
        assert (activations_0 == activations_1).all()
        print("Are the activations the same? ", (activations_0 == activations_1).all())

    if len(activations_0.shape) == 2:
        activations_0 = activations_0.t()
        activations_1 = activations_1.t()
    elif len(activations_0.shape) > 2:
        reorder_dim = [l for l in range(1, len(activations_0.shape))]
        reorder_dim.append(0)
        print("reorder_dim is ", reorder_dim)
        activations_0 = activations_0.permute(*reorder_dim).contiguous()
        activations_1 = activations_1.permute(*reorder_dim).contiguous()

    return activations_0, activations_1

def reduce_layer_name(layer_name):
    # print("layer0_name is ", layer0_name) It was features.0.weight
    # previous way assumed only one dot, so now I replace the stuff after last dot
    return layer_name.replace('.' + layer_name.split('.')[-1], '')

def get_layer_weights(layer_weight, is_conv):
    if is_conv:
        # For convolutional layers, it is (#out_channels, #in_channels, height, width)
        layer_weight_data = layer_weight.data.view(layer_weight.shape[0], layer_weight.shape[1], -1)
    else:
        layer_weight_data = layer_weight.data

    return layer_weight_data

def process_ground_metric_from_acts(args, is_conv, ground_metric_object, activations):
    print("inside refactored")
    if is_conv:
        # ASK gromov
        if not args.gromov:
            M0 = ground_metric_object.process(activations[0].view(activations[0].shape[0], -1),
                                             activations[1].view(activations[1].shape[0], -1))
        else:
            M0 = ground_metric_object.process(activations[0].view(activations[0].shape[0], -1),
                                              activations[0].view(activations[0].shape[0], -1))
            M1 = ground_metric_object.process(activations[1].view(activations[1].shape[0], -1),
                                              activations[1].view(activations[1].shape[0], -1))

        print("# of ground metric features is ", (activations[0].view(activations[0].shape[0], -1)).shape[1])
    else:
        if not args.gromov:
            M0 = ground_metric_object.process(activations[0], activations[1])
        else:
            M0 = ground_metric_object.process(activations[0], activations[0])
            M1 = ground_metric_object.process(activations[1], activations[1])

    if args.gromov:
        return M0, M1
    else:
        return M0, None


def custom_sinkhorn(args, mu, nu, cpuM):
    if not args.unbalanced:
        if args.sinkhorn_type == 'normal':
            T = ot.bregman.sinkhorn(mu, nu, cpuM, reg=args.reg)
        elif args.sinkhorn_type == 'stabilized':
            T = ot.bregman.sinkhorn_stabilized(mu, nu, cpuM, reg=args.reg)
        elif args.sinkhorn_type == 'epsilon':
            T = ot.bregman.sinkhorn_epsilon_scaling(mu, nu, cpuM, reg=args.reg)
        else:
            raise NotImplementedError
    else:
        T = ot.unbalanced.sinkhorn_knopp_unbalanced(mu, nu, cpuM, reg=args.reg, reg_m=args.reg_m)
    return T


def sanity_check_tmap(T):
    if not math.isclose(np.sum(T), 1.0, abs_tol=1e-7):
        print("Sum of transport map is ", np.sum(T))
        raise Exception('NAN inside Transport MAP. Most likely due to large ground metric values')

def get_updated_acts_v0(args, layer_shape, aligned_wt, model0_aligned_layers, networks, test_loader, layer_names):
    '''
    Return the updated activations of the 0th model with respect to the other one.

    :param args:
    :param layer_shape:
    :param aligned_wt:
    :param model0_aligned_layers:
    :param networks:
    :param test_loader:
    :param layer_names:
    :return:
    '''
    if layer_shape != aligned_wt.shape:
        updated_aligned_wt = aligned_wt.view(layer_shape)
    else:
        updated_aligned_wt = aligned_wt

    updated_model0, _ = update_model(args, networks[0], model0_aligned_layers + [updated_aligned_wt], test=True, test_loader=test_loader, idx=0)
    updated_activations = compute_activations.get_model_activations(args, [updated_model0, networks[1]], config=args.config, layer_name=reduce_layer_name(layer_names[0]), selective=True)

    updated_activations_0, updated_activations_1 = process_activations(args, updated_activations,
                                                                       layer_names[0], layer_names[1])
    return updated_activations_0, updated_activations_1

def get_updated_acts_v1(args, networks, test_loader, layer_names):
    '''
    Return the updated activations of the 0th model with respect to the other one.

    :param args:
    :param test_loader:
    :param layer_names:
    :return:
    '''
    updated_activations = compute_activations.get_model_activations(args, networks, config=args.config)

    updated_activations_0, updated_activations_1 = process_activations(args, updated_activations, layer_names[0], layer_names[1])
    return updated_activations_0, updated_activations_1

def check_layer_sizes(args, layer_idx, shape1, shape2, num_layers):
    if args.width_ratio == 1:
        return shape1 == shape2
    else: 
        raise ValueError(f"Different layer widths: {shape1} and {shape2}")

def compute_marginals(args, T_var, device, eps=1e-7):
    if args.correction:
        if not args.proper_marginals:
            
            # think of it as m x 1, scaling weights for m linear combinations of points in X
            marginals = torch.ones(T_var.shape)
            if args.gpu_id != -1:
                marginals = marginals.cuda(args.gpu_id)

            marginals = torch.matmul(T_var, marginals)
            marginals = 1 / (marginals + eps)
            print("marginals are ", marginals)

            T_var = T_var * marginals

        else:
            marginals_beta = T_var.t() @ torch.ones(T_var.shape[0], dtype=T_var.dtype).to(device)

            marginals = (1 / (marginals_beta + eps))
            print("shape of inverse marginals beta is ", marginals_beta.shape)
            print("inverse marginals beta is ", marginals_beta)

            T_var = T_var * marginals
            # i.e., how a neuron of 2nd model is constituted by the neurons of 1st model
            # this should all be ones, and number equal to number of neurons in 2nd model
            
            print(T_var.sum(dim=0))
            # assert (T_var.sum(dim=0) == torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)).all()

        print("T_var after correction ", T_var)
        print("T_var stats: max {}, min {}, mean {}, std {} ".format(T_var.max(), T_var.min(), T_var.mean(),
                                                                     T_var.std()))
    else:
        marginals = None

    return T_var, marginals

def get_current_layer_transport_map(args, mu, nu, M0, M1, idx, layer_shape, eps=1e-7, layer_name=None):

    if not args.gromov:
        cpuM = M0.data.cpu().numpy()
        if args.exact:
            T = ot.emd(mu, nu, cpuM)
        else:
            T = custom_sinkhorn(args, mu, nu, cpuM)

        if args.print_distances:
            ot_cost = np.multiply(T, cpuM).sum()
            print(f'At layer idx {idx} and shape {layer_shape}, the OT cost is ', ot_cost)
            if layer_name is not None:
                setattr(args, f'{layer_name}_layer_{idx}_cost', ot_cost)
            else:
                setattr(args, f'layer_{idx}_cost', ot_cost)
    
    else:
        cpuM0 = M0.data.cpu().numpy()
        cpuM1 = M1.data.cpu().numpy()

        assert not args.exact
        # if i understood correctly, this should be more efficient but less accurate
        T = ot.gromov.entropic_gromov_wasserstein(cpuM0, cpuM1, mu, nu, loss_fun=args.gromov_loss, epsilon=args.reg)

    if not args.unbalanced:
        sanity_check_tmap(T)

    if args.gpu_id != -1:
        T_var = torch.from_numpy(T).cuda(args.gpu_id).float()
    else:
        T_var = torch.from_numpy(T).float()

    if args.tmap_stats:
        print(
        "Tmap stats (before correction) \n: For layer {}, frobenius norm from the joe's transport map is {}".format(
            layer_name, torch.norm(T_var - torch.ones_like(T_var) / torch.numel(T_var), p='fro')
        ))

    print("shape of T_var is ", T_var.shape)
    print("T_var before correction ", T_var)

    return T_var

def get_neuron_importance_histogram(args, layer_weight, is_conv, eps=1e-9):
    print('shape of layer_weight is ', layer_weight.shape)
    if is_conv:
        layer = layer_weight.contiguous().view(layer_weight.shape[0], -1).cpu().numpy()
    else:
        layer = layer_weight.cpu().numpy()
    
    if args.importance == 'l1':
        importance_hist = np.linalg.norm(layer, ord=1, axis=-1).astype(
                    np.float64) + eps
    elif args.importance == 'l2':
        importance_hist = np.linalg.norm(layer, ord=2, axis=-1).astype(
                    np.float64) + eps
    else:
        raise NotImplementedError

    if not args.unbalanced:
        importance_hist = (importance_hist/importance_hist.sum())
        print('sum of importance hist is ', importance_hist.sum())
    # assert importance_hist.sum() == 1.0
    return importance_hist

def get_network_from_param_list(args, param_list, test_loader):

    print("using independent method")
    model_type = ModelType.RESNET18

    # check the test performance of the network before
    batch_size = 32
    max_epochs = 1
    datamodule_type = DataModuleType.CIFAR10
    datamodule_hparams = {'batch_size': batch_size, 'data_dir': BASE_DATA_DIR}

    model_type = ModelType.RESNET18
    model_hparams = {'num_classes': 10, 'num_channels': 3, 'bias': False}

    wandb_tags = ['RESNET-18', 'CIFAR_10', f"Batch size {batch_size}", "vanilla averaging"]

    new_model, datamodule, trainer = setup_training(f'RESNET-18 CIFAR-10 B32', model_type, model_hparams, datamodule_type, datamodule_hparams, max_epochs=max_epochs, wandb_tags=wandb_tags)

    datamodule.prepare_data()
    datamodule.setup('test')
    trainer.test(new_model, dataloaders=datamodule.test_dataloader())

    # set the weights of the new network
    # print("before", new_network.state_dict())
    print("len of model parameters and avg aligned layers is ", len(list(new_model.parameters())), len(param_list))
    assert len(list(new_model.parameters())) == len(param_list)

    layer_idx = 0
    model_state_dict = new_model.state_dict()

    print("len of model_state_dict is ", len(model_state_dict.items()))
    print("len of param_list is ", len(param_list))
    
    for param, (name, _) in zip(param_list, new_model.named_parameters()):
        new_model.state_dict()[name].copy_(param.data)

    trainer.test(new_model, dataloaders=datamodule.test_dataloader())

    wandb.finish()


    return new_model