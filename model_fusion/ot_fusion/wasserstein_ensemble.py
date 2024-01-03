'''
Source: https://github.com/sidak/otfusion
'''

import torch
import ot
from model_fusion.models import ModelType
from model_fusion.ot_fusion.ground_metric import GroundMetric
import model_fusion.ot_fusion.wasserstein_helpers as helpers

def get_otfused_model(args, networks, activations, datamodule_type, datamodule_hparams):

    if args.geom_ensemble_type == 'wts':
        avg_aligned_layers, aligned_base_model  = get_aligned_layers_wts(args, networks, datamodule_type, datamodule_hparams)
    
    elif args.geom_ensemble_type == 'acts':
        avg_aligned_layers, aligned_base_model = get_aligned_layers_acts(args, networks, activations, datamodule_type, datamodule_hparams)
        
    otfused_model = helpers.get_network_from_param_list(networks[0], avg_aligned_layers)

    return otfused_model, aligned_base_model

def get_aligned_layers_wts(args, networks, datamodule_type, datamodule_hparams, eps=1e-7):
    '''
    Two neural networks that have to be averaged in geometric manner (i.e. layerwise).
    The 1st network is aligned with respect to the other via wasserstein distance.
    Also this assumes that all the layers are either fully connected or convolutional *(with no bias)*

    :param networks: list of networks (only 2 networks supported)
    :param activations: If not None, use it to build the activation histograms.
    Otherwise assumes uniform distribution over neurons in a layer.
    :return: list of layer weights 'wassersteinized'
    '''

    avg_aligned_layers = []
    T_var = None

    if networks[0].type == ModelType.RESNET18:
        args.handle_skips = True

    if args.handle_skips:
        skip_T_var = None
        skip_T_var_idx = -1
        residual_T_var = None
        residual_T_var_idx = -1

    previous_layer_shape = None
    ground_metric_object = GroundMetric(args)

    if args.eval_aligned:
        model0_aligned_layers = []

    num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))
    for idx, ((layer0_name, fc_layer0_weight), (layer1_name, fc_layer1_weight)) in enumerate(zip(networks[0].named_parameters(), networks[1].named_parameters())):

        assert fc_layer0_weight.shape == fc_layer1_weight.shape
        # print("Previous layer shape is ", previous_layer_shape)
        previous_layer_shape = fc_layer1_weight.shape

        mu_cardinality = fc_layer0_weight.shape[0]
        nu_cardinality = fc_layer1_weight.shape[0]

        # mu = np.ones(fc_layer0_weight.shape[0])/fc_layer0_weight.shape[0]
        # nu = np.ones(fc_layer1_weight.shape[0])/fc_layer1_weight.shape[0]
        layer0_shape = fc_layer0_weight.shape
        layer_shape = fc_layer0_weight.shape
        if len(layer_shape) > 2:
            is_conv = True
            # For convolutional layers, it is (#out_channels, #in_channels, height, width)
            fc_layer0_weight_data = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], fc_layer0_weight.shape[1], -1)
            fc_layer1_weight_data = fc_layer1_weight.data.view(fc_layer1_weight.shape[0], fc_layer1_weight.shape[1], -1)
        else:
            is_conv = False
            fc_layer0_weight_data = fc_layer0_weight.data
            fc_layer1_weight_data = fc_layer1_weight.data

        if idx == 0:
            if is_conv:
                M = ground_metric_object.process(fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1), 
                                                 fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
               
            else:
                # print("layer data is ", fc_layer0_weight_data, fc_layer1_weight_data)
                M = ground_metric_object.process(fc_layer0_weight_data, fc_layer1_weight_data)

            aligned_wt = fc_layer0_weight_data
        
        else:

            # print("shape of layer: model 0", fc_layer0_weight_data.shape)
            # print("shape of layer: model 1", fc_layer1_weight_data.shape)
            # print("shape of previous transport map", T_var.shape)

            if is_conv:
                if args.handle_skips:
                    assert len(layer0_shape) == 4
                    
                    # save skip_level transport map if there is block ahead
                    if layer0_shape[1] != layer0_shape[0]:
                        if not (layer0_shape[2] == 1 and layer0_shape[3] == 1):
                            # print(f'saved skip T_var at layer {idx} with shape {layer0_shape}')
                            skip_T_var = T_var.clone()
                            skip_T_var_idx = idx
                        else:
                            # print(f'utilizing skip T_var saved from layer layer {skip_T_var_idx} with shape {skip_T_var.shape}')
                            # if it's a shortcut (128, 64, 1, 1)
                            residual_T_var = T_var.clone()
                            residual_T_var_idx = idx  # use this after the skip
                            T_var = skip_T_var
                        # print("shape of previous transport map now is", T_var.shape)
                    
                    else:
                        if residual_T_var is not None and (residual_T_var_idx == (idx - 1)):
                            T_var = (T_var + residual_T_var) / 2
                            # print("averaging multiple T_var's")
                        # else:
                            # print("doing nothing for skips")

                T_var_conv = T_var.unsqueeze(0).repeat(fc_layer0_weight_data.shape[2], 1, 1)
                # print("shape of T_var_conv is ", T_var_conv.shape)
                # print("shape of fc_layer0_weight_data is ", fc_layer0_weight_data.shape)
                aligned_wt = torch.bmm(fc_layer0_weight_data.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)

                M = ground_metric_object.process(
                    aligned_wt.contiguous().view(aligned_wt.shape[0], -1),
                    fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
                )
            
            else:
                if fc_layer0_weight.data.shape[1] != T_var.shape[0]:
                    # Handles the switch from convolutional layers to fc layers
                    fc_layer0_unflattened = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], T_var.shape[0], -1).permute(2, 0, 1)
                    aligned_wt = torch.bmm(fc_layer0_unflattened,
                                           T_var.unsqueeze(0).repeat(fc_layer0_unflattened.shape[0], 1, 1)
                                           ).permute(1, 2, 0)
                    aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
                else:
                    # print("layer data (aligned) is ", aligned_wt, fc_layer1_weight_data)
                    aligned_wt = torch.matmul(fc_layer0_weight.data, T_var)
                # M = cost_matrix(aligned_wt, fc_layer1_weight)
                M = ground_metric_object.process(aligned_wt, fc_layer1_weight)
                # print("ground metric is ", M)
            
            if args.skip_last_layer and idx == (num_layers - 1):
                # print("Simple averaging of last layer weights. NO transport map needs to be computed")
                avg_aligned_layers.append((aligned_wt + fc_layer1_weight)/2)
                
                return avg_aligned_layers

        if args.importance is None or (idx == num_layers -1):
            mu = helpers.get_histogram(args, 0, mu_cardinality, layer0_name)
            nu = helpers.get_histogram(args, 1, nu_cardinality, layer1_name)
        
        else:
            mu = helpers.get_neuron_importance_histogram(args, fc_layer0_weight_data, is_conv)
            nu = helpers.get_neuron_importance_histogram(args, fc_layer1_weight_data, is_conv)
            assert args.proper_marginals
        
        # print("computed mu, nu")
        # print(mu, nu)

        cpuM = M.data.cpu().numpy()
        if args.exact:
            T = ot.emd(mu, nu, cpuM)
        else:
            T = ot.bregman.sinkhorn(mu, nu, cpuM, reg=args.reg)

        T_var = torch.from_numpy(T).float()
        print("the transport map is ", T_var)

        if args.correction:
            if not args.proper_marginals:
                # think of it as m x 1, scaling weights for m linear combinations of points in X
                marginals = torch.ones(T_var.shape[0]) / T_var.shape[0]
                marginals = torch.diag(1.0/(marginals + eps))  # take inverse
                T_var = torch.matmul(T_var, marginals)
            
            else:
                marginals_beta = T_var.t() @ torch.ones(T_var.shape[0], dtype=T_var.dtype)
                marginals = (1 / (marginals_beta + eps))
                # print("shape of inverse marginals beta is ", marginals_beta.shape)
                # print("inverse marginals beta is ", marginals_beta)

                T_var = T_var * marginals
                
                # i.e., how a neuron of 2nd model is constituted by the neurons of 1st model
                # this should all be ones, and number equal to number of neurons in 2nd model
                # print(T_var.sum(dim=0))
                # assert (T_var.sum(dim=0) == torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)).all()

        # print("Ratio of trace to the matrix sum: ", torch.trace(T_var) / torch.sum(T_var))
        # print("Here, trace is {} and matrix sum is {} ".format(torch.trace(T_var), torch.sum(T_var)))
        setattr(args, 'trace_sum_ratio_{}'.format(layer0_name), (torch.trace(T_var) / torch.sum(T_var)).item())

        if args.past_correction:
            # print("this is past correction for weight mode")
            # print("Shape of aligned wt is ", aligned_wt.shape)
            # print("Shape of fc_layer0_weight_data is ", fc_layer0_weight_data.shape)
            t_fc0_model = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1))
        else:
            t_fc0_model = torch.matmul(T_var.t(), fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1))

        # Average the weights of aligned first layers
        geometric_fc = (t_fc0_model + fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))/2
        
        if is_conv and layer_shape != geometric_fc.shape:
            geometric_fc = geometric_fc.view(layer_shape)
        
        avg_aligned_layers.append(geometric_fc)

        # get the performance of the model 0 aligned with respect to the model 1
        if args.eval_aligned:
            
            if is_conv and layer_shape != t_fc0_model.shape:
                t_fc0_model = t_fc0_model.view(layer_shape)
            
            model0_aligned_layers.append(t_fc0_model)
    
    if args.eval_aligned:        
        aligned_base_model = helpers.eval_aligned_model(networks[0], model0_aligned_layers, datamodule_type, datamodule_hparams)

    return avg_aligned_layers, aligned_base_model 

def get_aligned_layers_acts(args, networks, activations, datamodule_type, datamodule_hparams, eps=1e-7):
    '''
    Average based on the activation vector over data samples. Obtain the transport map,
    and then based on which align the nodes and average the weights!
    Like before: two neural networks that have to be averaged in geometric manner (i.e. layerwise).
    The 1st network is aligned with respect to the other via wasserstein distance.
    Also this assumes that all the layers are either fully connected or convolutional *(with no bias)*
    :param networks: list of networks (only 2 networks supported)
    :param activations: If not None, use it to build the activation histograms.
    Otherwise assumes uniform distribution over neurons in a layer.
    :return: list of layer weights 'wassersteinized'
    '''


    avg_aligned_layers = []
    T_var = None
    if args.handle_skips:
        skip_T_var = None
        skip_T_var_idx = -1
        residual_T_var = None
        residual_T_var_idx = -1

    marginals_beta = None
    previous_layer_shape = None
    num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))
    ground_metric_object = GroundMetric(args)

    if args.eval_aligned:
        model0_aligned_layers = []

    networks_named_params = list(zip(networks[0].named_parameters(), networks[1].named_parameters()))
    idx = 0
    incoming_layer_aligned = True # for input
    while idx < num_layers:
        ((layer0_name, fc_layer0_weight), (layer1_name, fc_layer1_weight)) = networks_named_params[idx]
        # print("NUM LAYERS: ", num_layers)
        # print("\n--------------- At layer index {} ------------- \n ".format(idx))
        # layer shape is out x in
        # assert fc_layer0_weight.shape == fc_layer1_weight.shape
        assert helpers.check_layer_sizes(args, idx, fc_layer0_weight.shape, fc_layer1_weight.shape, num_layers)
        # print("Previous layer shape is ", previous_layer_shape)
        previous_layer_shape = fc_layer1_weight.shape

        # will have shape layer_size x act_num_samples
        layer0_name_reduced = helpers.reduce_layer_name(layer0_name)
        layer1_name_reduced = helpers.reduce_layer_name(layer1_name)

        # print(layer0_name, layer1_name)
        # print(layer0_name_reduced, layer1_name_reduced)

        # print("let's see the difference in layer names", layer0_name.replace('.' + layer0_name.split('.')[-1], ''), layer0_name_reduced)
        # print(activations[0])
        # print(activations[0][layer0_name.replace('.' + layer0_name.split('.')[-1], '')].shape, 'shape of activations generally')
        
        # for conv layer I need to make the act_num_samples dimension the last one, but it has the intermediate dimensions for height and width of channels, so that won't work.
        # So convert (num_samples, layer_size, ht, wt) -> (layer_size, ht, wt, num_samples)

        activations_0, activations_1 = helpers.process_activations(args, activations, layer0_name, layer1_name)

        # print("activations for 1st model are ", activations_0)
        # print("activations for 2nd model are ", activations_1)

        assert activations_0.shape[0] == fc_layer0_weight.shape[0]
        assert activations_1.shape[0] == fc_layer1_weight.shape[0]

        mu_cardinality = fc_layer0_weight.shape[0]
        nu_cardinality = fc_layer1_weight.shape[0]

        helpers.get_activation_distance_stats(activations_0, activations_1, layer0_name)

        layer0_shape = fc_layer0_weight.shape
        layer_shape = fc_layer1_weight.shape
        
        if len(layer_shape) > 2:
            is_conv = True
        else:
            is_conv = False

        fc_layer0_weight_data = helpers.get_layer_weights(fc_layer0_weight, is_conv)
        fc_layer1_weight_data = helpers.get_layer_weights(fc_layer1_weight, is_conv)

        if idx == 0 or incoming_layer_aligned:
            aligned_wt = fc_layer0_weight_data

        else:

            # print("shape of layer: model 0", fc_layer0_weight_data.shape)
            # print("shape of layer: model 1", fc_layer1_weight_data.shape)

            # print("shape of activations: model 0", activations_0.shape)
            # print("shape of activations: model 1", activations_1.shape)


            # print("shape of previous transport map", T_var.shape)

            if is_conv:
                
                if args.handle_skips:
                    
                    assert len(layer0_shape) == 4
                    
                    # save skip_level transport map if there is block ahead
                    if layer0_shape[1] != layer0_shape[0]:
                        if not (layer0_shape[2] == 1 and layer0_shape[3] == 1):
                            # print(f'saved skip T_var at layer {idx} with shape {layer0_shape}')
                            skip_T_var = T_var.clone()
                            skip_T_var_idx = idx
                        else:
                            # print(f'utilizing skip T_var saved from layer layer {skip_T_var_idx} with shape {skip_T_var.shape}')
                            # if it's a shortcut (128, 64, 1, 1)
                            residual_T_var = T_var.clone()
                            residual_T_var_idx = idx  # use this after the skip
                            T_var = skip_T_var
                        # print("shape of previous transport map now is", T_var.shape)
                    
                    else:
                        if residual_T_var is not None and (residual_T_var_idx == (idx - 1)):
                            T_var = (T_var + residual_T_var) / 2
                            print("averaging multiple T_var's")
                        # else:
                        #     print("doing nothing for skips")
                T_var_conv = T_var.unsqueeze(0).repeat(fc_layer0_weight_data.shape[2], 1, 1)
                aligned_wt = torch.bmm(fc_layer0_weight_data.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)

            else:
                
                if fc_layer0_weight.data.shape[1] != T_var.shape[0]:
                    # Handles the switch from convolutional layers to fc layers
                    # checks if the input has been reshaped
                    fc_layer0_unflattened = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], T_var.shape[0],-1).permute(2, 0, 1)
                    aligned_wt = torch.bmm(fc_layer0_unflattened,
                                           T_var.unsqueeze(0).repeat(fc_layer0_unflattened.shape[0], 1, 1)
                                           ).permute(1, 2, 0)
                    
                    aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
                
                else:
                    aligned_wt = torch.matmul(fc_layer0_weight.data, T_var)
                
        if args.importance is None or (idx == num_layers -1):
            mu = helpers.get_histogram(args, 0, mu_cardinality, layer0_name)
            nu = helpers.get_histogram(args, 1, nu_cardinality, layer1_name)
        else:
            mu = helpers.get_neuron_importance_histogram(args, fc_layer0_weight_data, is_conv)
            nu = helpers.get_neuron_importance_histogram(args, fc_layer1_weight_data, is_conv)
            assert args.proper_marginals
        
        # print("computed mu, nu")
        # print(mu, nu)

        # print("Refactored ground metric calc")
        M0 = helpers.process_ground_metric_from_acts(args, is_conv, ground_metric_object,[activations_0, activations_1])

        # print("# of ground metric features in 0 is  ", (activations_0.view(activations_0.shape[0], -1)).shape[1])
        # print("# of ground metric features in 1 is  ", (activations_1.view(activations_1.shape[0], -1)).shape[1])

        if args.skip_last_layer and idx == (num_layers - 1):

            if args.skip_last_layer_type == 'average':
                # print("Simple averaging of last layer weights. NO transport map needs to be computed")
                avg_aligned_layers.append(((aligned_wt + fc_layer1_weight)/2))
            
            elif args.skip_last_layer_type == 'second':
                # print("Just giving the weights of the second model. NO transport map needs to be computed")
                avg_aligned_layers.append(fc_layer1_weight)

            return avg_aligned_layers

        # print("ground metric (m0) is ", M0)

        T_var = helpers.get_current_layer_transport_map(args, mu, nu, M0, idx=idx, layer_shape=layer_shape, eps=eps, layer_name=layer0_name)
        T_var, marginals = helpers.compute_marginals(args, T_var, eps=eps)

        print("the transport map is ", T_var)
        # print("Ratio of trace to the matrix sum: ", torch.trace(T_var) / torch.sum(T_var))
        # print("Here, trace is {} and matrix sum is {} ".format(torch.trace(T_var), torch.sum(T_var)))
        setattr(args, 'trace_sum_ratio_{}'.format(layer0_name), (torch.trace(T_var) / torch.sum(T_var)).item())

        print("Shape of aligned wt is ", aligned_wt.shape)
        # print("Shape of fc_layer0_weight_data is ", fc_layer0_weight_data.shape)

        if args.past_correction:
            t_fc0_model = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1))
        else:
            t_fc0_model = torch.matmul(T_var.t(), fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1))

        # Average the weights of aligned first layers
        geometric_fc = (t_fc0_model + fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)) / 2
        
        if is_conv and layer_shape != geometric_fc.shape:
            geometric_fc = geometric_fc.view(layer_shape)
        
        avg_aligned_layers.append(geometric_fc)

        if args.eval_aligned:        
            
            if is_conv and layer_shape != t_fc0_model.shape:
                t_fc0_model = t_fc0_model.view(layer_shape)
            
            model0_aligned_layers.append(t_fc0_model)
            
        incoming_layer_aligned = False
        next_aligned_wt_reshaped = None

        # remove cached variables to prevent out of memory
        activations_0 = None
        activations_1 = None
        mu = None
        nu = None
        fc_layer0_weight_data = None
        fc_layer1_weight_data = None
        M0 = None
        M1 = None
        cpuM = None

        idx += 1

    if args.eval_aligned:        
        aligned_base_model = helpers.eval_aligned_model(networks[0], model0_aligned_layers, datamodule_type, datamodule_hparams)

    print = aligned_base_model
    
    return avg_aligned_layers, aligned_base_model

