import torch
from model_fusion.train import setup_training
from model_fusion.datasets import DataModuleType
from model_fusion.models import ModelType
from model_fusion.config import BASE_DATA_DIR


def get_model_activations(args, models, config=None, layer_name=None, selective=False, personal_dataset = None):

    # TODO change harcoded values

    if args.activation_histograms and args.act_num_samples > 0:

        batch_size = 1
        max_epochs = 1
        datamodule_type = DataModuleType.CIFAR10
        datamodule_hparams = {'batch_size': batch_size, 'data_dir': BASE_DATA_DIR}

        model_type = ModelType.RESNET18
        model_hparams = {'num_classes': 10, 'num_channels': 3, 'bias': False}

        wandb_tags = ['RESNET-18', 'CIFAR_10', f"Batch size {batch_size}", "prediction ensembling"]
        
        _, datamodule, trainer = setup_training(f'RESNET-18 CIFAR-10 B32', model_type, model_hparams, datamodule_type, datamodule_hparams, max_epochs=max_epochs, wandb_tags=wandb_tags)

        datamodule.prepare_data()
        datamodule.setup('fit')
        train_loader = datamodule.train_dataloader()

        if args.activation_mode is None:
            activations = compute_activations_across_models(args, models, train_loader, args.act_num_samples)
        else:
            if selective and args.update_acts:
                activations = compute_selective_activation(args, models, layer_name, train_loader, args.act_num_samples)
            else:
                activations = compute_activations_across_models_v1(args, models, train_loader, args.act_num_samples, mode=args.activation_mode)

    else:
        raise ValueError("No activations computed")

    return activations

def compute_activations_across_models(args, models, train_loader, num_samples):

    # hook that computes the mean activations across data samples
    def get_activation(activation, name):
        
        def hook(model, input, output):
                    
            if name not in activation:
                activation[name] = output.detach()
            else:
                activation[name] = (num_samples_processed * activation[name] + output.detach()) / (num_samples_processed + 1)

        return hook

    activations = {}

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
    for batch_idx, (data, target) in enumerate(train_loader):
        
        for idx, model in enumerate(models):
            model(data)
        
        num_samples_processed += 1
        
        if num_samples_processed == num_samples:
            break
    
    print("Activations computed across {} samples".format(num_samples_processed))
    return activations


def normalize_tensor(tens):
    tens_shape = tens.shape
    assert tens_shape[1] == 1
    tens = tens.view(tens_shape[0], 1, -1)
    norms = tens.norm(dim=-1)
    ntens = tens/norms.view(-1, 1, 1)
    ntens = ntens.view(tens_shape)
    return ntens

# TODO to be finished, for now use mode = None
def compute_activations_across_models_v1(args, models, train_loader, num_samples, mode='mean'):

    torch.manual_seed(args.activation_seed)

    # hook that computes the mean activations across data samples
    def get_activation(activation, name):
        
        def hook(model, input, output):
            
            if name not in activation:
                activation[name] = []

            activation[name].append(output.detach())

        return hook

    activations = {}
    forward_hooks = []
    
    param_names = [tupl[0].replace('.weight', '') for tupl in models[0].named_parameters()]
    
    for idx, model in enumerate(models):
        
        # Initialize the activation dictionary for each model
        activations[idx] = {}
        
        # Set forward hooks for all layers inside a model
        layer_hooks = []
        for name, layer in model.named_modules():
            if name == '' or name not in param_names:
                print("excluded")
                continue
            layer_hooks.append(layer.register_forward_hook(get_activation(activations[idx], name)))
            print("set forward hook for layer named: ", name)

        forward_hooks.append(layer_hooks)
        
        # Set the model in train mode
        model.train()

    # Run the same data samples ('num_samples' many) across all the models
    num_samples_processed = 0
    num_personal_idx = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if num_samples_processed == num_samples:
            break

        for idx, model in enumerate(models):
            model(data)

        num_samples_processed += 1

    relu = torch.nn.ReLU()
    maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    avgpool = torch.nn.AvgPool2d(kernel_size=1, stride=1)

    # Combine the activations generated across the number of samples to form importance scores
    # The importance calculated is based on the 'mode' flag: which is either of 'mean', 'std', 'meanstd'
    
    # TODO change harcoded values + check if correct
    model_cfg = ModelType.RESNET18
    for idx in range(len(models)):
        cfg_idx = 0
        for lnum, layer in enumerate(activations[idx]):
            print('***********')
            activations[idx][layer] = torch.stack(activations[idx][layer])
            print("min of act: {}, max: {}, mean: {}".format(torch.min(activations[idx][layer]), torch.max(activations[idx][layer]), torch.mean(activations[idx][layer])))

            # ASK prelu, pool, and relu acts
            if not args.prelu_acts and not lnum == (len(activations[idx])-1):
                print("applying relu ---------------")
                activations[idx][layer] = relu(activations[idx][layer])
                min_act, max_act, mean_act = torch.min(activations[idx][layer]), torch.max(activations[idx][layer]), torch.mean(activations[idx][layer]) 
                print("after RELU: min of act: {}, max: {}, mean: {}".format(min_act, max_act, mean_act))
                
            elif args.model_name == 'vgg11_nobias' and args.pool_acts and len(activations[idx][layer].shape)>3:
                
                if args.pool_relu:
                    print("applying relu ---------------")
                    activations[idx][layer] = relu(activations[idx][layer])

                activations[idx][layer] = activations[idx][layer].squeeze(1)

                # apply maxpool wherever the next thing in config list is 'M'
                if (cfg_idx + 1) < len(model_cfg):
                    if model_cfg[cfg_idx+1] == 'M':
                        print("applying maxpool ---------------")
                        activations[idx][layer] = maxpool(activations[idx][layer])
                        cfg_idx += 2
                    else:
                        cfg_idx += 1

                # apply avgpool only for the last layer
                if cfg_idx == len(model_cfg):
                    print("applying avgpool ---------------")
                    activations[idx][layer] = avgpool(activations[idx][layer])

                # unsqueeze back at axis 1
                activations[idx][layer] = activations[idx][layer].unsqueeze(1)

                print("checking stats after pooling")
                print("min of act: {}, max: {}, mean: {}".format(torch.min(activations[idx][layer]),
                                                                 torch.max(activations[idx][layer]),
                                                                 torch.mean(activations[idx][layer])))

            if mode == 'mean':
                activations[idx][layer] = activations[idx][layer].mean(dim=0)
            elif mode == 'std':
                activations[idx][layer] = activations[idx][layer].std(dim=0)
            elif mode == 'meanstd':
                activations[idx][layer] = activations[idx][layer].mean(dim=0) * activations[idx][layer].std(dim=0)

            # ASK standardize, center, normalize acts
            if args.standardize_acts:
                mean_acts = activations[idx][layer].mean(dim=0)
                std_acts = activations[idx][layer].std(dim=0)
                print("shape of mean, std, and usual acts are: ", mean_acts.shape, std_acts.shape, activations[idx][layer].shape)
                activations[idx][layer] = (activations[idx][layer] - mean_acts)/(std_acts + 1e-9)
            elif args.center_acts:
                mean_acts = activations[idx][layer].mean(dim=0)
                print("shape of mean and usual acts are: ", mean_acts.shape, activations[idx][layer].shape)
                activations[idx][layer] = (activations[idx][layer] - mean_acts)
            elif args.normalize_acts:
                print("normalizing the activation vectors")
                activations[idx][layer] = normalize_tensor(activations[idx][layer])
                print("min of act: {}, max: {}, mean: {}".format(torch.min(activations[idx][layer]),
                                                                 torch.max(activations[idx][layer]),
                                                                 torch.mean(activations[idx][layer])))

            print("activations for idx {} at layer {} have the following shape ".format(idx, layer), activations[idx][layer].shape)
            print('-----------')

    # Remove the hooks (as this was intefering with prediction ensembling)
    for idx in range(len(forward_hooks)):
        for hook in forward_hooks[idx]:
            hook.remove()


    return activations

# TODO to be finished, for now use mode = None
def compute_selective_activation(args, models, layer_name, train_loader, num_samples):
    torch.manual_seed(args.activation_seed)

    # hook that computes the mean activations across data samples
    def get_activation(activation, name):
        
        def hook(model, input, output):
            if name not in activation:
                activation[name] = []

            activation[name].append(output.detach())

        return hook

    activations = {}
    forward_hooks = []
    
    param_names = [tupl[0].replace('.weight', '') for tupl in models[0].named_parameters()]

    for idx, model in enumerate(models):

        # Initialize the activation dictionary for each model
        activations[idx] = {}
        
        # Set forward hooks for all layers inside a model
        layer_hooks = []
        for name, layer in model.named_modules():
            if name == '' or name not in param_names:
                print("excluded")
                continue
            else:
                layer_hooks.append(layer.register_forward_hook(get_activation(activations[idx], name)))
                print("set forward hook for layer named: ", name)

        forward_hooks.append(layer_hooks)
       
        # Set the model in train mode
        model.train()

    # Run the same data samples ('num_samples' many) across all the models
    num_samples_processed = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if num_samples_processed == num_samples:
            break
        
        for idx, model in enumerate(models):
            model(data)
        
        num_samples_processed += 1

    relu = torch.nn.ReLU()
    for idx in range(len(models)):
        for lnum, layer in enumerate(activations[idx]):
            print('***********')
            activations[idx][layer] = torch.stack(activations[idx][layer])
            min_act, max_act, mean_act = torch.min(activations[idx][layer]), torch.max(activations[idx][layer]), torch.mean(activations[idx][layer])
            print("min of act: {}, max: {}, mean: {}".format(min_act, max_act, mean_act))

            if not args.prelu_acts and not lnum == (len(activations[idx]) - 1):
                print("applying relu ---------------")
                activations[idx][layer] = relu(activations[idx][layer])
                print("after RELU: min of act: {}, max: {}, mean: {}".format(torch.min(activations[idx][layer]),
                                                                             torch.max(activations[idx][layer]),
                                                                             torch.mean(activations[idx][layer])))
            
            if args.standardize_acts:
                mean_acts = activations[idx][layer].mean(dim=0)
                std_acts = activations[idx][layer].std(dim=0)
                print("shape of mean, std, and usual acts are: ", mean_acts.shape, std_acts.shape, activations[idx][layer].shape)
                activations[idx][layer] = (activations[idx][layer] - mean_acts) / (std_acts + 1e-9)
            
            elif args.center_acts:
                mean_acts = activations[idx][layer].mean(dim=0)
                print("shape of mean and usual acts are: ", mean_acts.shape, activations[idx][layer].shape)
                activations[idx][layer] = (activations[idx][layer] - mean_acts)

            print("activations for idx {} at layer {} have the following shape ".format(idx, layer),
                  activations[idx][layer].shape)
            print('-----------')

    # Remove the hooks (as this was intefering with prediction ensembling)
    for idx in range(len(forward_hooks)):
        for hook in forward_hooks[idx]:
            hook.remove()

    return activations

