from Experiments.train_resnet18_mnist import train_resnet18_mnist
from Experiments.train_resnet18_cifar10 import train_resnet18_cifar10
from Experiments.train_resnet18_cifar100 import train_resnet18_cifar100
from Experiments.train_vgg11_mnist import train_vgg11_mnist
from Experiments.train_vgg11_cifar10 import train_vgg11_cifar10
from Experiments.train_vgg11_cifar100 import train_vgg11_cifar100


if __name__ == '__main__':
    min_epochs_cifar = 50
    max_epochs_cifar = 200
    min_epochs_mnist = 20
    max_epochs_mnist = 100

    # [model_seed, data_seed]
    seeds = [[42, 42], [42, 43], [43, 42]]

    # [batch size, learning_rate]
    resnet_cifar_configs = [
        [32, 0.025],
        [128, 0.01],
        [512, 0.04],
    ]

    # [batch size, learning_rate]
    resnet_mnist_configs = [
        [32, 0.025],
        [512, 0.1],
    ]

    # [batch size, learning_rate]
    vgg_cifar_configs = [
        [32, 0.025],
        [512, 0.01]
    ]

    for batch_size, lr in resnet_cifar_configs:
        for model_seed, data_seed in seeds:
            train_resnet18_cifar10(
                min_epochs=min_epochs_cifar, max_epochs=max_epochs_cifar,
                batch_size=batch_size, learning_rate=lr, model_seed=model_seed, data_seed=data_seed)

    for batch_size, lr in resnet_mnist_configs:
        for model_seed, data_seed in seeds:
            train_resnet18_mnist(
                min_epochs=min_epochs_mnist, max_epochs=max_epochs_mnist,
                batch_size=batch_size, learning_rate=lr, model_seed=model_seed, data_seed=data_seed)

    for batch_size, lr in vgg_cifar_configs:
        for model_seed, data_seed in seeds:
            train_vgg11_cifar10(
                min_epochs=min_epochs_cifar, max_epochs=max_epochs_cifar,
                batch_size=batch_size, learning_rate=lr, model_seed=model_seed, data_seed=data_seed)
