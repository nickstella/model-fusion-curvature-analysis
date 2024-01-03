from Experiments.train_resnet18_mnist import train_resnet18_mnist
from Experiments.train_resnet18_cifar10 import train_resnet18_cifar10
from Experiments.train_resnet18_cifar100 import train_resnet18_cifar100
from Experiments.train_vgg11_mnist import train_vgg11_mnist
from Experiments.train_vgg11_cifar10 import train_vgg11_cifar10
from Experiments.train_vgg11_cifar100 import train_vgg11_cifar100


IS_FILIPPO = False


if __name__ == '__main__':
    min_epochs_cifar = 50
    max_epochs_cifar = 200
    min_epochs_mnist = 20
    max_epochs_mnist = 100

    batch_sizes = [32, 128, 512]
    seeds = [[42, 42], [42, 43], [43, 42]]

    if not IS_FILIPPO:
        for batch_size in batch_sizes:
            for model_seed, data_seed in seeds:
                train_resnet18_mnist(
                    min_epochs=min_epochs_mnist, max_epochs=max_epochs_mnist,
                    batch_size=batch_size, model_seed=model_seed, data_seed=data_seed)

    if not IS_FILIPPO:
        for batch_size in batch_sizes[:1]:
            for model_seed, data_seed in seeds:
                train_resnet18_cifar10(
                    min_epochs=min_epochs_cifar, max_epochs=max_epochs_cifar,
                    batch_size=batch_size, model_seed=model_seed, data_seed=data_seed)

    if IS_FILIPPO:
        for batch_size in batch_sizes[1:]:
            for model_seed, data_seed in seeds:
                train_resnet18_cifar10(
                    min_epochs=min_epochs_cifar, max_epochs=max_epochs_cifar,
                    batch_size=batch_size, model_seed=model_seed, data_seed=data_seed)

    if not IS_FILIPPO:
        for batch_size in batch_sizes:
            for model_seed, data_seed in seeds:
                train_vgg11_mnist(
                    min_epochs=min_epochs_mnist, max_epochs=max_epochs_mnist,
                    batch_size=batch_size, model_seed=model_seed, data_seed=data_seed)

    if not IS_FILIPPO:
        for batch_size in batch_sizes[:1]:
            for model_seed, data_seed in seeds:
                train_vgg11_cifar10(
                    min_epochs=min_epochs_cifar, max_epochs=max_epochs_cifar,
                    batch_size=batch_size, model_seed=model_seed, data_seed=data_seed)

    if IS_FILIPPO:
        for batch_size in batch_sizes[1:]:
            for model_seed, data_seed in seeds:
                train_vgg11_cifar10(
                    min_epochs=min_epochs_cifar, max_epochs=max_epochs_cifar,
                    batch_size=batch_size, model_seed=model_seed, data_seed=data_seed)
