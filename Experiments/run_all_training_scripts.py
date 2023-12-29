from Experiments.train_resnet18_mnist import train_resnet18_mnist
from Experiments.train_resnet18_cifar10 import train_resnet18_cifar10
from Experiments.train_resnet18_cifar100 import train_resnet18_cifar100
from Experiments.train_vgg11_mnist import train_vgg11_mnist
from Experiments.train_vgg11_cifar10 import train_vgg11_cifar10
from Experiments.train_vgg11_cifar100 import train_vgg11_cifar100

if __name__ == '__main__':
    batch_sizes = [32, 64, 128, 256, 512]
    max_epochs = 500
    for batch_size in batch_sizes:
        train_resnet18_mnist(max_epochs=max_epochs, batch_size=batch_size)
        train_resnet18_cifar10(max_epochs=max_epochs, batch_size=batch_size)
        train_resnet18_cifar100(max_epochs=max_epochs, batch_size=batch_size)
        train_vgg11_mnist(max_epochs=max_epochs, batch_size=batch_size)
        train_vgg11_cifar10(max_epochs=max_epochs, batch_size=batch_size)
        train_vgg11_cifar100(max_epochs=max_epochs, batch_size=batch_size)
