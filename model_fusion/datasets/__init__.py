import enum

from lightning import LightningDataModule
from pl_bolts.datamodules import MNISTDataModule, CIFAR10DataModule
from torchvision.transforms import Resize, v2

from model_fusion.datasets.cifar100_datamodule import CIFAR100DataModule


class DataModuleType(enum.Enum):
    MNIST = 'mnist'
    CIFAR10 = 'cifar10'
    CIFAR100 = 'cifar100'

    def get_data_module(self, *args, **kwargs) -> LightningDataModule:
        if 'resize' in kwargs:
            new_size = kwargs.pop('resize')
            transforms = v2.Compose([v2.Resize(new_size), v2.ToTensor()])
            kwargs['train_transforms'] = transforms
            kwargs['val_transforms'] = transforms
            kwargs['test_transforms'] = transforms

        if self == DataModuleType.MNIST:
            return MNISTDataModule(*args, **kwargs)
        if self == DataModuleType.CIFAR10:
            return CIFAR10DataModule(*args, **kwargs)
        if self == DataModuleType.CIFAR100:
            return CIFAR100DataModule(*args, **kwargs)
        raise ValueError(f'Unknown dataset: {self}')
