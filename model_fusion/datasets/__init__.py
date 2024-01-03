import enum

import torch
from lightning import LightningDataModule
from pl_bolts.datamodules import MNISTDataModule, CIFAR10DataModule
from torchvision.transforms.v2 import Resize, Compose, RandomCrop, RandomHorizontalFlip, ToDtype, ToImage
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

from model_fusion.datasets.cifar100_datamodule import CIFAR100DataModule
from model_fusion.config import NUM_WORKERS

def get_cifar_transforms():
    train_transforms = Compose(
        [
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToImage(), ToDtype(torch.float32, scale=True),
            cifar10_normalization(),
        ]
    )
    test_transforms = Compose([ToImage(), ToDtype(torch.float32, scale=True), cifar10_normalization()])
    return train_transforms, test_transforms

class DataModuleType(enum.Enum):
    MNIST = 'mnist'
    CIFAR10 = 'cifar10'
    CIFAR100 = 'cifar100'

    def get_data_module(self, *args, **kwargs) -> LightningDataModule:
        if 'resize' in kwargs:
            new_size = kwargs.pop('resize')
            transforms = Compose([Resize(new_size), ToImage(), ToDtype(torch.float32, scale=True)])
            kwargs['train_transforms'] = transforms
            kwargs['val_transforms'] = transforms
            kwargs['test_transforms'] = transforms

        data_augmentation = kwargs.pop('data_augmentation', False)

        if self == DataModuleType.MNIST:
            return MNISTDataModule(normalize=True, num_workers=NUM_WORKERS, *args, **kwargs)
        if self == DataModuleType.CIFAR10:
            if data_augmentation:
                train_transforms, test_transforms = get_cifar_transforms()
                return CIFAR10DataModule(normalize=True, val_split=0.1,
                                        train_transforms=train_transforms,
                                        test_transforms=test_transforms,
                                        val_transforms=test_transforms,
                                        num_workers=NUM_WORKERS, *args, **kwargs)
            return CIFAR10DataModule(normalize=True, val_split=0.1,
                                     num_workers=NUM_WORKERS, *args, **kwargs)
        if self == DataModuleType.CIFAR100:
            if data_augmentation:
                train_transforms, test_transforms = get_cifar_transforms()
                return CIFAR100DataModule(normalize=True, val_split=0.1,
                                        train_transforms=train_transforms,
                                        test_transforms=test_transforms,
                                        val_transforms=test_transforms,
                                        num_workers=NUM_WORKERS, *args, **kwargs)
            return CIFAR100DataModule(normalize=True, val_split=0.1,
                                      num_workers=NUM_WORKERS, *args, **kwargs)
        raise ValueError(f'Unknown dataset: {self}')
