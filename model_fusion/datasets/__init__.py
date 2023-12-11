import enum

from model_fusion.datasets.mnist_datamodule import MNISTDataModule


class DataModuleType(enum.Enum):
    MNIST = 'mnist'

    def get_data_module(self, *args, **kwargs) -> MNISTDataModule:
        if self == DataModuleType.MNIST:
            return MNISTDataModule(*args, **kwargs)
        raise ValueError(f'Unknown dataset: {self}')
