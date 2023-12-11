# Adapted from https://lightning.ai/docs/pytorch/stable/data/datamodule.html
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

from model_fusion.config import BASE_DATA_DIR


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: Path | str = f"{BASE_DATA_DIR}",
                 batch_size: int = 32, num_workers: int = 0, *args, **kwargs):
        """Initialize the MNIST dataset module.
        Args:
            data_dir: The directory containing the MNIST dataset.
            batch_size: The batch size to use for the data loaders.
            num_workers: The number of workers to use for the data loaders."""
        super().__init__()
        self.data_dir = data_dir
        # We can turn this into an argument if we want to
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size,
                          num_workers=self.num_workers)
