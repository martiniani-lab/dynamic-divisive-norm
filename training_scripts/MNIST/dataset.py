import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl


class MnistDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, permuted=False, resize=1.0, seed=73):
        super().__init__()
        torch.manual_seed(seed)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.permuted = permuted

        self.h = int(28 * resize)
        self.w = int(28 * resize)
        self.channels = 1

        if self.permuted:
            self.idx_permute = torch.randperm(self.h * self.w, dtype=torch.int64, generator=torch.Generator().manual_seed(seed))
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.Normalize((0.1307,), (0.3081,)),
                    transforms.Resize((self.h, self.w), antialias=True),
                    transforms.Lambda(
                        lambda x: x.view(-1)[self.idx_permute].view(1, self.h, self.w)
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((self.h, self.w), antialias=True),
                ]
            )

    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        entire_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            transform=self.transform,
            download=False,
        )
        # split train and val data either from the checkpoint if available or randomly
        self.train_ds, self.val_ds = random_split(
            entire_dataset, [57000, 3000], generator=torch.Generator().manual_seed(73)
        )

        # note that this random split of dataset can cause discontinuity in training from a checkpoint
        self.test_ds = datasets.MNIST(
            root=self.data_dir,
            train=False,
            transform=self.transform,
            download=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
