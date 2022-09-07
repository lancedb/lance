#!/usr/bin/env python

"""Train and evaluate models on Oxford pet dataset.

"""

import io
import os
from typing import Optional

import click
import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch import optim
from torchvision.models.efficientnet import EfficientNet_B0_Weights

import lance
import lance.pytorch.data


def collate_fn(batch):
    # TODO: Labels should be converted via torch.LanceDataset
    labels = torch.randint(0, 31, size=(len(batch[1]),))
    # TODO: Image conversion should in torch.LanceDataset
    preprocessing = EfficientNet_B0_Weights.DEFAULT.transforms()
    images = [
        preprocessing(T.ToTensor()(Image.open(io.BytesIO(data)))) for data in batch[0]
    ]
    return torch.stack(images), labels


class Classification(pl.LightningModule):
    """Classification model to train"""

    def __init__(
        self,
        model: torch.nn.Module = torchvision.models.efficientnet_b0(),
        benchmark: Optional[str] = None,
    ) -> None:
        """Build a PyTorch classification model."""
        super().__init__()
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.benchmark = benchmark

    def training_step(self, batch, batch_idx):
        """
        https://github.com/pytorch/vision/blob/main/references/classification/train.py
        """
        images, labels = batch
        if self.benchmark == "io":
            # only test I/O
            pass
        else:
            output = self.model(images)
            loss = self.criterion(output, labels)
            return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


@click.command()
@click.option("-b", "--batch_size", default=64, help="batch size", show_default=True)
@click.option("-e", "--epochs", default=10, help="set max ephochs", show_default=True)
@click.option(
    "-w",
    "--num_workers",
    default=os.cpu_count(),
    help="set pytorch DataLoader number of workers",
    show_default=True,
)
@click.option("--benchmark", type=click.Choice(["io", "train"]), default="train")
@click.argument("dataset")
def train(dataset: str, batch_size: int, epochs: int, benchmark: str, num_workers: int):
    print(f"Running benchmark: {benchmark}")
    dataset = lance.pytorch.data.LanceDataset(
        dataset,
        columns=["image", "class"],
        batch_size=batch_size,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=None,
        collate_fn=collate_fn,
    )
    model = Classification(benchmark=benchmark)
    trainer = pl.Trainer(
        limit_train_batches=100, max_epochs=epochs, accelerator="gpu", devices=-1
    )
    trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == "__main__":
    train()
