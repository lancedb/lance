#!/usr/bin/env python

"""Train and evaluate models on Oxford pet dataset.

"""

from pydoc import cli
import click
import pytorch_lightning as pl
import torch
import torchvision
from torch import optim

import lance
import lance.pytorch.data


class EfficientNet(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = torchvision.models.efficientnet_b1()

    def training_step(self, batch, batch_idx):
        print(batch, batch_idx)
        x, y = batch
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


@click.command()
@click.option("-b", "--batch_size", default=64, help="batch size", show_default=True)
@click.argument("dataset")
def train(dataset: str, batch_size: int):
    dataset = lance.pytorch.data.LanceDataset(dataset, columns=["class"], batch_size=batch_size)
    train_loader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=None)
    model = EfficientNet()
    trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
    trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == "__main__":
    train()
