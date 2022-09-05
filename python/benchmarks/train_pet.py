#!/usr/bin/env python

"""Train and evaluate models on Oxford pet dataset.

"""

from pydoc import cli
import click
from torch import optim
import pytorch_lightning as pl

import lance


class EfficientNet(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()

    def training_step(self, batch, batch_idx):
        print(batch, batch_idx)
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


@click.command()
@click.option("-b", "--batch_size", default=64, help="batch size", show_default=True)
@click.argument("dataset")
def train(dataset: str, batch_size: int):
    model = EfficientNet()
    trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
    trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == "__main__":
    train()
