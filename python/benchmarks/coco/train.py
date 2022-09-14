#!/usr/bin/env python
#
# Training over coco dataset.

import os

import click
import pytorch_lightning as pl
import torch
from common import ObjectDetection, RawCocoDataset, collate_fn
from pytorch_lightning.loggers import TensorBoardLogger
from torchdata.datapipes.iter import IterableWrapper

import lance
import lance.pytorch.data
import transforms as T

# https://github.com/pytorch/vision/blob/24890d718f5a73586ef093371912b5b37a5b0d46/references/detection/presets.py#L37
transform = T.Compose(
    [
        T.RandomPhotometricDistort(),
        T.RandomZoomOut(fill=list((123.0, 117.0, 104.0))),
        T.RandomIoUCrop(),
        T.RandomHorizontalFlip(),
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
    ]
)


@click.command()
@click.argument("uri")
@click.option("-b", "--batch_size", default=4, help="set batch size", show_default=True)
@click.option(
    "-e",
    "--epoch",
    type=int,
    default=10,
    show_default=True,
    help="set how many epoch to run",
)
@click.option(
    "-w", "--num_workers", default=os.cpu_count(), help="set Pytorch DataLoader workers"
)
@click.option(
    "-m",
    "--model",
    "model_name",
    type=click.Choice(["ssd"]),
    help="Choose the network to run",
)
@click.option(
    "-f",
    "--format",
    "data_format",
    type=click.Choice(["lance", "raw"]),
    default="lance",
    help="File format to train on",
)
def train(
    uri: str,
    batch_size: int,
    epoch: int,
    num_workers: int,
    model_name: str,
    data_format: str,
):
    if data_format == "lance":
        dataset = lance.pytorch.data.LanceDataset(
            uri,
            columns=["image", "annotations.category_id", "annotations.bbox"],
            batch_size=batch_size,
            transform=transform,
            # filter=(pc.field("split") == "train")
        )
        dp = IterableWrapper(dataset).shuffle()
        train_loader = torch.utils.data.DataLoader(
            dp,
            num_workers=num_workers,
            batch_size=None,
            collate_fn=collate_fn,
        )
    elif data_format == "raw":
        dataset = RawCocoDataset(uri, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            dataset,
            shuffle=True,
            num_workers=num_workers,
            batch_size=batch_size,
            collate_fn=raw_collate_fn,
        )
    else:
        raise ValueError("Unsupported data format")

    logger = TensorBoardLogger("logs", name=f"coco_{model_name}")
    trainer = pl.Trainer(
        limit_train_batches=100,
        max_epochs=epoch,
        logger=logger,
        accelerator="gpu",
        devices=-1,
        auto_lr_find=True,
    )
    model = ObjectDetection()
    trainer.tune(model, train_dataloaders=train_loader)


if __name__ == "__main__":
    train()
