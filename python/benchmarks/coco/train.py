#!/usr/bin/env python
#
# Training over coco dataset.

import os
from typing import Callable

import click
import pytorch_lightning as pl
import torch
import torch.multiprocessing
import transforms as T
from common import ObjectDetection, RawCocoDataset, collate_fn
from pytorch_lightning.loggers import TensorBoardLogger
from torchdata.datapipes.iter import IterableWrapper

import lance
import lance.pytorch.data

torch.multiprocessing.set_sharing_strategy("file_system")


def prepare_target(*args):
    """Prepare dataset."""
    images, annotations = args
    # TODO: convert numpy to tensor from pytorch dataset
    return images, {
        "labels": annotations["category_id"].type(torch.int64),
        "boxes": annotations["bbox"],
    }


def get_transform(benchmark: str) -> Callable:
    if benchmark == "io":
        return prepare_target
    elif benchmark == "no_augmentation":
        return T.Compose(
            [prepare_target, T.PILToTensor(), T.ConvertImageDtype(torch.float)]
        )
    elif benchmark == "train":
        # https://github.com/pytorch/vision/blob/24890d718f5a73586ef093371912b5b37a5b0d46/references/detection/presets.py#L37
        return T.Compose(
            [
                prepare_target,
                T.RandomPhotometricDistort(),
                T.RandomZoomOut(fill=list((123.0, 117.0, 104.0))),
                T.RandomIoUCrop(),
                T.RandomHorizontalFlip(),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
            ]
        )
    else:
        raise ValueError("Unsupported benchmark: ", benchmark)


@click.command()
@click.argument("uri")
@click.option(
    "-b", "--batch_size", default=32, help="set batch size", show_default=True
)
@click.option(
    "-e",
    "--epoch",
    type=int,
    default=10,
    show_default=True,
    help="set how many epoch to run",
)
@click.option(
    "-w",
    "--num_workers",
    default=os.cpu_count(),
    help="set Pytorch DataLoader workers",
    show_default=True,
)
@click.option(
    "-m",
    "--model",
    "model_name",
    type=click.Choice(["ssd"]),
    help="Choose the network to run",
)
@click.option(
    "-B",
    "--benchmark",
    type=click.Choice(["io", "no_augmentation", "train"]),
    default="train",
    help="Specify the benchmark to run",
    show_default=True,
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
    benchmark: str,
):
    if data_format == "lance":
        dataset = lance.pytorch.data.Dataset(
            uri,
            columns=["image", "annotations.category_id", "annotations.bbox"],
            batch_size=2,
            transform=get_transform(benchmark),
            # filter=(pc.field("split") == "train")
        )
        dp = IterableWrapper(dataset).shuffle(buffer_size=64)
        train_loader = torch.utils.data.DataLoader(
            dp,
            num_workers=num_workers,
            batch_size=batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    elif data_format == "raw":
        dataset = RawCocoDataset(uri, transform=get_transform(benchmark))
        train_loader = torch.utils.data.DataLoader(
            dataset,
            # shuffle=True,
            num_workers=num_workers,
            batch_size=batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    else:
        raise ValueError("Unsupported data format")

    logger = TensorBoardLogger("logs", name=f"coco_{model_name}")
    trainer = pl.Trainer(
        max_epochs=epoch,
        logger=logger,
        accelerator="gpu",
        devices=-1,
        auto_lr_find=True,
    )
    model = ObjectDetection(benchmark=benchmark)
    if benchmark == "train":
        trainer.tune(model, train_dataloaders=train_loader)
    trainer.fit(model, train_dataloaders=train_loader)


if __name__ == "__main__":
    train()
