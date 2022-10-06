#!/usr/bin/env python

"""Train and evaluate models on Oxford pet dataset.

"""

import os

import click
import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms as T
from common import (
    NUM_CLASSES,
    Classification,
    RawOxfordPetDataset,
    collate_fn,
    raw_collate_fn,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torchdata.datapipes.iter import IterableWrapper
from torchvision.transforms.functional import InterpolationMode

import lance
import lance.pytorch.data


class TrainTransform(torch.nn.Module):
    """
    Image transform for training.

    Adding random augmentation.

    https://github.com/pytorch/vision/blob/a89b1957a62e2f68f001d5d60268743edbe164d8/references/classification/presets.py#L6
    """

    def __init__(
        self,
        crop_size: float,
        mean: tuple[float] = (0.485, 0.456, 0.406),
        std: tuple[float] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.transform = T.Compose(
            [
                T.RandomResizedCrop(
                    crop_size, interpolation=InterpolationMode.BILINEAR
                ),
                T.RandomHorizontalFlip(),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.transform(tensor)


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
@click.option("-m", "--model", type=click.Choice(["resnet"]), default="resnet")
@click.option(
    "--format",
    "-F",
    "data_format",
    type=click.Choice(["lance", "raw", "parquet"]),
    default="lance",
)
@click.option("--benchmark", type=click.Choice(["io", "train"]), default="train")
@click.argument("dataset")
def train(
    dataset: str,
    model: str,
    batch_size: int,
    epochs: int,
    benchmark: str,
    num_workers: int,
    data_format,
):

    transform = TrainTransform(crop_size=224)
    if model == "resnet":
        m = torchvision.models.resnet50(num_classes=NUM_CLASSES)
    elif model == "efficientnet":
        m = torchvision.models.efficientnet_b0(num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unsupported model: {model}")

    print(f"Running benchmark: {benchmark}")
    if data_format == "lance":
        dataset = lance.pytorch.data.LanceDataset(
            dataset,
            columns=["image", "class"],
            batch_size=batch_size,
            # filter=(pc.field("split") == "train")
        )
        dp = IterableWrapper(dataset).shuffle()
        train_loader = torch.utils.data.DataLoader(
            dp,
            num_workers=num_workers,
            batch_size=None,
            collate_fn=collate_fn(transform),
        )
    elif data_format == "raw":
        dataset = RawOxfordPetDataset(dataset, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            dataset,
            shuffle=True,
            num_workers=num_workers,
            batch_size=batch_size,
            collate_fn=raw_collate_fn,
        )
    else:
        raise ValueError("Unsupported data format")

    logger = TensorBoardLogger("logs", name=f"oxford_pet_{model}")

    model = Classification(benchmark=benchmark, backbone=m)
    trainer = pl.Trainer(
        limit_train_batches=100,
        max_epochs=epochs,
        logger=logger,
        accelerator="gpu",
        devices=-1,
        auto_lr_find=True,
    )
    trainer.tune(model, train_dataloaders=train_loader)
    trainer.fit(model=model, train_dataloaders=train_loader)
    trainer.save_checkpoint("model.ckpt")


if __name__ == "__main__":
    train()
