#!/usr/bin/env python

"""Train and evaluate models on Oxford pet dataset.

"""

import io
import os
import time
from typing import Callable, Optional

import click
import pyarrow.fs
import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from torch import optim
from torchdata.datapipes.iter import IterableWrapper
from torchvision.models.efficientnet import EfficientNet_B0_Weights
from torchvision.models.resnet import ResNet50_Weights

import lance
import lance.pytorch.data

NUM_CLASSES = 38


class TrainTransform(torch.nn.Module):
    """
    Image transform for training.
    I
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
                T.RandomResizedCrop(crop_size, interpolation=InterpolationMode.BILINEAR),
                T.RandomHorizontalFlip(),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.transform(tensor)


def raw_collate_fn(batch):
    images = []
    labels = []
    for img, label in batch:
        images.append(img)
        labels.append(label)
    return torch.stack(images), torch.tensor(labels)


class RawOxfordPetDataset(torch.utils.data.Dataset):
    """Build OxfordPet dataset from raw metadata.

    The main difference between this and the one from torchvision is that
    this one can directly read from cloud storage.
    """

    def __init__(
        self, root: str, split: str = "trainval", transform: Optional[Callable] = None
    ):
        """Constructor.

        Parameters
        ----------
        root : str
            Root URI of the dataset, i.e., `s3://eto-public/datasets/oxford_pet/`.
            We expect two sub-directories under the root: "annotations" and "images".
        split : str
            The split to load, either "trailval" or "test"
        """
        self.root = root
        self.split = split
        self.transform = transform

        image_list_file = os.path.join(root, "annotations", f"{split}.txt")
        fs, path = pyarrow.fs.FileSystem.from_uri(image_list_file)
        self.fs = fs
        self.images = []
        self.labels = []
        with fs.open_input_stream(path) as input:
            content = input.readall().decode("utf-8")
            for line in content.split("\n"):
                comps = line.split()
                if comps:
                    self.images.append(comps[0])
                    self.labels.append(torch.tensor(int(comps[1]), dtype=torch.long))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_uri = os.path.join(self.root, "images", f"{self.images[idx]}.jpg")
        fs, path = pyarrow.fs.FileSystem.from_uri(image_uri)
        with fs.open_input_stream(path) as fobj:
            img = Image.open(io.BytesIO(fobj.readall())).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]


def collate_fn(transform):
    def _collate_fn(batch):
        # TODO: Labels should be converted via torch.LanceDataset
        labels = torch.from_numpy(batch[1])
        # TODO: Image conversion should in torch.LanceDataset
        images = [
            transform(Image.open(io.BytesIO(data)).convert("RGB")) for data in batch[0]
        ]
        return torch.stack(images), labels

    return _collate_fn


class Classification(pl.LightningModule):
    """Classification model to train"""

    def __init__(
        self,
        model: torch.nn.Module = torchvision.models.efficientnet_b0(
            num_classes=NUM_CLASSES
        ),
        learning_rate=1e-3,
        benchmark: Optional[str] = None,
    ) -> None:
        """Build a PyTorch classification model."""
        super().__init__()
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.benchmark = benchmark
        self.fit_start_time = 0
        self.learning_rate = learning_rate

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.fit_start_time = time.time()

    def on_fit_end(self) -> None:
        super().on_fit_end()
        print(f"Training finished in {time.time() - self.fit_start_time} seconds")

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
            self.log("loss", loss)
            return loss

    def configure_optimizers(self):
        # Use hyperparameters from https://github.com/pytorch/vision/tree/main/references/classification
        #
        optimizer = optim.SGD(self.parameters(), lr=(self.learning_rate), momentum=0.9, weight_decay=1e-4)
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
@click.option(
    "-m",
    "--model",
    type=click.Choice(["resnet", "efficientnet"]),
    default="resnet",
    show_default=True,
)
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
    if model == "resnet":
        m = torchvision.models.resnet50(num_classes=NUM_CLASSES)
        transform = TrainTransform(crop_size=224)
    elif model == "efficientnet":
        m = torchvision.models.efficientnet_b0(num_classes=NUM_CLASSES)
        transform = TrainTransform(crop_size=224)
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

    model = Classification(benchmark=benchmark, model=m)
    trainer = pl.Trainer(
        limit_train_batches=100,
        max_epochs=epochs,
        accelerator="gpu",
        devices=-1,
        auto_lr_find=True,
    )
    trainer.tune(model, train_dataloaders=train_loader)
    trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == "__main__":
    train()
