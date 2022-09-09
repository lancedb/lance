#!/usr/bin/env python

"""Train and evaluate models on Oxford pet dataset.

"""

import io
import os
import time
from typing import Callable, Optional

import click
import pyarrow.compute as pc
import pyarrow.fs
import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch import optim
from torchvision.models.efficientnet import EfficientNet_B0_Weights

import lance
import lance.pytorch.data

transform = T.Compose([EfficientNet_B0_Weights.DEFAULT.transforms()])


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


def collate_fn(batch):
    # TODO: Labels should be converted via torch.LanceDataset
    labels = torch.randint(0, 31, size=(len(batch[1]),))
    # TODO: Image conversion should in torch.LanceDataset
    images = [
        transform(Image.open(io.BytesIO(data)).convert("RGB")) for data in batch[0]
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
        self.fit_start_time = 0

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
    batch_size: int,
    epochs: int,
    benchmark: str,
    num_workers: int,
    data_format,
):
    print(f"Running benchmark: {benchmark}")
    if data_format == "lance":
        dataset = lance.pytorch.data.LanceDataset(
            dataset,
            columns=["image", "class"],
            batch_size=batch_size,
            # filter=(pc.field("split") == "train")
        )
        train_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_size=None,
            collate_fn=collate_fn,
        )
    elif data_format == "raw":
        dataset = RawOxfordPetDataset(dataset, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            collate_fn=raw_collate_fn,
        )
    else:
        raise ValueError("Unsupported data format")

    model = Classification(benchmark=benchmark)
    trainer = pl.Trainer(
        limit_train_batches=100, max_epochs=epochs, accelerator="gpu", devices=-1
    )
    trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == "__main__":
    train()
