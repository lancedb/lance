#!/usr/bin/env python3

import io
import os
import time
from typing import Callable, Optional

import pyarrow
import pytorch_lightning as pl
import torch
import torchvision
from PIL import Image

import lance.pytorch.data

NUM_CLASSES = 38


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


class Classification(pl.LightningModule):
    """Classification model to train"""

    def __init__(
        self,
        backbone: Optional[torch.nn.Module] = None,
        learning_rate=0.1,
        benchmark: Optional[str] = None,
    ) -> None:
        """Build a PyTorch classification model."""
        super().__init__()
        self.backbone = torchvision.models.resnet50(num_classes=NUM_CLASSES)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.benchmark = benchmark
        self.fit_start_time = 0
        self.learning_rate = learning_rate

    @staticmethod
    def get(name: str, **kwargs):
        if name == "resnet":
            return Classification(backbone=torchvision.models.resnet50(num_classes=NUM_CLASSES))
        elif name == "efficientnet":
            return Classification(backbone=torchvision.models.efficientnet_b0(num_classes=NUM_CLASSES))
        else:
            raise ValueError(f"Unsupported model: {name}")

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
            output = self.backbone(images)
            loss = self.criterion(output, labels)
            self.log_dict({"loss": loss})
            return loss

    def configure_optimizers(self):
        # Use hyperparameters from https://github.com/pytorch/vision/tree/main/references/classification
        #
        optimizer = torch.optim.SGD(
            self.parameters(), lr=(self.learning_rate), momentum=0.9, weight_decay=1e-4
        )
        return optimizer


def collate_fn(transform):
    def _collate_fn(batch):
        # TODO: convert label to int64 from Dataset?
        labels = torch.from_numpy(batch[1]).to(torch.int64)
        # TODO: Image conversion should in torch.LanceDataset
        images = [
            transform(Image.open(io.BytesIO(data)).convert("RGB")) for data in batch[0]
        ]
        return torch.stack(images), labels

    return _collate_fn


def raw_collate_fn(batch):
    images = []
    labels = []
    for img, label in batch:
        images.append(img)
        labels.append(label)
    return torch.stack(images), torch.tensor(labels)
