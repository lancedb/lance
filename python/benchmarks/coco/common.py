#!/usr/bin/env python

from typing import Callable, Optional

import pytorch_lightning as pl
import torch
import torchvision


class RawCocoDataset(torch.utils.data.Dataset):
    """Raw COCO dataset.

    It differs from torchvision COCO dataset because it allows to read cloud storage.
    """

    def __init__(
        self, root: str, split: str = "train", transform: Optional[Callable] = None
    ):
        import pycocotools

        self.root = root
        self.split = split

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        pass


class ObjectDetection(pl.LightningModule):
    """General object detection model for training and inference benchmarks."""
    def __init__(self):
        super().__init__()
        self.backbond = torchvision.models.detection.ssd300_vgg16()

    def training_step(self, *args, **kwargs):
        return super().training_step(*args, **kwargs)

    def configure_optimizers(self):
        # Use hyperparameters from
        # https://github.com/pytorch/vision/blob/main/references/detection
        #
        optimizer = torch.optim.SGD(
            self.parameters(), lr=(self.learning_rate), momentum=0.9, weight_decay=0.0005
        )
        return optimizer