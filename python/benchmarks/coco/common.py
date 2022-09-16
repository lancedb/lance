#!/usr/bin/env python

import io
from typing import Callable, Optional

import PIL
import pytorch_lightning as pl
import torch
import torchvision
import transforms as T


def collate_fn(batch):
    # Collate for training
    # TODO: consider what can be auto inferred.
    #
    target = []
    # https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssd300_vgg16.html\
    #   #torchvision.models.detection.ssd300_vgg16
    for raw_annotation in batch[1]:
        labels, boxes = [], []
        for elem in raw_annotation:
            labels.append(elem["category_id"])
            boxes.append(elem["bbox"])
        annotations = {
            "labels": torch.tensor(labels),
            "boxes": torch.tensor(boxes)
        }
        target.append(annotations)
    images = batch[0]
    return images, target


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
    """General object detection model for training and inference benchmarks.

    Hyper parameters are referred to

    https://github.com/pytorch/vision/blob/main/references/detection
    """
    def __init__(self):
        super().__init__()
        self.backbond = torchvision.models.detection.ssd300_vgg16()
        self.lr = 0

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.backbond(images, targets)
        return loss_dict

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005
        )
        return optimizer
