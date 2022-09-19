#!/usr/bin/env python

import io
import json
import os
import sys
from typing import Callable, Dict, Optional

import numpy as np
import PIL
import pyarrow as pa
import pytorch_lightning as pl
import torch
import torchvision
import transforms as T

from lance.io import open_uri


def collate_fn(batch):
    if not isinstance(batch, list):
        # In case setting DataLoader.batch_size=None
        batch = [batch]
    images = []
    targets = []
    for record in batch:
        images.append(record[0])
        targets.append(record[1])
    return images, targets


class RawCocoDataset(torch.utils.data.Dataset):
    """Raw COCO dataset.

    It differs from torchvision COCO dataset because it allows to read cloud storage.
    """

    def __init__(
        self, root: str, split: str = "train", transform: Optional[Callable] = None
    ):
        self.root = root
        self.split = split
        self.transform = transform

        self.annotation_file = os.path.join(
            self.root, "annotations", f"instances_{self.split}2017.json"
        )

        self.images = {}
        with open_uri(self.annotation_file, mode="r") as fobj:
            annotation_payload = json.loads(fobj.read())
            for img in annotation_payload["images"]:
                img["annotations"] = []
                self.images[img["id"]] = img
            for ann in annotation_payload["annotations"]:
                image_id = ann["image_id"]
                self.images[image_id]["annotations"].append(ann)

        # Remove all empty annotations images.
        # https://github.com/pytorch/vision/blob/a4f53308b2d0f1aa9191686e326f45c26053f686/references/detection/coco_utils.py#L243-L244
        for key in list(self.images.keys()):
            img = self.images[key]
            if not img["annotations"]:
                del self.images[key]

        self.image_ids = sorted(self.images.keys())

    def _load_image(self, img: Dict):
        image_uri = os.path.join(self.root, f"{self.split}2017", img["file_name"])
        with open_uri(image_uri) as fobj:
            return PIL.Image.open(io.BytesIO(fobj.read())).convert("RGB")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img = self.images[image_id]
        image = self._load_image(img)
        labels = []
        boxes = []
        for ann in img["annotations"]:
            labels.append(torch.tensor(ann["category_id"]))
            bbox = ann["bbox"]
            boxes.append(np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]))

        target = {
            "category_id": np.array(labels),
            "bbox": np.array(boxes),
        }
        # print(image, target)

        if self.transform:
            image, target = self.transform(image, target)
        return image, target


class ObjectDetection(pl.LightningModule):
    """General object detection model for training and inference benchmarks.

    Hyper parameters are referred to

    https://github.com/pytorch/vision/blob/main/references/detection
    """

    def __init__(self):
        super().__init__()
        self.backbond = torchvision.models.detection.ssd300_vgg16()
        self.lr = 0.1

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.backbond(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        if torch.isinf(loss) or torch.isnan(loss):
            print(f"Loss is {loss}, exit")
            sys.exit(1)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005
        )
        return optimizer
