#!/usr/bin/env python

"""PyTorch Vision transforms that is useful for data augmentation for
Object Detection training tasks.

https://github.com/pytorch/vision/blob/main/references/detection/transforms.py

"""

# TODO: This file should be shared with other object detection modules.

from typing import Callable, Dict, List, Optional, Tuple

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T

__all__ = ["Compose", "PILToTensor", "ConvertImageDtype", "RandomHorizontalFlip"]


class Compose:
    """Compose several transforms together, apply them on both and target.

    See: `torchvision.transform.Compose`
    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class PILToTensor(torch.nn.Module):
    """Convert PIL image to PyTorch Tensors."""

    def forward(
        self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        image = F.pil_to_tensor(image)
        return image, target


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    """Randomly flip the image horizontally.

    See
    ---
    https://github.com/pytorch/vision/blob/main/references/detection/transforms.py
    """

    def forward(
        self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                _, _, width = F.get_dimensions(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                # if "masks" in target:
                #     target["masks"] = target["masks"].flip(-1)
                # if "keypoints" in target:
                #     keypoints = target["keypoints"]
                #     keypoints = _flip_coco_person_keypoints(keypoints, width)
                #     target["keypoints"] = keypoints
        return image, target


class ConvertImageDtype(torch.nn.Module):
    """Convert Image data type
    """

    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(
        self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        image = F.convert_image_dtype(image, self.dtype)
        return image, target
