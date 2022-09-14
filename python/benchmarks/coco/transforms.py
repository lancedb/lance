#!/usr/bin/env python

"""PyTorch Vision transforms for General Object Detection Tasks.

https://github.com/pytorch/vision/blob/main/references/detection/transforms.py

"""

# TODO: This file should be shared with other object detection modules.

from typing import Callable, List, Optional, Dict, Tuple

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T

__all__ = ["Compose"]


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
    """Convert PIL image to PyTorch Tensors.

    """

    def forward(
        self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        image = F.pil_to_tensor(image)
        return image, target
