#  Copyright (c) 2022. Lance Developers
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import numpy as np

import pyarrow as pa

from lance.types.base import LanceType


class Box2dType(LanceType):
    """
    A rectangular box in 2D space (usually used for bounding boxes).
    Represented as 2 Point2Ds (top-left and bottom-right corners)
    """

    def __init__(self):
        super(Box2dType, self).__init__(
            pa.struct(
                [
                    pa.field("xmin", pa.float64()),
                    pa.field("ymin", pa.float64()),
                    pa.field("xmax", pa.float64()),
                    pa.field("ymax", pa.float64()),
                ]
            ),
            "box2d",
        )

    def __arrow_ext_class__(self):
        return Box2dArray

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return Box2dType()


class Box2dArray(pa.ExtensionArray):

    def iou(self, others: "Box2dArray") -> np.ndarray:
        """
        Compute the intersection-over-union between these bounding boxes
        and the given bounding boxes.

        Return
        ------
        ious: a 2-d np.ndarray
            If n = len(self), m = len(others) then returned ious should
            have shape (n, m)
        """
        if not isinstance(others, Box2dArray):
            raise TypeError("Only box2d knows iou")
        area_self = self.area()
        if self is others:
            area_others = area_self
        else:
            area_others = others.area()
        xmin_inter = np.maximum(self.xmin[:, np.newaxis], others.xmin)
        ymin_inter = np.maximum(self.ymin[:, np.newaxis], others.ymin)
        xmax_inter = np.minimum(self.xmax[:, np.newaxis], others.xmax)
        ymax_inter = np.minimum(self.ymax[:, np.newaxis], others.ymax)
        intersection = (np.maximum(xmax_inter - xmin_inter + 1, 0) *
                        np.maximum(ymax_inter - ymin_inter + 1, 0))
        union = area_self[:, np.newaxis] + area_others - intersection
        return intersection / union

    def area(self) -> np.ndarray:
        """Compute the area of boxes"""
        return (self.xmax - self.xmin + 1) * (self.ymax - self.ymin + 1)

    @property
    def xmin(self) -> np.ndarray:
        """Return a numpy array of the min X-coord of each box"""
        return self.storage.field("xmin").to_numpy()

    @property
    def xmax(self) -> np.ndarray:
        """Return a numpy array of the max X-coord of each box"""
        return self.storage.field("xmax").to_numpy()

    @property
    def ymin(self) -> np.ndarray:
        """Return a numpy array of the min Y-coord of each box"""
        return self.storage.field("ymin").to_numpy()

    @property
    def ymax(self) -> np.ndarray:
        """Return a numpy array of the max Y-coord of each box"""
        return self.storage.field("ymax").to_numpy()
