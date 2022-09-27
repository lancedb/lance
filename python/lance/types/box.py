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
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from lance.types.base import LanceType


class Box2dType(LanceType):
    """
    A bounding box in 2-dimensional space. Each box is represented by
    2 2-dimensional points
    """

    def __init__(self):
        super(Box2dType, self).__init__(
            pa.list_(pa.float64(), list_size=4),
            "box2d",
        )

    @property
    def ndims(self):
        return 2

    def __arrow_ext_class__(self):
        return Box2dArray

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return Box2dType()

    def __arrow_ext_serialize__(self):
        return b""


class Box3dType(LanceType):
    """
    A bounding box in 3-dimensional space. Each box is represented by
    2 3-dimensional points
    """

    def __init__(self):
        super(Box3dType, self).__init__(
            pa.list_(pa.float64(), list_size=6),
            "box3d",
        )

    @property
    def ndims(self):
        return 3

    def __arrow_ext_class__(self):
        return Box3dArray

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return Box3dType()

    def __arrow_ext_serialize__(self):
        return b""


class BoxNdArray(pa.ExtensionArray, ABC):
    @property
    @abstractmethod
    def ndims(self):
        """The number of dimensions of each bounding box in this array"""
        pass

    def flatten(self):
        """
        Return a flattened 1-d array
        """
        # If this is a slice, the default Arrow behavior does not
        # return the sliced values properly
        # TODO .values doesn't deal with NAs
        values = self.storage.values
        # If this is a full-length array then just default Arrow
        if len(self) == len(values) / (self.ndims * 2):
            return values
        else:
            # If this is a slice then we need to compute manually
            return pc.list_flatten(self.storage)

    def to_numpy(self, zero_copy_only=True):
        """
        Return a homogeneous 2-d numpy array where each row represents the
        2 points defining each bounding box

        For a 2-d box, each row will have (xmin, ymin, xmax, ymax)
        For a 3-d box, each row will have (xmin, ymin, zmin, xmax, ymax, zmax)
        """
        return (
            self.flatten()
            .to_numpy(zero_copy_only=zero_copy_only)
            .reshape((len(self), (self.ndims * 2)))
        )

    def get_axis_len(self, axis: Union[int, str]):
        """
        Get the length along the given axis

        Parameters
        ----------
        axis: int or str
            0 is 'x', 1 is 'y', 2 is 'z`

        Return
        ------
        len: int
        """
        axis = self._sanitize_axis(axis)
        return self.get_max(axis) - self.get_min(axis) + 1

    def get_min(self, axis: Union[int, str]):
        """
        Get a 1-d numpy array of the minimums for each box along given axis

        Parameters
        ----------
        axis: int or str
            0 is 'x', 1 is 'y', 2 is 'z`
        """
        axis = self._sanitize_axis(axis)
        return self.to_numpy()[:, axis]

    def get_max(self, axis: Union[int, str]):
        """
        Get a 1-d numpy array of the maximums for each box along given axis

        Parameters
        ----------
        axis: int or str
            0 is 'x', 1 is 'y', 2 is 'z`
        """
        axis = self._sanitize_axis(axis)
        return self.to_numpy()[:, self.ndims + axis]

    def _sanitize_axis(self, axis):
        if isinstance(axis, str):
            axis = {"x": 0, "y": 1, "z": 2}[axis]
        if axis < 0 or axis >= self.ndims:
            raise ValueError(
                f"Axis {axis} is greater the number of box " f"dimensions in this array"
            )
        return axis

    @property
    def xmin(self) -> np.ndarray:
        """Return a numpy array of the min X-coord of each box"""
        return self.get_min("x")

    @property
    def ymin(self) -> np.ndarray:
        """Return a numpy array of the min Y-coord of each box"""
        return self.get_min("y")

    @property
    def xmax(self) -> np.ndarray:
        """Return a numpy array of the max X-coord of each box"""
        return self.get_max("x")

    @property
    def ymax(self) -> np.ndarray:
        """Return a numpy array of the max Y-coord of each box"""
        return self.get_max("y")

    def _box_sizes(self):
        return np.multiply.reduce([self.get_axis_len(i) for i in range(self.ndims)])

    def iou(self, others: "BoxNdArray") -> np.ndarray:
        """
        Compute the intersection-over-union between these bounding boxes
        and the given bounding boxes.

        Return
        ------
        ious: an np.ndarray
            If n = len(self), m = len(others) then returned ious should
            have shape (n, m)
        """
        if not isinstance(others, BoxNdArray):
            raise TypeError("Need BoxNDArray to compute IOU")
        if self.ndims != others.ndims:
            raise ValueError("ndims must be equal to compute IOU")
        size_self = self._box_sizes()
        if self is others:
            size_others = size_self
        else:
            size_others = others._box_sizes()

        min_inter = [
            np.maximum.outer(self.get_min(axis), others.get_min(axis))
            for axis in range(self.ndims)
        ]
        max_inter = [
            np.minimum.outer(self.get_max(axis), others.get_max(axis))
            for axis in range(self.ndims)
        ]
        intersection = np.multiply.reduce(
            [np.maximum(max_inter[i] - min_inter[i] + 1, 0) for i in range(self.ndims)]
        )
        union = np.add.outer(size_self, size_others) - intersection
        return intersection / union


class Box2dArray(BoxNdArray):
    """Array of 2d bounding boxes"""

    @property
    def ndims(self):
        return 2

    def area(self) -> np.ndarray:
        """Compute the area of boxes"""
        return self._box_sizes()


class Box3dArray(BoxNdArray):
    """Array of 3d bounding boxes"""

    @property
    def ndims(self):
        return 3

    def volume(self) -> np.ndarray:
        """Compute the area of boxes"""
        return self._box_sizes()

    @property
    def zmin(self) -> np.ndarray:
        """Return a numpy array of the min Z-coord of each box"""
        return self.get_min("z")

    @property
    def zmax(self) -> np.ndarray:
        """Return a numpy array of the max Z-coord of each box"""
        return self.get_max("z")
