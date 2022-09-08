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
import pyarrow as pa

from .base import LanceType


class Point2dType(LanceType):
    """
    A Point in 2D space. Represented as 2 floating point numbers
    """

    def __init__(self):
        super(Point2dType, self).__init__(
            pa.struct([pa.field("x", pa.float64()), pa.field("y", pa.float64())]),
            "point2d",
        )

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return Point2dType()


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

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return Box2dType()


class LabelType(LanceType):
    """
    A label used for classification. This is backed by a dictionary type
    to make it easier for translating between human-readable strings and
    integer classes used in the models
    """

    def __init__(self):
        super(LabelType, self).__init__(pa.dictionary(pa.int8(), pa.string()), "label")

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, type_self, storage_type, serialized):
        return LabelType()
