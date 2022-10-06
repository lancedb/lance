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
"""Extension types for computer vision"""

import pyarrow as pa
from pyarrow import ArrowKeyError

from lance.types.base import Point2dType, Point3dType, Polygon2dType, Polygon3dType
from lance.types.box import Box2dArray, Box2dType, Box3dArray, Box3dType
from lance.types.image import (
    Image,
    ImageArray,
    ImageBinary,
    ImageBinaryType,
    ImageType,
    ImageUri,
    ImageUriType,
    is_image_type,
)
from lance.types.label import LabelArray, LabelType


def register_extension_types():
    types = [
        ImageUriType(),
        ImageBinaryType(),
        Point2dType(),
        Point3dType(),
        Box2dType(),
        Box3dType(),
        LabelType(),
    ]
    for t in types:
        try:
            pa.register_extension_type(t)
        except ArrowKeyError:
            # already registered
            pass
