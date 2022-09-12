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
import platform

import pyarrow as pa
from pyarrow import ArrowKeyError

from lance.types.base import Point2dType
from lance.types.box import Box2dArray, Box2dType
from lance.types.image import (
    Image,
    ImageBinary,
    ImageBinaryType,
    ImageType,
    ImageUri,
    ImageUriType,
)
from lance.types.label import LabelArray, LabelType


def register_extension_types():
    if platform.system() != "Linux":
        raise NotImplementedError("Extension types are only supported on Linux for now")
    try:
        pa.register_extension_type(ImageUriType())
        pa.register_extension_type(ImageBinaryType())
        pa.register_extension_type(Point2dType())
        pa.register_extension_type(Box2dType())
        pa.register_extension_type(LabelType())
    except ArrowKeyError:
        # already registered
        pass
