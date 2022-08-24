#  Copyright 2022 Lance Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Arrow extension types for Lance
"""
from abc import ABC, abstractproperty
import platform

import pandas as pd

import pyarrow as pa


__all__ = ["ImageType", "ImageUriType", "ImageBinaryType", "Point2dType", "Box2dType"]


# Arrow extension type
from pyarrow import ArrowKeyError


class LanceType(pa.ExtensionType, ABC):
    def __init__(self, storage_type, extension_name):
        if platform.system() != "Linux":
            raise NotImplementedError(
                "Extension types are enabled for linux only for now"
            )
        super(LanceType, self).__init__(storage_type, extension_name)


class ImageType(LanceType):
    @abstractproperty
    def storage_type(self):
        pass

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def from_storage(cls, storage_type):
        if storage_type == pa.utf8():
            return ImageUriType()
        elif storage_type in (pa.binary(), pa.large_binary()):
            return ImageBinaryType()
        else:
            raise NotImplementedError(f"Unrecognized image storage type {storage_type}")


class ImageUriType(ImageType):
    @property
    def storage_type(self):
        return pa.utf8()

    def __init__(self):
        super(ImageUriType, self).__init__(self.storage_type, "image[uri]")

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return ImageUriType()


class ImageBinaryType(ImageType):
    @property
    def storage_type(self):
        return pa.binary()

    def __init__(self):
        super(ImageBinaryType, self).__init__(self.storage_type, "image[binary]")

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return ImageBinaryType()


# TODO turn these into fixed sized list arrays once GH#101 is done
class Point2dType(LanceType):
    def __init__(self):
        super(Point2dType, self).__init__(pa.list_(pa.float64()), "point2d")

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return Point2dType()


# TODO turn these into fixed sized list arrays once GH#101 is done
class Box2dType(LanceType):
    def __init__(self):
        super(Box2dType, self).__init__(pa.list_(pa.float64()), "box2d")

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return Box2dType()


def register_extension_types():
    if platform.system() != "Linux":
        pass
    try:
        pa.register_extension_type(ImageUriType())
        pa.register_extension_type(ImageBinaryType())
        pa.register_extension_type(Point2dType())
        pa.register_extension_type(Box2dType())
    except ArrowKeyError:
        # already registered
        pass
