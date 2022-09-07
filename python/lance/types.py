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
import platform
from abc import ABC, abstractproperty, abstractmethod
from typing import Optional

import pandas as pd
import pyarrow as pa
from pyarrow import ArrowKeyError

__all__ = [
    "ImageType",
    "ImageUriType",
    "ImageBinaryType",
    "Point2dType",
    "Box2dType",
    "LabelType",
]


class LanceType(pa.ExtensionType, ABC):
    def __init__(self, storage_type, extension_name):
        if platform.system() != "Linux":
            raise NotImplementedError(
                "Extension types are enabled for linux only for now"
            )
        super(LanceType, self).__init__(storage_type, extension_name)


class ImageType(LanceType):
    """
    Base type for Image's. Images can be either stored as a Uri pointer or
    bytes directly. Use ImageType.from_storage(storage_arr.type) to get an
    instance of the correct type (only works for utf8 or binary/large-binary
    storage types).
    """

    def __arrow_ext_serialize__(self):
        return b""

    def __arrow_ext_scalar_class__(self):
        return ImageScalar

    @classmethod
    def from_storage(cls, storage_type):
        # TODO consider parameterizing types to support utf* variants
        #      and also large binary (for geo or medical imaging)
        if storage_type == pa.utf8():
            return ImageUriType()
        elif storage_type == pa.binary():
            return ImageBinaryType()
        else:
            raise NotImplementedError(f"Unrecognized image storage type {storage_type}")


class ImageUriType(ImageType):
    """
    Represents an externalized Image containing just the uri. Storage type is
    utf8
    """

    def __init__(self):
        super(ImageUriType, self).__init__(pa.utf8(), "image[uri]")

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return ImageUriType()


class ImageBinaryType(ImageType):
    """
    Represents an inlined Image containing the actual image bytes. Storage type
    is binary.

    TODO: add support for large binary
    """

    def __init__(self):
        super(ImageBinaryType, self).__init__(pa.binary(), "image[binary]")

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return ImageBinaryType()


class Image:

    def __init__(self, data: [bytes, str]):
        if isinstance(data, bytes):
            self._bytes: Optional[bytes] = data
            self._uri: Optional[str] = None
        elif isinstance(data, str):
            self._bytes: Optional[bytes] = None
            self._uri: Optional[str] = data

    @property
    def uri(self) -> str:
        return self.uri

    @property
    def bytes(self) -> bytes:
        return self.bytes

    def to_pil(self):
        import io
        from PIL import Image
        if self.bytes is not None:
            return Image.open(io.BytesIO(self.bytes))
        elif self.uri is not None:
            with open_uri(self.uri) as fp:
                return Image.open(fp)


class ImageScalar(pa.ExtensionScalar, ABC):

    @abstractmethod
    def as_py(self) -> Image:
        return Image(self.value.as_py())


# TODO turn these into fixed sized list arrays once GH#101 is done
class Point2dType(LanceType):
    """
    A Point in 2D space. Represented as 2 floating point numbers
    """

    def __init__(self):
        super(Point2dType, self).__init__(pa.list_(pa.float64()), "point2d")

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return Point2dType()


# TODO turn these into fixed sized list arrays once GH#101 is done
class Box2dType(LanceType):
    """
    A rectangular box in 2D space (usually used for bounding boxes).
    Represented as 2 Point2Ds (top-left and bottom-right corners)
    """

    def __init__(self):
        super(Box2dType, self).__init__(pa.list_(pa.float64()), "box2d")

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, openstorage_type, serialized):
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
