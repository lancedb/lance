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
from abc import ABC

import pandas as pd

import pyarrow as pa


__all__ = ["ImageType", "Point2dType", "Box2dType"]


# Arrow extension type
from pyarrow import ArrowKeyError


class LanceType(pa.ExtensionType, ABC):
    pass


# Pandas extension type
class LanceDtype(pd.api.extensions.ExtensionDtype, ABC):
    pass


class ImageType(LanceType):

    URI_TYPE = "uri"
    BINARY_TYPE = "binary"
    # TODO support Tensor type?

    @classmethod
    def get_storage_type(cls, image_type):
        if image_type == ImageType.URI_TYPE:
            storage_type = pa.utf8()
        elif image_type == ImageType.BINARY_TYPE:
            storage_type = pa.binary()
        else:
            raise NotImplementedError(
                "Lance ImageType must be either 'uri' or 'binary'")
        return storage_type

    def __init__(self, image_type=URI_TYPE):
        self._image_type = image_type
        storage_type = ImageType.get_storage_type(image_type)
        super(ImageType, self).__init__(storage_type, f'lance.image[{image_type}]')

    @property
    def image_type(self):
        return self._image_type

    def __arrow_ext_serialize__(self):
        return f"image_type={self.image_type}".encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        serialized = serialized.decode()
        assert serialized.startswith("image_type=")
        image_type = serialized.split("=")[1]
        assert image_type == cls.URI_TYPE or image_type == cls.BINARY_TYPE
        return ImageType(image_type)


# TODO turn these into fixed sized list arrays once GH#101 is done
class Point2dType(LanceType):

    def __init__(self):
        super(Point2dType, self).__init__(
            pa.list_(pa.float64()),
            'lance.point2d')

    def __arrow_ext_serialize__(self):
        return b''

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return Point2dType()


# TODO turn these into fixed sized list arrays once GH#101 is done
class Box2dType(LanceType):

    def __init__(self):
        super(Box2dType, self).__init__(
            pa.list_(pa.float64()),
            'lance.box2d')

    def __arrow_ext_serialize__(self):
        return b''

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return Box2dType()


def register_extension_types():
    try:
        pa.register_extension_type(ImageType(ImageType.URI_TYPE))
        pa.register_extension_type(ImageType(ImageType.BINARY_TYPE))
        pa.register_extension_type(Point2dType())
        pa.register_extension_type(Box2dType())
    except ArrowKeyError:
        # already registered
        pass
    