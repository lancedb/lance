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
from abc import abstractmethod, ABC
import io
from pathlib import Path
from typing import IO

import numpy as np
from PIL import Image as PILImage
import pyarrow as pa
from pyarrow import fs

from lance.io import open_uri, copy
from lance.types.base import LanceType


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


class Image(ABC):
    """
    Represents an Image that can either be in-memory or stored at some uri.
    This class allows us to convert to/from Image bytes, uri, numpy, and PIL
    representations
    """

    def __new__(cls, data: [bytes | str]):
        if isinstance(data, bytes):
            img = object.__new__(EmbeddedImage)
        elif isinstance(data, str):
            img = object.__new__(ImageRef)
        else:
            raise TypeError(f"{cls.__name__} can only handle bytes or str "
                            f"but got {type(data)}")
        img.__init__(data)
        return img

    @classmethod
    def from_dict(cls, data: dict):
        """
        Construct Image from result of Image.to_dict
        """
        return Image(data.get("data", data.get("uri")))

    @abstractmethod
    def open(self) -> IO:
        """Open for reading"""
        pass

    def to_pil(self, mode="RGB") -> "PILImage":
        """Get a PIL instance converted to RGB by default"""
        return PILImage.open(self.open()).convert(mode=mode)

    def to_numpy(self) -> np.ndarray:
        """Get the 3d numpy array representation of this image"""
        with self.to_pil() as pil_img:
            return np.asarray(pil_img)

    @abstractmethod
    def to_dict(self) -> dict:
        """Return a json friendly dict"""
        pass

    @abstractmethod
    def to_embedded(self) -> "Image":
        """Return an in-memory Image instance"""
        pass

    @abstractmethod
    def save(self, uri: [str | Path]) -> "Image":
        """Write this image to the given uri and return the new Image"""
        pass


class EmbeddedImage(Image):
    """
    In-memory Image
    """

    def __init__(self, data: bytes):
        self._bytes = data

    @property
    def bytes(self) -> bytes:
        return self._bytes

    def open(self) -> IO:
        return io.BytesIO(self.bytes)

    def to_dict(self) -> dict:
        return {"data": self.bytes}

    def to_embedded(self):
        return self

    def save(self, uri):
        filesystem, path = fs.FileSystem.from_uri(uri)
        with filesystem.open_output_stream(path) as fobj:
            fobj.write(self.bytes)
        return ImageRef(uri)

    def __eq__(self, other):
        return isinstance(other, EmbeddedImage) and other.bytes == self.bytes


class ImageRef(Image):
    """
    Image reference
    """

    def __init__(self, uri: str):
        self._uri = uri

    @property
    def uri(self):
        return self._uri

    def open(self) -> IO:
        return open_uri(self.uri)

    def to_dict(self) -> dict:
        return {"uri": self.uri}

    def to_embedded(self):
        with self.open() as img:
            return Image(img.read())

    def save(self, uri):
        copy(self.uri, uri)
        return ImageRef(uri)

    def __eq__(self, other):
        return isinstance(other, ImageRef) and other.uri == self.uri


class ImageScalar(pa.ExtensionScalar):
    """Used by ExtensionArray.to_pylist()"""

    def as_py(self) -> Image:
        return Image(self.value.as_py())
