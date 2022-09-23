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

import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa
from PIL import Image as PILImage
from pyarrow import fs

from lance.io import copy, open_uri
from lance.types.base import LanceType


class ImageType(LanceType, ABC):
    """
    Base type for Image's. Images can be either stored as a Uri pointer or
    bytes directly. Use ImageType.from_storage(storage_arr.type) to get an
    instance of the correct type (only works for utf8 or binary/large-binary
    storage types).
    """

    def __arrow_ext_serialize__(self):
        return b""

    def __arrow_ext_class__(self):
        return ImageArray

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

    def __arrow_ext_scalar_class__(self):
        return ImageUriScalar


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

    def __arrow_ext_scalar_class__(self):
        return ImageBinaryScalar


class Image(ABC):
    """
    Represents an Image that can either be in-memory or stored at some uri.
    This class allows us to convert to/from Image bytes, uri, numpy, and PIL
    representations
    """

    @staticmethod
    def create(data: Optional[Union[bytes, bytearray, str]]):
        if pd.isna(data):
            return None
        if isinstance(data, bytearray):
            data = bytes(data)

        if isinstance(data, bytes):
            img = ImageBinary(data)
        elif isinstance(data, str):
            img = ImageUri(data)
        else:
            raise TypeError(
                f"Image can only handle bytes or str " f"but got {type(data)}"
            )
        return img

    @classmethod
    def from_dict(cls, data: dict):
        """
        Construct Image from result of Image.to_dict
        """
        return Image.create(data.get("data", data.get("uri")))

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
    def save(self, uri: Union[str, Path]) -> "Image":
        """Write this image to the given uri and return the new Image"""
        pass

    @abstractmethod
    def to_arrow_storage(self):
        """Get the Arrow scalar storage value"""
        pass

    @property
    @abstractmethod
    def arrow_dtype(self):
        """Return the Arrow Extension dtype"""
        pass

    def display(self, **kwargs):
        """Return the jupyter compatible viz for this image"""
        import base64

        from IPython.display import Image as IPyImage

        with self.open() as fobj:
            data = fobj.read()
            inferred_format = IPyImage(data).format
            encoded = base64.b64encode(data).decode("utf-8")
            url = f"data:image;base64,{encoded}"
            return IPyImage(url=url, format=inferred_format)

    def _repr_mimebundle_(self, include=None, exclude=None):
        return self.display()._repr_mimebundle_(include=include, exclude=exclude)


class ImageBinary(Image):
    """
    An In-memory Image
    """

    DTYPE = ImageBinaryType()

    def __init__(self, data: bytes):
        self._bytes = data

    @classmethod
    def from_numpy(cls, arr: np.ndarray, format: str = "png"):
        """Construct an Image from a numpy array."""
        img: PILImage = PILImage.fromarray(arr).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format=format)
        return ImageBinary(data=buf.getvalue())

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
        return ImageUri(uri)

    def __eq__(self, other):
        return isinstance(other, ImageBinary) and other.bytes == self.bytes

    def __repr__(self):
        return "Image(<embedded>)"

    def to_arrow_storage(self):
        return self.bytes

    @property
    def arrow_dtype(self):
        return ImageBinary.DTYPE


class ImageUri(Image):
    """
    An externalized image represented by its uri
    """

    DTYPE = ImageUriType()

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
            return Image.create(img.read())

    def save(self, uri):
        return ImageUri(copy(self.uri, uri))

    def __eq__(self, other):
        return isinstance(other, ImageUri) and other.uri == self.uri

    def __repr__(self):
        return f"Image({self.uri})"

    def to_arrow_storage(self):
        return self.uri

    @property
    def arrow_dtype(self):
        return ImageUri.DTYPE


class ImageBinaryScalar(pa.ExtensionScalar):
    """Used by ExtensionArray.to_pylist()"""

    def as_py(self) -> Optional[Image]:
        if pd.isna(self.value):
            return None
        return ImageBinary(self.value.as_py())


class ImageUriScalar(pa.ExtensionScalar):
    """Used by ExtensionArray.to_pylist()"""

    def as_py(self) -> Optional[Image]:
        if pd.isna(self.value):
            return None
        return ImageUri(self.value.as_py())


class ImageArray(pa.ExtensionArray):
    @staticmethod
    def from_pandas(obj, mask=None, type=None, safe=True, memory_pool=None):
        """
        Create an ImageArray instance from a variety of formats. Specifically
        this knows how to convert sequence of Image instances to ImageArray

        Parameters
        ----------
        obj : ndarray, pandas.Series, array-like
        mask : array (boolean), optional
            Indicate which values are null (True) or not null (False).
        type : pyarrow.DataType
            Explicit type to attempt to coerce to, otherwise will be inferred
            from the data.
        safe : bool, default True
            Check for overflows or other unsafe conversions.
        memory_pool : pyarrow.MemoryPool, optional
            If not passed, will allocate memory from the currently-set default
            memory pool.

        Returns
        -------
        array : pyarrow.Array or pyarrow.ChunkedArray
            ChunkedArray is returned if object data overflows binary buffer.
        """
        if isinstance(obj, pa.ChunkedArray):
            chunks = [ImageArray.from_pandas(c) for c in obj.chunks]
            return pa.chunked_array(chunks, chunks[0].type)

        if isinstance(obj, pa.Array):
            if pa.types.is_binary(obj.type):
                return ImageArray.from_storage(ImageBinary.DTYPE, obj)
            elif pa.types.is_string(obj.type):
                return ImageArray.from_storage(ImageUri.DTYPE, obj)

        if isinstance(obj, (list, tuple, np.ndarray)) and len(obj) > 0:
            first = obj[0]
            if isinstance(first, Image):
                return ImageArray.from_images(obj, mask, type, safe, memory_pool)
            elif isinstance(first, bytes):
                storage = pa.array(
                    obj, mask=mask, type=pa.binary(), safe=safe, memory_pool=memory_pool
                )
                return ImageArray.from_pandas(storage)
            elif isinstance(first, str):
                storage = pa.array(
                    obj, mask=mask, type=pa.string(), safe=safe, memory_pool=memory_pool
                )
                return ImageArray.from_pandas(storage)

        return pa.ExtensionArray.from_pandas(
            obj, mask=mask, type=type, safe=safe, memory_pool=memory_pool
        )

    @staticmethod
    def from_images(images, type=None, mask=None, safe=True, memory_pool=None):
        """
        Create an ImageArray from Image instances

        Parameters
        ----------
        images : sequence / iterable / ndarray / pandas.Series of Image
        type : DataType, default None
            If not specified then use the arrow_dtype of the first Image
            instance
        mask : array[bool], optional
            Indicate which values are null (True) or not null (False).
        safe : bool, default True
            Check for overflows or other unsafe conversions.
        memory_pool : pyarrow.MemoryPool, optional
            If not passed, will allocate memory from the currently-set default
            memory pool.
        """
        if len(images) > 0:
            type = images[0].arrow_dtype
            storage = pa.array(
                [im.to_arrow_storage() if im is not None else None for im in images],
                mask=mask,
                safe=safe,
                memory_pool=memory_pool,
            )
        else:
            storage = pa.array(
                [], type=type, mask=mask, safe=safe, memory_pool=memory_pool
            )
        return pa.ExtensionArray.from_storage(type, storage)


def _ensure_type(images, typ):
    for im in images:
        if not isinstance(im, typ):
            raise TypeError(f"Expecting {typ} but got {type(im)}")


def is_image_type(t: pa.DataType) -> bool:
    """Returns True if the type is an image type"""
    return isinstance(t, ImageType)
