# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import io
from dataclasses import dataclass
from typing import IO, Any, Iterator, Optional, Union

import pyarrow as pa

from .lance import LanceBlobFile


@dataclass(frozen=True)
class Blob:
    """
    A logical blob value for writing Lance blob columns.

    A blob can be represented either by inlined bytes or by an external URI.
    """

    data: Optional[bytes] = None
    uri: Optional[str] = None

    def __post_init__(self) -> None:
        if self.data is not None and self.uri is not None:
            raise ValueError("Blob cannot have both data and uri")
        if self.uri == "":
            raise ValueError("Blob uri cannot be empty")

    @staticmethod
    def from_bytes(data: Union[bytes, bytearray, memoryview]) -> "Blob":
        return Blob(data=bytes(data))

    @staticmethod
    def from_uri(uri: str) -> "Blob":
        if uri == "":
            raise ValueError("Blob uri cannot be empty")
        return Blob(uri=uri)

    @staticmethod
    def empty() -> "Blob":
        return Blob(data=b"")


class BlobType(pa.ExtensionType):
    """
    A PyArrow extension type for Lance blob columns.

    This is the "logical" type users write. Lance will store it in a compact
    descriptor format, and reads will return descriptors by default.
    """

    def __init__(self) -> None:
        storage_type = pa.struct(
            [
                pa.field("data", pa.large_binary(), nullable=True),
                pa.field("uri", pa.utf8(), nullable=True),
            ]
        )
        pa.ExtensionType.__init__(self, storage_type, "lance.blob.v2")

    def __arrow_ext_serialize__(self) -> bytes:
        return b""

    @classmethod
    def __arrow_ext_deserialize__(
        cls, storage_type: pa.DataType, serialized: bytes
    ) -> "BlobType":
        return BlobType()

    def __arrow_ext_class__(self):
        return BlobArray

    def __reduce__(self):
        # Workaround to ensure pickle works in earlier versions of PyArrow
        # https://github.com/apache/arrow/issues/35599
        return type(self).__arrow_ext_deserialize__, (
            self.storage_type,
            self.__arrow_ext_serialize__(),
        )


try:
    pa.register_extension_type(BlobType())
except pa.ArrowKeyError:
    # Already registered in this interpreter.
    pass


class BlobArray(pa.ExtensionArray):
    """
    A PyArrow extension array for Lance blob columns.

    Construct with :meth:`from_pylist` or use :func:`blob_array`.
    """

    @classmethod
    def from_pylist(cls, values: list[Any]) -> "BlobArray":
        data_values: list[Optional[bytes]] = []
        uri_values: list[Optional[str]] = []
        null_mask: list[bool] = []

        for v in values:
            if v is None:
                data_values.append(None)
                uri_values.append(None)
                null_mask.append(True)
                continue

            if isinstance(v, Blob):
                data_values.append(v.data)
                uri_values.append(v.uri)
                null_mask.append(False)
                continue

            if isinstance(v, str):
                if v == "":
                    raise ValueError("Blob uri cannot be empty")
                data_values.append(None)
                uri_values.append(v)
                null_mask.append(False)
                continue

            if isinstance(v, (bytes, bytearray, memoryview)):
                data_values.append(bytes(v))
                uri_values.append(None)
                null_mask.append(False)
                continue

            raise TypeError(
                "BlobArray values must be bytes-like, str (URI), Blob, or None; "
                f"got {type(v)}"
            )

        data_arr = pa.array(data_values, type=pa.large_binary())
        uri_arr = pa.array(uri_values, type=pa.utf8())
        mask_arr = pa.array(null_mask, type=pa.bool_())
        storage = pa.StructArray.from_arrays(
            [data_arr, uri_arr], names=["data", "uri"], mask=mask_arr
        )
        return pa.ExtensionArray.from_storage(BlobType(), storage)  # type: ignore[return-value]


def blob_array(values: list[Any]) -> BlobArray:
    """
    Construct a blob array from Python values.

    Each value must be one of:
    - bytes-like: inline bytes
    - str: an external URI
    - Blob: explicit inline/uri/empty
    - None: null
    """

    return BlobArray.from_pylist(values)


def blob_field(name: str, *, nullable: bool = True) -> pa.Field:
    """Construct an Arrow field for a Lance blob column."""
    return pa.field(name, BlobType(), nullable=nullable)


class BlobIterator:
    def __init__(self, binary_iter: Iterator[pa.BinaryScalar]):
        self.binary_iter = binary_iter

    def __next__(self) -> Optional[IO[bytes]]:
        value = next(self.binary_iter)
        if value is None:
            return None
        return io.BytesIO(value.as_py())


class BlobColumn:
    """
    A utility to wrap a Pyarrow binary column and iterate over the rows as
    file-like objects.

    This can be useful for working with medium-to-small binary objects that need
    to interface with APIs that expect file-like objects.  For very large binary
    objects (4-8MB or more per value) you might be better off creating a blob column
    and using :py:meth:`lance.Dataset.take_blobs` to access the blob data.
    """

    def __init__(self, blob_column: Union[pa.Array, pa.ChunkedArray]):
        if not isinstance(blob_column, (pa.Array, pa.ChunkedArray)):
            raise ValueError(
                "Expected a pyarrow.Array or pyarrow.ChunkedArray, "
                f"got {type(blob_column)}"
            )

        if not pa.types.is_large_binary(blob_column.type) and not pa.types.is_binary(
            blob_column.type
        ):
            raise ValueError(f"Expected a binary array, got {blob_column.type}")

        self.blob_column = blob_column

    def __iter__(self) -> Iterator[IO[bytes]]:
        return BlobIterator(iter(self.blob_column))


class BlobFile(io.RawIOBase):
    """Represents a blob in a Lance dataset as a file-like object."""

    def __init__(self, inner: LanceBlobFile):
        """
        Internal only:  To obtain a BlobFile use
        :py:meth:`lance.dataset.Dataset.take_blobs`.
        """
        self.inner = inner

    ## Note: most methods undocumented since they are defined by
    ## the base class.
    def close(self) -> None:
        self.inner.close()

    @property
    def closed(self) -> bool:
        return self.inner.is_closed()

    def readable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.inner.seek(offset)
        elif whence == io.SEEK_CUR:
            self.inner.seek(self.inner.tell() + offset)
        elif whence == io.SEEK_END:
            self.inner.seek(self.inner.size() + offset)
        else:
            raise ValueError(f"Invalid whence: {whence}")

        return self.inner.tell()

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.inner.tell()

    def size(self) -> int:
        """
        Returns the size of the blob in bytes.
        """
        return self.inner.size()

    def readall(self) -> bytes:
        return self.inner.readall()

    def readinto(self, b: bytearray) -> int:
        return self.inner.read_into(b)

    def __repr__(self) -> str:
        return f"<BlobFile size={self.size()}>"
