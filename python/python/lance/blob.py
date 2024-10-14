# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import io
from typing import IO, Iterator, Optional, Union

import pyarrow as pa

from lance.lance import LanceBlobFile


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
    and using :ref:`lance.Dataset.take_blobs` to access the blob data.
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
    """
    Represents a blob in a Lance dataset as a file-like object.
    """

    def __init__(self, inner: LanceBlobFile):
        """
        Internal only:  To obtain a BlobFile use :ref:`lance.Dataset.take_blobs`.
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
