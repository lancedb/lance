# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import io
from typing import IO, Iterator, Optional, Union

import pyarrow as pa


class BlobIterator:
    def __init__(self, binary_iter: Iterator[pa.BinaryScalar]):
        self.binary_iter = binary_iter

    def __next__(self) -> Optional[IO[bytes]]:
        value = next(self.binary_iter)
        if value is None:
            return None
        return io.BytesIO(value.as_py())


class BlobColumn:
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
