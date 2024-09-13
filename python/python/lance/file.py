# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from typing import Optional, Union

import pyarrow as pa

from .lance import (
    LanceBufferDescriptor,
    LanceColumnMetadata,
    LanceFileMetadata,
    LancePageMetadata,
)
from .lance import (
    LanceFileReader as _LanceFileReader,
)
from .lance import (
    LanceFileWriter as _LanceFileWriter,
)


class ReaderResults:
    """
    Utility class for converting results from Lance's internal
    format (RecordBatchReader) to a desired format such
    as a pyarrow Table, etc.
    """

    def __init__(self, reader: pa.RecordBatchReader):
        """
        Creates a new instance, not meant for external use
        """
        self.reader = reader

    def to_batches(self) -> pa.RecordBatchReader:
        """
        Return the results as a pyarrow RecordBatchReader
        """
        return self.reader

    def to_table(self) -> pa.Table:
        """
        Return the results as a pyarrow Table
        """
        return self.reader.read_all()


class LanceFileReader:
    """
    A file reader for reading Lance files

    This class is used to read Lance data files, a low level structure
    optimized for storing multi-modal tabular data.  If you are working with
    Lance datasets then you should use the LanceDataset class instead.
    """

    # TODO: make schema optional
    def __init__(self, path: str):
        """
        Creates a new file reader to read the given file

        Parameters
        ----------

        path: str
            The path to read, can be a pathname for local storage
            or a URI to read from cloud storage.
        """
        self._reader = _LanceFileReader(path)

    def read_all(self, *, batch_size: int = 1024, batch_readahead=16) -> ReaderResults:
        """
        Reads the entire file

        Parameters
        ----------
        batch_size: int, default 1024
            The file will be read in batches.  This parameter controls
            how many rows will be in each batch (except the final batch)

            Smaller batches will use less memory but might be slightly
            slower because there is more per-batch overhead
        """
        return ReaderResults(self._reader.read_all(batch_size, batch_readahead))

    def read_range(
        self, start: int, num_rows: int, *, batch_size: int = 1024, batch_readahead=16
    ) -> ReaderResults:
        """
        Read a range of rows from the file

        Parameters
        ----------
        start: int
            The offset of the first row to start reading
        num_rows: int
            The number of rows to read from the file
        batch_size: int, default 1024
            The file will be read in batches.  This parameter controls
            how many rows will be in each batch (except the final batch)

            Smaller batches will use less memory but might be slightly
            slower because there is more per-batch overhead
        """
        return ReaderResults(
            self._reader.read_range(start, num_rows, batch_size, batch_readahead)
        )

    def take_rows(
        self, indices, *, batch_size: int = 1024, batch_readahead=16
    ) -> ReaderResults:
        """
        Read a specific set of rows from the file

        Parameters
        ----------
        indices: List[int]
            The indices of the rows to read from the file in ascending order
        batch_size: int, default 1024
            The file will be read in batches.  This parameter controls
            how many rows will be in each batch (except the final batch)

            Smaller batches will use less memory but might be slightly
            slower because there is more per-batch overhead
        """
        for i in range(len(indices) - 1):
            if indices[i] > indices[i + 1]:
                raise ValueError(
                    f"Indices must be sorted in ascending order for \
                                 file API, got {indices[i]} > {indices[i+1]}"
                )

        return ReaderResults(
            self._reader.take_rows(indices, batch_size, batch_readahead)
        )

    def metadata(self) -> LanceFileMetadata:
        """
        Return metadata describing the file contents
        """
        return self._reader.metadata()

    def read_global_buffer(self, index: int) -> bytes:
        """
        Read a global buffer from the file at a given index

        Parameters
        ----------
        index: int
            The index of the global buffer to read

        Returns
        -------
        bytes
            The contents of the global buffer
        """
        return self._reader.read_global_buffer(index)


class LanceFileWriter:
    """
    A file writer for writing Lance data files

    This class is used to write Lance data files, a low level structure
    optimized for storing multi-modal tabular data.  If you are working with
    Lance datasets then you should use the LanceDataset class instead.
    """

    def __init__(
        self,
        path: str,
        schema: Optional[pa.Schema] = None,
        *,
        data_cache_bytes: Optional[int] = None,
        version: Optional[str] = None,
        **kwargs,
    ):
        """
        Create a new LanceFileWriter to write to the given path

        Parameters
        ----------
        path: str
            The path to write to.  Can be a pathname for local storage
            or a URI for remote storage.
        schema: pa.Schema
            The schema of data that will be written.  If not specified then
            the schema will be inferred from the first batch.  If the schema
            is not specified and no data is written then the write will fail.
        data_cache_bytes: int
            How many bytes (per column) to cache before writing a page.  The
            default is an appropriate value based on the filesystem.
        version: str
            The version of the file format to write.  If not specified then
            the latest stable version will be used.  Newer versions are more
            efficient but may not be readable by older versions of the software.
        """
        self._writer = _LanceFileWriter(
            path, schema, data_cache_bytes=data_cache_bytes, version=version, **kwargs
        )
        self.closed = False

    def write_batch(self, batch: Union[pa.RecordBatch, pa.Table]) -> None:
        """
        Write a batch of data to the file

        parameters
        ----------
        batch: Union[pa.RecordBatch, pa.Table]
            The data to write to the file
        """
        if isinstance(batch, pa.Table):
            for batch in batch.to_batches():
                self._writer.write_batch(batch)
        else:
            self._writer.write_batch(batch)

    def close(self) -> int:
        """
        Write the file metadata and close the file

        Returns the number of rows written to the file
        """
        if self.closed:
            return
        self.closed = True
        return self._writer.finish()

    def add_schema_metadata(self, key: str, value: str) -> None:
        """
        Add a metadata (key/value pair) entry to the schema. This method allows you to
        alter the schema metadata. It must be called before `close` is called.

        Parameters
        ----------
        key: str
            The key to add.
        value: str
            The value to add.
        """
        self._writer.add_schema_metadata(key, value)

    def add_global_buffer(self, data: bytes) -> int:
        """
        Add a global buffer to the file. The global buffer can contain any
        arbitrary bytes.

        Parameters
        ----------
        data: bytes
            The data to write to the file.

        Returns
        -------
        int
            The index of the global buffer. This will always start at 1
            and increment by 1 each time this method is called.
        """
        return self._writer.add_global_buffer(data)

    def __enter__(self) -> "LanceFileWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


__all__ = [
    "LanceFileReader",
    "LanceFileWriter",
    "LanceFileMetadata",
    "LanceColumnMetadata",
    "LancePageMetadata",
    "LanceBufferDescriptor",
]
