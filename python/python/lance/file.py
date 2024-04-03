#  Copyright (c) 2023. Lance Developers
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
from typing import Union

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
    def __init__(self, path: str, schema: pa.Schema):
        """
        Creates a new file reader to read the given file

        Parameters
        ----------

        path: str
            The path to read, can be a pathname for local storage
            or a URI to read from cloud storage.
        schema: pa.Schema
            The desired projection schema
        """
        self._reader = _LanceFileReader(path, schema)

    def read_all(self, *, batch_size: int = 1024) -> ReaderResults:
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
        return ReaderResults(self._reader.read_all(batch_size))

    def read_range(
        self, start: int, num_rows: int, *, batch_size: int = 1024
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
        return ReaderResults(self._reader.read_range(start, num_rows, batch_size))

    def metadata(self) -> LanceFileMetadata:
        """
        Return metadata describing the file contents
        """
        return self._reader.metadata()


class LanceFileWriter:
    """
    A file writer for writing Lance data files

    This class is used to write Lance data files, a low level structure
    optimized for storing multi-modal tabular data.  If you are working with
    Lance datasets then you should use the LanceDataset class instead.
    """

    def __init__(self, path: str, schema: pa.Schema, **kwargs):
        """
        Create a new LanceFileWriter to write to the given path

        Parameters
        ----------
        path: str
            The path to write to.  Can be a pathname for local storage
            or a URI for remote storage.
        schema: pa.Schema
            The schema of data that will be written
        """
        self._writer = _LanceFileWriter(path, schema, **kwargs)
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

    def close(self) -> None:
        """
        Write the file metadata and close the file
        """
        if self.closed:
            return
        self.closed = True
        self._writer.finish()

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
