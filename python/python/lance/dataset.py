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
#

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset
from pyarrow import RecordBatch, Schema
from pyarrow._compute import Expression

from .lance import _Dataset, _Scanner, _write_dataset


class LanceDataset(pa.dataset.Dataset):
    """A dataset in Lance format where the data is stored at the given uri"""

    def __init__(self, uri: Union[str, Path], version: Optional[int] = None):
        if isinstance(uri, Path):
            uri = str(uri.absolute())
        self._uri = uri
        self._ds = _Dataset(uri, version)

    @property
    def uri(self) -> str:
        """
        The location of the data
        """
        return self._uri

    def scanner(
        self,
        columns: Optional[list[str]] = None,
        limit: int = 0,
        offset: Optional[int] = None,
        nearest: Optional[dict] = None,
    ) -> LanceScanner:
        """
        Return a Scanner that can support various pushdowns

        Parameters
        ----------
        columns: list of str, default None
            List of column names to be fetched.
            All columns if None or unspecified.
        limit: int, default 0
            Fetch up to this many rows. All rows if 0 or unspecified.
        offset: int, default None
            Fetch starting with this row. 0 if None or unspecified.
        nearest: dict, default None
            Get the rows corresponding to the K most similar vectors
            nearest should look like {
              "column": <embedding col name>,
              "q": <query vector as pa.Float32Array>,
              "k": 10,
              "nprobes": 1,
              "refine_factor": 1
            }
        """
        return (
            ScannerBuilder(self)
            .columns(columns)
            .limit(limit)
            .offset(offset)
            .nearest(**(nearest or {}))
            .to_scanner()
        )

    @property
    def schema(self) -> pa.Schema:
        """
        The pyarrow Schema for this dataset
        """
        return self._ds.schema

    def to_table(
        self,
        columns: Optional[list[str]] = None,
        limit: int = 0,
        offset: Optional[int] = None,
        nearest: Optional[dict] = None,
    ) -> pa.Table:
        """Read the data into memory and return a pyarrow Table.

        Parameters
        ----------
        columns: list of str, default None
            List of column names to be fetched.
            All columns if None or unspecified.
        limit: int, default 0
            Fetch up to this many rows. All rows if 0 or unspecified.
        offset: int, default None
            Fetch starting with this row. 0 if None or unspecified.
        nearest: dict, default None
            Get the rows corresponding to the K most similar vectors
            nearest should look like {
              "column": <embedding col name>,
              "q": <query vector as pa.Float32Array>,
              "k": 10,
              "nprobes": 1,
              "refine_factor": 1
            }

        """
        return self.scanner(
            columns=columns, limit=limit, offset=offset, nearest=nearest
        ).to_table()

    @property
    def partition_expression(self):
        """
        An Expression which evaluates to true for all data viewed by this
        Dataset.
        """
        raise NotImplementedError("partitioning not yet supported")

    def replace_schema(self, schema: Schema):
        """
        Return a copy of this Dataset with a different schema.

        The copy will view the same Fragments. If the new schema is not
        compatible with the original dataset's schema then an error will
        be raised.

        Parameters
        ----------
        schema : Schema
            The new dataset schema.
        """
        raise NotImplementedError("not changing schemas yet")

    def get_fragments(self, filter: Expression = None):
        """Returns an iterator over the fragments in this dataset.

        Parameters
        ----------
        filter : Expression, default None
            Return fragments matching the optional filter, either using the
            partition_expression or internal information like Parquet's
            statistics.

        Returns
        -------
        fragments : iterator of Fragment
        """
        raise NotImplementedError("Rust Fragments not yet exposed")

    def to_batches(self, **kwargs):
        """
        Read the dataset as materialized record batches.

        Parameters
        ----------
        **kwargs : dict, optional
            Arguments for `Scanner.from_dataset`.

        Returns
        -------
        record_batches : iterator of RecordBatch
        """
        return self.scanner(**kwargs).to_batches()

    def take(self, indices, **kwargs):
        """
        Select rows of data by index.

        Parameters
        ----------
        indices : Array or array-like
            indices of rows to select in the dataset.
        **kwargs : dict, optional
            See scanner() method for full parameter description.

        Returns
        -------
        table : Table
        """
        # TODO expose take from Rust
        # kwargs['take'] = indices
        return self.scanner(**kwargs).to_table().take(indices)

    def head(self, num_rows, **kwargs):
        """
        Load the first N rows of the dataset.

        Parameters
        ----------
        num_rows : int
            The number of rows to load.
        **kwargs : dict, optional
            See scanner() method for full parameter description.

        Returns
        -------
        table : Table
        """
        kwargs["limit"] = num_rows
        return self.scanner(**kwargs).to_table()

    def count_rows(self, **kwargs):
        """Count rows matching the scanner filter.

        Parameters
        ----------
        **kwargs : dict, optional
            See scanner() method for full parameter description.

        Returns
        -------
        count : int

        """
        return self._ds.count_rows()

    def join(
        self,
        right_dataset,
        keys,
        right_keys=None,
        join_type="left outer",
        left_suffix=None,
        right_suffix=None,
        coalesce_keys=True,
        use_threads=True,
    ):
        """
        Perform a join between this dataset and another one.

        Result of the join will be a new dataset, where further
        operations can be applied.

        Parameters
        ----------
        right_dataset : dataset
            The dataset to join to the current one, acting as the right dataset
            in the join operation.
        keys : str or list[str]
            The columns from current dataset that should be used as keys
            of the join operation left side.
        right_keys : str or list[str], default None
            The columns from the right_dataset that should be used as keys
            on the join operation right side.
            When ``None`` use the same key names as the left dataset.
        join_type : str, default "left outer"
            The kind of join that should be performed, one of
            ("left semi", "right semi", "left anti", "right anti",
            "inner", "left outer", "right outer", "full outer")
        left_suffix : str, default None
            Which suffix to add to right column names. This prevents confusion
            when the columns in left and right datasets have colliding names.
        right_suffix : str, default None
            Which suffic to add to the left column names. This prevents confusion
            when the columns in left and right datasets have colliding names.
        coalesce_keys : bool, default True
            If the duplicated keys should be omitted from one of the sides
            in the join result.
        use_threads : bool, default True
            Whenever to use multithreading or not.

        Returns
        -------
        InMemoryDataset
        """
        raise NotImplementedError("Versioning not yet supported in Rust")

    def versions(self):
        """
        Return all versions in this dataset
        """
        versions = self._ds.versions()
        for v in versions:
            v["timestamp"] = datetime.fromtimestamp(v["timestamp"])
        return versions

    def create_index(
        self, column: str, index_type: str, name: Optional[str] = None, **kwargs
    ):
        """Create index on column

        ***Experimental API***

        Parameters
        ----------
        column : str
            The column to be indexed.
        index_type : str
            The type of the index. Only "``IVF_PQ``" is supported now.
        name : str, optional
            The index name. If not provided, it will be generated from the
            column name.
        kwargs :
            Parameters passed to the index building process.


        Accepted keyword parameters:

        - **num_partitions**: the number of partitions of IVF (Inverted File Index).
        - **num_sub_vectors**: the number of sub-vectors used in Product Quantization.

        Examples
        --------

        .. code-block:: python

            import lance

            dataset = lance.dataset("/tmp/sift.lance")
            dataset.create_index(
                "vector",
                "IVF_PQ",
                num_partitions=256,
                num_sub_vectors=16
            )

        References
        ----------
        * `Faiss Index <https://github.com/facebookresearch/faiss/wiki/Faiss-indexes>`_
        * IVF introduced in `Video Google: a text retrieval approach to object matching in videos <https://ieeexplore.ieee.org/abstract/document/1238663>`_
        * `Product quantization for nearest neighbor search <https://hal.inria.fr/inria-00514462v2/document>`_

        """
        # Only support building index for 1 column from the API aspect, however
        # the internal implementation might support building multi-column index later.
        if isinstance(column, str):
            column = [column]
        self._ds.create_index(column, index_type, name, kwargs)


class ScannerBuilder:
    def __init__(self, ds: LanceDataset):
        self.ds = ds
        self._limit = 0
        self._offset = None
        self._columns = None
        self._nearest = None

    def limit(self, n: int = 0) -> ScannerBuilder:
        if int(n) < 0:
            raise ValueError("Limit must be non-negative")
        self._limit = n
        return self

    def offset(self, n: Optional[int] = None) -> ScannerBuilder:
        if n is not None and int(n) < 0:
            raise ValueError("Offset must be non-negative")
        self._offset = n
        return self

    def columns(self, cols: Optional[list[str]] = None) -> ScannerBuilder:
        if cols is not None and len(cols) == 0:
            cols = None
        self._columns = cols
        return self

    def nearest(
        self,
        column: Optional[str] = None,
        q: Optional[pa.FloatingPointArray] = None,
        k: Optional[int] = None,
        nprobes: Optional[int] = None,
        refine_factor: Optional[int] = None,
    ) -> ScannerBuilder:
        if column is None or q is None:
            self._nearest = None
            return self

        if self.ds.schema.get_field_index(column) < 0:
            raise ValueError(f"Embedding column {column} not in dataset")
        if isinstance(q, (np.ndarray, list, tuple)):
            q = pa.FloatingPointArray.from_pandas(q, type=pa.float32())
        if k is not None and int(k) <= 0:
            raise ValueError(f"Nearest-K must be > 0 but got {k}")
        if nprobes is not None and int(nprobes) <= 0:
            raise ValueError(f"Nearest-K must be > 0 but got {nprobes}")
        if refine_factor is not None and int(refine_factor) < 1:
            raise ValueError(f"Refine factor must be 1 or more got {refine_factor}")
        self._nearest = {
            "column": column,
            "q": q,
            "k": k,
            "nprobes": nprobes,
            "refine_factor": refine_factor,
        }
        return self

    def to_scanner(self) -> LanceScanner:
        scanner = self.ds._ds.scanner(
            self._columns, self._limit, self._offset, self._nearest
        )
        return LanceScanner(scanner, self.ds)


class LanceScanner(pa.dataset.Scanner):
    def __init__(self, scanner: _Scanner, dataset: LanceDataset):
        self._scanner = scanner
        self._ds = dataset

    def to_table(self) -> pa.Table:
        """
        Read the data into memory and return a pyarrow Table.
        """
        return self.to_reader().read_all()

    def to_reader(self) -> pa.RecordBatchReader:
        return self._scanner.to_pyarrow()

    def to_batches(self) -> Iterator[RecordBatch]:
        def _iterator(batch_iter):
            for batch in batch_iter:
                yield batch.record_batch

        # Don't make ourselves a generator so errors are raised immediately
        return _iterator(self.to_reader())

    @property
    def projected_schema(self) -> Schema:
        return self._scanner.schema

    @staticmethod
    def from_dataset(*args, **kwargs):
        """
        Create Scanner from Dataset,

        Parameters
        ----------
        dataset : Dataset
            Dataset to scan.
        columns : list of str, default None
            The columns to project. This can be a list of column names to
            include (order and duplicates will be preserved), or a dictionary
            with {new_column_name: expression} values for more advanced
            projections.

            The list of columns or expressions may use the special fields
            `__batch_index` (the index of the batch within the fragment),
            `__fragment_index` (the index of the fragment within the dataset),
            `__last_in_fragment` (whether the batch is last in fragment), and
            `__filename` (the name of the source file or a description of the
            source fragment).

            The columns will be passed down to Datasets and corresponding data
            fragments to avoid loading, copying, and deserializing columns
            that will not be required further down the compute chain.
            By default all of the available columns are projected. Raises
            an exception if any of the referenced column names does not exist
            in the dataset's Schema.
        filter : Expression, default None
            Scan will return only the rows matching the filter.
            If possible the predicate will be pushed down to exploit the
            partition information or internal metadata found in the data
            source, e.g. Parquet statistics. Otherwise filters the loaded
            RecordBatches before yielding them.
        batch_size : int, default 128Ki
            The maximum row count for scanned record batches. If scanned
            record batches are overflowing memory then this method can be
            called to reduce their size.
        batch_readahead : int, default 16
            The number of batches to read ahead in a file. This might not work
            for all file formats. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_readahead : int, default 4
            The number of files to read ahead. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        use_threads : bool, default True
            If enabled, then maximum parallelism will be used determined by
            the number of available CPU cores.
        use_async : bool, default True
            This flag is deprecated and is being kept for this release for
            backwards compatibility.  It will be removed in the next
            release.
        memory_pool : MemoryPool, default None
            For memory allocations, if required. If not specified, uses the
            default pool.
        fragment_scan_options : FragmentScanOptions, default None
            Options specific to a particular scan and fragment type, which
            can change between different scans of the same dataset.
        """
        raise NotImplementedError("from dataset")

    @staticmethod
    def from_fragment(*args, **kwargs):
        """
        Create Scanner from Fragment,

        Parameters
        ----------
        fragment : Fragment
            fragment to scan.
        schema : Schema, optional
            The schema of the fragment.
        columns : list of str, default None
            The columns to project. This can be a list of column names to
            include (order and duplicates will be preserved), or a dictionary
            with {new_column_name: expression} values for more advanced
            projections.

            The list of columns or expressions may use the special fields
            `__batch_index` (the index of the batch within the fragment),
            `__fragment_index` (the index of the fragment within the dataset),
            `__last_in_fragment` (whether the batch is last in fragment), and
            `__filename` (the name of the source file or a description of the
            source fragment).

            The columns will be passed down to Datasets and corresponding data
            fragments to avoid loading, copying, and deserializing columns
            that will not be required further down the compute chain.
            By default all of the available columns are projected. Raises
            an exception if any of the referenced column names does not exist
            in the dataset's Schema.
        filter : Expression, default None
            Scan will return only the rows matching the filter.
            If possible the predicate will be pushed down to exploit the
            partition information or internal metadata found in the data
            source, e.g. Parquet statistics. Otherwise filters the loaded
            RecordBatches before yielding them.
        batch_size : int, default 128Ki
            The maximum row count for scanned record batches. If scanned
            record batches are overflowing memory then this method can be
            called to reduce their size.
        batch_readahead : int, default 16
            The number of batches to read ahead in a file. This might not work
            for all file formats. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        use_threads : bool, default True
            If enabled, then maximum parallelism will be used determined by
            the number of available CPU cores.
        use_async : bool, default True
            This flag is deprecated and is being kept for this release for
            backwards compatibility.  It will be removed in the next
            release.
        memory_pool : MemoryPool, default None
            For memory allocations, if required. If not specified, uses the
            default pool.
        fragment_scan_options : FragmentScanOptions, default None
            Options specific to a particular scan and fragment type, which
            can change between different scans of the same dataset.
        """
        raise NotImplementedError("from fragment")

    @staticmethod
    def from_batches(*args, **kwargs):
        """
        Create a Scanner from an iterator of batches.

        This creates a scanner which can be used only once. It is
        intended to support writing a dataset (which takes a scanner)
        from a source which can be read only once (e.g. a
        RecordBatchReader or generator).

        Parameters
        ----------
        source : Iterator
            The iterator of Batches.
        schema : Schema
            The schema of the batches.
        columns : list of str or dict, default None
                The columns to project.
        filter : Expression, default None
            Scan will return only the rows matching the filter.
        batch_size : int, default 128Ki
            The maximum row count for scanned record batches.
        use_threads : bool, default True
            If enabled, then maximum parallelism will be used determined by
            the number of available CPU cores.
        use_async : bool, default True
            This flag is deprecated and is being kept for this release for
            backwards compatibility.  It will be removed in the next
            release.
        memory_pool : MemoryPool, default None
            For memory allocations, if required. If not specified, uses the
            default pool.
        fragment_scan_options : FragmentScanOptions
            The fragment scan options.
        """
        raise NotImplementedError("from batches")

    @property
    def dataset_schema(self):
        """The schema with which batches will be read from fragments."""
        raise NotImplementedError("")

    def scan_batches(self):
        """
        Consume a Scanner in record batches with corresponding fragments.

        Returns
        -------
        record_batches : iterator of TaggedRecordBatch
        """
        lst = []
        reader = self.to_reader()
        while True:
            batch = reader.read_next_batch()
            if batch is None:
                reader.close()
                break
            lst.append(batch)
        return lst

    def take(self, indices):
        """
        Select rows of data by index.

        Will only consume as many batches of the underlying dataset as
        needed. Otherwise, this is equivalent to
        ``to_table().take(indices)``.

        Parameters
        ----------
        indices : Array or array-like
            indices of rows to select in the dataset.

        Returns
        -------
        Table
        """
        raise NotImplementedError("take")

    def head(self, num_rows):
        """
        Load the first N rows of the dataset.

        Parameters
        ----------
        num_rows : int
            The number of rows to load.

        Returns
        -------
        Table
        """
        return self.to_table()[:num_rows]

    def count_rows(self):
        """Count rows matching the scanner filter.

        Returns
        -------
        count : int

        """
        return self._ds.count_rows()


ReaderLike = Union[
    pa.Table,
    pa.dataset.Dataset,
    pa.dataset.Scanner,
    pa.RecordBatchReader,
    LanceDataset,
    LanceScanner,
]


def write_dataset(
    data_obj: ReaderLike,
    uri: Union[str, Path],
    mode: str = "create",
    max_rows_per_file: int = 1024 * 1024,
    max_rows_per_group: int = 1024,
) -> LanceDataset:
    """Write a given data_obj to the given uri

    Parameters
    ----------
    data_obj: Reader-like
        The data to be written. Acceptable types are:
        - Pyarrow Table, Dataset, Scanner, or RecordBatchReader
        - LanceDataset or LanceScanner
    uri: str or Path
        Where to write the dataset to (directory)
    mode: str
        **create** - create a new dataset (raises if uri already exists).

        **overwrite** - create a new snapshot version

        **append** - create a new version that is the concat of the input the
        latest version (raises if uri does not exist)
    max_rows_per_file: int, default 1024 * 1024
        The max number of rows to write before starting a new file
    max_rows_per_group: int, default 1024
        The max number of rows before starting a new group (in the same file)

    """
    if isinstance(data_obj, pa.Table):
        reader = data_obj.to_reader()
    elif isinstance(data_obj, pa.dataset.Dataset):
        reader = pa.dataset.Scanner.from_dataset(data_obj).to_reader()
    elif isinstance(data_obj, pa.dataset.Scanner):
        reader = data_obj.to_reader()
    elif isinstance(data_obj, pa.RecordBatchReader):
        reader = data_obj
    else:
        raise TypeError(f"Unknown data_obj type {type(data_obj)}")
    # TODO add support for passing in LanceDataset and LanceScanner here

    params = {
        "mode": mode,
        "max_rows_per_file": max_rows_per_file,
        "max_rows_per_group": max_rows_per_group,
    }
    if isinstance(uri, Path):
        uri = str(uri.absolute())
    _write_dataset(reader, str(uri), params)
    return LanceDataset(str(uri))
