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

import os
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

import numpy as np
import pyarrow as pa
import pyarrow.dataset
from pyarrow import RecordBatch, Schema
from pyarrow._compute import Expression

from .commit import CommitLock
from .fragment import FragmentMetadata, LanceFragment
from .lance import __version__ as __version__
from .lance import _Dataset, _Scanner, _write_dataset

try:
    import pandas as pd

    ReaderLike = Union[
        pd.Timestamp,
        pa.Table,
        pa.dataset.Dataset,
        pa.dataset.Scanner,
        Iterable[RecordBatch],
        pa.RecordBatchReader,
    ]
except ImportError:
    pd = None
    ReaderLike = Union[
        pa.Table,
        pa.dataset.Dataset,
        pa.dataset.Scanner,
        Iterable[RecordBatch],
        pa.RecordBatchReader,
    ]


class LanceDataset(pa.dataset.Dataset):
    """A dataset in Lance format where the data is stored at the given uri."""

    def __init__(
        self,
        uri: Union[str, Path],
        version: Optional[int] = None,
        block_size: Optional[int] = None,
        index_cache_size: Optional[int] = None,
        metadata_cache_size: Optional[int] = None,
    ):
        uri = os.fspath(uri) if isinstance(uri, Path) else uri
        self._uri = uri
        self._ds = _Dataset(
            uri, version, block_size, index_cache_size, metadata_cache_size
        )

    def __reduce__(self):
        return LanceDataset, (self.uri, self._ds.version())

    def __getstate__(self):
        return self.uri, self._ds.version()

    def __setstate__(self, state):
        self._uri, version = state
        self._ds = _Dataset(self._uri, version)

    @property
    def uri(self) -> str:
        """
        The location of the data
        """
        return self._uri

    @lru_cache(maxsize=None)
    def list_indices(self) -> List[Dict[str, Any]]:
        return self._ds.load_indices()

    @property
    def has_index(self):
        return len(self.list_indices()) > 0

    def scanner(
        self,
        columns: Optional[list[str]] = None,
        filter: Optional[Union[str, pa.compute.Expression]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        nearest: Optional[dict] = None,
        batch_size: Optional[int] = None,
        batch_readahead: Optional[int] = None,
        fragment_readahead: Optional[int] = None,
        scan_in_order: bool = True,
        fragments: Optional[Iterable[LanceFragment]] = None,
    ) -> LanceScanner:
        """Return a Scanner that can support various pushdowns.

        Parameters
        ----------
        columns: list of str, default None
            List of column names to be fetched.
            All columns if None or unspecified.
        filter: pa.compute.Expression or str
            Expression or str that is a valid SQL where clause. See
            `Lance filter pushdown <https://lancedb.github.io/lance/read_and_write.html#filter-push-down>`_
            for valid SQL expressions.
        limit: int, default None
            Fetch up to this many rows. All rows if None or unspecified.
        offset: int, default None
            Fetch starting with this row. 0 if None or unspecified.
        nearest: dict, default None
            Get the rows corresponding to the K most similar vectors. Example:

            .. code-block:: python

                {
                    "column": <embedding col name>,
                    "q": <query vector as pa.Float32Array>,
                    "k": 10,
                    "nprobes": 1,
                    "refine_factor": 1
                }
        batch_size: int, default None
            The number of rows to fetch per batch.
        batch_readahead: int, optional
            The number of batches to read ahead.
        fragment_readahead: int, optional
            The number of fragments to read ahead.
        scan_in_order: bool, default True
            Whether to read the fragments and batches in order. If false,
            throughput may be higher, but batches will be returned out of order
            and memory use might increase.
        fragments: iterable of LanceFragment, default None
            If specified, only scan these fragments. If scan_in_order is True, then
            the fragments will be scanned in the order given.

        Notes
        -----

        For now, if BOTH filter and nearest is specified, then:
        1. nearest is executed first.
        2. The results are filtered afterwards.

        For debugging ANN results, you can choose to not use the index
        even if present by specifying ``use_index=False``. For example,
        the following will always return exact KNN results:

        .. code-block:: python

            dataset.to_table(nearest={
                "column": "vector",
                "k": 10,
                "q": <query vector>,
                "use_index": False
            }

        """
        return (
            ScannerBuilder(self)
            .columns(columns)
            .filter(filter)
            .limit(limit)
            .offset(offset)
            .nearest(**(nearest or {}))
            .batch_size(batch_size)
            .batch_readahead(batch_readahead)
            .fragment_readahead(fragment_readahead)
            .scan_in_order(scan_in_order)
            .with_fragments(fragments)
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
        filter: Optional[Union[str, pa.compute.Expression]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        nearest: Optional[dict] = None,
        batch_size: Optional[int] = None,
        batch_readahead: Optional[int] = None,
        fragment_readahead: Optional[int] = None,
        scan_in_order: bool = True,
    ) -> pa.Table:
        """Read the data into memory as a pyarrow Table.

        Parameters
        ----------
        columns: list of str, default None
            List of column names to be fetched.
            All columns if None or unspecified.
        filter : pa.compute.Expression or str
            Expression or str that is a valid SQL where clause. See
            `Lance filter pushdown <https://lancedb.github.io/lance/read_and_write.html#filter-push-down>`_
            for valid SQL expressions.
        limit: int, default None
            Fetch up to this many rows. All rows if None or unspecified.
        offset: int, default None
            Fetch starting with this row. 0 if None or unspecified.
        nearest: dict, default None
            Get the rows corresponding to the K most similar vectors. Example:

            .. code-block:: python

                {
                    "column": <embedding col name>,
                    "q": <query vector as pa.Float32Array>,
                    "k": 10,
                    "metric": "cosine",
                    "nprobes": 1,
                    "refine_factor": 1
                }

        batch_size: int, optional
            The number of rows to read at a time.
        batch_readahead: int, optional
            The number of batches to read ahead.
        fragment_readahead: int, optional
            The number of fragments to read ahead.
        scan_in_order: bool, default True
            Whether to read the fragments and batches in order. If false,
            throughput may be higher, but batches will be returned out of order
            and memory use might increase.

        Notes
        -----
        For now, if BOTH filter and nearest is specified, then:
        1. nearest is executed first.
        2. The results are filtered afterwards.
        """
        return self.scanner(
            columns=columns,
            filter=filter,
            limit=limit,
            offset=offset,
            nearest=nearest,
            batch_size=batch_size,
            batch_readahead=batch_readahead,
            fragment_readahead=fragment_readahead,
            scan_in_order=scan_in_order,
        ).to_table()

    @property
    def partition_expression(self):
        """
        Not implemented (just override pyarrow dataset to prevent segfault)
        """
        raise NotImplementedError("partitioning not yet supported")

    def replace_schema(self, schema: Schema):
        """
        Not implemented (just override pyarrow dataset to prevent segfault)
        """
        raise NotImplementedError("not changing schemas yet")

    def get_fragments(
        self, filter: Optional[Expression] = None
    ) -> Iterator[pa.dataset.Fragment]:
        """Get all fragments from the dataset.


        Note: filter is not supported yet.
        """
        if filter is not None:
            raise ValueError("get_fragments() does not support filter yet")
        return [
            LanceFragment(self, fragment_id=None, fragment=f)
            for f in self._ds.get_fragments()
        ]

    def get_fragment(self, fragment_id: int) -> Optional[LanceFragment]:
        """Get the fragment with fragment id."""
        raw_fragment = self._ds.get_fragment(fragment_id)
        if raw_fragment is None:
            return None
        return LanceFragment(self, fragment_id=None, fragment=raw_fragment)

    def to_batches(
        self,
        columns: Optional[list[str]] = None,
        filter: Optional[Union[str, pa.compute.Expression]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        nearest: Optional[dict] = None,
        batch_size: Optional[int] = None,
        batch_readahead: Optional[int] = None,
        fragment_readahead: Optional[int] = None,
        scan_in_order: bool = True,
        **kwargs,
    ) -> Iterator[pa.RecordBatch]:
        """Read the dataset as materialized record batches.

        Parameters
        ----------
        **kwargs : dict, optional
            Arguments for ``Scanner.from_dataset``.

        Returns
        -------
        record_batches : Iterator of RecordBatch
        """
        return self.scanner(
            columns=columns,
            filter=filter,
            limit=limit,
            offset=offset,
            nearest=nearest,
            batch_size=batch_size,
            batch_readahead=batch_readahead,
            fragment_readahead=fragment_readahead,
            scan_in_order=scan_in_order,
        ).to_batches()

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
        # kwargs['take'] = indices
        return pa.Table.from_batches([self._ds.take(indices)])

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
            See py:method:`scanner` method for full parameter description.

        Returns
        -------
        count : int
            The total number of rows in the dataset.

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
        Not implemented (just override pyarrow dataset to prevent segfault)
        """
        raise NotImplementedError("Versioning not yet supported in Rust")

    def merge(
        self,
        data_obj: ReaderLike,
        left_on: str,
        right_on: Optional[str] = None,
        schema=None,
    ):
        """
        Merge another dataset into this one.

        Performs a left join, where the dataset is the left side and data_obj
        is the right side. Rows existing in the dataset but not on the left will
        be filled with null values, unless Lance doesn't support null values for
        some types, in which case an error will be raised.

        Parameters
        ----------
        data_obj: Reader-like
            The data to be merged. Acceptable types are:
            - Pandas DataFrame, Pyarrow Table, Dataset, Scanner,
            Iterator[RecordBatch], or RecordBatchReader
        left_on: str
            The name of the column in the dataset to join on.
        right_on: str or None
            The name of the column in data_obj to join on. If None, defaults to
            left_on.

        Examples
        --------

        >>> import lance
        >>> import pyarrow as pa
        >>> df = pa.table({'x': [1, 2, 3], 'y': ['a', 'b', 'c']})
        >>> dataset = lance.write_dataset(df, "dataset")
        >>> dataset.to_table().to_pandas()
           x  y
        0  1  a
        1  2  b
        2  3  c
        >>> new_df = pa.table({'x': [1, 2, 3], 'z': ['d', 'e', 'f']})
        >>> dataset.merge(new_df, 'x')
        >>> dataset.to_table().to_pandas()
           x  y  z
        0  1  a  d
        1  2  b  e
        2  3  c  f
        """
        if right_on is None:
            right_on = left_on

        reader = _coerce_reader(data_obj, schema)

        self._ds.merge(reader, left_on, right_on)

    def delete(self, predicate: Union[str, pa.compute.Expression]):
        """
        Delete rows from the dataset.

        This marks rows as deleted, but does not physically remove them from the
        files. This keeps the existing indexes still valid.

        Parameters
        ----------
        predicate : str or pa.compute.Expression
            The predicate to use to select rows to delete. May either be a SQL
            string or a pyarrow Expression.

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> table = pa.table({"a": [1, 2, 3], "b": ["a", "b", "c"]})
        >>> dataset = lance.write_dataset(table, "example")
        >>> dataset.delete("a = 1 or b in ('a', 'b')")
        >>> dataset.to_table()
        pyarrow.Table
        a: int64
        b: string
        ----
        a: [[3]]
        b: [["c"]]
        """
        if isinstance(predicate, pa.compute.Expression):
            predicate = str(predicate)
        self._ds.delete(predicate)

    def versions(self):
        """
        Return all versions in this dataset.
        """
        versions = self._ds.versions()
        for v in versions:
            # TODO: python datetime supports only microsecond precision. When a
            # separate Version object is implemented, expose the precise timestamp
            # (ns) to python.
            ts_nanos = v["timestamp"]
            v["timestamp"] = datetime.fromtimestamp(ts_nanos // 1e9) + timedelta(
                microseconds=(ts_nanos % 1e9) // 1e3
            )
        return versions

    @property
    def version(self) -> int:
        """
        Returns the currently checked out version of the dataset
        """
        return self._ds.version()

    def create_index(
        self,
        column: str,
        index_type: str,
        name: Optional[str] = None,
        metric: str = "L2",
        replace: bool = False,
        num_partitions: Optional[int] = None,
        ivf_centroids: Optional[Union[np.ndarray, pa.FixedSizeListArray]] = None,
        num_sub_vectors: Optional[int] = None,
        **kwargs,
    ) -> LanceDataset:
        """Create index on column.

        **Experimental API**

        Parameters
        ----------
        column : str
            The column to be indexed.
        index_type : str
            The type of the index. Only ``"IVF_PQ"`` is supported now.
        name : str, optional
            The index name. If not provided, it will be generated from the
            column name.
        metric : str
            The distance metric type, i.e., "L2" (alias to "euclidean"), "cosine"
            or "dot" (dot product). Default is "L2".
        replace : bool
            Replace the existing index if it exists.
        num_partitions : int, optional
            The number of partitions of IVF (Inverted File Index).
        ivf_centroids : ``np.ndarray`` or ``pyarrow.FixedSizeListArray``. Optional.
            A ``num_partitions x dimension`` array of K-mean centroids for IVF
            clustering. If not provided, a new Kmean model will be trained.
        num_sub_vectors : int, optional
            The number of sub-vectors for PQ (Product Quantization).
        kwargs :
            Parameters passed to the index building process.

        If ``index_type`` is "IVF_PQ", then the following parameters are required:
        - **num_partitions**
        - **num_sub_vectors**

        Optional parameters for "IVF_PQ":
        - **use_opq**: whether to use OPQ (Optimized Product Quantization).
            Must have feature 'opq' enabled in Rust.
        - **max_opq_iterations**: the maximum number of iterations for training OPQ.
        - **ivf_centroids**: K-mean centroids for IVF clustering.

        If ``index_type`` is "DISKANN", then the following parameters are optional:

        - **r**: out-degree bound
        - **l**: number of levels in the graph.
        - **alpha**: distance threshold for the graph.

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
        * IVF introduced in `Video Google: a text retrieval approach to object matching
          in videos <https://ieeexplore.ieee.org/abstract/document/1238663>`_
        * `Product quantization for nearest neighbor search
          <https://hal.inria.fr/inria-00514462v2/document>`_

        """
        # Only support building index for 1 column from the API aspect, however
        # the internal implementation might support building multi-column index later.
        if isinstance(column, str):
            column = [column]

        # validate args
        for c in column:
            if c not in self.schema.names:
                raise KeyError(f"{c} not found in schema")
            field = self.schema.field(c)
            if not pa.types.is_fixed_size_list(field.type):
                raise TypeError(
                    f"Vector column {c} must be FixedSizeListArray, got {field.type}"
                )
            if not pa.types.is_float32(field.type.value_type):
                raise TypeError(
                    f"Vector column {c} must have float32 value type, "
                    f"got {field.type.value_type}"
                )

        if not isinstance(metric, str) or metric.lower() not in [
            "l2",
            "cosine",
            "euclidean",
            "dot",
        ]:
            raise ValueError(f"Metric {metric} not supported.")
        index_type = index_type.upper()
        if index_type not in ["IVF_PQ", "DISKANN"]:
            raise NotImplementedError(
                f"Only IVF_PQ or DiskANN index_types supported. Got {index_type}"
            )
        if index_type == "IVF_PQ":
            if num_partitions is None or num_sub_vectors is None:
                raise ValueError(
                    "num_partitions and num_sub_vectors are required for IVF_PQ"
                )
            kwargs["num_partitions"] = num_partitions
            kwargs["num_sub_vectors"] = num_sub_vectors
            if ivf_centroids is not None:
                # User provided IVF centroids
                if isinstance(ivf_centroids, np.ndarray):
                    if (
                        len(ivf_centroids.shape) != 2
                        or ivf_centroids.shape[0] != num_partitions
                    ):
                        raise ValueError(
                            f"Ivf centroids must be 2D array: (clusters, dim), "
                            f"got {ivf_centroids.shape}"
                        )
                    if ivf_centroids.dtype != np.float32:
                        raise TypeError(
                            f"IVF centroids must be float32, got {ivf_centroids.dtype}"
                        )
                    dim = ivf_centroids.shape[1]
                    values = pa.array(ivf_centroids.reshape(-1), type=pa.float32())
                    ivf_centroids = pa.FixedSizeListArray.from_arrays(values, dim)
                # Convert it to RecordBatch because Rust side only accepts RecordBatch.
                ivf_centroids_batch = pa.RecordBatch.from_arrays(
                    [ivf_centroids], ["_ivf_centroids"]
                )
                kwargs["ivf_centroids"] = ivf_centroids_batch

        kwargs["replace"] = replace

        self._ds.create_index(column, index_type, name, metric, kwargs)
        return LanceDataset(self.uri)

    @staticmethod
    def _commit(
        base_uri: Union[str, Path],
        new_schema: pa.Schema,
        fragments: Iterable[FragmentMetadata],
        mode: str = "append",
    ) -> LanceDataset:
        """Create a new version of dataset with collected fragments.

        This method allows users to commit a version of dataset in a distributed
        environment.

        Parameters
        ----------
        new_schema : pa.Schema
            The schema for the new version of dataset.
        fragments : list[FragmentMetadata]
            The fragments to create new version of dataset.

        Returns
        -------
        LanceDataset
            A new version of Lance Dataset.


        Note
        -----
        This method is for internal use only.
        """
        if isinstance(base_uri, Path):
            base_uri = str(base_uri)
        if not isinstance(new_schema, pa.Schema):
            raise TypeError(f"schema must be pyarrow.Schema, got {type(new_schema)}")
        if not isinstance(fragments, list):
            raise TypeError(
                f"fragments must be list[FragmentMetadata], got {type(fragments)}"
            )
        if len(fragments) > 0 and not all(
            isinstance(f, FragmentMetadata) for f in fragments
        ):
            raise TypeError(
                f"fragments must be list[FragmentMetadata], got {type(fragments[0])}"
            )
        raw_fragments = [f._metadata for f in fragments]
        # TODO: make fragments as a generator
        _Dataset.commit(base_uri, new_schema, raw_fragments)
        return LanceDataset(base_uri)


class ScannerBuilder:
    def __init__(self, ds: LanceDataset):
        self.ds = ds
        self._limit = 0
        self._filter = None
        self._offset = None
        self._columns = None
        self._nearest = None
        self._batch_size = None
        self._batch_readahead = None
        self._fragment_readahead = None
        self._scan_in_order = True
        self._fragments = None

    def batch_size(self, batch_size: int) -> ScannerBuilder:
        """Set batch size for Scanner"""
        self._batch_size = batch_size
        return self

    def batch_readahead(self, nbatches: Optional[int] = None) -> ScannerBuilder:
        if nbatches is not None and int(nbatches) < 0:
            raise ValueError("batch_readahead must be non-negative")
        self._batch_readahead = nbatches
        return self

    def fragment_readahead(self, nfragments: Optional[int] = None) -> ScannerBuilder:
        if nfragments is not None and int(nfragments) < 0:
            raise ValueError("fragment_readahead must be non-negative")
        self._fragment_readahead = nfragments
        return self

    def scan_in_order(self, scan_in_order: bool = True) -> ScannerBuilder:
        """
        Whether to scan the dataset in order of fragments and batches.

        If set to False, the scanner may read fragments concurrently and yield
        batches out of order. This may improve performance since it allows more
        concurrency in the scan, but can also use more memory.
        """
        self._scan_in_order = scan_in_order
        return self

    def limit(self, n: Optional[int] = None) -> ScannerBuilder:
        if n is not None and int(n) < 0:
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

    def filter(self, filter: Union[str, pa.compute.Expression]) -> ScannerBuilder:
        if isinstance(filter, pa.compute.Expression):
            filter = str(filter)
        self._filter = filter
        return self

    def with_fragments(
        self, fragments: Optional[Iterable[LanceFragment]]
    ) -> ScannerBuilder:
        if fragments is not None:
            inner_fragments = []
            for f in fragments:
                if isinstance(f, LanceFragment):
                    inner_fragments.append(f._fragment)
                else:
                    raise TypeError(
                        f"fragments must be an iterable of LanceFragment. "
                        f"Got {type(f)} instead."
                    )
            fragments = inner_fragments

        self._fragments = fragments
        return self

    def nearest(
        self,
        column: Optional[str] = None,
        q: Optional[pa.FloatingPointArray] = None,
        k: Optional[int] = None,
        metric: Optional[str] = None,
        nprobes: Optional[int] = None,
        refine_factor: Optional[int] = None,
        use_index: bool = True,
    ) -> ScannerBuilder:
        if column is None or q is None:
            self._nearest = None
            return self

        if self.ds.schema.get_field_index(column) < 0:
            raise ValueError(f"Embedding column {column} not in dataset")
        if isinstance(q, (np.ndarray, list, tuple)):
            q = np.array(q).astype("float64")  # workaround for GH-608
            q = pa.FloatingPointArray.from_pandas(q, type=pa.float32())
        if not isinstance(q, pa.FloatingPointArray):
            raise TypeError("query vector must be list-like or pa.FloatingPointArray")
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
            "metric": metric,
            "nprobes": nprobes,
            "refine_factor": refine_factor,
            "use_index": use_index,
        }
        return self

    def to_scanner(self) -> LanceScanner:
        scanner = self.ds._ds.scanner(
            self._columns,
            self._filter,
            self._limit,
            self._offset,
            self._nearest,
            self._batch_size,
            self._batch_readahead,
            self._fragment_readahead,
            self._scan_in_order,
            self._fragments,
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
        yield from self.to_reader()

    @property
    def projected_schema(self) -> Schema:
        return self._scanner.schema

    @staticmethod
    def from_dataset(*args, **kwargs):
        """
        Not implemented
        """
        raise NotImplementedError("from dataset")

    @staticmethod
    def from_fragment(*args, **kwargs):
        """
        Not implemented
        """
        raise NotImplementedError("from fragment")

    @staticmethod
    def from_batches(*args, **kwargs):
        """
        Not implemented
        """
        raise NotImplementedError("from batches")

    @property
    def dataset_schema(self) -> Schema:
        """The schema with which batches will be read from fragments."""
        return self._ds.schema

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
        Not implemented
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


def write_dataset(
    data_obj: ReaderLike,
    uri: Union[str, Path],
    schema: Optional[pa.Schema] = None,
    mode: str = "create",
    max_rows_per_file: int = 1024 * 1024,
    max_rows_per_group: int = 1024,
    commit_lock: Optional[CommitLock] = None,
) -> LanceDataset:
    """Write a given data_obj to the given uri

    Parameters
    ----------
    data_obj: Reader-like
        The data to be written. Acceptable types are:
        - Pandas DataFrame, Pyarrow Table, Dataset, Scanner, or RecordBatchReader
    uri: str or Path
        Where to write the dataset to (directory)
    schema: Schema, optional
        If specified and the input is a pandas DataFrame, use this schema
        instead of the default pandas to arrow table conversion.
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
    reader = _coerce_reader(data_obj, schema)
    # TODO add support for passing in LanceDataset and LanceScanner here

    params = {
        "mode": mode,
        "max_rows_per_file": max_rows_per_file,
        "max_rows_per_group": max_rows_per_group,
    }

    if commit_lock:
        if not callable(commit_lock):
            raise TypeError(f"commit_lock must be a function, got {type(commit_lock)}")
        params["commit_handler"] = commit_lock

    uri = os.fspath(uri) if isinstance(uri, Path) else uri
    _write_dataset(reader, uri, params)
    return LanceDataset(uri)


def _coerce_reader(
    data_obj: ReaderLike, schema: Optional[pa.Schema] = None
) -> pa.RecordBatchReader:
    if pd and isinstance(data_obj, pd.DataFrame):
        return pa.Table.from_pandas(data_obj, schema=schema).to_reader()
    elif isinstance(data_obj, pa.Table):
        return data_obj.to_reader()
    elif isinstance(data_obj, pa.RecordBatch):
        return pa.Table.from_batches([data_obj]).to_reader()
    elif isinstance(data_obj, pa.dataset.Dataset):
        return pa.dataset.Scanner.from_dataset(data_obj).to_reader()
    elif isinstance(data_obj, pa.dataset.Scanner):
        return data_obj.to_reader()
    elif isinstance(data_obj, pa.RecordBatchReader):
        return data_obj
    elif (
        type(data_obj).__module__.startswith("polars")
        and data_obj.__class__.__name__ == "DataFrame"
    ):
        return data_obj.to_arrow().to_reader()
    # for other iterables, assume they are of type Iterable[RecordBatch]
    elif isinstance(data_obj, Iterable):
        if schema is not None:
            data = _casting_recordbatch_iter(data_obj, schema)
            return pa.RecordBatchReader.from_batches(schema, data)
        else:
            raise ValueError(
                "Must provide schema to write dataset from RecordBatch iterable"
            )
    else:
        raise TypeError(
            f"Unknown data type {type(data_obj)}. "
            "Please check "
            "https://lancedb.github.io/lance/read_and_write.html "
            "to see supported types."
        )


def _casting_recordbatch_iter(
    input_iter: Iterable[pa.RecordBatch], schema: pa.Schema
) -> Iterable[pa.RecordBatch]:
    """
    Wrapper around an iterator of record batches. If the batches don't match the
    schema, try to cast them to the schema. If that fails, raise an error.

    This is helpful for users who might have written the iterator with default
    data types in PyArrow, but specified more specific types in the schema. For
    example, PyArrow defaults to float64 for floating point types, but Lance
    uses float32 for vectors.
    """
    for batch in input_iter:
        if not isinstance(batch, pa.RecordBatch):
            raise TypeError(f"Expected RecordBatch, got {type(batch)}")
        if batch.schema != schema:
            try:
                # RecordBatch doesn't have a cast method, but table does.
                batch = pa.Table.from_batches([batch]).cast(schema).to_batches()[0]
            except pa.lib.ArrowInvalid:
                raise ValueError(
                    f"Input RecordBatch iterator yielded a batch with schema that "
                    f"does not match the expected schema.\nExpected:\n{schema}\n"
                    f"Got:\n{batch.schema}"
                )
        yield batch
