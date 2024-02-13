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

import copy
import json
import os
import pickle
import random
import sqlite3
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    TypedDict,
    Union,
)

import pyarrow as pa
import pyarrow.dataset
from pyarrow import RecordBatch, Schema

from .dependencies import (
    _check_for_hugging_face,
    _check_for_numpy,
    _check_for_pandas,
    torch,
)
from .dependencies import numpy as np
from .dependencies import pandas as pd
from .fragment import FragmentMetadata, LanceFragment
from .lance import (
    CleanupStats,
    _Dataset,
    _MergeInsertBuilder,
    _Operation,
    _Scanner,
    _write_dataset,
)
from .lance import CompactionMetrics as CompactionMetrics
from .lance import __version__ as __version__
from .optimize import Compaction
from .util import td_to_micros

if TYPE_CHECKING:
    from pyarrow._compute import Expression

    from .commit import CommitLock
    from .progress import FragmentWriteProgress

    ReaderLike = Union[
        pd.Timestamp,
        pa.Table,
        pa.dataset.Dataset,
        pa.dataset.Scanner,
        Iterable[RecordBatch],
        pa.RecordBatchReader,
    ]

    QueryVectorLike = Union[
        pd.Series,
        pa.Array,
        pa.Scalar,
        np.ndarray,
        Iterable[float],
    ]


class MergeInsertBuilder(_MergeInsertBuilder):
    def execute(self, data_obj: ReaderLike, *, schema: Optional[pa.Schema] = None):
        """Executes the merge insert operation

        There is no return value but the original dataset will be updated.

        Parameters
        ----------

        data_obj: ReaderLike
            The new data to use as the source table for the operation.  This parameter
            can be any source of data (e.g. table / dataset) that
            :func:`~lance.write_dataset` accepts.
        schema: Optional[pa.Schema]
            The schema of the data.  This only needs to be supplied whenever the data
            source is some kind of generator.
        """
        reader = _coerce_reader(data_obj, schema)
        super(MergeInsertBuilder, self).execute(reader)

    # These next three overrides exist only to document the methods

    def when_matched_update_all(self, condition: Optional[str] = None):
        """
        Configure the operation to update matched rows

        After this method is called, when the merge insert operation executes,
        any rows that match both the source table and the target table will be
        updated.  The rows from the target table will be removed and the rows
        from the source table will be added.

        An optional condition may be specified.  This should be an SQL filter
        and, if present, then only matched rows that also satisfy this filter will
        be updated.  The SQL filter should use the prefix `target.` to refer to
        columns in the target table and the prefix `source.` to refer to columns
        in the source table.  For example, `source.last_update < target.last_update`.

        If a condition is specified and rows do not satisfy the condition then these
        rows will not be updated.  Failure to satisfy the filter does not cause
        a "matched" row to become a "not matched" row.
        """
        return super(MergeInsertBuilder, self).when_matched_update_all(condition)

    def when_not_matched_insert_all(self):
        """
        Configure the operation to insert not matched rows

        After this method is called, when the merge insert operation executes,
        any rows that exist only in the source table will be inserted into
        the target table.
        """
        return super(MergeInsertBuilder, self).when_not_matched_insert_all()

    def when_not_matched_by_source_delete(self, expr: Optional[str] = None):
        """
        Configure the operation to delete source rows that do not match

        After this method is called, when the merge insert operation executes,
        any rows that exist only in the target table will be deleted.  An
        optional filter can be specified to limit the scope of the delete
        operation.  If given (as an SQL filter) then only rows which match
        the filter will be deleted.
        """
        return super(MergeInsertBuilder, self).when_not_matched_by_source_delete(expr)


class LanceDataset(pa.dataset.Dataset):
    """A dataset in Lance format where the data is stored at the given uri."""

    def __init__(
        self,
        uri: Union[str, Path],
        version: Optional[int] = None,
        block_size: Optional[int] = None,
        index_cache_size: Optional[int] = None,
        metadata_cache_size: Optional[int] = None,
        commit_lock: Optional[CommitLock] = None,
    ):
        uri = os.fspath(uri) if isinstance(uri, Path) else uri
        self._uri = uri
        self._ds = _Dataset(
            uri, version, block_size, index_cache_size, metadata_cache_size, commit_lock
        )

    def __reduce__(self):
        return LanceDataset, (self.uri, self._ds.version())

    def __getstate__(self):
        return self.uri, self._ds.version()

    def __setstate__(self, state):
        self._uri, version = state
        self._ds = _Dataset(self._uri, version)

    def __len__(self):
        return self.count_rows()

    @property
    def uri(self) -> str:
        """
        The location of the data
        """
        return self._uri

    def list_indices(self) -> List[Dict[str, Any]]:
        if getattr(self, "_list_indices_res", None) is None:
            self._list_indices_res = self._ds.load_indices()
        return self._list_indices_res

    def index_statistics(self, index_name: str) -> Dict[str, Any]:
        warnings.warn(
            "LanceDataset.index_statistics() is deprecated, "
            + "use LanceDataset.stats.index_stats() instead",
            DeprecationWarning,
        )
        return json.loads(self._ds.index_statistics(index_name))

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
        *,
        prefilter: bool = False,
        with_row_id: bool = False,
        use_stats: bool = True,
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
            The max size of batches returned.
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
        prefilter: bool, default False
            If True then the filter will be applied before the vector query is run.
            This will generate more correct results but it may be a more costly
            query.  It's generally good when the filter is highly selective.

            If False then the filter will be applied after the vector query is run.
            This will perform well but the results may have fewer than the requested
            number of rows (or be empty) if the rows closest to the query do not
            match the filter.  It's generally good when the filter is not very
            selective.

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
        builder = (
            ScannerBuilder(self)
            .columns(columns)
            .filter(filter)
            .prefilter(prefilter)
            .limit(limit)
            .offset(offset)
            .batch_size(batch_size)
            .batch_readahead(batch_readahead)
            .fragment_readahead(fragment_readahead)
            .scan_in_order(scan_in_order)
            .with_fragments(fragments)
            .with_row_id(with_row_id)
            .use_stats(use_stats)
        )
        if nearest is not None:
            builder = builder.nearest(**nearest)
        return builder.to_scanner()

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
        *,
        prefilter: bool = False,
        with_row_id: bool = False,
        use_stats: bool = True,
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
        prefilter: bool, default False
            Run filter before the vector search.
        with_row_id: bool, default False
            Return physical row ID.
        use_stats: bool, default True
            Use stats pushdown during filters.

        Notes
        -----
        If BOTH filter and nearest is specified, then:
        1. nearest is executed first.
        2. The results are filtered afterward, unless pre-filter sets to True.
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
            prefilter=prefilter,
            with_row_id=with_row_id,
            use_stats=use_stats,
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

    def get_fragments(self, filter: Optional[Expression] = None) -> List[LanceFragment]:
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
        *,
        prefilter: bool = False,
        with_row_id: bool = False,
        use_stats: bool = True,
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
            prefilter=prefilter,
            with_row_id=with_row_id,
            use_stats=use_stats,
        ).to_batches()

    def sample(
        self,
        num_rows: int,
        columns: Optional[List[str]] = None,
        randomize_order: bool = True,
        **kwargs,
    ) -> pa.Table:
        """Select a random sample of data

        Parameters
        ----------
        num_rows: int
            number of rows to retrieve
        columns: list of strings, optional
            list of column names to be fetched.  All columns are fetched
            if not specified.
        **kwargs : dict, optional
            see scanner() method for full parameter description.

        Returns
        -------
        table : Table
        """
        total_num_rows = self.count_rows()
        indices = random.sample(range(total_num_rows), num_rows)
        if not randomize_order:
            # Sort the indices in order to increase the locality and thus reduce
            # the number of random reads.
            indices = sorted(indices)
        return self.take(indices, columns, **kwargs)

    def take(
        self,
        indices: Union[List[int], pa.Array],
        columns: Optional[List[str]] = None,
        **kwargs,
    ) -> pa.Table:
        """Select rows of data by index.

        Parameters
        ----------
        indices : Array or array-like
            indices of rows to select in the dataset.
        columns: list of strings, optional
            List of column names to be fetched. All columns are fetched
            if not specified.
        **kwargs : dict, optional
            See scanner() method for full parameter description.

        Returns
        -------
        table : Table
        """
        return pa.Table.from_batches([self._ds.take(indices, columns)])

    def _take_rows(
        self,
        row_ids: Union[List[int], pa.Array],
        columns: Optional[List[str]] = None,
        **kargs,
    ) -> pa.Table:
        """Select rows by row_ids.

        **Unstable API**. Internal use only

        Parameters
        ----------
        row_ids : List Array or array-like
            row IDs to select in the dataset.
        columns: list of strings, optional
            List of column names to be fetched. All columns are fetched
            if not specified.
        **kwargs : dict, optional
            See scanner() method for full parameter description.

        Returns
        -------
        table : Table
        """
        return pa.Table.from_batches([self._ds.take_rows(row_ids, columns)])

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

    def count_rows(
        self, filter: Optional[Union[str, pa.compute.Expression]] = None, **kwargs
    ) -> int:
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
        if filter is None:
            return self._ds.count_rows()
        else:
            return self.scanner(filter=filter).count_rows()

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

    def alter_columns(self, *alterations: Iterable[Dict[str, Any]]):
        """Alter column names and nullability.

        Parameters
        ----------
        alterations : Iterable[Dict[str, Any]]
            A sequence of dictionaries, each with the following keys:
            - "path": str
                The column path to alter. For a top-level column, this is the name.
                For a nested column, this is the dot-separated path, e.g. "a.b.c".
            - "name": str, optional
                The new name of the column. If not specified, the column name is
                not changed.
            - "nullable": bool, optional
                Whether the column should be nullable. If not specified, the column
                nullability is not changed. Only non-nullable columns can be changed
                to nullable. Currently, you cannot change a nullable column to
                non-nullable.

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> schema = pa.schema([pa.field('a', pa.int64()),
        ...                     pa.field('b', pa.string(), nullable=False)])
        >>> table = pa.table({"a": [1, 2, 3], "b": ["a", "b", "c"]})
        >>> dataset = lance.write_dataset(table, "example")
        >>> dataset.alter_columns({"path": "a", "name": "x"},
        ...                       {"path": "b", "nullable": True})
        >>> dataset.to_table().to_pandas()
           x  b
        0  1  a
        1  2  b
        2  3  c
        """
        self._ds.alter_columns(list(alterations))

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

        See Also
        --------
        LanceDataset.add_columns :
            Add new columns by computing batch-by-batch.
        """
        if right_on is None:
            right_on = left_on

        reader = _coerce_reader(data_obj, schema)

        self._ds.merge(reader, left_on, right_on)

    def add_columns(
        self,
        transforms: Dict[str, str] | BatchUDF,
        read_columns: List[str] | None = None,
    ):
        """
        Add new columns with defined values.

        There are two ways to specify the new columns. First, you can provide
        SQL expressions for each new column. Second you can provide a UDF that
        takes a batch of existing data and returns a new batch with the new
        columns. These new columns will be appended to the dataset.

        See the :func:`lance.add_columns_udf` decorator for more information on
        writing UDFs.

        Parameters
        ----------
        transforms : dict or AddColumnsUDF
            If this is a dictionary, then the keys are the names of the new
            columns and the values are SQL expression strings. These strings can
            reference existing columns in the dataset.
            If this is a AddColumnsUDF, then it is a UDF that takes a batch of
            existing data and returns a new batch with the new columns.
        read_columns : list of str, optional
            The names of the columns that the UDF will read. If None, then the
            UDF will read all columns. This is only used when transforms is a
            UDF. Otherwise, the read columns are inferred from the SQL expressions.

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> table = pa.table({"a": [1, 2, 3]})
        >>> dataset = lance.write_dataset(table, "my_dataset")
        >>> @lance.batch_udf()
        ... def double_a(batch):
        ...     df = batch.to_pandas()
        ...     return pd.DataFrame({'double_a': 2 * df['a']})
        >>> dataset.add_columns(double_a)
        >>> dataset.to_table().to_pandas()
           a  double_a
        0  1         2
        1  2         4
        2  3         6
        >>> dataset.add_columns({"triple_a": "a * 3"})
        >>> dataset.to_table().to_pandas()
           a  double_a  triple_a
        0  1         2         3
        1  2         4         6
        2  3         6         9

        See Also
        --------
        LanceDataset.merge :
            Merge a pre-computed set of columns into the dataset.
        """
        if isinstance(transforms, BatchUDF):
            if transforms.output_schema is None:
                # Infer the schema based on the first batch
                sample_batch = transforms(
                    next(iter(self.to_batches(limit=1, columns=read_columns)))
                )
                if isinstance(sample_batch, pd.DataFrame):
                    sample_batch = pa.RecordBatch.from_pandas(sample_batch)
                transforms.output_schema = sample_batch.schema
                del sample_batch
        elif isinstance(transforms, dict):
            for k, v in transforms.items():
                if not isinstance(k, str):
                    raise TypeError(f"Column names must be a string. Got {type(k)}")
                if not isinstance(v, str):
                    raise TypeError(
                        f"Column expressions must be a string. Got {type(k)}"
                    )
        else:
            raise TypeError("transforms must be a dict or AddColumnsUDF")

        self._ds.add_columns(transforms, read_columns)

        if isinstance(transforms, BatchUDF):
            if transforms.cache is not None:
                transforms.cache.cleanup()

    def drop_columns(self, columns: List[str]):
        """Drop one or more columns from the dataset

        Parameters
        ----------
        columns : list of str
            The names of the columns to drop. These can be nested column references
            (e.g. "a.b.c") or top-level column names (e.g. "a").

        This is a metadata-only operation and does not remove the data from the
        underlying storage. In order to remove the data, you must subsequently
        call ``compact_files`` to rewrite the data without the removed columns and
        then call ``cleanup_files`` to remove the old files.

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> table = pa.table({"a": [1, 2, 3], "b": ["a", "b", "c"]})
        >>> dataset = lance.write_dataset(table, "example")
        >>> dataset.drop_columns(["a"])
        >>> dataset.to_table().to_pandas()
           b
        0  a
        1  b
        2  c
        """
        self._ds.drop_columns(columns)
        # Indices might have changed
        self._list_indices_res = None

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

    def merge_insert(
        self,
        on: Union[str, Iterable[str]],
    ):
        """
        Returns a builder that can be used to create a "merge insert" operation

        This operation can add rows, update rows, and remove rows in a single
        transaction. It is a very generic tool that can be used to create
        behaviors like "insert if not exists", "update or insert (i.e. upsert)",
        or even replace a portion of existing data with new data (e.g. replace
        all data where month="january")

        The merge insert operation works by combining new data from a
        **source table** with existing data in a **target table** by using a
        join.  There are three categories of records.

        "Matched" records are records that exist in both the source table and
        the target table. "Not matched" records exist only in the source table
        (e.g. these are new data). "Not matched by source" records exist only
        in the target table (this is old data).

        The builder returned by this method can be used to customize what
        should happen for each category of data.

        Please note that the data will be reordered as part of this
        operation.  This is because updated rows will be deleted from the
        dataset and then reinserted at the end with the new values.  The
        order of the newly inserted rows may fluctuate randomly because a
        hash-join operation is used internally.

        Parameters
        ----------

        on: Union[str, Iterable[str]]
            A column (or columns) to join on.  This is how records from the
            source table and target table are matched.  Typically this is some
            kind of key or id column.

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> table = pa.table({"a": [2, 1, 3], "b": ["a", "b", "c"]})
        >>> dataset = lance.write_dataset(table, "example")
        >>> new_table = pa.table({"a": [2, 3, 4], "b": ["x", "y", "z"]})
        >>> # Perform a "upsert" operation
        >>> dataset.merge_insert("a")             \\
        ...        .when_matched_update_all()     \\
        ...        .when_not_matched_insert_all() \\
        ...        .execute(new_table)
        >>> dataset.to_table().sort_by("a").to_pandas()
           a  b
        0  1  b
        1  2  x
        2  3  y
        3  4  z
        """
        return MergeInsertBuilder(self._ds, on)

    def update(
        self,
        updates: Dict[str, str],
        where: Optional[str] = None,
    ):
        """
        Update column values for rows matching where.

        Parameters
        ----------
        updates : dict of str to str
            A mapping of column names to a SQL expression.
        where : str, optional
            A SQL predicate indicating which rows should be updated.

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> table = pa.table({"a": [1, 2, 3], "b": ["a", "b", "c"]})
        >>> dataset = lance.write_dataset(table, "example")
        >>> dataset.update(dict(a = 'a + 2'), where="b != 'a'")
        >>> dataset.to_table().to_pandas()
           a  b
        0  1  a
        1  4  b
        2  5  c
        """
        if isinstance(where, pa.compute.Expression):
            where = str(where)
        self._ds.update(updates, where)

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

    @property
    def latest_version(self) -> int:
        """
        Returns the latest version of the dataset.
        """
        return self._ds.latest_version()

    def checkout_version(self, version) -> "LanceDataset":
        """
        Load the given version of the dataset.

        Unlike the :func:`dataset` constructor, this will re-use the
        current cache.
        This is a no-op if the dataset is already at the given version.
        """
        ds = copy.copy(self)
        if version != ds.version:
            ds._ds = self._ds.checkout_version(version)
        return ds

    def restore(self):
        """
        Restore the currently checked out version as the latest version of the dataset.

        This creates a new commit.
        """
        self._ds.restore()

    def cleanup_old_versions(
        self,
        older_than: Optional[timedelta] = None,
        *,
        delete_unverified: bool = False,
    ) -> CleanupStats:
        """
        Cleans up old versions of the dataset.

        Some dataset changes, such as overwriting, leave behind data that is not
        referenced by the latest dataset version.  The old data is left in place
        to allow the dataset to be restored back to an older version.

        This method will remove older versions and any data files they reference.
        Once this cleanup task has run you will not be able to checkout or restore
        these older versions.

        Parameters
        ----------

        older_than: timedelta, optional
            Only versions older than this will be removed.  If not specified, this
            will default to two weeks.

        delete_unverified: bool, default False
            Files leftover from a failed transaction may appear to be part of an
            in-progress operation (e.g. appending new data) and these files will
            not be deleted unless they are at least 7 days old.  If delete_unverified
            is True then these files will be deleted regardless of their age.

            This should only be set to True if you can guarantee that no other process
            is currently working on this dataset.  Otherwise the dataset could be put
            into a corrupted state.
        """
        if older_than is None:
            older_than = timedelta(days=14)
        return self._ds.cleanup_old_versions(
            td_to_micros(older_than), delete_unverified
        )

    def create_scalar_index(
        self,
        column: str,
        index_type: Literal["BTREE"],
        name: Optional[str] = None,
        *,
        replace: bool = True,
    ):
        """Create a scalar index on a column.

        Scalar indices, like vector indices, can be used to speed up scans.  A scalar
        index can speed up scans that contain filter expressions on the indexed column.
        For example, the following scan will be faster if the column ``my_col`` has
        a scalar index:

        .. code-block:: python

            import lance

            dataset = lance.dataset("/tmp/images.lance")
            my_table = dataset.scanner(filter="my_col != 7").to_table()

        Scalar indices can also speed up scans containing a vector search and a
        prefilter:

        .. code-block::python

            import lance

            dataset = lance.dataset("/tmp/images.lance")
            my_table = dataset.scanner(
                nearest=dict(
                   column="vector",
                   q=[1, 2, 3, 4],
                   k=10,
                )
                filter="my_col != 7",
                prefilter=True
            )

        Scalar indices can only speed up scans for basic filters using
        equality, comparison, range (e.g. ``my_col BETWEEN 0 AND 100``), and set
        membership (e.g. `my_col IN (0, 1, 2)`)

        Scalar indices can be used if the filter contains multiple indexed columns and
        the filter criteria are AND'd or OR'd together
        (e.g. ``my_col < 0 AND other_col> 100``)

        Scalar indices may be used if the filter contains non-indexed columns but,
        depending on the structure of the filter, they may not be usable.  For example,
        if the column ``not_indexed`` does not have a scalar index then the filter
        ``my_col = 0 OR not_indexed = 1`` will not be able to use any scalar index on
        ``my_col``.

        To determine if a scan is making use of a scalar index you can use
        ``explain_plan`` to look at the query plan that lance has created.  Queries
        that use scalar indices will either have a ``ScalarIndexQuery`` relation or a
        ``MaterializeIndex`` operator.

        Currently, the only type of scalar index available is ``BTREE``. This index
        combines is inspired by the btree data structure although only the first few
        layers of the btree are cached in memory.

        **Experimental API**

        Parameters
        ----------
        column : str
            The column to be indexed.  Must be a boolean, integer, float,
            or string column.
        index_type : str
            The type of the index.  Only ``"BTREE"`` is supported now.
        name : str, optional
            The index name. If not provided, it will be generated from the
            column name.
        replace : bool, default True
            Replace the existing index if it exists.

        Examples
        --------

        .. code-block:: python

            import lance

            dataset = lance.dataset("/tmp/images.lance")
            dataset.create_index(
                "category",
                "BTREE",
            )
        """
        if isinstance(column, str):
            column = [column]

        if len(column) > 1:
            raise NotImplementedError(
                "Scalar indices currently only support a single column"
            )

        column = column[0]
        if column not in self.schema.names:
            raise KeyError(f"{column} not found in schema")

        field = self.schema.field(column)
        if (
            not pa.types.is_integer(field.type)
            and not pa.types.is_floating(field.type)
            and not pa.types.is_boolean(field.type)
            and not pa.types.is_string(field.type)
        ):
            raise TypeError(
                f"Scalar index column {column} must be int, float, bool, or str"
            )

        index_type = index_type.upper()
        if index_type != "BTREE":
            raise NotImplementedError((
                'Only "BTREE" is supported for ',
                f"index_type.  Received {index_type}",
            ))

        self._ds.create_index([column], index_type, name, replace)

    def create_index(
        self,
        column: Union[str, List[str]],
        index_type: str,
        name: Optional[str] = None,
        metric: str = "L2",
        replace: bool = False,
        num_partitions: Optional[int] = None,
        ivf_centroids: Optional[
            Union[np.ndarray, pa.FixedSizeListArray, pa.FixedShapeTensorArray]
        ] = None,
        pq_codebook: Optional[
            Union[np.ndarray, pa.FixedSizeListArray, pa.FixedShapeTensorArray]
        ] = None,
        num_sub_vectors: Optional[int] = None,
        accelerator: Optional[Union[str, "torch.Device"]] = None,
        index_cache_size: Optional[int] = None,
        shuffle_partition_batches: Optional[int] = None,
        shuffle_partition_concurrency: Optional[int] = None,
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
        ivf_centroids : ``np.ndarray``, ``pyarrow.FixedSizeListArray``
        or ``pyarrow.FixedShapeTensorArray``. Optional.
            A ``num_partitions x dimension`` array of K-mean centroids for IVF
            clustering. If not provided, a new Kmean model will be trained.
        pq_codebook : ``np.ndarray``, ``pyarrow.FixedSizeListArray``
        or ``pyarrow.FixedShapeTensorArray``. Optional.
            A ``num_sub_vectors x (2 ^ nbits * dimensions // num_sub_vectors)``
            array of K-mean centroids for PQ codebook.
            Note: nbits is always 8 for now.
            If not provided, a new PQ model will be trained.
        num_sub_vectors : int, optional
            The number of sub-vectors for PQ (Product Quantization).
        accelerator : str or ``torch.Device``, optional
            If set, use an accelerator to speed up the training process.
            Accepted accelerator: "cuda" (Nvidia GPU) and "mps" (Apple Silicon GPU).
            If not set, use the CPU.
        index_cache_size : int, optional
            The size of the index cache in number of entries. Default value is 256.
        shuffle_partition_batches : int, optional
            The number of batches, using the row group size of the dataset, to include
            in each shuffle partition. Default value is 10240.

            Assuming the row group size is 1024, each shuffle partition will hold
            10240 * 1024 = 10,485,760 rows. By making this value smaller, this shuffle
            will consume less memory but will take longer to complete, and vice versa.
        shuffle_partition_concurrency : int, optional
            The number of shuffle partitions to process concurrently. Default value is 2

            By making this value smaller, this shuffle will consume less memory but will
            take longer to complete, and vice versa.
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

        Experimental Accelerator (GPU) support:

        - *accelerate*: use GPU to train IVF partitions.
            Only supports CUDA (Nvidia) or MPS (Apple) currently.
            Requires PyTorch being installed.

        .. code-block:: python

            import lance

            dataset = lance.dataset("/tmp/sift.lance")
            dataset.create_index(
                "vector",
                "IVF_PQ",
                num_partitions=256,
                num_sub_vectors=16,
                accelerator="cuda"
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
            if not (
                pa.types.is_fixed_size_list(field.type)
                or (
                    isinstance(field.type, pa.FixedShapeTensorType)
                    and len(field.type.shape) == 1
                )
            ):
                raise TypeError(
                    f"Vector column {c} must be FixedSizeListArray "
                    f"1-dimensional FixedShapeTensorArray, got {field.type}"
                )
            if not pa.types.is_floating(field.type.value_type):
                raise TypeError(
                    f"Vector column {c} must have floating value type, "
                    f"got {field.type.value_type}"
                )

        if not isinstance(metric, str) or metric.lower() not in [
            "l2",
            "cosine",
            "euclidean",
            "dot",
        ]:
            raise ValueError(f"Metric {metric} not supported.")

        kwargs["metric_type"] = metric

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
            if isinstance(num_partitions, float):
                warnings.warn("num_partitions is float, converting to int")
                num_partitions = int(num_partitions)
            elif not isinstance(num_partitions, int):
                raise TypeError(
                    f"num_partitions must be int, got {type(num_partitions)}"
                )
            kwargs["num_partitions"] = num_partitions
            kwargs["num_sub_vectors"] = num_sub_vectors

            if accelerator is not None and ivf_centroids is None:
                # Use accelerator to train ivf centroids
                from .vector import train_ivf_centroids_on_accelerator

                ivf_centroids, partitions_file = train_ivf_centroids_on_accelerator(
                    self,
                    column[0],
                    num_partitions,
                    metric,
                    accelerator,
                )
                kwargs["precomputed_partitions_file"] = partitions_file

            if (ivf_centroids is None) and (pq_codebook is not None):
                raise ValueError(
                    "ivf_centroids must be specified when pq_codebook is provided"
                )

            if ivf_centroids is not None:
                # User provided IVF centroids
                if _check_for_numpy(ivf_centroids) and isinstance(
                    ivf_centroids, np.ndarray
                ):
                    if (
                        len(ivf_centroids.shape) != 2
                        or ivf_centroids.shape[0] != num_partitions
                    ):
                        raise ValueError(
                            f"Ivf centroids must be 2D array: (clusters, dim), "
                            f"got {ivf_centroids.shape}"
                        )
                    if ivf_centroids.dtype not in [np.float16, np.float32, np.float64]:
                        raise TypeError(
                            "IVF centroids must be floating number"
                            + f"got {ivf_centroids.dtype}"
                        )
                    dim = ivf_centroids.shape[1]
                    values = pa.array(ivf_centroids.reshape(-1))
                    ivf_centroids = pa.FixedSizeListArray.from_arrays(values, dim)
                # Convert it to RecordBatch because Rust side only accepts RecordBatch.
                ivf_centroids_batch = pa.RecordBatch.from_arrays(
                    [ivf_centroids], ["_ivf_centroids"]
                )
                kwargs["ivf_centroids"] = ivf_centroids_batch

            if pq_codebook is not None:
                # User provided IVF centroids
                if _check_for_numpy(pq_codebook) and isinstance(
                    pq_codebook, np.ndarray
                ):
                    if (
                        len(pq_codebook.shape) != 3
                        or pq_codebook.shape[0] != num_sub_vectors
                        or pq_codebook.shape[1] != 256
                    ):
                        raise ValueError(
                            f"PQ codebook must be 3D array: (sub_vectors, 256, dim), "
                            f"got {pq_codebook.shape}"
                        )
                    if pq_codebook.dtype not in [np.float16, np.float32, np.float64]:
                        raise TypeError(
                            "PQ codebook must be floating number"
                            + f"got {pq_codebook.dtype}"
                        )
                    values = pa.array(pq_codebook.reshape(-1))
                    pq_codebook = pa.FixedSizeListArray.from_arrays(
                        values, num_sub_vectors * 256
                    )
                pq_codebook_batch = pa.RecordBatch.from_arrays(
                    [pq_codebook], ["_pq_codebook"]
                )
                kwargs["pq_codebook"] = pq_codebook_batch

        if shuffle_partition_batches is not None:
            kwargs["shuffle_partition_batches"] = shuffle_partition_batches
        if shuffle_partition_concurrency is not None:
            kwargs["shuffle_partition_concurrency"] = shuffle_partition_concurrency

        self._ds.create_index(column, index_type, name, replace, kwargs)
        return LanceDataset(self.uri, index_cache_size=index_cache_size)

    @staticmethod
    def _commit(
        base_uri: Union[str, Path],
        operation: LanceOperation.BaseOperation,
        read_version: Optional[int] = None,
        commit_lock: Optional[CommitLock] = None,
    ) -> LanceDataset:
        warnings.warn(
            "LanceDataset._commit() is deprecated, use LanceDataset.commit()"
            " instead",
            DeprecationWarning,
        )
        return LanceDataset.commit(base_uri, operation, read_version, commit_lock)

    @staticmethod
    def commit(
        base_uri: Union[str, Path],
        operation: LanceOperation.BaseOperation,
        read_version: Optional[int] = None,
        commit_lock: Optional[CommitLock] = None,
    ) -> LanceDataset:
        """Create a new version of dataset

        This method is an advanced method which allows users to describe a change
        that has been made to the data files.  This method is not needed when using
        Lance to apply changes (e.g. when using :py:class:`LanceDataset` or
        :py:func:`write_dataset`.)

        It's current purpose is to allow for changes being made in a distributed
        environment where no single process is doing all of the work.  For example,
        a distributed bulk update or a distributed bulk modify operation.

        Once all of the changes have been made, this method can be called to make
        the changes visible by updating the dataset manifest.

        Warnings
        --------
        This is an advanced API and doesn't provide the same level of validation
        as the other APIs. For example, it's the responsibility of the caller to
        ensure that the fragments are valid for the schema.

        Parameters
        ----------
        base_uri: str or Path
            The base uri of the dataset
        operation: BaseOperation
            The operation to apply to the dataset.  This describes what changes
            have been made. See available operations under :class:`LanceOperation`.
        read_version: int, optional
            The version of the dataset that was used as the base for the changes.
            This is not needed for overwrite or restore operations.
        commit_lock : CommitLock, optional
            A custom commit lock.  Only needed if your object store does not support
            atomic commits.  See the user guide for more details.

        Returns
        -------
        LanceDataset
            A new version of Lance Dataset.

        Examples
        --------

        Creating a new dataset with the :class:`LanceOperation.Overwrite` operation:

        >>> import lance
        >>> import pyarrow as pa
        >>> tab1 = pa.table({"a": [1, 2], "b": ["a", "b"]})
        >>> tab2 = pa.table({"a": [3, 4], "b": ["c", "d"]})
        >>> fragment1 = lance.fragment.LanceFragment.create("example", tab1)
        >>> fragment2 = lance.fragment.LanceFragment.create("example", tab2)
        >>> fragments = [fragment1, fragment2]
        >>> operation = lance.LanceOperation.Overwrite(tab1.schema, fragments)
        >>> dataset = lance.LanceDataset.commit("example", operation)
        >>> dataset.to_table().to_pandas()
           a  b
        0  1  a
        1  2  b
        2  3  c
        3  4  d
        """
        # TODO: mode is never used!
        if isinstance(base_uri, Path):
            base_uri = str(base_uri)

        if commit_lock:
            if not callable(commit_lock):
                raise TypeError(
                    f"commit_lock must be a function, got {type(commit_lock)}"
                )

        _Dataset.commit(base_uri, operation._to_inner(), read_version, commit_lock)
        return LanceDataset(base_uri)

    def validate(self):
        """
        Validate the dataset.

        This checks the integrity of the dataset and will raise an exception if
        the dataset is corrupted.
        """
        self._ds.validate()

    @property
    def optimize(self) -> "DatasetOptimizer":
        return DatasetOptimizer(self)

    @property
    def stats(self) -> "LanceStats":
        """
        **Experimental API**
        """
        return LanceStats(self._ds)


# LanceOperation is a namespace for operations that can be applied to a dataset.
class LanceOperation:
    @staticmethod
    def _validate_fragments(fragments):
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

    class BaseOperation(ABC):
        """
        Base class for operations that can be applied to a dataset.

        See available operations under :class:`LanceOperation`.
        """

        @abstractmethod
        def _to_inner(self):
            raise NotImplementedError()

    @dataclass
    class Overwrite(BaseOperation):
        """
        Overwrite or create a new dataset.

        Attributes
        ----------
        new_schema: pyarrow.Schema
            The schema of the new dataset.
        fragments: list[FragmentMetadata]
            The fragments that make up the new dataset.

        Warning
        -------
        This is an advanced API for distributed operations. To overwrite or
        create new dataset on a single machine, use :func:`lance.write_dataset`.

        Examples
        --------

        To create or overwrite a dataset, first use
        :meth:`lance.fragment.LanceFragment.create` to create fragments. Then
        collect the fragment metadata into a list and pass it along with the
        schema to this class. Finally, pass the operation to the
        :meth:`LanceDataset.commit` method to create the new dataset.

        >>> import lance
        >>> import pyarrow as pa
        >>> tab1 = pa.table({"a": [1, 2], "b": ["a", "b"]})
        >>> tab2 = pa.table({"a": [3, 4], "b": ["c", "d"]})
        >>> fragment1 = lance.fragment.LanceFragment.create("example", tab1)
        >>> fragment2 = lance.fragment.LanceFragment.create("example", tab2)
        >>> fragments = [fragment1, fragment2]
        >>> operation = lance.LanceOperation.Overwrite(tab1.schema, fragments)
        >>> dataset = lance.LanceDataset.commit("example", operation)
        >>> dataset.to_table().to_pandas()
           a  b
        0  1  a
        1  2  b
        2  3  c
        3  4  d
        """

        new_schema: pa.Schema
        fragments: Iterable[FragmentMetadata]

        def __post_init__(self):
            if not isinstance(self.new_schema, pa.Schema):
                raise TypeError(
                    f"schema must be pyarrow.Schema, got {type(self.new_schema)}"
                )
            LanceOperation._validate_fragments(self.fragments)

        def _to_inner(self):
            raw_fragments = [f._metadata for f in self.fragments]
            return _Operation.overwrite(self.new_schema, raw_fragments)

    @dataclass
    class Append(BaseOperation):
        """
        Append new rows to the dataset.

        Attributes
        ----------
        fragments: list[FragmentMetadata]
            The fragments that contain the new rows.

        Warning
        -------
        This is an advanced API for distributed operations. To append to a
        dataset on a single machine, use :func:`lance.write_dataset`.

        Examples
        --------

        To append new rows to a dataset, first use
        :meth:`lance.fragment.LanceFragment.create` to create fragments. Then
        collect the fragment metadata into a list and pass it to this class.
        Finally, pass the operation to the :meth:`LanceDataset.commit`
        method to create the new dataset.

        >>> import lance
        >>> import pyarrow as pa
        >>> tab1 = pa.table({"a": [1, 2], "b": ["a", "b"]})
        >>> dataset = lance.write_dataset(tab1, "example")
        >>> tab2 = pa.table({"a": [3, 4], "b": ["c", "d"]})
        >>> fragment = lance.fragment.LanceFragment.create("example", tab2)
        >>> operation = lance.LanceOperation.Append([fragment])
        >>> dataset = lance.LanceDataset.commit("example", operation,
        ...                                     read_version=dataset.version)
        >>> dataset.to_table().to_pandas()
           a  b
        0  1  a
        1  2  b
        2  3  c
        3  4  d
        """

        fragments: Iterable[FragmentMetadata]

        def __post_init__(self):
            LanceOperation._validate_fragments(self.fragments)

        def _to_inner(self):
            raw_fragments = [f._metadata for f in self.fragments]
            return _Operation.append(raw_fragments)

    @dataclass
    class Delete(BaseOperation):
        """
        Remove fragments or rows from the dataset.

        Attributes
        ----------
        updated_fragments: list[FragmentMetadata]
            The fragments that have been updated with new deletion vectors.
        deleted_fragment_ids: list[int]
            The ids of the fragments that have been deleted entirely. These are
            the fragments where :meth:`LanceFragment.delete()` returned None.
        predicate: str
            The original SQL predicate used to select the rows to delete.

        Warning
        -------
        This is an advanced API for distributed operations. To delete rows from
        dataset on a single machine, use :meth:`lance.LanceDataset.delete`.

        Examples
        --------

        To delete rows from a dataset, call :meth:`lance.fragment.LanceFragment.delete`
        on each of the fragments. If that returns a new fragment, add that to
        the ``updated_fragments`` list. If it returns None, that means the whole
        fragment was deleted, so add the fragment id to the ``deleted_fragment_ids``.
        Finally, pass the operation to the :meth:`LanceDataset.commit` method to
        complete the deletion operation.

        >>> import lance
        >>> import pyarrow as pa
        >>> table = pa.table({"a": [1, 2], "b": ["a", "b"]})
        >>> dataset = lance.write_dataset(table, "example")
        >>> table = pa.table({"a": [3, 4], "b": ["c", "d"]})
        >>> dataset = lance.write_dataset(table, "example", mode="append")
        >>> dataset.to_table().to_pandas()
           a  b
        0  1  a
        1  2  b
        2  3  c
        3  4  d
        >>> predicate = "a >= 2"
        >>> updated_fragments = []
        >>> deleted_fragment_ids = []
        >>> for fragment in dataset.get_fragments():
        ...     new_fragment = fragment.delete(predicate)
        ...     if new_fragment is not None:
        ...         updated_fragments.append(new_fragment)
        ...     else:
        ...         deleted_fragment_ids.append(fragment.fragment_id)
        >>> operation = lance.LanceOperation.Delete(updated_fragments,
        ...                                         deleted_fragment_ids,
        ...                                         predicate)
        >>> dataset = lance.LanceDataset.commit("example", operation,
        ...                                     read_version=dataset.version)
        >>> dataset.to_table().to_pandas()
           a  b
        0  1  a
        """

        updated_fragments: Iterable[FragmentMetadata]
        deleted_fragment_ids: Iterable[int]
        predicate: str

        def __post_init__(self):
            LanceOperation._validate_fragments(self.updated_fragments)

        def _to_inner(self):
            raw_updated_fragments = [f._metadata for f in self.updated_fragments]
            return _Operation.delete(
                raw_updated_fragments, self.deleted_fragment_ids, self.predicate
            )

    @dataclass
    class Merge(BaseOperation):
        """
        Operation that adds columns. Unlike Overwrite, this should not change
        the structure of the fragments, allowing existing indices to be kept.

        Attributes
        ----------
        fragments: iterable of FragmentMetadata
            The fragments that make up the new dataset.
        schema: pyarrow.Schema
            The schema of the new dataset.

        Warning
        -------
        This is an advanced API for distributed operations. To overwrite or
        create new dataset on a single machine, use :func:`lance.write_dataset`.

        Examples
        --------

        To add new columns to a dataset, first define a method that will create
        the new columns based on the existing columns. Then use
        :meth:`lance.fragment.LanceFragment.add_columns`

        >>> import lance
        >>> import pyarrow as pa
        >>> import pyarrow.compute as pc
        >>> table = pa.table({"a": [1, 2, 3, 4], "b": ["a", "b", "c", "d"]})
        >>> dataset = lance.write_dataset(table, "example")
        >>> dataset.to_table().to_pandas()
           a  b
        0  1  a
        1  2  b
        2  3  c
        3  4  d
        >>> def double_a(batch: pa.RecordBatch) -> pa.RecordBatch:
        ...     doubled = pc.multiply(batch["a"], 2)
        ...     return pa.record_batch([doubled], ["a_doubled"])
        >>> fragments = []
        >>> for fragment in dataset.get_fragments():
        ...     new_fragment = fragment.add_columns(double_a, columns=['a'])
        ...     fragments.append(new_fragment)
        >>> new_schema = table.schema.append(pa.field("a_doubled", pa.int64()))
        >>> operation = lance.LanceOperation.Merge(fragments, new_schema)
        >>> dataset = lance.LanceDataset.commit("example", operation,
        ...                                     read_version=dataset.version)
        >>> dataset.to_table().to_pandas()
           a  b  a_doubled
        0  1  a          2
        1  2  b          4
        2  3  c          6
        3  4  d          8
        """

        fragments: Iterable[FragmentMetadata]
        schema: pa.Schema

        def __post_init__(self):
            LanceOperation._validate_fragments(self.fragments)

        def _to_inner(self):
            raw_fragments = [f._metadata for f in self.fragments]
            return _Operation.merge(raw_fragments, self.schema)

    @dataclass
    class Restore(BaseOperation):
        """
        Operation that restores a previous version of the dataset.
        """

        version: int

        def _to_inner(self):
            return _Operation.restore(self.version)


class ScannerBuilder:
    def __init__(self, ds: LanceDataset):
        self.ds = ds
        self._limit = 0
        self._filter = None
        self._substrait_filter = None
        self._prefilter = None
        self._offset = None
        self._columns = None
        self._nearest = None
        self._batch_size: Optional[int] = None
        self._batch_readahead: Optional[int] = None
        self._fragment_readahead: Optional[int] = None
        self._scan_in_order = True
        self._fragments = None
        self._with_row_id = False
        self._use_stats = True

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
        self._columns = cols
        return self

    def filter(self, filter: Union[str, pa.compute.Expression]) -> ScannerBuilder:
        if isinstance(filter, pa.compute.Expression):
            try:
                from pyarrow.substrait import serialize_expressions

                fields_without_lists = []
                counter = 0
                # Pyarrow cannot handle fixed size lists when converting
                # types to Substrait. So we can't use those in our filter,
                # which is ok for now but we need to replace them with some
                # kind of placeholder because Substrait is going to use
                # ordinal field references and we want to make sure those are
                # correct.
                for field in self.ds.schema:
                    if pa.types.is_fixed_size_list(field.type):
                        pos = counter
                        counter += 1
                        fields_without_lists.append(
                            pa.field(f"__unlikely_name_placeholder_{pos}", pa.int8())
                        )
                    else:
                        fields_without_lists.append(field)
                # Serialize the pyarrow compute expression toSubstrait and use
                # that as a filter.
                scalar_schema = pa.schema(fields_without_lists)
                self._substrait_filter = serialize_expressions(
                    [filter], ["my_filter"], scalar_schema
                )
            except ImportError:
                # serialize_expressions was introduced in pyarrow 14.  Fallback to
                # stringifying the expression if pyarrow is too old
                self._filter = str(filter)
        else:
            self._filter = filter
        return self

    def prefilter(self, prefilter: bool) -> ScannerBuilder:
        self._prefilter = prefilter
        return self

    def with_row_id(self, with_row_id: bool = True) -> ScannerBuilder:
        """Enable returns with physical row IDs."""
        self._with_row_id = with_row_id
        return self

    def use_stats(self, use_stats: bool = True) -> ScannerBuilder:
        """
        Enable use of statistics for query planning.

        Disabling statistics is used for debugging and benchmarking purposes.
        This should be left on for normal use.
        """
        self._use_stats = use_stats
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
        column: str,
        q: QueryVectorLike,
        k: Optional[int] = None,
        metric: Optional[str] = None,
        nprobes: Optional[int] = None,
        refine_factor: Optional[int] = None,
        use_index: bool = True,
    ) -> ScannerBuilder:
        q = _coerce_query_vector(q)

        if self.ds.schema.get_field_index(column) < 0:
            raise ValueError(f"Embedding column {column} is not in the dataset")

        column_field = self.ds.schema.field(column)
        column_type = column_field.type
        if hasattr(column_type, "storage_type"):
            column_type = column_type.storage_type
        if not pa.types.is_fixed_size_list(column_type):
            raise TypeError(
                f"Query column {column} must be a vector. Got {column_field.type}."
            )
        if len(q) != column_type.list_size:
            raise ValueError(
                f"Query vector size {len(q)} does not match index column size"
                f" {column_type.list_size}"
            )

        if k is not None and int(k) <= 0:
            raise ValueError(f"Nearest-K must be > 0 but got {k}")
        if nprobes is not None and int(nprobes) <= 0:
            raise ValueError(f"Nprobes must be > 0 but got {nprobes}")
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
            self._prefilter,
            self._limit,
            self._offset,
            self._nearest,
            self._batch_size,
            self._batch_readahead,
            self._fragment_readahead,
            self._scan_in_order,
            self._fragments,
            self._with_row_id,
            self._use_stats,
            self._substrait_filter,
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
        return self._scanner.count_rows()

    def explain_plan(self, verbose=False) -> str:
        """Return the execution plan for this scanner.

        Parameters
        ----------
        verbose : bool, default False
            Use a verbose output format.

        Returns
        -------
        plan : str
        """

        return self._scanner.explain_plan(verbose=verbose)


class DatasetOptimizer:
    def __init__(self, dataset: LanceDataset):
        self._dataset = dataset

    def compact_files(
        self,
        *,
        target_rows_per_fragment: int = 1024 * 1024,
        max_rows_per_group: int = 1024,
        materialize_deletions: bool = True,
        materialize_deletions_threshold: float = 0.1,
        num_threads: Optional[int] = None,
    ) -> CompactionMetrics:
        """Compacts small files in the dataset, reducing total number of files.

        This does a few things:
         * Removes deleted rows from fragments
         * Removes dropped columns from fragments
         * Merges small fragments into larger ones

        This method preserves the insertion order of the dataset. This may mean
        it leaves small fragments in the dataset if they are not adjacent to
        other fragments that need compaction. For example, if you have fragments
        with row counts 5 million, 100, and 5 million, the middle fragment will
        not be compacted because the fragments it is adjacent to do not need
        compaction.

        Parameters
        ----------
        target_rows_per_fragment: int, default 1024*1024
            The target number of rows per fragment. This is the number of rows
            that will be in each fragment after compaction.
        max_rows_per_group: int, default 1024
            Max number of rows per group. This does not affect which fragments
            need compaction, but does affect how they are re-written if selected.
        materialize_deletions: bool, default True
            Whether to compact fragments with soft deleted rows so they are no
            longer present in the file.
        materialize_deletions_threshold: float, default 0.1
            The fraction of original rows that are soft deleted in a fragment
            before the fragment is a candidate for compaction.
        num_threads: int, optional
            The number of threads to use when performing compaction. If not
            specified, defaults to the number of cores on the machine.

        Returns
        -------
        CompactionMetrics
            Metrics about the compaction process

        See Also
        --------
        lance.optimize.Compaction
        """
        opts = dict(
            target_rows_per_fragment=target_rows_per_fragment,
            max_rows_per_group=max_rows_per_group,
            materialize_deletions=materialize_deletions,
            materialize_deletions_threshold=materialize_deletions_threshold,
            num_threads=num_threads,
        )
        return Compaction.execute(self._dataset, opts)

    def optimize_indices(self, **kwargs):
        """Optimizes index performance.

        As new data arrives it is not added to existing indexes automatically.
        When searching we need to perform an indexed search of the old data plus
        an expensive unindexed search on the new data.  As the amount of new
        unindexed data grows this can have an impact on search latency.
        This function will add the new data to existing indexes, restoring the
        performance.  This function does not retrain the index, it only assigns
        the new data to existing partitions.  This means an update is much quicker
        than retraining the entire index but may have less accuracy (especially
        if the new data exhibits new patterns, concepts, or trends)
        """
        self._dataset._ds.optimize_indices(**kwargs)


class DatasetStats(TypedDict):
    num_deleted_rows: int
    num_fragments: int
    num_small_files: int


class LanceStats:
    """
    Statistics about a LanceDataset.
    """

    def __init__(self, dataset: _Dataset):
        self._ds = dataset

    def dataset_stats(self, max_rows_per_group: int = 1024) -> DatasetStats:
        """
        Statistics about the dataset.
        """
        return {
            "num_deleted_rows": self._ds.count_deleted_rows(),
            "num_fragments": self._ds.count_fragments(),
            "num_small_files": self._ds.num_small_files(max_rows_per_group),
        }

    def index_stats(self, index_name: str) -> Dict[str, Any]:
        """
        Statistics about an index.

        Parameters
        ----------
        index_name: str
            The name of the index to get statistics for.
        """
        index_stats = json.loads(self._ds.index_statistics(index_name))
        return index_stats


def write_dataset(
    data_obj: ReaderLike,
    uri: Union[str, Path],
    schema: Optional[pa.Schema] = None,
    mode: str = "create",
    *,
    max_rows_per_file: int = 1024 * 1024,
    max_rows_per_group: int = 1024,
    max_bytes_per_file: int = 90 * 1024 * 1024 * 1024,
    commit_lock: Optional[CommitLock] = None,
    progress: Optional[FragmentWriteProgress] = None,
) -> LanceDataset:
    """Write a given data_obj to the given uri

    Parameters
    ----------
    data_obj: Reader-like
        The data to be written. Acceptable types are:
        - Pandas DataFrame, Pyarrow Table, Dataset, Scanner, or RecordBatchReader
        - Huggingface dataset
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
    max_bytes_per_file: int, default 90 * 1024 * 1024 * 1024
        The max number of bytes to write before starting a new file. This is a
        soft limit. This limit is checked after each group is written, which
        means larger groups may cause this to be overshot meaningfully. This
        defaults to 90 GB, since we have a hard limit of 100 GB per file on
        object stores.
    commit_lock : CommitLock, optional
        A custom commit lock.  Only needed if your object store does not support
        atomic commits.  See the user guide for more details.
    progress: FragmentWriteProgress, optional
        *Experimental API*. Progress tracking for writing the fragment. Pass
        a custom class that defines hooks to be called when each fragment is
        starting to write and finishing writing.
    """
    if _check_for_hugging_face(data_obj):
        # Huggingface datasets
        from .dependencies import datasets

        if isinstance(data_obj, datasets.Dataset):
            if schema is None:
                schema = data_obj.features.arrow_schema
            data_obj = data_obj.data.to_batches()

    reader = _coerce_reader(data_obj, schema)
    _validate_schema(reader.schema)
    # TODO add support for passing in LanceDataset and LanceScanner here

    params = {
        "mode": mode,
        "max_rows_per_file": max_rows_per_file,
        "max_rows_per_group": max_rows_per_group,
        "max_bytes_per_file": max_bytes_per_file,
        "progress": progress,
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
    if _check_for_pandas(data_obj) and isinstance(data_obj, pd.DataFrame):
        return pa.Table.from_pandas(data_obj, schema=schema).to_reader()
    elif isinstance(data_obj, pa.Table):
        return data_obj.to_reader()
    elif isinstance(data_obj, pa.RecordBatch):
        return pa.Table.from_batches([data_obj]).to_reader()
    elif isinstance(data_obj, LanceDataset):
        return data_obj.scanner().to_reader()
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


def _coerce_query_vector(query: QueryVectorLike):
    if isinstance(query, pa.Scalar):
        if isinstance(query, pa.ExtensionScalar):
            # If it's an extension scalar then convert to storage
            query = query.value
        if isinstance(query.type, pa.FixedSizeListType):
            query = query.values
    elif isinstance(query, (list, tuple)) or (
        _check_for_numpy(query),
        isinstance(query, np.ndarray),
    ):
        query = np.array(query).astype("float64")  # workaround for GH-608
        query = pa.FloatingPointArray.from_pandas(query, type=pa.float32())
    elif not isinstance(query, pa.Array):
        try:
            query = pa.array(query)
        except:  # noqa: E722
            raise TypeError(
                "Query vectors should be an array of floats, "
                f"got {type(query)} which we cannot coerce to a "
                "float array"
            )

    # At this point `query` should be an arrow array
    if not isinstance(query, pa.FloatingPointArray):
        if pa.types.is_integer(query.type):
            query = query.cast(pa.float32())
        else:
            raise TypeError(
                "query vector must be list-like or pa.FloatingPointArray "
                f"but received {query.type}"
            )

    return query


def _validate_schema(schema: pa.Schema):
    """
    Make sure the metadata is valid utf8
    """
    if schema.metadata is not None:
        _validate_metadata(schema.metadata)


def _validate_metadata(metadata: dict):
    """
    Make sure the metadata values are valid utf8 (can be nested)

    Raises ValueError if not valid utf8
    """
    for k, v in metadata.items():
        if isinstance(v, bytes):
            try:
                v.decode("utf8")
            except UnicodeDecodeError:
                raise ValueError(
                    f"Metadata key {k} is not valid utf8. "
                    "Consider base64 encode for generic binary metadata."
                )
        elif isinstance(v, dict):
            _validate_metadata(v)


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


class BatchUDF:
    """A user-defined function that can be passed to :meth:`LanceDataset.add_columns`.

    Use :func:`lance.add_columns_udf` decorator to wrap a function with this class.
    """

    def __init__(self, func, output_schema=None, checkpoint_file=None):
        self.func = func
        self.output_schema = output_schema
        if checkpoint_file is not None:
            self.cache = BatchUDFCheckpoint(checkpoint_file)
        else:
            self.cache = None

    def __call__(self, batch: pa.RecordBatch):
        # Directly call inner function. This is to allow the user to test the
        # function and have it behave exactly as it was written.
        return self.func(batch)

    def _call(self, batch: pa.RecordBatch):
        if self.output_schema is None:
            raise ValueError(
                "output_schema must be provided when using a function that "
                "returns a RecordBatch"
            )
        result = self.func(batch)

        if _check_for_pandas(result):
            if isinstance(result, pd.DataFrame):
                result = pa.RecordBatch.from_pandas(result)
        assert result.schema == self.output_schema, (
            f"Output schema of function does not match the expected schema. "
            f"Expected:\n{self.output_schema}\nGot:\n{result.schema}"
        )
        return result


def batch_udf(output_schema=None, checkpoint_file=None):
    """
    Create a user defined function (UDF) that adds columns to a dataset.

    This function is used to add columns to a dataset. It takes a function that
    takes a single argument, a RecordBatch, and returns a RecordBatch. The
    function is called once for each batch in the dataset. The function should
    not modify the input batch, but instead create a new batch with the new
    columns added.

    Parameters
    ----------
    output_schema : Schema, optional
        The schema of the output RecordBatch. This is used to validate the
        output of the function. If not provided, the schema of the first output
        RecordBatch will be used.
    checkpoint_file : str or Path, optional
        If specified, this file will be used as a cache for unsaved results of
        this UDF. If the process fails, and you call add_columns again with this
        same file, it will resume from the last saved state. This is useful for
        long running processes that may fail and need to be resumed. This file
        may get very large. It will hold up to an entire data files' worth of
        results on disk, which can be multiple gigabytes of data.

    Returns
    -------
    AddColumnsUDF
    """

    def inner(func):
        return BatchUDF(func, output_schema, checkpoint_file)

    return inner


class BatchUDFCheckpoint:
    """A cache for BatchUDF results to avoid recomputation.

    This is backed by a SQLite database.
    """

    class BatchInfo(NamedTuple):
        fragment_id: int
        batch_index: int

    def __init__(self, path):
        self.path = path
        # We don't re-use the connection because it's not thread safe
        conn = sqlite3.connect(path)
        # One table to store the results for each batch.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS batches
            (fragment_id INT, batch_index INT, result BLOB)
            """
        )
        # One table to store fully written (but not committed) fragments.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS fragments (fragment_id INT, data BLOB)"
        )
        conn.commit()

    def cleanup(self):
        os.remove(self.path)

    def get_batch(self, info: BatchInfo) -> Optional[pa.RecordBatch]:
        conn = sqlite3.connect(self.path)
        cursor = conn.execute(
            "SELECT result FROM batches WHERE fragment_id = ? AND batch_index = ?",
            (info.fragment_id, info.batch_index),
        )
        row = cursor.fetchone()
        if row is not None:
            return pickle.loads(row[0])
        return None

    def insert_batch(self, info: BatchInfo, batch: pa.RecordBatch):
        conn = sqlite3.connect(self.path)
        conn.execute(
            "INSERT INTO batches (fragment_id, batch_index, result) VALUES (?, ?, ?)",
            (info.fragment_id, info.batch_index, pickle.dumps(batch)),
        )
        conn.commit()

    def get_fragment(self, fragment_id: int) -> Optional[str]:
        """Retrieves a fragment as a JSON string."""
        conn = sqlite3.connect(self.path)
        cursor = conn.execute(
            "SELECT data FROM fragments WHERE fragment_id = ?", (fragment_id,)
        )
        row = cursor.fetchone()
        if row is not None:
            return row[0]
        return None

    def insert_fragment(self, fragment_id: int, fragment: str):
        """Save a JSON string of a fragment to the cache."""
        # Clear all batches for the fragment
        conn = sqlite3.connect(self.path)
        conn.execute(
            "INSERT INTO fragments (fragment_id, data) VALUES (?, ?)",
            (fragment_id, fragment),
        )
        conn.execute("DELETE FROM batches WHERE fragment_id = ?", (fragment_id,))
        conn.commit()
