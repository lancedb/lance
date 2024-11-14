# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

import copy
import json
import logging
import os
import random
import time
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
    Optional,
    Set,
    TypedDict,
    Union,
)

import pyarrow as pa
import pyarrow.dataset
from pyarrow import RecordBatch, Schema

from .blob import BlobFile
from .dependencies import (
    _check_for_hugging_face,
    _check_for_numpy,
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
    _RewriteGroup,
    _RewrittenIndex,
    _Scanner,
    _write_dataset,
)
from .lance import CompactionMetrics as CompactionMetrics
from .lance import __version__ as __version__
from .lance import _Session as Session
from .optimize import Compaction
from .schema import LanceSchema
from .types import _coerce_reader
from .udf import BatchUDF, normalize_transform
from .udf import BatchUDFCheckpoint as BatchUDFCheckpoint
from .udf import batch_udf as batch_udf
from .util import td_to_micros

if TYPE_CHECKING:
    from pyarrow._compute import Expression

    from .commit import CommitLock
    from .progress import FragmentWriteProgress
    from .types import ReaderLike

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

        This function updates the original dataset and returns a dictionary with
        information about merge statistics - i.e. the number of inserted, updated,
        and deleted rows.

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

        return super(MergeInsertBuilder, self).execute(reader)

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
        version: Optional[int | str] = None,
        block_size: Optional[int] = None,
        index_cache_size: Optional[int] = None,
        metadata_cache_size: Optional[int] = None,
        commit_lock: Optional[CommitLock] = None,
        storage_options: Optional[Dict[str, str]] = None,
        serialized_manifest: Optional[bytes] = None,
        default_scan_options: Optional[Dict[str, Any]] = None,
    ):
        uri = os.fspath(uri) if isinstance(uri, Path) else uri
        self._uri = uri
        self._ds = _Dataset(
            uri,
            version,
            block_size,
            index_cache_size,
            metadata_cache_size,
            commit_lock,
            storage_options,
            serialized_manifest,
        )
        self._default_scan_options = default_scan_options

    @classmethod
    def __deserialize__(
        cls,
        uri: str,
        version: int,
        manifest: bytes,
        default_scan_options: Optional[Dict[str, Any]],
    ):
        return cls(
            uri,
            version,
            serialized_manifest=manifest,
            default_scan_options=default_scan_options,
        )

    def __reduce__(self):
        return type(self).__deserialize__, (
            self.uri,
            self._ds.version(),
            self._ds.serialized_manifest(),
            self._default_scan_options,
        )

    def __getstate__(self):
        return (
            self.uri,
            self._ds.version(),
            self._ds.serialized_manifest(),
            self._default_scan_options,
        )

    def __setstate__(self, state):
        self._uri, version, manifest, default_scan_options = state
        self._ds = _Dataset(
            self._uri,
            version,
            manifest=manifest,
            default_scan_options=default_scan_options,
        )

    def __copy__(self):
        ds = LanceDataset.__new__(LanceDataset)
        ds._uri = self._uri
        ds._ds = copy.copy(self._ds)
        ds._default_scan_options = self._default_scan_options
        return ds

    def __len__(self):
        return self.count_rows()

    @property
    def uri(self) -> str:
        """
        The location of the data
        """
        return self._uri

    @property
    def tags(self) -> Tags:
        return Tags(self._ds)

    def list_indices(self) -> List[Dict[str, Any]]:
        return self._ds.load_indices()

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

    def _apply_default_scan_options(self, builder: ScannerBuilder):
        if self._default_scan_options:
            builder.apply_defaults(self._default_scan_options)
        return builder

    def scanner(
        self,
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        filter: Optional[Union[str, pa.compute.Expression]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        nearest: Optional[dict] = None,
        batch_size: Optional[int] = None,
        batch_readahead: Optional[int] = None,
        fragment_readahead: Optional[int] = None,
        scan_in_order: bool = None,
        fragments: Optional[Iterable[LanceFragment]] = None,
        full_text_query: Optional[Union[str, dict]] = None,
        *,
        prefilter: bool = None,
        with_row_id: bool = None,
        with_row_address: bool = None,
        use_stats: bool = None,
        fast_search: bool = None,
        io_buffer_size: Optional[int] = None,
        late_materialization: Optional[bool | List[str]] = None,
        use_scalar_index: Optional[bool] = None,
    ) -> LanceScanner:
        """Return a Scanner that can support various pushdowns.

        Parameters
        ----------
        columns: list of str, or dict of str to str default None
            List of column names to be fetched.
            Or a dictionary of column names to SQL expressions.
            All columns are fetched if None or unspecified.
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
        io_buffer_size: int, default None
            The size of the IO buffer.  See ``ScannerBuilder.io_buffer_size``
            for more information.
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
        use_scalar_index: bool, default True
            Lance will automatically use scalar indices to optimize a query.  In some
            corner cases this can make query performance worse and this parameter can
            be used to disable scalar indices in these cases.
        late_materialization: bool or List[str], default None
            Allows custom control over late materialization.  Late materialization
            fetches non-query columns using a take operation after the filter.  This
            is useful when there are few results or columns are very large.

            Early materialization can be better when there are many results or the
            columns are very narrow.

            If True, then all columns are late materialized.
            If False, then all columns are early materialized.
            If a list of strings, then only the columns in the list are
              late materialized.

            The default uses a heuristic that assumes filters will select about 0.1%
            of the rows.  If your filter is more selective (e.g. find by id) you may
            want to set this to True.  If your filter is not very selective (e.g.
            matches 20% of the rows) you may want to set this to False.
        full_text_query: str or dict, optional
            query string to search for, the results will be ranked by BM25.
            e.g. "hello world", would match documents containing "hello" or "world".
            or a dictionary with the following keys:
            - columns: list[str]
                The columns to search,
                currently only supports a single column in the columns list.
            - query: str
                The query string to search for.
        fast_search:  bool, default False
            If True, then the search will only be performed on the indexed data, which
            yields faster search time.

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
        builder = ScannerBuilder(self)
        builder = self._apply_default_scan_options(builder)

        # Calls the setter if the user provided a non-None value
        # We need to avoid calling the setter with a None value so
        # we don't override any defaults from _default_scan_options
        def setopt(opt, val):
            if val is not None:
                opt(val)

        setopt(builder.filter, filter)
        setopt(builder.prefilter, prefilter)
        setopt(builder.limit, limit)
        setopt(builder.offset, offset)
        setopt(builder.batch_size, batch_size)
        setopt(builder.io_buffer_size, io_buffer_size)
        setopt(builder.batch_readahead, batch_readahead)
        setopt(builder.fragment_readahead, fragment_readahead)
        setopt(builder.scan_in_order, scan_in_order)
        setopt(builder.with_fragments, fragments)
        setopt(builder.late_materialization, late_materialization)
        setopt(builder.with_row_id, with_row_id)
        setopt(builder.with_row_address, with_row_address)
        setopt(builder.use_stats, use_stats)
        setopt(builder.use_scalar_index, use_scalar_index)
        setopt(builder.fast_search, fast_search)

        # columns=None has a special meaning. we can't treat it as "user didn't specify"
        if self._default_scan_options is None:
            # No defaults, use user-provided, if any
            builder = builder.columns(columns)
        else:
            default_columns = self._default_scan_options.get("columns", None)
            if default_columns is None:
                # No default_columns, use user-provided, if any
                builder = builder.columns(columns)
            else:
                if columns is not None:
                    # User supplied None, fallback to default (no way to override
                    # default to None)
                    builder = builder.columns(columns)
                else:
                    # User supplied non-None, use that
                    builder = builder.columns(default_columns)

        if full_text_query is not None:
            if isinstance(full_text_query, str):
                builder = builder.full_text_search(full_text_query)
            else:
                builder = builder.full_text_search(**full_text_query)
        if nearest is not None:
            builder = builder.nearest(**nearest)
        return builder.to_scanner()

    @property
    def schema(self) -> pa.Schema:
        """
        The pyarrow Schema for this dataset
        """
        if self._default_scan_options is None:
            return self._ds.schema
        else:
            return self.scanner().projected_schema

    @property
    def lance_schema(self) -> "LanceSchema":
        """
        The LanceSchema for this dataset
        """
        return self._ds.lance_schema

    @property
    def data_storage_version(self) -> str:
        """
        The version of the data storage format this dataset is using
        """
        return self._ds.data_storage_version

    def to_table(
        self,
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
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
        with_row_address: bool = False,
        use_stats: bool = True,
        fast_search: bool = False,
        full_text_query: Optional[Union[str, dict]] = None,
        io_buffer_size: Optional[int] = None,
        late_materialization: Optional[bool | List[str]] = None,
        use_scalar_index: Optional[bool] = None,
    ) -> pa.Table:
        """Read the data into memory as a pyarrow Table.

        Parameters
        ----------
        columns: list of str, or dict of str to str default None
            List of column names to be fetched.
            Or a dictionary of column names to SQL expressions.
            All columns are fetched if None or unspecified.
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
        io_buffer_size: int, default None
            The size of the IO buffer.  See ``ScannerBuilder.io_buffer_size``
            for more information.
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
        late_materialization: bool or List[str], default None
            Allows custom control over late materialization.  See
            ``ScannerBuilder.late_materialization`` for more information.
        use_scalar_index: bool, default True
            Allows custom control over scalar index usage.  See
            ``ScannerBuilder.use_scalar_index`` for more information.
        with_row_id: bool, default False
            Return row ID.
        with_row_address: bool, default False
            Return row address
        use_stats: bool, default True
            Use stats pushdown during filters.
        full_text_query: str or dict, optional
            query string to search for, the results will be ranked by BM25.
            e.g. "hello world", would match documents contains "hello" or "world".
            or a dictionary with the following keys:
            - columns: list[str]
                The columns to search,
                currently only supports a single column in the columns list.
            - query: str
                The query string to search for.

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
            io_buffer_size=io_buffer_size,
            batch_readahead=batch_readahead,
            fragment_readahead=fragment_readahead,
            late_materialization=late_materialization,
            use_scalar_index=use_scalar_index,
            scan_in_order=scan_in_order,
            prefilter=prefilter,
            with_row_id=with_row_id,
            with_row_address=with_row_address,
            use_stats=use_stats,
            fast_search=fast_search,
            full_text_query=full_text_query,
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
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
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
        with_row_address: bool = False,
        use_stats: bool = True,
        full_text_query: Optional[Union[str, dict]] = None,
        io_buffer_size: Optional[int] = None,
        late_materialization: Optional[bool | List[str]] = None,
        use_scalar_index: Optional[bool] = None,
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
            io_buffer_size=io_buffer_size,
            batch_readahead=batch_readahead,
            fragment_readahead=fragment_readahead,
            late_materialization=late_materialization,
            use_scalar_index=use_scalar_index,
            scan_in_order=scan_in_order,
            prefilter=prefilter,
            with_row_id=with_row_id,
            with_row_address=with_row_address,
            use_stats=use_stats,
            full_text_query=full_text_query,
        ).to_batches()

    def sample(
        self,
        num_rows: int,
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        randomize_order: bool = True,
        **kwargs,
    ) -> pa.Table:
        """Select a random sample of data

        Parameters
        ----------
        num_rows: int
            number of rows to retrieve
        columns: list of str, or dict of str to str default None
            List of column names to be fetched.
            Or a dictionary of column names to SQL expressions.
            All columns are fetched if None or unspecified.
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
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        **kwargs,
    ) -> pa.Table:
        """Select rows of data by index.

        Parameters
        ----------
        indices : Array or array-like
            indices of rows to select in the dataset.
        columns: list of str, or dict of str to str default None
            List of column names to be fetched.
            Or a dictionary of column names to SQL expressions.
            All columns are fetched if None or unspecified.
        **kwargs : dict, optional
            See scanner() method for full parameter description.

        Returns
        -------
        table : Table
        """
        columns_with_transform = None
        if isinstance(columns, dict):
            columns_with_transform = list(columns.items())
            columns = None
        return pa.Table.from_batches(
            [self._ds.take(indices, columns, columns_with_transform)]
        )

    def _take_rows(
        self,
        row_ids: Union[List[int], pa.Array],
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        **kargs,
    ) -> pa.Table:
        """Select rows by row_ids.

        **Unstable API**. Internal use only

        Parameters
        ----------
        row_ids : List Array or array-like
            row IDs to select in the dataset.
        columns: list of str, or dict of str to str default None
            List of column names to be fetched.
            Or a dictionary of column names to SQL expressions.
            All columns are fetched if None or unspecified.
        **kwargs : dict, optional
            See scanner() method for full parameter description.

        Returns
        -------
        table : Table
        """
        columns_with_transform = None
        if isinstance(columns, dict):
            columns_with_transform = list(columns.items())
            columns = None
        return pa.Table.from_batches(
            [self._ds.take_rows(row_ids, columns, columns_with_transform)]
        )

    def take_blobs(
        self,
        row_ids: Union[List[int], pa.Array],
        blob_column: str,
    ) -> List[BlobFile]:
        """
        Select blobs by row_ids.

        Parameters
        ----------
        row_ids : List Array or array-like
            row IDs to select in the dataset.
        blob_column : str
            The name of the blob column to select.

        Returns
        -------
        blob_files : List[BlobFile]
        """
        lance_blob_files = self._ds.take_blobs(row_ids, blob_column)
        return [BlobFile(lance_blob_file) for lance_blob_file in lance_blob_files]

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
        if isinstance(filter, pa.compute.Expression):
            # TODO: consolidate all to use scanner
            return self.scanner(filter=filter).count_rows()

        return self._ds.count_rows(filter)

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
        """Alter column name, data type, and nullability.

        Columns that are renamed can keep any indices that are on them. If a
        column has an IVF_PQ index, it can be kept if the column is casted to
        another type. However, other index types don't support casting at this
        time.

        Column types can be upcasted (such as int32 to int64) or downcasted
        (such as int64 to int32). However, downcasting will fail if there are
        any values that cannot be represented in the new type. In general,
        columns can be casted to same general type: integers to integers,
        floats to floats, and strings to strings. However, strings, binary, and
        list columns can be casted between their size variants. For example,
        string to large string, binary to large binary, and list to large list.

        Columns that are renamed can keep any indices that are on them. However, if
        the column is casted to a different type, it's indices will be dropped.

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
            - "data_type": pyarrow.DataType, optional
                The new data type to cast the column to. If not specified, the column
                data type is not changed.

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
        >>> dataset.alter_columns({"path": "x", "data_type": pa.int32()})
        >>> dataset.schema
        x: int32
        b: string
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
        transforms: Dict[str, str] | BatchUDF | ReaderLike,
        read_columns: List[str] | None = None,
        reader_schema: Optional[pa.Schema] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Add new columns with defined values.

        There are several ways to specify the new columns. First, you can provide
        SQL expressions for each new column. Second you can provide a UDF that
        takes a batch of existing data and returns a new batch with the new
        columns. These new columns will be appended to the dataset.

        You can also provide a RecordBatchReader which will read the new column
        values from some external source.  This is often useful when the new column
        values have already been staged to files (often by some distributed process)

        See the :func:`lance.add_columns_udf` decorator for more information on
        writing UDFs.

        Parameters
        ----------
        transforms : dict or AddColumnsUDF or ReaderLike
            If this is a dictionary, then the keys are the names of the new
            columns and the values are SQL expression strings. These strings can
            reference existing columns in the dataset.
            If this is a AddColumnsUDF, then it is a UDF that takes a batch of
            existing data and returns a new batch with the new columns.
        read_columns : list of str, optional
            The names of the columns that the UDF will read. If None, then the
            UDF will read all columns. This is only used when transforms is a
            UDF. Otherwise, the read columns are inferred from the SQL expressions.
        reader_schema: pa.Schema, optional
            Only valid if transforms is a `ReaderLike` object.  This will be used to
            determine the schema of the reader.
        batch_size: int, optional
            The number of rows to read at a time from the source dataset when applying
            the transform.  This is ignored if the dataset is a v1 dataset.

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
        transforms = normalize_transform(transforms, self, read_columns, reader_schema)
        if isinstance(transforms, pa.RecordBatchReader):
            self._ds.add_columns_from_reader(transforms, batch_size)
            return
        else:
            self._ds.add_columns(transforms, read_columns, batch_size)

            if isinstance(transforms, BatchUDF):
                if transforms.cache is not None:
                    transforms.cache.cleanup()

    def drop_columns(self, columns: List[str]):
        """Drop one or more columns from the dataset

        This is a metadata-only operation and does not remove the data from the
        underlying storage. In order to remove the data, you must subsequently
        call ``compact_files`` to rewrite the data without the removed columns and
        then call ``cleanup_old_versions`` to remove the old files.

        Parameters
        ----------
        columns : list of str
            The names of the columns to drop. These can be nested column references
            (e.g. "a.b.c") or top-level column names (e.g. "a").

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
        >>> dataset.merge_insert("a")     \\
        ...             .when_matched_update_all()     \\
        ...             .when_not_matched_insert_all() \\
        ...             .execute(new_table)
        {'num_inserted_rows': 1, 'num_updated_rows': 2, 'num_deleted_rows': 0}
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
    ) -> Dict[str, Any]:
        """
        Update column values for rows matching where.

        Parameters
        ----------
        updates : dict of str to str
            A mapping of column names to a SQL expression.
        where : str, optional
            A SQL predicate indicating which rows should be updated.

        Returns
        -------
        updates : dict
            A dictionary containing the number of rows updated.

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> table = pa.table({"a": [1, 2, 3], "b": ["a", "b", "c"]})
        >>> dataset = lance.write_dataset(table, "example")
        >>> update_stats = dataset.update(dict(a = 'a + 2'), where="b != 'a'")
        >>> update_stats["num_updated_rows"] = 2
        >>> dataset.to_table().to_pandas()
           a  b
        0  1  a
        1  4  b
        2  5  c
        """
        if isinstance(where, pa.compute.Expression):
            where = str(where)
        return self._ds.update(updates, where)

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

    def checkout_version(self, version: int | str) -> "LanceDataset":
        """
        Load the given version of the dataset.

        Unlike the :func:`dataset` constructor, this will re-use the
        current cache.
        This is a no-op if the dataset is already at the given version.

        Parameters
        ----------
        version: int | str,
            The version to check out. A version number (`int`) or a tag
            (`str`) can be provided.

        Returns
        -------
        LanceDataset
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
        error_if_tagged_old_versions: bool = True,
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

        error_if_tagged_old_versions: bool, default True
            Some versions may have tags associated with them. Tagged versions will
            not be cleaned up, regardless of how old they are. If this argument
            is set to `True` (the default), an exception will be raised if any
            tagged versions match the parameters. Otherwise, tagged versions will
            be ignored without any error and only untagged versions will be
            cleaned up.
        """
        if older_than is None:
            older_than = timedelta(days=14)
        return self._ds.cleanup_old_versions(
            td_to_micros(older_than), delete_unverified, error_if_tagged_old_versions
        )

    def create_scalar_index(
        self,
        column: str,
        index_type: Union[
            Literal["BTREE"],
            Literal["BITMAP"],
            Literal["LABEL_LIST"],
            Literal["INVERTED"],
            Literal["FTS"],
        ],
        name: Optional[str] = None,
        *,
        replace: bool = True,
        **kwargs,
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

        Vector search with pre-filers can also benefit from scalar indices. For example,

        .. code-block:: python

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


        There are 4 types of scalar indices available today.

        * ``BTREE``. The most common type is ``BTREE``. This index is inspired
          by the btree data structure although only the first few layers of the btree
          are cached in memory.  It will
          perform well on columns with a large number of unique values and few rows per
          value.
        * ``BITMAP``. This index stores a bitmap for each unique value in the column.
          This index is useful for columns with a small number of unique values and
          many rows per value.
        * ``LABEL_LIST``. A special index that is used to index list
          columns whose values have small cardinality.  For example, a column that
          contains lists of tags (e.g. ``["tag1", "tag2", "tag3"]``) can be indexed
          with a ``LABEL_LIST`` index.  This index can only speedup queries with
          ``array_has_any`` or ``array_has_all`` filters.
        * ``FTS/INVERTED``. It is used to index document columns. This index
          can conduct full-text searches. For example, a column that contains any word
          of query string "hello world". The results will be ranked by BM25.

        Note that the ``LANCE_BYPASS_SPILLING`` environment variable can be used to
        bypass spilling to disk. Setting this to true can avoid memory exhaustion
        issues (see https://github.com/apache/datafusion/issues/10073 for more info).

        **Experimental API**

        Parameters
        ----------
        column : str
            The column to be indexed.  Must be a boolean, integer, float,
            or string column.
        index_type : str
            The type of the index.  One of ``"BTREE"``, ``"BITMAP"``,
            ``"LABEL_LIST"``, "FTS" or ``"INVERTED"``.
        name : str, optional
            The index name. If not provided, it will be generated from the
            column name.
        replace : bool, default True
            Replace the existing index if it exists.

        Optional Parameters
        -------------------
        with_position: bool, default True
            This is for the ``INVERTED`` index. If True, the index will store the
            positions of the words in the document, so that you can conduct phrase
            query. This will significantly increase the index size.
            It won't impact the performance of non-phrase queries even if it is set to
            True.
        base_tokenizer: str, default "simple"
            This is for the ``INVERTED`` index. The base tokenizer to use. The value
            can be:
            * "simple": splits tokens on whitespace and punctuation.
            * "whitespace": splits tokens on whitespace.
            * "raw": no tokenization.
        language: str, default "English"
            This is for the ``INVERTED`` index. The language for stemming
            and stop words. This is only used when `stem` or `remove_stop_words` is true
        max_token_length: Optional[int], default 40
            This is for the ``INVERTED`` index. The maximum token length.
            Any token longer than this will be removed.
        lower_case: bool, default True
            This is for the ``INVERTED`` index. If True, the index will convert all
            text to lowercase.
        stem: bool, default False
            This is for the ``INVERTED`` index. If True, the index will stem the
            tokens.
        remove_stop_words: bool, default False
            This is for the ``INVERTED`` index. If True, the index will remove
            stop words.
        ascii_folding: bool, default False
            This is for the ``INVERTED`` index. If True, the index will convert
            non-ascii characters to ascii characters if possible.
            This would remove accents like "é" -> "e".

        Examples
        --------

        .. code-block:: python

            import lance

            dataset = lance.dataset("/tmp/images.lance")
            dataset.create_index(
                "category",
                "BTREE",
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

        index_type = index_type.upper()
        if index_type not in ["BTREE", "BITMAP", "LABEL_LIST", "INVERTED"]:
            raise NotImplementedError(
                (
                    'Only "BTREE", "LABEL_LIST", "INVERTED", '
                    'or "BITMAP" are supported for '
                    f"scalar columns.  Received {index_type}",
                )
            )

        field = self.schema.field(column)
        if index_type in ["BTREE", "BITMAP"]:
            if (
                not pa.types.is_integer(field.type)
                and not pa.types.is_floating(field.type)
                and not pa.types.is_boolean(field.type)
                and not pa.types.is_string(field.type)
                and not pa.types.is_temporal(field.type)
            ):
                raise TypeError(
                    f"BTREE/BITMAP index column {column} must be int",
                    ", float, bool, str, or temporal",
                )
        elif index_type == "LABEL_LIST":
            if not pa.types.is_list(field.type):
                raise TypeError(f"LABEL_LIST index column {column} must be a list")
        elif index_type in ["INVERTED", "FTS"]:
            if not pa.types.is_string(field.type) and not pa.types.is_large_string(
                field.type
            ):
                raise TypeError(
                    f"INVERTED index column {column} must be string or large string"
                )

        if pa.types.is_duration(field.type):
            raise TypeError(
                f"Scalar index column {column} cannot currently be a duration"
            )

        self._ds.create_index([column], index_type, name, replace, None, kwargs)

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
        # experimental parameters
        ivf_centroids_file: Optional[str] = None,
        precomputed_partition_dataset: Optional[str] = None,
        storage_options: Optional[Dict[str, str]] = None,
        filter_nan: bool = True,
        one_pass_ivfpq: bool = False,
        **kwargs,
    ) -> LanceDataset:
        """Create index on column.

        **Experimental API**

        Parameters
        ----------
        column : str
            The column to be indexed.
        index_type : str
            The type of the index.
            ``"IVF_PQ, IVF_HNSW_PQ and IVF_HNSW_SQ"`` are supported now.
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
        storage_options : optional, dict
            Extra options that make sense for a particular storage connection. This is
            used to store connection parameters like credentials, endpoint, etc.
        filter_nan: bool
            Defaults to True. False is UNSAFE, and will cause a crash if any null/nan
            values are present (and otherwise will not). Disables the null filter used
            for nullable columns. Obtains a small speed boost.
        one_pass_ivfpq: bool
            Defaults to False. If enabled, index type must be "IVF_PQ". Reduces disk IO.
        kwargs :
            Parameters passed to the index building process.

        The SQ (Scalar Quantization) is available for only "IVF_HNSW_SQ" index type,
        this quantization method is used to reduce the memory usage of the index,
        it maps the float vectors to integer vectors, each integer is of ``num_bits``,
        now only 8 bits are supported.

        If ``index_type`` is "IVF_*", then the following parameters are required:
            num_partitions

        If ``index_type`` is with "PQ", then the following parameters are required:
            num_sub_vectors

        Optional parameters for "IVF_PQ":
            ivf_centroids :
                K-mean centroids for IVF clustering.

        Optional parameters for "IVF_HNSW_*":
            max_level : int
                the maximum number of levels in the graph.
            m : int
                the number of edges per node in the graph.
            ef_construction : int
                the number of nodes to examine during the construction.

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

        .. code-block:: python

            import lance

            dataset = lance.dataset("/tmp/sift.lance")
            dataset.create_index(
                "vector",
                "IVF_HNSW_SQ",
                num_partitions=256,
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
            if pa.types.is_fixed_size_list(field.type):
                dimension = field.type.list_size
            elif (
                isinstance(field.type, pa.FixedShapeTensorType)
                and len(field.type.shape) == 1
            ):
                dimension = field.type.shape[0]
            else:
                raise TypeError(
                    f"Vector column {c} must be FixedSizeListArray "
                    f"1-dimensional FixedShapeTensorArray, got {field.type}"
                )

            if num_sub_vectors is not None and dimension % num_sub_vectors != 0:
                raise ValueError(
                    f"dimension ({dimension}) must be divisible by num_sub_vectors"
                    f" ({num_sub_vectors})"
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
        valid_index_types = ["IVF_PQ", "IVF_HNSW_PQ", "IVF_HNSW_SQ"]
        if index_type not in valid_index_types:
            raise NotImplementedError(
                f"Only {valid_index_types} index types supported. " f"Got {index_type}"
            )
        if index_type != "IVF_PQ" and one_pass_ivfpq:
            raise ValueError(
                f'one_pass_ivfpq requires index_type="IVF_PQ", got {index_type}'
            )

        # Handle timing for various parts of accelerated builds
        timers = {}
        if one_pass_ivfpq and accelerator is not None:
            from .vector import (
                one_pass_assign_ivf_pq_on_accelerator,
                one_pass_train_ivf_pq_on_accelerator,
            )

            logging.info("Doing one-pass ivfpq accelerated computations")

            timers["ivf+pq_train:start"] = time.time()
            (
                ivf_centroids,
                ivf_kmeans,
                pq_codebook,
                pq_kmeans_list,
            ) = one_pass_train_ivf_pq_on_accelerator(
                self,
                column[0],
                num_partitions,
                metric,
                accelerator,
                num_sub_vectors=num_sub_vectors,
                batch_size=20480,
                filter_nan=filter_nan,
            )
            timers["ivf+pq_train:end"] = time.time()
            ivfpq_train_time = timers["ivf+pq_train:end"] - timers["ivf+pq_train:start"]
            logging.info("ivf+pq training time: %ss", ivfpq_train_time)
            timers["ivf+pq_assign:start"] = time.time()
            shuffle_output_dir, shuffle_buffers = one_pass_assign_ivf_pq_on_accelerator(
                self,
                column[0],
                metric,
                accelerator,
                ivf_kmeans,
                pq_kmeans_list,
                batch_size=20480,
                filter_nan=filter_nan,
            )
            timers["ivf+pq_assign:end"] = time.time()
            ivfpq_assign_time = (
                timers["ivf+pq_assign:end"] - timers["ivf+pq_assign:start"]
            )
            logging.info("ivf+pq transform time: %ss", ivfpq_assign_time)

            kwargs["precomputed_shuffle_buffers"] = shuffle_buffers
            kwargs["precomputed_shuffle_buffers_path"] = os.path.join(
                shuffle_output_dir, "data"
            )
        if index_type.startswith("IVF"):
            if (ivf_centroids is not None) and (ivf_centroids_file is not None):
                raise ValueError(
                    "ivf_centroids and ivf_centroids_file"
                    " cannot be provided at the same time"
                )

            if ivf_centroids_file is not None:
                from pyarrow.fs import FileSystem

                fs, path = FileSystem.from_uri(ivf_centroids_file)
                with fs.open_input_file(path) as f:
                    ivf_centroids = np.load(f)
                num_partitions = ivf_centroids.shape[0]

            if num_partitions is None:
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

            if (precomputed_partition_dataset is not None) and (ivf_centroids is None):
                raise ValueError(
                    "ivf_centroids must be provided when"
                    " precomputed_partition_dataset is provided"
                )
            if precomputed_partition_dataset is not None:
                logging.info("Using provided precomputed partition dataset")
                precomputed_ds = LanceDataset(
                    precomputed_partition_dataset, storage_options=storage_options
                )
                if not (
                    "PQ" in index_type
                    and pq_codebook is None
                    and accelerator is not None
                    and "precomputed_partitions_file" in kwargs
                ):
                    # In this case, the precomputed partitions file would be used
                    # without being turned into a set of precomputed buffers, so it
                    # needs to have a very specific format
                    if len(precomputed_ds.get_fragments()) != 1:
                        raise ValueError(
                            "precomputed_partition_dataset must have only one fragment"
                        )
                    files = precomputed_ds.get_fragments()[0].data_files()
                    if len(files) != 1:
                        raise ValueError(
                            "precomputed_partition_dataset must have only one files"
                        )
                kwargs["precomputed_partitions_file"] = precomputed_partition_dataset

            if accelerator is not None and ivf_centroids is None and not one_pass_ivfpq:
                logging.info("Computing new precomputed partition dataset")
                # Use accelerator to train ivf centroids
                from .vector import (
                    compute_partitions,
                    train_ivf_centroids_on_accelerator,
                )

                timers["ivf_train:start"] = time.time()
                ivf_centroids, kmeans = train_ivf_centroids_on_accelerator(
                    self,
                    column[0],
                    num_partitions,
                    metric,
                    accelerator,
                    filter_nan=filter_nan,
                )
                timers["ivf_train:end"] = time.time()
                ivf_train_time = timers["ivf_train:end"] - timers["ivf_train:start"]
                logging.info("ivf training time: %ss", ivf_train_time)
                timers["ivf_assign:start"] = time.time()
                num_sub_vectors_cur = None
                if "PQ" in index_type and pq_codebook is None:
                    # compute residual subspace columns in the same pass
                    num_sub_vectors_cur = num_sub_vectors
                partitions_file = compute_partitions(
                    self,
                    column[0],
                    kmeans,
                    batch_size=20480,
                    num_sub_vectors=num_sub_vectors_cur,
                    filter_nan=filter_nan,
                )
                timers["ivf_assign:end"] = time.time()
                ivf_assign_time = timers["ivf_assign:end"] - timers["ivf_assign:start"]
                logging.info("ivf transform time: %ss", ivf_assign_time)
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

        if "PQ" in index_type:
            if num_sub_vectors is None:
                raise ValueError(
                    "num_partitions and num_sub_vectors are required for IVF_PQ"
                )
            kwargs["num_sub_vectors"] = num_sub_vectors

            if (
                pq_codebook is None
                and accelerator is not None
                and "precomputed_partitions_file" in kwargs
                and not one_pass_ivfpq
            ):
                logging.info("Computing new precomputed shuffle buffers for PQ.")
                partitions_file = kwargs["precomputed_partitions_file"]
                del kwargs["precomputed_partitions_file"]

                partitions_ds = LanceDataset(partitions_file)
                # Use accelerator to train pq codebook
                from .vector import (
                    compute_pq_codes,
                    train_pq_codebook_on_accelerator,
                )

                timers["pq_train:start"] = time.time()
                pq_codebook, kmeans_list = train_pq_codebook_on_accelerator(
                    partitions_ds,
                    metric,
                    accelerator=accelerator,
                    num_sub_vectors=num_sub_vectors,
                )
                timers["pq_train:end"] = time.time()
                pq_train_time = timers["pq_train:end"] - timers["pq_train:start"]
                logging.info("pq training time: %ss", pq_train_time)
                timers["pq_assign:start"] = time.time()
                shuffle_output_dir, shuffle_buffers = compute_pq_codes(
                    partitions_ds,
                    kmeans_list,
                    batch_size=20480,
                )
                timers["pq_assign:end"] = time.time()
                pq_assign_time = timers["pq_assign:end"] - timers["pq_assign:start"]
                logging.info("pq transform time: %ss", pq_assign_time)
                # Save disk space
                if precomputed_partition_dataset is not None and os.path.exists(
                    partitions_file
                ):
                    logging.info(
                        "Temporary partitions file stored at %s,"
                        "you may want to delete it.",
                        partitions_file,
                    )

                kwargs["precomputed_shuffle_buffers"] = shuffle_buffers
                kwargs["precomputed_shuffle_buffers_path"] = os.path.join(
                    shuffle_output_dir, "data"
                )

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
                        values, pq_codebook.shape[2]
                    )
                pq_codebook_batch = pa.RecordBatch.from_arrays(
                    [pq_codebook], ["_pq_codebook"]
                )
                kwargs["pq_codebook"] = pq_codebook_batch

        if shuffle_partition_batches is not None:
            kwargs["shuffle_partition_batches"] = shuffle_partition_batches
        if shuffle_partition_concurrency is not None:
            kwargs["shuffle_partition_concurrency"] = shuffle_partition_concurrency

        timers["final_create_index:start"] = time.time()
        self._ds.create_index(
            column, index_type, name, replace, storage_options, kwargs
        )
        timers["final_create_index:end"] = time.time()
        final_create_index_time = (
            timers["final_create_index:end"] - timers["final_create_index:start"]
        )
        logging.info("Final create_index rust time: %ss", final_create_index_time)
        # Save disk space
        if "precomputed_shuffle_buffers_path" in kwargs.keys() and os.path.exists(
            kwargs["precomputed_shuffle_buffers_path"]
        ):
            logging.info(
                "Temporary shuffle buffers stored at %s, you may want to delete it.",
                kwargs["precomputed_shuffle_buffers_path"],
            )
        return self

    def session(self) -> Session:
        """
        Return the dataset session, which holds the dataset's state.
        """
        return self._ds.session()

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
        storage_options: Optional[Dict[str, str]] = None,
        enable_v2_manifest_paths: Optional[bool] = None,
        detached: Optional[bool] = False,
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
        storage_options : optional, dict
            Extra options that make sense for a particular storage connection. This is
            used to store connection parameters like credentials, endpoint, etc.
        enable_v2_manifest_paths : bool, optional
            If True, and this is a new dataset, uses the new V2 manifest paths.
            These paths provide more efficient opening of datasets with many
            versions on object stores. This parameter has no effect if the dataset
            already exists. To migrate an existing dataset, instead use the
            :meth:`migrate_manifest_paths_v2` method. Default is False. WARNING:
            turning this on will make the dataset unreadable for older versions
            of Lance (prior to 0.17.0).
        detached : bool, optional
            If True, then the commit will not be part of the dataset lineage.  It will
            never show up as the latest dataset and the only way to check it out in the
            future will be to specifically check it out by version.  The version will be
            a random version that is only unique amongst detached commits.  The caller
            should store this somewhere as there will be no other way to obtain it in
            the future.

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

        new_ds = _Dataset.commit(
            base_uri,
            operation._to_inner(),
            read_version,
            commit_lock,
            storage_options=storage_options,
            enable_v2_manifest_paths=enable_v2_manifest_paths,
            detached=detached,
        )
        return LanceDataset(
            base_uri, version=new_ds.version(), storage_options=storage_options
        )

    def validate(self):
        """
        Validate the dataset.

        This checks the integrity of the dataset and will raise an exception if
        the dataset is corrupted.
        """
        self._ds.validate()

    def migrate_manifest_paths_v2(self):
        """
        Migrate the manifest paths to the new format.

        This will update the manifest to use the new v2 format for paths.

        This function is idempotent, and can be run multiple times without
        changing the state of the object store.

        DANGER: this should not be run while other concurrent operations are happening.
        And it should also run until completion before resuming other operations.
        """
        self._ds.migrate_manifest_paths_v2()

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
        schema: LanceSchema or pyarrow.Schema
            The schema of the new dataset. Passing a LanceSchema is preferred,
            and passing a pyarrow.Schema is deprecated.

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
        ...     new_fragment, new_schema = fragment.merge_columns(double_a,
        ...                                                       columns=['a'])
        ...     fragments.append(new_fragment)
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
        schema: LanceSchema | pa.Schema

        def __post_init__(self):
            LanceOperation._validate_fragments(self.fragments)

        def _to_inner(self):
            raw_fragments = [f._metadata for f in self.fragments]
            if isinstance(self.schema, pa.Schema):
                warnings.warn(
                    "Passing a pyarrow.Schema to Merge is deprecated. "
                    "Please use a LanceSchema instead.",
                    DeprecationWarning,
                )
                self.schema = LanceSchema.from_pyarrow(self.schema)
            return _Operation.merge(raw_fragments, self.schema)

    @dataclass
    class Restore(BaseOperation):
        """
        Operation that restores a previous version of the dataset.
        """

        version: int

        def _to_inner(self):
            return _Operation.restore(self.version)

    @dataclass
    class RewriteGroup:
        """
        Collection of rewritten files
        """

        old_fragments: Iterable[FragmentMetadata]
        new_fragments: Iterable[FragmentMetadata]

        def _to_inner(self):
            old_fragments = [f._metadata for f in self.old_fragments]
            new_fragments = [f._metadata for f in self.new_fragments]
            return _RewriteGroup(old_fragments, new_fragments)

    @dataclass
    class RewrittenIndex:
        """
        An index that has been rewritten
        """

        old_id: str
        new_id: str

        def _to_inner(self):
            return _RewrittenIndex(self.old_id, self.new_id)

    @dataclass
    class Rewrite(BaseOperation):
        """
        Operation that rewrites one or more files and indices into one
        or more files and indices.

        Attributes
        ----------
        groups: list[RewriteGroup]
            Groups of files that have been rewritten.
        rewritten_indices: list[RewrittenIndex]
            Indices that have been rewritten.

        Warning
        -------
        This is an advanced API not intended for general use.
        """

        groups: Iterable[LanceOperation.RewriteGroup]
        rewritten_indices: Iterable[LanceOperation.RewrittenIndex]

        def __post_init__(self):
            all_frags = [old for group in self.groups for old in group.old_fragments]
            all_frags += [new for group in self.groups for new in group.new_fragments]
            LanceOperation._validate_fragments(all_frags)

        def _to_inner(self):
            groups = [group._to_inner() for group in self.groups]
            rewritten_indices = [index._to_inner() for index in self.rewritten_indices]
            return _Operation.rewrite(groups, rewritten_indices)

    @dataclass
    class CreateIndex(BaseOperation):
        """
        Operation that creates an index on the dataset.
        """

        uuid: str
        name: str
        fields: List[int]
        dataset_version: int
        fragment_ids: Set[int]

        def _to_inner(self):
            return _Operation.create_index(
                self.uuid,
                self.name,
                self.fields,
                self.dataset_version,
                self.fragment_ids,
            )


class ScannerBuilder:
    def __init__(self, ds: LanceDataset):
        self.ds = ds
        self._limit = None
        self._filter = None
        self._substrait_filter = None
        self._prefilter = False
        self._late_materialization = None
        self._offset = None
        self._columns = None
        self._columns_with_transform = None
        self._nearest = None
        self._batch_size: Optional[int] = None
        self._io_buffer_size: Optional[int] = None
        self._batch_readahead: Optional[int] = None
        self._fragment_readahead: Optional[int] = None
        self._scan_in_order = True
        self._fragments = None
        self._with_row_id = False
        self._with_row_address = False
        self._use_stats = True
        self._fast_search = False
        self._full_text_query = None
        self._use_scalar_index = None

    def apply_defaults(self, default_opts: Dict[str, Any]) -> ScannerBuilder:
        for key, value in default_opts.items():
            setter = getattr(self, key, None)
            if setter is None:
                raise ValueError(f"Unknown option {key}")
            setter(value)

    def batch_size(self, batch_size: int) -> ScannerBuilder:
        """Set batch size for Scanner"""
        self._batch_size = batch_size
        return self

    def io_buffer_size(self, io_buffer_size: int) -> ScannerBuilder:
        """
        Set the I/O buffer size for the Scanner

        This is the amount of RAM that will be reserved for holding I/O received from
        storage before it is processed.  This is used to control the amount of memory
        used by the scanner.  If the buffer is full then the scanner will block until
        the buffer is processed.

        Generally this should scale with the number of concurrent I/O threads.  The
        default is 2GiB which comfortably provides enough space for somewhere between
        32 and 256 concurrent I/O threads.

        This value is not a hard cap on the amount of RAM the scanner will use.  Some
        space is used for the compute (which can be controlled by the batch size) and
        Lance does not keep track of memory after it is returned to the user.

        Currently, if there is a single batch of data which is larger than the io buffer
        size then the scanner will deadlock.  This is a known issue and will be fixed in
        a future release.

        This parameter is only used when reading v2 files
        """
        self._io_buffer_size = io_buffer_size
        return self

    def batch_readahead(self, nbatches: Optional[int] = None) -> ScannerBuilder:
        """
        This parameter is ignored when reading v2 files
        """
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

        This parameter is ignored when using v2 files.  In the v2 file format
        there is no penalty to scanning in order and so all scans will scan in
        order.
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

    def columns(
        self, cols: Optional[Union[List[str], Dict[str, str]]] = None
    ) -> ScannerBuilder:
        if cols is None:
            self._columns = None
        elif isinstance(cols, dict):
            self._columns_with_transform = list(cols.items())
        elif isinstance(cols, list):
            self._columns = cols
        else:
            raise TypeError(
                f"columns must be a list or dict[name, expression], got {type(cols)}"
            )
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
        """Enable returns with row IDs."""
        self._with_row_id = with_row_id
        return self

    def with_row_address(self, with_row_address: bool = True) -> ScannerBuilder:
        """
        Enables returns with row addresses.

        Row addresses are a unique but unstable identifier for each row in the
        dataset that consists of the fragment id (upper 32 bits) and the row
        offset in the fragment (lower 32 bits).  Row IDs are generally preferred
        since they do not change when a row is modified or compacted.  However,
        row addresses may be useful in some advanced use cases.
        """
        self._with_row_address = with_row_address
        return self

    def late_materialization(
        self, late_materialization: bool | List[str]
    ) -> ScannerBuilder:
        self._late_materialization = late_materialization
        return self

    def use_stats(self, use_stats: bool = True) -> ScannerBuilder:
        """
        Enable use of statistics for query planning.

        Disabling statistics is used for debugging and benchmarking purposes.
        This should be left on for normal use.
        """
        self._use_stats = use_stats
        return self

    def use_scalar_index(self, use_scalar_index: bool = True) -> ScannerBuilder:
        """
        Set whether scalar indices should be used in a query

        Scans will use scalar indices, when available, to optimize queries with filters.
        However, in some corner cases, scalar indices may make performance worse.  This
        parameter allows users to disable scalar indices in these cases.
        """
        self._use_scalar_index = use_scalar_index
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
        ef: Optional[int] = None,
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
        if ef is not None and int(ef) <= 0:
            # `ef` should be >= `k`, but `k` could be None so we can't check it here
            # the rust code will check it
            raise ValueError(f"ef must be > 0 but got {ef}")
        self._nearest = {
            "column": column,
            "q": q,
            "k": k,
            "metric": metric,
            "nprobes": nprobes,
            "refine_factor": refine_factor,
            "use_index": use_index,
            "ef": ef,
        }
        return self

    def fast_search(self, flag: bool) -> ScannerBuilder:
        """Enable fast search, which only perform search on the indexed data.

        Users can use `Table::optimize()` or `create_index()` to include the new data
        into index, thus make new data searchable.
        """
        self._fast_search = flag
        return self

    def full_text_search(
        self,
        query: str,
        columns: Optional[List[str]] = None,
    ) -> ScannerBuilder:
        """
        Filter rows by full text searching. *Experimental API*,
        may remove it after we support to do this within `filter` SQL-like expression

        Must create inverted index on the given column before searching,
        """
        self._full_text_query = {"query": query, "columns": columns}
        return self

    def to_scanner(self) -> LanceScanner:
        scanner = self.ds._ds.scanner(
            self._columns,
            self._columns_with_transform,
            self._filter,
            self._prefilter,
            self._limit,
            self._offset,
            self._nearest,
            self._batch_size,
            self._io_buffer_size,
            self._batch_readahead,
            self._fragment_readahead,
            self._scan_in_order,
            self._fragments,
            self._with_row_id,
            self._with_row_address,
            self._use_stats,
            self._substrait_filter,
            self._fast_search,
            self._full_text_query,
            self._late_materialization,
            self._use_scalar_index,
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
        max_bytes_per_file: Optional[int] = None,
        materialize_deletions: bool = True,
        materialize_deletions_threshold: float = 0.1,
        num_threads: Optional[int] = None,
        batch_size: Optional[int] = None,
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

            This setting only affects datasets using the legacy storage format.
            The newer format does not require row groups.
        max_bytes_per_file: Optional[int], default None
            Max number of bytes in a single file.  This does not affect which
            fragments need compaction, but does affect how they are re-written if
            selected.  If this value is too small you may end up with fragments
            that are smaller than `target_rows_per_fragment`.

            The default will use the default from ``write_dataset``.
        materialize_deletions: bool, default True
            Whether to compact fragments with soft deleted rows so they are no
            longer present in the file.
        materialize_deletions_threshold: float, default 0.1
            The fraction of original rows that are soft deleted in a fragment
            before the fragment is a candidate for compaction.
        num_threads: int, optional
            The number of threads to use when performing compaction. If not
            specified, defaults to the number of cores on the machine.
        batch_size: int, optional
            The batch size to use when scanning input fragments.  You may want
            to reduce this if you are running out of memory during compaction.

            The default will use the same default from ``scanner``.

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
            max_bytes_per_file=max_bytes_per_file,
            materialize_deletions=materialize_deletions,
            materialize_deletions_threshold=materialize_deletions_threshold,
            num_threads=num_threads,
            batch_size=batch_size,
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

        Parameters
        ----------
        num_indices_to_merge: int, default 1
            The number of indices to merge.
            If set to 0, new delta index will be created.
        index_names: List[str], default None
            The names of the indices to optimize.
            If None, all indices will be optimized.
        """
        self._dataset._ds.optimize_indices(**kwargs)


class Tags:
    """
    Dataset tag manager.
    """

    def __init__(self, dataset: _Dataset):
        self._ds = dataset

    def list(self) -> dict[str, int]:
        """
        List all dataset tags.

        Returns
        -------
        dict[str, int]
            A dictionary mapping tag names to version numbers.
        """
        return self._ds.tags()

    def create(self, tag: str, version: int) -> None:
        """
        Create a tag for a given dataset version.

        Parameters
        ----------
        tag: str,
            The name of the tag to create. This name must be unique among all tag
            names for the dataset.
        version: int,
            The dataset version to tag.
        """
        self._ds.create_tag(tag, version)

    def delete(self, tag: str) -> None:
        """
        Delete tag from the dataset.

        Parameters
        ----------
        tag: str,
            The name of the tag to delete.

        """
        self._ds.delete_tag(tag)

    def update(self, tag: str, version: int) -> None:
        """
        Update tag to a new version.

        Parameters
        ----------
        tag: str,
            The name of the tag to update.
        version: int,
            The new dataset version to tag.
        """
        self._ds.update_tag(tag, version)


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
    storage_options: Optional[Dict[str, str]] = None,
    data_storage_version: Optional[str] = None,
    use_legacy_format: Optional[bool] = None,
    enable_v2_manifest_paths: bool = False,
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
    storage_options : optional, dict
        Extra options that make sense for a particular storage connection. This is
        used to store connection parameters like credentials, endpoint, etc.
    data_storage_version: optional, str, default None
        The version of the data storage format to use. Newer versions are more
        efficient but require newer versions of lance to read.  The default (None)
        will use the latest stable version.  See the user guide for more details.
    use_legacy_format : optional, bool, default None
        Deprecated method for setting the data storage version. Use the
        `data_storage_version` parameter instead.
    enable_v2_manifest_paths : bool, optional
        If True, and this is a new dataset, uses the new V2 manifest paths.
        These paths provide more efficient opening of datasets with many
        versions on object stores. This parameter has no effect if the dataset
        already exists. To migrate an existing dataset, instead use the
        :meth:`LanceDataset.migrate_manifest_paths_v2` method. Default is False.
    """
    if use_legacy_format is not None:
        warnings.warn(
            "use_legacy_format is deprecated, use data_storage_version instead",
            DeprecationWarning,
        )
        if use_legacy_format:
            data_storage_version = "legacy"
        else:
            data_storage_version = "stable"

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
        "storage_options": storage_options,
        "data_storage_version": data_storage_version,
        "enable_v2_manifest_paths": enable_v2_manifest_paths,
    }

    if commit_lock:
        if not callable(commit_lock):
            raise TypeError(f"commit_lock must be a function, got {type(commit_lock)}")
        params["commit_handler"] = commit_lock

    uri = os.fspath(uri) if isinstance(uri, Path) else uri
    inner_ds = _write_dataset(reader, uri, params)

    ds = LanceDataset.__new__(LanceDataset)
    ds._ds = inner_ds
    ds._uri = uri
    ds._default_scan_options = None
    return ds


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
