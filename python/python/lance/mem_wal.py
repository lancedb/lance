# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
Python wrappers for Lance MemWAL functionality.

The MemWAL feature enables high-throughput, low-latency writes to a Lance
dataset via an LSM-tree structure.  Data flows through three levels:

1. **WAL** – append-only durable log (raw writes)
2. **Active MemTable** – in-memory write buffer
3. **Flushed MemTable** – Lance files written to object store
4. **Base table** – canonical Lance dataset files (after merge_insert)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

import pyarrow as pa

from .lance import (
    _ExecutionPlan,
    _LsmPointLookupPlanner,
    _LsmScanner,
    _LsmVectorSearchPlanner,
    _MergedGeneration,
    _RegionSnapshot,
    _RegionWriter,
)
from .types import _coerce_reader

if TYPE_CHECKING:
    import lance

__all__ = [
    "RegionField",
    "RegionSpec",
    "MergedGeneration",
    "RegionSnapshot",
    "RegionWriter",
    "LsmScanner",
    "ExecutionPlan",
    "LsmPointLookupPlanner",
    "LsmVectorSearchPlanner",
]


# ---------------------------------------------------------------------------
# RegionSpec
# ---------------------------------------------------------------------------


@dataclass
class RegionField:
    """Defines one derived field used in region partitioning.

    Parameters
    ----------
    field_id : str
        Identifier for the derived field.
    source_ids : list of int
        Source field IDs used as inputs.
    transform : str, optional
        Optional transform name applied to the source fields.
    expression : str, optional
        Optional expression used to derive the field value.
    result_type : str
        Output type name for the derived field.
    parameters : dict of str to str, optional
        Extra transform parameters.
    """

    field_id: str
    source_ids: List[int]
    transform: Optional[str] = None
    expression: Optional[str] = None
    result_type: str = ""
    parameters: Dict[str, str] = field(default_factory=dict)


@dataclass
class RegionSpec:
    """Partitioning specification for deriving MemWAL region IDs."""

    spec_id: int
    fields: List[RegionField]


@dataclass
class MergedGeneration:
    """Identifies a flushed MemWAL generation that has been merged.

    Pass a list of these to mark_generations_as_merged
    so Lance knows which generations are now in the base table.

    Parameters
    ----------
    region_id : str
        UUID string for the write region.
    generation : int
        Generation number (from
        :attr:`RegionSnapshot.flushed_generations`).
    """

    region_id: str
    generation: int


class RegionSnapshot:
    """Snapshot of a MemWAL region's state, used when constructing scanners.

    Parameters
    ----------
    region_id : str
        UUID string for the write region.
    """

    def __init__(self, region_id: str) -> None:
        self._raw = _RegionSnapshot(region_id)

    @property
    def region_id(self) -> str:
        """UUID string for this region."""
        return self._raw.region_id

    def with_spec_id(self, spec_id: int) -> "RegionSnapshot":
        """Set the RegionSpec ID."""
        self._raw = self._raw.with_spec_id(spec_id)
        return self

    def with_current_generation(self, generation: int) -> "RegionSnapshot":
        """Set the current (active) generation number."""
        self._raw = self._raw.with_current_generation(generation)
        return self

    def with_flushed_generation(self, generation: int, path: str) -> "RegionSnapshot":
        """Add a flushed generation with its storage path."""
        self._raw = self._raw.with_flushed_generation(generation, path)
        return self

    def __repr__(self) -> str:
        return repr(self._raw)


class RegionWriter:
    """Stateful writer for one MemWAL region.

    Obtain an instance via mem_wal_writer.
    Use as a context manager so the writer is closed automatically::

        with dataset.mem_wal_writer(region_id) as writer:
            writer.put(batch)

    Parameters
    ----------
    _raw : _RegionWriter
        Internal PyO3 object — do not construct directly.
    """

    def __init__(self, _raw: _RegionWriter) -> None:
        self._raw = _raw

    @property
    def region_id(self) -> str:
        """UUID string for this writer's region."""
        return self._raw.region_id

    def put(self, data, *, schema: Optional[pa.Schema] = None) -> None:
        """Write data to the MemWAL.

        Parameters
        ----------
        data : ReaderLike
            Any Arrow-compatible data `pyarrow.Table`,
            `pyarrow.RecordBatch`, ``RecordBatchReader``, pandas
            DataFrame, etc.
        schema : pa.Schema, optional
            Schema hint, needed when *data* is a generator.

        Raises
        ------
        IOError
            On WAL flush failure.
        RuntimeError
            If the writer has already been closed.
        """
        reader = _coerce_reader(data, schema)
        self._raw.put(reader)

    def close(self) -> None:
        """Flush and close the writer.

        After ``close()``, calling :meth:`put` raises an error.
        Automatically called when used as a context manager.
        """
        self._raw.close()

    def stats(self) -> dict:
        """Return a snapshot of write statistics.

        Returns
        -------
        dict
            Keys: ``put_count``, ``put_time_ms``, ``wal_flush_count``,
            ``wal_flush_bytes``, ``wal_flush_time_ms``,
            ``memtable_flush_count``, ``memtable_flush_rows``,
            ``memtable_flush_time_ms``.
        """
        return self._raw.stats()

    def memtable_stats(self) -> dict:
        """Return current MemTable statistics.

        Returns
        -------
        dict
            Keys: ``row_count``, ``batch_count``, ``estimated_size_bytes``,
            ``generation``.
        """
        return self._raw.memtable_stats()

    def lsm_scanner(
        self, region_snapshots: Optional[List[RegionSnapshot]] = None
    ) -> "LsmScanner":
        """Create an LSM scanner that includes the active MemTable.

        This scanner covers the base table, the given flushed generations,
        and the current active MemTable — providing strong read-your-writes
        consistency.

        Parameters
        ----------
        region_snapshots : list of RegionSnapshot, optional
            Snapshots of other regions to include.  This writer's own region
            is automatically included.

        Returns
        -------
        LsmScanner
        """
        raw_snaps = [s._raw for s in (region_snapshots or [])]
        return LsmScanner(self._raw.lsm_scanner(raw_snaps))

    def __enter__(self) -> "RegionWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False


class LsmScanner:
    """LSM-aware scanner covering all data levels.

    Deduplicates by primary key, always returning the newest version of
    each row across base table, flushed MemTables, and the active MemTable.

    Obtain an instance from `RegionWriter.lsm_scanner` (includes
    active MemTable) or `LsmScanner.from_snapshots` (flushed only).

    The builder methods (`project`, `filter`, `limit`)
    return ``self`` for chaining.

    Examples
    --------
    scanner = LsmScanner.from_snapshots(dataset, [snapshot])
    scanner.project(["id", "value"]).filter("value > 0.5")
    table = scanner.to_table()
    """

    def __init__(self, _raw: _LsmScanner) -> None:
        self._raw = _raw

    @staticmethod
    def from_snapshots(
        dataset: "lance.LanceDataset",
        region_snapshots: List[RegionSnapshot],
    ) -> "LsmScanner":
        """Create a scanner from dataset and region snapshots.

        Does **not** include the active MemTable; use
        `RegionWriter.lsm_scanner` for that.

        Parameters
        ----------
        dataset : LanceDataset
            The base dataset to scan.
        region_snapshots : list of RegionSnapshot
            Region snapshots specifying flushed generations to include.
        """
        raw = _LsmScanner.from_snapshots(
            dataset._ds, [s._raw for s in region_snapshots]
        )
        return LsmScanner(raw)

    def project(self, columns: List[str]) -> "LsmScanner":
        """Select specific columns to return."""
        self._raw = self._raw.project(columns)
        return self

    def filter(self, expr: str) -> "LsmScanner":
        """Set a SQL filter expression (e.g. ``"value > 0.5"``)."""
        self._raw = self._raw.filter(expr)
        return self

    def limit(self, n: int, offset: Optional[int] = None) -> "LsmScanner":
        """Limit rows returned, optionally with an offset."""
        self._raw = self._raw.limit(n, offset)
        return self

    def with_row_address(self) -> "LsmScanner":
        """Include the ``_rowaddr`` internal column in results."""
        self._raw = self._raw.with_row_address()
        return self

    def with_memtable_gen(self) -> "LsmScanner":
        """Include the ``_memtable_gen`` internal column in results."""
        self._raw = self._raw.with_memtable_gen()
        return self

    def to_batch(self) -> pa.RecordBatch:
        """Execute the scan and return a single merged :class:`~pyarrow.RecordBatch`."""
        return self._raw.to_batch()

    def to_batches(self) -> List[pa.RecordBatch]:
        """Execute the scan and return a list of :class:`~pyarrow.RecordBatch`."""
        return list(self._raw.to_batches())

    def to_table(self) -> pa.Table:
        """Execute the scan and return a :class:`~pyarrow.Table`."""
        batch = self.to_batch()
        return pa.Table.from_batches([batch])

    def count_rows(self) -> int:
        """Return the row count without loading all column data."""
        return self._raw.count_rows()


class ExecutionPlan:
    """Executable physical plan returned by MemWAL planners.

    This wraps the Rust/DataFusion physical plan object. Planner classes only
    construct plans; execution happens through this class.

    Parameters
    ----------
    _raw : _ExecutionPlan
    """

    def __init__(self, _raw: _ExecutionPlan) -> None:
        self._raw = _raw

    @property
    def schema(self) -> pa.Schema:
        """Output schema of this physical plan."""
        return self._raw.schema

    @property
    def dataset_schema(self) -> pa.Schema:
        """Base dataset schema used to construct this plan."""
        return self._raw.dataset_schema

    def explain(self) -> str:
        """Return the physical plan as an indented string."""
        return self._raw.explain()

    def to_reader(self) -> pa.RecordBatchReader:
        """Execute the plan and return a streaming reader."""
        return self._raw.to_reader()

    def to_batches(self) -> List[pa.RecordBatch]:
        """Execute the plan and return all record batches."""
        return list(self._raw.to_batches())

    def to_table(self) -> pa.Table:
        """Execute the plan and return a table."""
        return self.to_reader().read_all()


class LsmPointLookupPlanner:
    """Plans primary-key point lookups across all LSM levels.

    More efficient than `LsmScanner` for known-PK lookups due to
    bloom filter optimizations and short-circuit evaluation.

    Parameters
    ----------
    dataset : LanceDataset
        The base dataset.
    region_snapshots : list of RegionSnapshot
        Region snapshots specifying flushed generations to include.
    pk_columns : list of str, optional
        Primary key column names.  Inferred from schema metadata if omitted.

    Examples
    --------
    planner = LsmPointLookupPlanner(dataset, [snapshot])
    plan = planner.plan_lookup(pa.array([42], type=pa.int64()))
    result = plan.to_table()
    """

    def __init__(
        self,
        dataset: "lance.LanceDataset",
        region_snapshots: List[RegionSnapshot],
        pk_columns: Optional[List[str]] = None,
    ) -> None:
        self._raw = _LsmPointLookupPlanner(
            dataset._ds,
            [s._raw for s in region_snapshots],
            pk_columns,
        )

    def plan_lookup(
        self,
        pk_value: pa.Array,
        columns: Optional[List[str]] = None,
    ) -> ExecutionPlan:
        """Plan a point lookup by primary key value.

        Parameters
        ----------
        pk_value : pa.Array
            For single-column primary keys, a PyArrow array with exactly one
            element. For composite primary keys, a single-row
            `pyarrow.StructArray` with one field per primary-key column.
        columns : list of str, optional
            Columns to project. Returns all columns if omitted.

        Returns
        -------
        ExecutionPlan
            Physical plan for the lookup. Execute it via `to_table`,
            `to_reader`, or `to_batches`.
        """
        return ExecutionPlan(self._raw.plan_lookup(pk_value, columns))


class LsmVectorSearchPlanner:
    """Plans IVF-PQ vector KNN search across all LSM levels.

    Results include staleness filtering to return only the latest version
    of each row. The output schema includes the ``_distance`` column.

    Parameters
    ----------
    dataset : LanceDataset
        The base dataset.
    region_snapshots : list of RegionSnapshot
        Region snapshots specifying flushed generations to include.
    vector_column : str
        Name of the ``FixedSizeList<float32>`` vector column.
    pk_columns : list of str, optional
        Primary key columns.  Inferred from schema metadata if omitted.
    distance_type : str, optional
        Distance metric — one of ``"l2"`` (default), ``"cosine"``,
        ``"dot"``, ``"hamming"``.

    Examples
    --------
    import numpy as np
    planner = LsmVectorSearchPlanner(dataset, [snapshot], "vector")
    query = pa.array(np.random.rand(128).astype("float32"))
    plan = planner.plan_search(query, k=10)
    result = plan.to_table()
    """

    def __init__(
        self,
        dataset: "lance.LanceDataset",
        region_snapshots: List[RegionSnapshot],
        vector_column: str,
        pk_columns: Optional[List[str]] = None,
        distance_type: Optional[str] = None,
    ) -> None:
        kwargs = {}
        if pk_columns is not None:
            kwargs["pk_columns"] = pk_columns
        if distance_type is not None:
            kwargs["distance_type"] = distance_type
        self._raw = _LsmVectorSearchPlanner(
            dataset._ds,
            [s._raw for s in region_snapshots],
            vector_column,
            **kwargs,
        )

    def plan_search(
        self,
        query: pa.Array,
        k: int = 10,
        nprobes: int = 20,
        columns: Optional[List[str]] = None,
    ) -> ExecutionPlan:
        """Plan a KNN vector search.

        Parameters
        ----------
        query : pa.Array
            A flat ``Float32Array`` of length ``vector_dim``.
        k : int, optional
            Number of nearest neighbours to return (default: 10).
        nprobes : int, optional
            Number of IVF partitions to probe (default: 20).
        columns : list of str, optional
            Columns to project.  Returns all columns + ``_distance`` if
            omitted.

        Returns
        -------
        ExecutionPlan
            Physical plan for the vector search. Execute it via
            `to_table`, `to_reader`, or `to_batches`.
        """
        return ExecutionPlan(self._raw.plan_search(query, k, nprobes, columns))


def _unwrap_region_id(region_id: str) -> str:
    """Validate region_id is a UUID string."""
    import uuid as _uuid

    _uuid.UUID(region_id)  # raises ValueError if invalid
    return region_id


def _to_raw_merged_generations(
    generations: Iterable[MergedGeneration],
) -> list:
    """Convert Python MergedGeneration list to PyO3 _MergedGeneration list."""
    return [_MergedGeneration(g.region_id, g.generation) for g in generations]
