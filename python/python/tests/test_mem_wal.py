# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
import math
import os
import time
import uuid

import lance
import pyarrow as pa
from lance.mem_wal import (
    LsmPointLookupPlanner,
    LsmScanner,
    RegionSnapshot,
)

_PK_META = {"lance-schema:unenforced-primary-key": "true"}
_LOOKUP_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64(), nullable=False, metadata=_PK_META),
        pa.field("name", pa.utf8()),
    ]
)


def _lookup_table(ids, prefix: str) -> pa.Table:
    """Build a table where name = '{prefix}_{id}' for each id."""
    return pa.table(
        {
            "id": pa.array(ids, pa.int64()),
            "name": pa.array([f"{prefix}_{i}" for i in ids], pa.utf8()),
        },
        schema=_LOOKUP_SCHEMA,
    )


def _write_flushed_gen(base_path: str, region_id: str, gen_folder: str, data: pa.Table):
    """Write a flushed-generation Lance dataset at the expected sub-path.

    The collector resolves flushed generation paths as:
        {base_dataset_path}/_mem_wal/{region_id}/{gen_folder}
    """
    gen_path = os.path.join(base_path, "_mem_wal", region_id, gen_folder)
    lance.write_dataset(data, gen_path, schema=_LOOKUP_SCHEMA)


def test_point_lookup_with_memtables(tmp_path):
    """
    Lookup against a base table that has one flushed generation containing an
    update.  The flushed version must win over the base table version.

    Setup
    -----
    base   : ids [1, 2, 3]  names ["base_1",  "base_2",  "base_3"]
    gen_1  : ids [2]        names ["gen1_2"]          ← update to id=2

    RegionSnapshot: flushed_generation(gen=1, path="gen_1"), current_generation=2
    """
    ds_path = str(tmp_path / "base")
    region_id = str(uuid.uuid4())

    # --- Base dataset ---
    base_ds = lance.write_dataset(
        _lookup_table([1, 2, 3], "base"), ds_path, schema=_LOOKUP_SCHEMA
    )
    base_ds.initialize_mem_wal()

    # --- Flushed generation: overwrites id=2 ---
    _write_flushed_gen(ds_path, region_id, "gen_1", _lookup_table([2], "gen1"))

    # --- RegionSnapshot describing the flushed state ---
    snap = (
        RegionSnapshot(region_id)
        .with_flushed_generation(1, "gen_1")
        .with_current_generation(2)
    )

    planner = LsmPointLookupPlanner(base_ds, [snap])
    assert not hasattr(planner, "lookup")

    # id=2 must return the flushed version
    plan = planner.plan_lookup(pa.array([2], type=pa.int64()))
    assert plan.schema.names == ["id", "name"]
    assert plan.dataset_schema.names == ["id", "name"]
    assert "Take" in plan.explain() or "Scan" in plan.explain()
    result = plan.to_table()
    assert len(result) == 1, "Expected exactly one row for id=2"
    assert result.column("name")[0].as_py() == "gen1_2", (
        "Flushed generation must win over base table"
    )

    # id=1 is only in the base table
    result_base = planner.plan_lookup(pa.array([1], type=pa.int64())).to_table()
    assert len(result_base) == 1
    assert result_base.column("name")[0].as_py() == "base_1"

    # id=99 does not exist anywhere
    result_miss = planner.plan_lookup(pa.array([99], type=pa.int64())).to_table()
    assert len(result_miss) == 0, "Non-existent key must return empty result"


def test_lsm_scanner_with_memtables(tmp_path):
    """
    Full-scan via LsmScanner.from_snapshots deduplicates rows across LSM
    levels: each primary key appears exactly once, from its newest level.

    base  : ids [1, 2, 3]  names ["base_1",  "base_2",  "base_3"]
    gen_1 : ids [2]        names ["gen1_2"]    ← overwrites id=2

    Expected result: 3 unique rows — id=2 from gen_1, id=1 and id=3 from base.
    """
    ds_path = str(tmp_path / "base")
    region_id = str(uuid.uuid4())

    base_ds = lance.write_dataset(
        _lookup_table([1, 2, 3], "base"), ds_path, schema=_LOOKUP_SCHEMA
    )
    base_ds.initialize_mem_wal()

    _write_flushed_gen(ds_path, region_id, "gen_1", _lookup_table([2], "gen1"))

    snap = (
        RegionSnapshot(region_id)
        .with_flushed_generation(1, "gen_1")
        .with_current_generation(2)
    )

    scanner = LsmScanner.from_snapshots(base_ds, [snap])
    table = scanner.to_table()

    assert len(table) == 3, f"Expected 3 deduplicated rows, got {len(table)}"
    name_by_id = {row["id"]: row["name"] for row in table.to_pylist()}

    assert name_by_id[1] == "base_1"
    assert name_by_id[2] == "gen1_2", "Flushed gen must overwrite base for id=2"
    assert name_by_id[3] == "base_3"


def test_region_writer_lsm_scanner_includes_own_flushed_generations(tmp_path):
    ds_path = str(tmp_path / "base")
    region_id = str(uuid.uuid4())
    ds = lance.write_dataset(_lookup_table([0], "base"), ds_path, schema=_LOOKUP_SCHEMA)
    ds.initialize_mem_wal()

    with ds.mem_wal_writer(
        region_id,
        durable_write=True,
        max_wal_buffer_size=1,
        max_wal_flush_interval_ms=10,
        max_memtable_batches=1,
    ) as writer:
        writer.put(_lookup_table([1, 2], "writer"))

        deadline = time.time() + 5
        while True:
            table = writer.lsm_scanner().to_table()
            name_by_id = {row["id"]: row["name"] for row in table.to_pylist()}
            if name_by_id.get(1) == "writer_1" and name_by_id.get(2) == "writer_2":
                break
            if time.time() >= deadline:
                assert False, "writer.lsm_scanner() did not include flushed writer rows"
            time.sleep(0.05)


_VDIM = 4  # matches Rust test fixture dimension


def _vector_search_schema():
    """Schema for vector-search tests: id (int32 PK) + vector column."""
    pk_meta = {"lance-schema:unenforced-primary-key": "true"}
    return pa.schema(
        [
            pa.field("id", pa.int32(), nullable=False, metadata=pk_meta),
            pa.field("vector", pa.list_(pa.float32(), _VDIM)),
        ]
    )


def _vector_search_table(ids):
    """Build a table with deterministic vectors matching the Rust fixture.

    For id=N the vector is [N*0.1, N*0.1+0.1, N*0.1+0.2, N*0.1+0.3].
    """
    flat = []
    for i in ids:
        base = i * 0.1
        flat.extend([base, base + 0.1, base + 0.2, base + 0.3])
    storage = pa.array(flat, type=pa.float32())
    vectors = pa.FixedSizeListArray.from_arrays(storage, _VDIM)
    return pa.table(
        {"id": pa.array(ids, pa.int32()), "vector": vectors},
        schema=_vector_search_schema(),
    )


VECTOR_DIM = 32
ROWS_PER_BATCH = 50
NUM_WRITE_ROUNDS = 3
BATCHES_PER_ROUND = 3


def _e2e_schema():
    """Schema for the e2e test: id (PK), vector, text."""
    pk_meta = {"lance-schema:unenforced-primary-key": "true"}
    return pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False, metadata=pk_meta),
            pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
            pa.field("text", pa.utf8()),
        ]
    )


def _e2e_batch(schema, start_id: int, num_rows: int) -> pa.RecordBatch:
    """Deterministic RecordBatch for the e2e test."""
    ids = list(range(start_id, start_id + num_rows))
    flat_vecs = [
        math.sin((start_id + i) * 0.1 + d * 0.01)
        for i in range(num_rows)
        for d in range(VECTOR_DIM)
    ]
    storage = pa.array(flat_vecs, type=pa.float32())
    vector_array = pa.FixedSizeListArray.from_arrays(storage, VECTOR_DIM)
    texts = [f"Sample text for row {start_id + i}" for i in range(num_rows)]
    return pa.record_batch(
        {
            "id": pa.array(ids, pa.int64()),
            "vector": vector_array,
            "text": pa.array(texts, pa.utf8()),
        },
        schema=schema,
    )


def test_region_writer_e2e_correctness(tmp_path):
    """
    End-to-end correctness test for RegionWriter covering:
    - Multi-round writes that trigger WAL and MemTable flushes
    - File-system layout verification (_mem_wal/<region_id>/wal/ and manifest/)
    - Flushed generation data readable via LsmScanner
    - New writer created after close can write and scan correctly

    Mirrors Rust test: region_writer_tests::test_region_writer_e2e_correctness
    """
    schema = _e2e_schema()
    ds_path = str(tmp_path / "ds")

    # 500 seed rows so BTree index training succeeds
    initial_batch = _e2e_batch(schema, start_id=0, num_rows=500)
    ds = lance.write_dataset(
        pa.Table.from_batches([initial_batch]), ds_path, schema=schema
    )

    ds.create_scalar_index("id", "BTREE", name="id_btree")
    ds.initialize_mem_wal(maintained_indexes=["id_btree"])

    # Small buffers to trigger WAL and MemTable flushes during the test
    region_id = str(uuid.uuid4())
    writer = ds.mem_wal_writer(
        region_id,
        durable_write=True,
        sync_indexed_write=True,
        max_wal_buffer_size=10 * 1024,  # 10 KB
        max_wal_flush_interval_ms=50,
        max_memtable_size=80,  # flush after ~80 rows
    )

    total_rows_written = 0
    for _round in range(NUM_WRITE_ROUNDS):
        start_id = 500 + total_rows_written
        for i in range(BATCHES_PER_ROUND):
            batch = _e2e_batch(schema, start_id + i * ROWS_PER_BATCH, ROWS_PER_BATCH)
            writer.put(pa.Table.from_batches([batch]))
        total_rows_written += BATCHES_PER_ROUND * ROWS_PER_BATCH
        time.sleep(0.15)  # allow async WAL/memtable flush

    writer.close()

    # === Stats ===
    stats = writer.stats()
    assert stats["put_count"] == NUM_WRITE_ROUNDS * BATCHES_PER_ROUND
    assert stats["wal_flush_count"] >= 1, "Expected at least one WAL flush"

    closed_memtable_stats = writer.memtable_stats()
    assert closed_memtable_stats["row_count"] == 0
    assert closed_memtable_stats["batch_count"] == 0
    assert closed_memtable_stats["generation"] >= 1

    # === File-system layout ===
    mem_wal_dir = os.path.join(ds_path, "_mem_wal", region_id)
    assert os.path.isdir(mem_wal_dir), f"MemWAL directory missing: {mem_wal_dir}"

    wal_dir = os.path.join(mem_wal_dir, "wal")
    assert os.path.isdir(wal_dir), "WAL sub-directory missing"
    wal_files = os.listdir(wal_dir)
    assert len(wal_files) >= 1
    assert all(f.endswith(".arrow") for f in wal_files), (
        f"All WAL files should have .arrow extension, got: {wal_files}"
    )

    manifest_dir = os.path.join(mem_wal_dir, "manifest")
    assert os.path.isdir(manifest_dir), "Manifest sub-directory missing"
    assert len(os.listdir(manifest_dir)) >= 1

    # === Generation counter advanced ===
    mt_stats = writer.memtable_stats()
    assert mt_stats["generation"] >= 1

    # === New writer: write and read back via active MemTable scanner ===
    ds2 = lance.dataset(ds_path)
    region_id2 = str(uuid.uuid4())
    with ds2.mem_wal_writer(
        region_id2, durable_write=False, sync_indexed_write=True
    ) as writer2:
        verify_batch = _e2e_batch(schema, start_id=10000, num_rows=10)
        writer2.put(pa.Table.from_batches([verify_batch]))
        result = writer2.lsm_scanner().to_table()

    assert len(result) >= 10
    new_ids = result.column("id").to_pylist()
    assert 10000 in new_ids
    assert 10009 in new_ids
