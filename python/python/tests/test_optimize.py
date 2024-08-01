# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
import pickle
import random
import re
import threading
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
from lance.lance import Compaction
from lance.optimize import RewriteResult
from lance.vector import vec_to_table


def test_dataset_optimize(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    data = pa.table({"a": range(1000), "b": range(1000)})

    dataset = lance.write_dataset(data, base_dir, max_rows_per_file=100)
    assert dataset.version == 1
    assert len(dataset.get_fragments()) == 10

    metrics = dataset.optimize.compact_files(
        target_rows_per_fragment=1000,
        materialize_deletions=False,
        num_threads=1,
    )

    assert metrics.fragments_removed == 10
    assert metrics.fragments_added == 1
    assert metrics.files_removed == 10
    assert metrics.files_added == 1

    assert dataset.version == 3


def create_table(min, max, nvec, ndim=8):
    mat = np.random.uniform(min, max, (nvec, ndim))
    tbl = vec_to_table(data=mat)
    # Add id column for filtering
    tbl = pa.Table.from_pydict(
        {
            "vector": tbl.column(0).chunk(0),
            "id": np.arange(0, nvec),
        }
    )
    return tbl


def test_compact_with_write(tmp_path: Path):
    # This test creates a dataset with a manifest containing fragments
    # that are not in sorted order (by id)
    #
    # We do this by runnign compaction concurrently with append
    #
    # This is because compaction first reserves a fragment id.  Then the
    # concurrent writes grab later ids and commit them.  Then the compaction
    # commits with its earlier id.
    #
    # In the next compaction we should detect this, and reorder the fragments
    # when writing the compacted file.
    base_dir = tmp_path / "dataset"
    NUM_FRAGS = 5
    ROWS_PER_FRAG = 300

    # First, create some data
    data = create_table(min=0, max=1, nvec=ROWS_PER_FRAG)
    dataset = lance.write_dataset(data, base_dir)
    for _ in range(NUM_FRAGS):
        lance.write_dataset(data, base_dir, mode="append")

    # Now, run compaction at the same time as creating new data
    def do_compaction():
        dataset = lance.dataset(base_dir)
        dataset.optimize.compact_files()

    compact_thread = threading.Thread(target=do_compaction)
    compact_thread.start()
    for _ in range(NUM_FRAGS):
        lance.write_dataset(data, base_dir, mode="append")
    compact_thread.join()

    # Now, run compaction again, this should succeed
    dataset = lance.dataset(base_dir)
    dataset.optimize.compact_files()

    assert dataset.to_table().num_rows == ROWS_PER_FRAG * (NUM_FRAGS * 2 + 1)


def test_index_remapping(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    data = create_table(min=0, max=1, nvec=300)

    dataset = lance.write_dataset(data, base_dir, max_rows_per_file=150)
    dataset.create_index(
        "vector", index_type="IVF_PQ", num_partitions=2, num_sub_vectors=2
    )
    assert len(dataset.get_fragments()) == 2

    sample_query_indices = random.sample(range(300), 50)
    vecs = data.column("vector").chunk(0)
    sample_queries = [
        {"column": "vector", "q": vecs[i].values, "k": 5} for i in sample_query_indices
    ]

    def has_target(target, results):
        for item in results:
            if item.values == target:
                return True
        return False

    def check_index(has_knn_combined):
        for query in sample_queries:
            results = dataset.to_table(nearest=query).column("vector")
            assert has_target(query["q"], results)
            plan = dataset.scanner(nearest=query).explain_plan()
            assert ("KNNVectorDistance" in plan) == has_knn_combined

    # Original state is 2 indexed fragments of size 150.  This should not require
    # a combined scan
    check_index(has_knn_combined=False)

    # Compact the 2 fragments into 1.  Combined scan still not needed.
    dataset.optimize.compact_files()
    assert len(dataset.get_fragments()) == 1
    check_index(has_knn_combined=False)

    # Add a new fragment and recalculate the index
    extra_data = create_table(min=1000, max=1001, nvec=100)
    dataset = lance.write_dataset(
        extra_data, base_dir, mode="append", max_rows_per_file=100
    )
    dataset.create_index(
        "vector", index_type="IVF_PQ", num_partitions=2, num_sub_vectors=2, replace=True
    )

    # Combined scan should not be needed
    assert len(dataset.get_fragments()) == 2
    check_index(has_knn_combined=False)

    # Add a new unindexed fragment
    extra_data = create_table(min=1000, max=1001, nvec=100)
    dataset = lance.write_dataset(
        extra_data, base_dir, mode="append", max_rows_per_file=100
    )
    assert len(dataset.get_fragments()) == 3

    # Compaction should not combine the unindexed fragment with the indexed fragment
    dataset.optimize.compact_files()
    assert len(dataset.get_fragments()) == 2

    # Now a combined scan is required
    check_index(has_knn_combined=True)


def test_index_remapping_multiple_rewrite_tasks(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    ds = lance.write_dataset(
        create_table(min=0, max=1, nvec=300), base_dir, max_rows_per_file=150
    )
    ds = ds.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=2,
    )
    assert ds.has_index
    ds = lance.write_dataset(
        create_table(min=0, max=1, nvec=300),
        base_dir,
        mode="append",
        max_rows_per_file=150,
    )

    ds.delete("id % 4 == 0")
    fragments = list(ds.get_fragments())
    assert len(fragments) == 4

    # We have a dataset with 4 small fragments.  2 are indexed and
    # 2 are not.  The indexed fragments and the non-indexed fragments
    # cannot be combined and so we should end up with 2 fragments after
    # compaction
    ds.optimize.compact_files()

    fragments = list(ds.get_fragments())
    assert len(fragments) == 2

    index = ds.list_indices()[0]
    index_frag_ids = list(index["fragment_ids"])
    frag_ids = [frag.fragment_id for frag in fragments]

    assert len(index_frag_ids) == 1
    assert index_frag_ids[0] in frag_ids


def test_dataset_distributed_optimize(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    data = pa.table({"a": range(800), "b": range(800)})

    dataset = lance.write_dataset(data, base_dir, max_rows_per_file=200)
    fragments = dataset.get_fragments()
    assert len(fragments) == 4

    plan = Compaction.plan(
        dataset, options=dict(target_rows_per_fragment=400, num_threads=1)
    )
    assert plan.read_version == 1
    assert plan.num_tasks() == 2
    assert plan.tasks[0].fragments == [frag.metadata for frag in fragments[0:2]]
    assert plan.tasks[1].fragments == [frag.metadata for frag in fragments[2:4]]
    assert repr(plan) == "CompactionPlan(read_version=1, tasks=<2 compaction tasks>)"
    # Plan can be pickled
    assert pickle.loads(pickle.dumps(plan)) == plan

    pickled_task = pickle.dumps(plan.tasks[0])
    task = pickle.loads(pickled_task)
    assert task == plan.tasks[0]

    result1 = plan.tasks[0].execute(dataset)
    result1.metrics.fragments_removed == 2
    result1.metrics.fragments_added == 1

    pickled_result = pickle.dumps(result1)
    result = pickle.loads(pickled_result)
    assert isinstance(result, RewriteResult)
    assert result == result1
    assert re.match(
        r"RewriteResult\(read_version=1, new_fragments=\[.+\], old_fragments=\[.+\]\)",
        repr(result),
    )

    metrics = Compaction.commit(dataset, [result1])
    assert metrics.fragments_removed == 2
    assert metrics.fragments_added == 1
    # Compaction occurs in two transactions so it increments the version by 2.
    assert dataset.version == 3
