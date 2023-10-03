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
import pickle
import random
import re
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
    print(tbl)
    return tbl


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

    first = sample_queries[0]

    def has_target(target, results):
        for item in results:
            if item.values == target:
                return True
        return False

    def check_index(has_knn_combined):
        print(dataset.scanner(nearest=first).explain_plan())
        for query in sample_queries:
            results = dataset.to_table(nearest=query).column("vector")
            assert has_target(query["q"], results)
            plan = dataset.scanner(nearest=query).explain_plan()
            assert ("KNNFlat" in plan) == has_knn_combined

    # Original state is 2 indexed fragments of size 150.  This should not require
    # a combined scan
    check_index(has_knn_combined=False)

    # Compact the 2 fragments into 1.  Combined scan still not needed.
    dataset.optimize.compact_files()
    assert len(dataset.get_fragments()) == 1
    print(dataset.get_fragments())
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
    assert dataset.version == 3
