#  Copyright 2023 Lance Developers
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import platform
import random
import string
import time
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from lance.vector import vec_to_table


def create_table(nvec=1000, ndim=128):
    mat = np.random.randn(nvec, ndim)
    price = np.random.rand(nvec) * 100

    def gen_str(n):
        return "".join(random.choices(string.ascii_letters + string.digits, k=n))

    meta = np.array([gen_str(100) for _ in range(nvec)])
    tbl = (
        vec_to_table(data=mat)
        .append_column("price", pa.array(price))
        .append_column("meta", pa.array(meta))
        .append_column("id", pa.array(range(nvec)))
    )
    return tbl


@pytest.fixture()
def dataset(tmp_path):
    tbl = create_table()
    yield lance.write_dataset(tbl, tmp_path)


@pytest.fixture()
def indexed_dataset(tmp_path):
    tbl = create_table()
    dataset = lance.write_dataset(tbl, tmp_path)
    yield dataset.create_index(
        "vector", index_type="IVF_PQ", num_partitions=4, num_sub_vectors=16
    )


def run(ds, q=None, assert_func=None):
    if q is None:
        q = np.random.randn(128)
    project = [None, ["price"], ["vector", "price"], ["vector", "meta", "price"]]
    refine = [None, 1, 2]
    filters = [None, pc.field("price") > 50.0]
    times = []

    for columns in project:
        expected_columns = []
        if columns is None:
            expected_columns.extend(ds.schema.names)
        else:
            expected_columns.extend(columns)
        for c in ["vector", "_distance"]:
            if c not in expected_columns:
                expected_columns.append(c)

        for filter_ in filters:
            for rf in refine:
                start = time.time()
                rs = ds.to_table(
                    columns=columns,
                    nearest={
                        "column": "vector",
                        "q": q,
                        "k": 15,
                        "nprobes": 1,
                        "refine_factor": rf,
                    },
                    filter=filter_,
                )
                end = time.time()
                times.append(end - start)
                assert rs.column_names == expected_columns
                if filter_ is not None:
                    inmem = pa.dataset.dataset(rs)
                    assert len(inmem.to_table(filter=filter_)) == len(rs)
                else:
                    assert len(rs) == 15
                    distances = rs["_distance"].to_numpy()
                    assert (distances.max() - distances.min()) > 1e-6
                    if assert_func is not None:
                        assert_func(rs)
    return times


def test_flat(dataset):
    print(run(dataset))


def test_ann(indexed_dataset):
    print(run(indexed_dataset))


def test_ann_append(tmp_path):
    tbl = create_table()
    dataset = lance.write_dataset(tbl, tmp_path)
    dataset = dataset.create_index(
        "vector", index_type="IVF_PQ", num_partitions=4, num_sub_vectors=16
    )
    new_data = create_table(nvec=10)
    dataset = lance.write_dataset(new_data, dataset.uri, mode="append")
    q = new_data["vector"][0].as_py()

    def func(rs: pa.Table):
        assert rs["vector"][0].as_py() == q

    print(run(dataset, q=np.array(q), assert_func=func))


@pytest.mark.cuda
def test_create_index_using_cuda(tmp_path):
    tbl = create_table()
    dataset = lance.write_dataset(tbl, tmp_path)
    dataset = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=16,
        accelerator="cuda",
    )
    q = np.random.randn(128)
    expected = dataset.to_table(
        columns=["id"],
        nearest={
            "column": "vector",
            "q": q,
            "k": 10,  # Use non-default k
        },
    )["id"].to_numpy()
    assert len(expected) == 10


def test_create_index_unsupported_accelerator(tmp_path):
    tbl = create_table()
    dataset = lance.write_dataset(tbl, tmp_path)
    with pytest.raises(ValueError):
        dataset.create_index(
            "vector",
            index_type="IVF_PQ",
            num_partitions=4,
            num_sub_vectors=16,
            accelerator="no-supported",
        )

    with pytest.raises(ValueError):
        dataset.create_index(
            "vector",
            index_type="IVF_PQ",
            num_partitions=4,
            num_sub_vectors=16,
            accelerator="0cuda",
        )

    with pytest.raises(ValueError):
        dataset.create_index(
            "vector",
            index_type="IVF_PQ",
            num_partitions=4,
            num_sub_vectors=16,
            accelerator="cuda-0",
        )

    with pytest.raises(ValueError):
        dataset.create_index(
            "vector",
            index_type="IVF_PQ",
            num_partitions=4,
            num_sub_vectors=16,
            accelerator="cuda:",
        )

    with pytest.raises(ValueError):
        dataset.create_index(
            "vector",
            index_type="IVF_PQ",
            num_partitions=4,
            num_sub_vectors=16,
            accelerator="cuda:abc",
        )


def test_use_index(dataset, tmp_path):
    ann_ds = lance.write_dataset(dataset.to_table(), tmp_path / "indexed.lance")
    ann_ds = ann_ds.create_index(
        "vector", index_type="IVF_PQ", num_partitions=4, num_sub_vectors=16
    )
    q = np.random.randn(128)
    expected = dataset.to_table(
        columns=["id"],
        nearest={
            "column": "vector",
            "q": q,
            "k": 12,  # Use non-default k
        },
    )["id"].to_numpy()

    actual = ann_ds.to_table(
        columns=["id"],
        nearest={"column": "vector", "q": q, "k": 12, "use_index": False},
    )["id"].to_numpy()

    assert np.all(expected == actual)

    # Can omit k but provide limit
    actual = ann_ds.to_table(
        columns=["id"],
        nearest={"column": "vector", "q": q, "use_index": False},
        limit=12,
    )["id"].to_numpy()
    assert np.all(expected == actual)


def test_nearest_errors(dataset, tmp_path):
    import pandas as pd

    with pytest.raises(ValueError, match="does not match index column size"):
        dataset.to_table(
            columns=["id"],
            nearest={"column": "vector", "q": np.random.randn(127), "k": 10},
        )

    df = pd.DataFrame({"a": [5], "b": [10]})
    ds = lance.write_dataset(pa.Table.from_pandas(df), tmp_path / "dataset.lance")

    with pytest.raises(TypeError, match="must be a vector"):
        ds.to_table(nearest={"column": "a", "q": np.random.randn(128), "k": 10})


def test_has_index(dataset, tmp_path):
    assert not dataset.has_index
    ann_ds = lance.write_dataset(dataset.to_table(), tmp_path / "indexed.lance")
    ann_ds = ann_ds.create_index(
        "vector", index_type="IVF_PQ", num_partitions=4, num_sub_vectors=16
    )
    assert ann_ds.has_index

    assert ann_ds.list_indices()[0]["fields"] == ["vector"]


def test_create_dot_index(dataset, tmp_path):
    assert not dataset.has_index
    ann_ds = lance.write_dataset(dataset.to_table(), tmp_path / "indexed.lance")
    ann_ds = ann_ds.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=16,
        metric="dot",
    )
    assert ann_ds.has_index


def test_pre_populated_ivf_centroids(dataset, tmp_path: Path):
    centroids = np.random.randn(5, 128).astype(np.float32)  # IVF5
    dataset_with_index = dataset.create_index(
        ["vector"],
        index_type="IVF_PQ",
        ivf_centroids=centroids,
        num_partitions=5,
        num_sub_vectors=8,
    )

    q = np.random.randn(128)
    actual = dataset_with_index.to_table(
        columns=["id"],
        nearest={"column": "vector", "q": q, "k": 10, "use_index": False},
    )["id"].to_numpy()
    assert len(actual) == 10

    index_meta = dataset_with_index.list_indices()[0]
    index_uuid = index_meta["uuid"]
    assert len(index_uuid) == 36
    assert index_meta["fragment_ids"] == {0}

    expected_filepath = str(tmp_path / "_indices" / index_uuid / "index.idx")
    if platform.system() == "Windows":
        expected_filepath = expected_filepath.replace("\\", "/")
    expected_statistics = {
        "index_type": "IVF",
        "uuid": index_uuid,
        "uri": expected_filepath,
        "metric_type": "l2",
        "num_partitions": 5,
        "sub_index": {
            "dimension": 128,
            "index_type": "PQ",
            "metric_type": "l2",
            "nbits": 8,
            "num_sub_vectors": 8,
        },
    }

    with pytest.raises(KeyError, match='Index "non-existent_idx" not found'):
        assert dataset_with_index.index_statistics("non-existent_idx")
    with pytest.raises(KeyError, match='Index "" not found'):
        assert dataset_with_index.index_statistics("")
    with pytest.raises(TypeError):
        dataset_with_index.index_statistics()

    actual_statistics = dataset_with_index.index_statistics("vector_idx")
    partitions = actual_statistics.pop("partitions")
    assert actual_statistics == expected_statistics

    assert len(partitions) == 5
    partition_keys = {"index", "length", "offset", "centroid"}
    assert all([p["index"] == i for i, p in enumerate(partitions)])
    assert all([partition_keys == set(p.keys()) for p in partitions])
    assert all([all([isinstance(c, float) for c in p["centroid"]]) for p in partitions])


def test_optimize_index(dataset, tmp_path):
    dataset_uri = tmp_path / "dataset.lance"
    assert not dataset.has_index
    ds = lance.write_dataset(dataset.to_table(), dataset_uri)
    ds = ds.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=2,
    )

    assert ds.has_index

    # New data
    tbl = create_table(nvec=200)
    ds = lance.write_dataset(tbl, dataset_uri, mode="append")

    assert len(ds) == 1200
    assert ds.has_index

    indices_dir = dataset_uri / "_indices"
    assert len(list(indices_dir.iterdir())) == 1

    ds = ds.optimize.optimize_indices()
    assert len(list(indices_dir.iterdir())) == 2


def create_uniform_table(min, max, nvec, offset, ndim=8):
    mat = np.random.uniform(min, max, (nvec, ndim))
    # rowid = np.arange(offset, offset + nvec)
    tbl = vec_to_table(data=mat)
    tbl = pa.Table.from_pydict(
        {
            "vector": tbl.column(0).chunk(0),
            "filterable": np.arange(offset, offset + nvec),
        }
    )
    return tbl


def test_optimize_index_recall(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    data = create_uniform_table(min=0, max=1, nvec=300, offset=0)

    dataset = lance.write_dataset(data, base_dir, max_rows_per_file=150)
    dataset.create_index(
        "vector", index_type="IVF_PQ", num_partitions=2, num_sub_vectors=2
    )
    assert len(dataset.get_fragments()) == 2

    sample_indices = random.sample(range(300), 50)
    sample_query_indices = sample_indices[0:40]
    sample_delete_indices = sample_indices[40:]
    vecs = data.column("vector").chunk(0)
    sample_queries = [
        {"column": "vector", "q": vecs[i].values, "k": 5} for i in sample_query_indices
    ]
    sample_delete_queries = [
        {"column": "vector", "q": vecs[i].values, "k": 5} for i in sample_delete_indices
    ]

    def has_target(target, results):
        for item in results:
            if item.values == target:
                return True
        return False

    def check_index(has_knn_combined, delete_has_happened):
        for query in sample_queries:
            results = dataset.to_table(nearest=query).column("vector")
            assert has_target(query["q"], results)
            plan = dataset.scanner(nearest=query).explain_plan()
            assert ("KNNFlat" in plan) == has_knn_combined
        for query in sample_delete_queries:
            results = dataset.to_table(nearest=query).column("vector")
            assert delete_has_happened != has_target(query["q"], results)

    # Original state is 2 indexed fragments of size 150.  This should not require
    # a combined scan
    check_index(has_knn_combined=False, delete_has_happened=False)

    # Add a new fragment, now a combined scan is required
    extra_data = create_uniform_table(min=1000, max=1001, nvec=100, offset=300)
    dataset = lance.write_dataset(
        extra_data, base_dir, mode="append", max_rows_per_file=100
    )
    check_index(has_knn_combined=True, delete_has_happened=False)

    for row_id in sample_delete_indices:
        dataset.delete(f"filterable == {row_id}")

    # Delete some rows, combined KNN still needed
    check_index(has_knn_combined=True, delete_has_happened=True)

    # Optimize the index, combined KNN should no longer be needed
    dataset.optimize.optimize_indices()
    check_index(has_knn_combined=False, delete_has_happened=True)


def test_knn_with_deletions(tmp_path):
    dims = 5
    values = pa.array(
        [x for val in range(50) for x in [float(val)] * 5], type=pa.float32()
    )
    tbl = pa.Table.from_pydict(
        {
            "vector": pa.FixedSizeListArray.from_arrays(values, dims),
            "filterable": pa.array(range(50)),
        }
    )
    dataset = lance.write_dataset(tbl, tmp_path, max_rows_per_group=10)

    dataset.delete("not (filterable % 5 == 0)")

    # Do KNN with k=100, should return 10 vectors
    expected = [
        [0.0] * 5,
        [5.0] * 5,
        [10.0] * 5,
        [15.0] * 5,
        [20.0] * 5,
        [25.0] * 5,
        [30.0] * 5,
        [35.0] * 5,
        [40.0] * 5,
        [45.0] * 5,
    ]

    results = dataset.to_table(
        nearest={"column": "vector", "q": [0.0] * 5, "k": 100}
    ).column("vector")
    assert len(results) == 10

    assert expected == [r.as_py() for r in results]
