# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

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

torch = pytest.importorskip("torch")
from lance.util import validate_vector_index  # noqa: E402
from lance.vector import vec_to_table  # noqa: E402


def create_table(nvec=1000, ndim=128, nans=0):
    mat = np.random.randn(nvec, ndim)
    if nans > 0:
        nans_mat = np.empty((nans, ndim))
        nans_mat[:] = np.nan
        mat = np.concatenate((mat, nans_mat), axis=0)
    price = np.random.rand(nvec + nans) * 100

    def gen_str(n):
        return "".join(random.choices(string.ascii_letters + string.digits, k=n))

    meta = np.array([gen_str(100) for _ in range(nvec + nans)])
    tbl = (
        vec_to_table(data=mat)
        .append_column("price", pa.array(price))
        .append_column("meta", pa.array(meta))
        .append_column("id", pa.array(range(nvec + nans)))
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
        # TODO: _distance shouldn't be returned by default either
        if "_distance" not in expected_columns:
            expected_columns.append("_distance")

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
    run(dataset)


def test_ann(indexed_dataset):
    run(indexed_dataset)


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
        if "vector" not in rs:
            return
        assert rs["vector"][0].as_py() == q

    run(dataset, q=np.array(q), assert_func=func)


def test_index_with_nans(tmp_path):
    # 1024 rows, the entire table should be sampled
    tbl = create_table(nvec=1000, nans=24)

    dataset = lance.write_dataset(tbl, tmp_path)
    dataset = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=16,
        accelerator=torch.device("cpu"),
    )
    validate_vector_index(dataset, "vector")


def test_index_with_no_centroid_movement(tmp_path):
    # this test makes the centroids essentially [1..]
    # this makes sure the early stop condition in the index building code
    # doesn't do divide by zero
    mat = np.concatenate([np.ones((256, 32))])

    tbl = vec_to_table(data=mat)

    dataset = lance.write_dataset(tbl, tmp_path)
    dataset = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=1,
        num_sub_vectors=4,
        accelerator=torch.device("cpu"),
    )
    validate_vector_index(dataset, "vector")


def test_index_with_pq_codebook(tmp_path):
    tbl = create_table(nvec=1024, ndim=128)
    dataset = lance.write_dataset(tbl, tmp_path)
    pq_codebook = np.random.randn(4, 256, 128 // 4).astype(np.float32)

    dataset = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=1,
        num_sub_vectors=4,
        ivf_centroids=np.random.randn(1, 128).astype(np.float32),
        pq_codebook=pq_codebook,
    )
    validate_vector_index(dataset, "vector", refine_factor=10, pass_threshold=0.99)

    pq_codebook = pa.FixedShapeTensorArray.from_numpy_ndarray(pq_codebook)

    dataset = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=1,
        num_sub_vectors=4,
        ivf_centroids=np.random.randn(1, 128).astype(np.float32),
        pq_codebook=pq_codebook,
        replace=True,
    )
    validate_vector_index(dataset, "vector", refine_factor=10, pass_threshold=0.99)


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


def test_create_ivf_hnsw_pq_index(dataset, tmp_path):
    assert not dataset.has_index
    ann_ds = lance.write_dataset(dataset.to_table(), tmp_path / "indexed.lance")
    ann_ds = ann_ds.create_index(
        "vector",
        index_type="IVF_HNSW_PQ",
        num_partitions=4,
        num_sub_vectors=16,
    )
    assert ann_ds.list_indices()[0]["fields"] == ["vector"]


def test_create_ivf_hnsw_sq_index(dataset, tmp_path):
    assert not dataset.has_index
    ann_ds = lance.write_dataset(dataset.to_table(), tmp_path / "indexed.lance")
    ann_ds = ann_ds.create_index(
        "vector",
        index_type="IVF_HNSW_SQ",
        num_partitions=4,
        num_sub_vectors=16,
    )
    assert ann_ds.list_indices()[0]["fields"] == ["vector"]


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
        # increase 1 miss of index_cache.metadata_cache
        assert dataset_with_index.stats.index_stats("non-existent_idx")
    with pytest.raises(KeyError, match='Index "" not found'):
        # increase 1 miss of index_cache.metadata_cache
        assert dataset_with_index.stats.index_stats("")
    with pytest.raises(TypeError):
        dataset_with_index.stats.index_stats()

    # increase 1 hit of index_cache.metadata_cache
    actual_statistics = dataset_with_index.stats.index_stats("vector_idx")
    assert actual_statistics["num_indexed_rows"] == 1000
    assert actual_statistics["num_unindexed_rows"] == 0

    idx_stats = actual_statistics["indices"][0]
    partitions = idx_stats.pop("partitions")
    idx_stats.pop("centroids")
    assert idx_stats == expected_statistics
    assert len(partitions) == 5
    partition_keys = {"size"}
    assert all([partition_keys == set(p.keys()) for p in partitions])


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
            plan = dataset.scanner(nearest=query).explain_plan(verbose=True)
            assert ("KNNVectorDistance" in plan) == has_knn_combined
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


def test_index_cache_size(tmp_path):
    rng = np.random.default_rng(seed=42)

    def query_index(ds, ntimes, q=None):
        ndim = ds.schema[0].type.list_size
        for _ in range(ntimes):
            ds.to_table(
                nearest={
                    "column": "vector",
                    "q": q if q is not None else rng.standard_normal(ndim),
                },
            )

    tbl = create_table(nvec=1024, ndim=16)
    dataset = lance.write_dataset(tbl, tmp_path / "test")

    dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=128,
        num_sub_vectors=2,
        index_cache_size=10,
    )

    indexed_dataset = lance.dataset(tmp_path / "test", index_cache_size=0)
    # when there is no hit, the hit rate is hard coded to 1.0
    assert np.isclose(indexed_dataset._ds.index_cache_hit_rate(), 1.0)
    query_index(indexed_dataset, 1)
    # index cache is size=0, there should be no hit
    assert np.isclose(indexed_dataset._ds.index_cache_hit_rate(), 0.0)

    indexed_dataset = lance.dataset(tmp_path / "test", index_cache_size=1)
    # query using the same vector, we should get a very high hit rate
    query_index(indexed_dataset, 100, q=rng.standard_normal(16))
    assert indexed_dataset._ds.index_cache_hit_rate() > 0.99

    last_hit_rate = indexed_dataset._ds.index_cache_hit_rate()

    # send a few queries with different vectors, the hit rate should drop
    query_index(indexed_dataset, 128)

    assert last_hit_rate > indexed_dataset._ds.index_cache_hit_rate()


def test_f16_index(tmp_path: Path):
    DIM = 64
    uri = tmp_path / "f16data.lance"
    f16_data = np.random.uniform(0, 1, 2048 * DIM).astype(np.float16)
    fsl = pa.FixedSizeListArray.from_arrays(f16_data, DIM)
    tbl = pa.Table.from_pydict({"vector": fsl})
    dataset = lance.write_dataset(tbl, uri)
    dataset.create_index(
        "vector", index_type="IVF_PQ", num_partitions=4, num_sub_vectors=2
    )

    q = np.random.uniform(0, 1, DIM).astype(np.float16)
    rst = dataset.to_table(
        nearest={
            "column": "vector",
            "q": q,
            "k": 10,
        }
    )

    assert rst.schema.field("vector").type.value_type == pa.float16()
    assert len(rst) == 10


def test_vector_with_nans(tmp_path: Path):
    DIM = 32
    TOTAL = 2048
    data = np.random.uniform(0, 1, TOTAL * DIM).astype(np.float32)

    # Put the 1st vector as NaN.
    np.put(data, range(DIM, 2 * DIM), np.nan)
    fsl = pa.FixedSizeListArray.from_arrays(data, DIM)
    tbl = pa.Table.from_pydict({"vector": fsl})

    dataset = lance.write_dataset(tbl, tmp_path)
    row = dataset._take_rows([1])
    assert row["vector"]

    ds = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=2,
        num_sub_vectors=2,
        replace=True,
    )
    tbl = ds.to_table(
        nearest={"column": "vector", "q": data[0:DIM], "k": TOTAL, "nprobes": 2},
        with_row_id=True,
    )
    assert len(tbl) == TOTAL - 1
    assert 1 not in tbl["_rowid"].to_numpy(), "Row with ID 1 is not in the index"


def test_validate_vector_index(tmp_path: Path):
    # make sure the sanity check is correctly catchting issues
    ds = lance.write_dataset(create_table(), tmp_path)
    validate_vector_index(ds, "vector", sample_size=100)

    called = False

    def direct_first_call_to_new_table(*args, **kwargs):
        nonlocal called
        if called:
            return ds.to_table(*args, **kwargs)
        called = True
        return create_table()

    # return a new random table so things fail
    ds.sample = direct_first_call_to_new_table
    with pytest.raises(ValueError, match="Vector index failed sanity check"):
        validate_vector_index(ds, "vector", sample_size=100)


def test_dynamic_projection_with_vectors_index(tmp_path: Path):
    ds = lance.write_dataset(create_table(), tmp_path)
    ds = ds.create_index(
        "vector", index_type="IVF_PQ", num_partitions=4, num_sub_vectors=16
    )

    res = ds.to_table(
        nearest={
            "column": "vector",
            "q": np.random.randn(128),
        },
        columns={
            "vec": "vector",
            "vec_f16": "_cast_list_f16(vector)",
        },
    )

    # TODO: _distance shouldn't be returned by default
    assert res.column_names == ["vec", "vec_f16", "_distance"]

    original = np.stack(res["vec"].to_numpy())
    casted = np.stack(res["vec_f16"].to_numpy())

    assert (original.astype(np.float16) == casted).all()


def test_index_cast_centroids(tmp_path):
    tbl = create_table(nvec=1000)

    dataset = lance.write_dataset(tbl, tmp_path)
    dataset = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=16,
        accelerator=torch.device("cpu"),
    )

    # Get the centroids
    index_name = dataset.list_indices()[0]["name"]
    index_stats = dataset.stats.index_stats(index_name)
    centroids = index_stats["indices"][0]["centroids"]
    values = pa.array([x for arr in centroids for x in arr], pa.float32())
    centroids = pa.FixedSizeListArray.from_arrays(values, 128)

    dataset.alter_columns(dict(path="vector", data_type=pa.list_(pa.float16(), 128)))

    # centroids are f32, but the column is now f16
    dataset = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=16,
        accelerator=torch.device("cpu"),
        ivf_centroids=centroids,
    )
