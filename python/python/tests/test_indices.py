# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
import os

import lance
import numpy as np
import pyarrow as pa
import pytest
from lance.file import LanceFileReader
from lance.indices import IndicesBuilder, IvfModel, PqModel

NUM_ROWS_PER_FRAGMENT = 10000
DIMENSION = 128
NUM_SUBVECTORS = 8
NUM_FRAGMENTS = 3
NUM_ROWS = NUM_ROWS_PER_FRAGMENT * NUM_FRAGMENTS
NUM_PARTITIONS = round(np.sqrt(NUM_ROWS))


@pytest.fixture(
    params=[np.float16, np.float32, np.float64],
    ids=["f16", "f32", "f64"],
)
def rand_dataset(tmpdir, request):
    vectors = np.random.randn(NUM_ROWS, DIMENSION).astype(request.param)
    vectors.shape = -1
    vectors = pa.FixedSizeListArray.from_arrays(vectors, DIMENSION)
    table = pa.Table.from_arrays([vectors], names=["vectors"])
    uri = str(tmpdir / "dataset")

    ds = lance.write_dataset(table, uri, max_rows_per_file=NUM_ROWS_PER_FRAGMENT)
    return ds


def test_ivf_centroids(tmpdir, rand_dataset):
    ivf = IndicesBuilder(rand_dataset, "vectors").train_ivf(sample_rate=16)

    assert ivf.distance_type == "l2"
    assert len(ivf.centroids) == NUM_PARTITIONS

    ivf.save(str(tmpdir / "ivf"))
    reloaded = IvfModel.load(str(tmpdir / "ivf"))
    assert reloaded.distance_type == "l2"
    assert ivf.centroids == reloaded.centroids


@pytest.mark.cuda
def test_ivf_centroids_cuda(rand_dataset):
    ivf = IndicesBuilder(rand_dataset, "vectors").train_ivf(
        sample_rate=16, accelerator="cuda"
    )

    assert ivf.distance_type == "l2"
    assert len(ivf.centroids) == NUM_PARTITIONS


def test_ivf_centroids_distance_type(tmpdir, rand_dataset):
    def check(distance_type):
        ivf = IndicesBuilder(rand_dataset, "vectors").train_ivf(
            sample_rate=16, distance_type=distance_type
        )
        assert ivf.distance_type == distance_type
        ivf.save(str(tmpdir / "ivf"))
        reloaded = IvfModel.load(str(tmpdir / "ivf"))
        assert reloaded.distance_type == distance_type

    check("l2")
    check("cosine")
    check("dot")


def test_num_partitions(rand_dataset):
    ivf = IndicesBuilder(rand_dataset, "vectors").train_ivf(
        sample_rate=16, num_partitions=10
    )
    assert ivf.num_partitions == 10


@pytest.fixture
def rand_ivf(rand_dataset):
    dtype = rand_dataset.schema.field("vectors").type.value_type.to_pandas_dtype()
    centroids = np.random.rand(DIMENSION * 100).astype(dtype)
    centroids = pa.FixedSizeListArray.from_arrays(centroids, DIMENSION)
    return IvfModel(centroids, "l2")


def test_gen_pq(tmpdir, rand_dataset, rand_ivf):
    pq = IndicesBuilder(rand_dataset, "vectors").train_pq(rand_ivf, sample_rate=2)
    assert pq.dimension == DIMENSION
    assert pq.num_subvectors == NUM_SUBVECTORS

    pq.save(str(tmpdir / "pq"))
    reloaded = PqModel.load(str(tmpdir / "pq"))
    assert pq.dimension == reloaded.dimension
    assert pq.codebook == reloaded.codebook


@pytest.mark.cuda
def test_assign_partitions(rand_dataset, rand_ivf):
    builder = IndicesBuilder(rand_dataset, "vectors")

    partitions_uri = builder.assign_ivf_partitions(rand_ivf, accelerator="cuda")

    partitions = lance.dataset(partitions_uri)
    found_row_ids = set()
    for batch in partitions.to_batches():
        row_ids = batch["row_id"]
        for row_id in row_ids:
            found_row_ids.add(row_id)
        part_ids = batch["partition"]
        for part_id in part_ids:
            assert part_id.as_py() < 100
    assert len(found_row_ids) == rand_dataset.count_rows()


@pytest.fixture
def rand_pq(rand_dataset, rand_ivf):
    dtype = rand_dataset.schema.field("vectors").type.value_type.to_pandas_dtype()
    codebook = np.random.rand(DIMENSION * 256).astype(dtype)
    codebook = pa.FixedSizeListArray.from_arrays(codebook, DIMENSION)
    pq = PqModel(NUM_SUBVECTORS, codebook)
    return pq


def test_vector_transform(tmpdir, rand_dataset, rand_ivf, rand_pq):
    fragments = list(rand_dataset.get_fragments())

    builder = IndicesBuilder(rand_dataset, "vectors")
    uri = str(tmpdir / "transformed")
    builder.transform_vectors(rand_ivf, rand_pq, uri, fragments=fragments)

    reader = LanceFileReader(uri)
    assert reader.metadata().num_rows == (NUM_ROWS_PER_FRAGMENT * len(fragments))
    data = next(reader.read_all(batch_size=10000).to_batches())

    row_id = data.column("_rowid")
    assert row_id.type == pa.uint64()

    pq_code = data.column("__pq_code")
    assert pq_code.type == pa.list_(pa.uint8(), 8)

    part_id = data.column("__ivf_part_id")
    assert part_id.type == pa.uint32()

    # We need to close the file to be able to overwrite it on Windows.
    del reader

    # test when fragments = None
    builder.transform_vectors(rand_ivf, rand_pq, uri, fragments=None)
    reader = LanceFileReader(uri)

    assert reader.metadata().num_rows == (NUM_ROWS_PER_FRAGMENT * NUM_FRAGMENTS)


@pytest.mark.cuda
def test_vector_transform_with_precomputed_partitions(
    tmpdir, rand_dataset, rand_ivf, rand_pq
):
    fragments = list(rand_dataset.get_fragments())
    builder = IndicesBuilder(rand_dataset, "vectors")

    partitions = builder.assign_ivf_partitions(rand_ivf, accelerator="cuda")

    uri = str(tmpdir / "transformed")
    builder.transform_vectors(
        rand_ivf, rand_pq, uri, fragments=fragments, partition_ds_uri=partitions
    )

    reader = LanceFileReader(uri)
    assert reader.metadata().num_rows == (NUM_ROWS_PER_FRAGMENT * len(fragments))
    data = next(reader.read_all(batch_size=10000).to_batches())

    row_id = data.column("_rowid")
    assert row_id.type == pa.uint64()

    pq_code = data.column("__pq_code")
    assert pq_code.type == pa.list_(pa.uint8(), 8)

    part_id = data.column("__ivf_part_id")
    assert part_id.type == pa.uint32()

    # We need to close the file to be able to overwrite it on Windows.
    del reader

    # test when fragments = None
    builder.transform_vectors(rand_ivf, rand_pq, uri, fragments=None)
    reader = LanceFileReader(uri)

    assert reader.metadata().num_rows == (NUM_ROWS_PER_FRAGMENT * NUM_FRAGMENTS)


def test_shuffle_vectors(tmpdir, rand_dataset, rand_ivf, rand_pq):
    builder = IndicesBuilder(rand_dataset, "vectors")
    uri = str(tmpdir / "transformed_shuffle")
    builder.transform_vectors(rand_ivf, rand_pq, uri, fragments=None)

    # test shuffle for transformed vectors
    filenames = builder.shuffle_transformed_vectors(
        ["transformed_shuffle"], str(tmpdir), rand_ivf, "sorted"
    )

    for fname in filenames:
        full_path = str(tmpdir / fname)
        assert os.path.getsize(full_path) > 0


def test_load_shuffled_vectors(tmpdir, rand_dataset, rand_ivf, rand_pq):
    fragments = list(rand_dataset.get_fragments())

    fragments1 = fragments[:1]
    fragments2 = fragments[1:]

    builder = IndicesBuilder(rand_dataset, "vectors")

    uri_1 = str(tmpdir / "transformed1")
    builder.transform_vectors(rand_ivf, rand_pq, uri_1, fragments=fragments1)
    filenames1 = builder.shuffle_transformed_vectors(
        ["transformed1"], str(tmpdir), rand_ivf, "frags1_sorted"
    )

    uri_2 = str(tmpdir / "transformed2")
    builder.transform_vectors(rand_ivf, rand_pq, uri_2, fragments=fragments2)
    filenames2 = builder.shuffle_transformed_vectors(
        ["transformed2"], str(tmpdir), rand_ivf, "frags2_sorted"
    )

    sorted_filenames = filenames1 + filenames2
    builder.load_shuffled_vectors(sorted_filenames, str(tmpdir), rand_ivf, rand_pq)

    final_ds = lance.dataset(str(tmpdir / "dataset"))
    assert final_ds.has_index
    assert final_ds.list_indices()[0]["fields"] == ["vectors"]
    assert len(final_ds.list_indices()[0]["fragment_ids"]) == NUM_FRAGMENTS
