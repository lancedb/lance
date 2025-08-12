# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Tests for creating empty indices with train=False."""

import lance
import pyarrow as pa
import pyarrow.compute as pc


def test_create_empty_scalar_index():
    data = pa.table({"id": range(100)})
    dataset = lance.write_dataset(data, "memory://")

    # Passing train=False to create an empty index
    dataset.create_scalar_index("id", "BTREE", train=False)

    # Verify index exists and has correct stats
    indices = dataset.list_indices()
    assert len(indices) == 1
    assert indices[0]["type"] == "BTree"
    stats = dataset.stats.index_stats(indices[0]["name"])
    assert stats["num_indexed_rows"] == 0
    assert stats["num_unindexed_rows"] == dataset.count_rows()


def test_create_empty_vector_index():
    dim = 32
    values = pc.random(100 * dim).cast(pa.float32())
    vectors = pa.FixedSizeListArray.from_arrays(values, dim)
    data = pa.table({"vector": vectors})
    dataset = lance.write_dataset(data, "memory://")

    # Currently, vector indices with train=False may fall back to train=True
    try:
        dataset.create_index(
            "vector", "IVF_PQ", num_partitions=10, num_sub_vectors=8, train=False
        )
    except ValueError as e:
        # If it fails, it should be with the "not implemented" message
        error_msg = str(e).lower()
        assert "not yet implemented" in error_msg or "not implemented" in error_msg
