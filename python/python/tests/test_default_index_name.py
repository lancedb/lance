# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import lance
import numpy as np
import pyarrow as pa


def test_default_vector_index_name(tmp_path):
    # Create simple vector table
    dim = 16
    data = np.random.randn(1024, dim).astype(np.float32)
    tbl = pa.Table.from_pydict(
        {
            "vector": pa.array(data.tolist(), type=pa.list_(pa.float32(), dim)),
            "id": pa.array(list(range(1024))),
        }
    )

    ds = lance.write_dataset(tbl, tmp_path)
    ds = ds.create_index(
        "vector", index_type="IVF_PQ", num_partitions=2, num_sub_vectors=4
    )

    # Default name should include index type
    idx_meta = ds.list_indices()[0]
    assert idx_meta["name"] == "vector_ivf_pq_idx"

    # Stats should be retrievable via the default name
    stats = ds.stats.index_stats("vector_ivf_pq_idx")
    assert stats["indices"][0]["index_type"] == "IVF_PQ"


def test_default_scalar_index_name(tmp_path):
    tbl = pa.Table.from_pydict(
        {"doc": pa.array(["hello world", "test", "lance index"], type=pa.string())}
    )
    ds = lance.write_dataset(tbl, tmp_path)
    ds.create_scalar_index("doc", index_type="INVERTED")

    # Default name should include index type
    stats = ds.stats.index_stats("doc_inverted_idx")
    assert stats["index_type"] == "Inverted"
