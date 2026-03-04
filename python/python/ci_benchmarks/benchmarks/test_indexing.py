# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
from pathlib import Path

import lance
import pyarrow as pa
import pytest
from lance._datagen import rand_batches


@pytest.mark.parametrize(
    "data_type", [pa.int64(), pa.string()], ids=["int64", "string"]
)
@pytest.mark.parametrize("index_type", ["btree", "bitmap", "zonemap", "bloomfilter"])
@pytest.mark.io_memory_benchmark()
def test_io_mem_build_scalar_index(
    io_mem_benchmark, data_type: pa.DataType, index_type: str, tmp_path: Path
):
    metadata = None
    if index_type == "bitmap":
        metadata = {b"lance-datagen:cardinality": b"1000"}
    schema = pa.schema([pa.field("col", data_type, metadata=metadata)])

    # 100MB
    data = rand_batches(schema, num_batches=100, batch_size_bytes=1024 * 1024)
    ds = lance.write_dataset(data, tmp_path)

    def build_index(ds):
        ds.create_scalar_index("col", index_type, replace=True)

    io_mem_benchmark(build_index, ds, warmup=False)


@pytest.mark.parametrize("with_positions", [True, False])
@pytest.mark.io_memory_benchmark()
def test_io_mem_build_fts(io_mem_benchmark, with_positions: bool, tmp_path: Path):
    schema = pa.schema(
        [
            pa.field(
                "text", pa.string(), metadata={"lance-datagen:content-type": "sentence"}
            )
        ]
    )
    # 100MB
    data = rand_batches(schema, num_batches=100, batch_size_bytes=1024 * 1024)
    ds = lance.write_dataset(data, tmp_path)

    def build_index(ds):
        ds.create_scalar_index("text", "INVERTED", with_position=True, replace=True)

    io_mem_benchmark(build_index, ds, warmup=False)


@pytest.mark.io_memory_benchmark()
def test_io_mem_build_ivf_pq(io_mem_benchmark, tmp_path: Path):
    schema = pa.schema([pa.field("vector", pa.list_(pa.float32(), 1024))])
    # 1GB
    data = rand_batches(schema, num_batches=100, batch_size_bytes=10 * 1024 * 1024)
    ds = lance.write_dataset(data, tmp_path)

    def build_index(ds):
        ds.create_index(
            "vector",
            index_type="IVF_PQ",
            num_partitions=32,
            num_sub_vectors=4,
            replace=True,
        )

    io_mem_benchmark(build_index, ds, warmup=False)
