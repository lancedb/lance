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
import shutil
from pathlib import Path

import lance
import pyarrow as pa
import pytest
from lance._datagen import rand_batches

N_DIMS = 768
KiB = 1024
MiB = 1024 * KiB
GiB = 1024 * MiB


# Mostly vector data.  One id column, a caption, and an embedding vector
def create_captioned_image_data(num_bytes):
    schema = pa.schema([
        pa.field("int32", pa.int32()),
        pa.field("text", pa.utf8()),
        pa.field("vector", pa.list_(pa.float32(), N_DIMS)),
    ])
    return schema, rand_batches(
        schema, num_batches=8, batch_size_bytes=int(num_bytes / 8)
    )


# Purely scalar data (schema based on TPC-H lineitem table)
def create_scalar_data(num_bytes):
    schema = pa.schema([
        pa.field("l_orderkey", pa.int64()),
        pa.field("l_partkey", pa.int64()),
        pa.field("l_suppkey", pa.int64()),
        pa.field("l_linenumber", pa.int64()),
        pa.field("l_quantity", pa.float64()),
        pa.field("l_extendedprice", pa.float64()),
        pa.field("l_discount", pa.float64()),
        pa.field("l_tax", pa.float64()),
        pa.field("l_returnflag", pa.utf8()),
        pa.field("l_linestatus", pa.utf8()),
        pa.field("l_shipdate", pa.date32()),
        pa.field("l_commitdate", pa.date32()),
        pa.field("l_receiptdate", pa.date32()),
        pa.field("l_shipinstruct", pa.utf8()),
        pa.field("l_shipmode", pa.utf8()),
        pa.field("l_comment", pa.utf8()),
    ])
    return schema, rand_batches(
        schema, num_batches=8, batch_size_bytes=int(num_bytes / 8)
    )


def do_write_dataset(data, tmp_path, schema):
    shutil.rmtree(tmp_path)
    lance.write_dataset(data, tmp_path, schema=schema)


def write_dataset_benchmark(benchmark, tmpdir_factory, data_fn):
    tmp_path = Path(tmpdir_factory.mktemp("dataset_ops"))
    schema, data = data_fn(num_bytes=1 * GiB)
    benchmark(do_write_dataset, data, tmp_path, schema)


@pytest.mark.benchmark(group="create_dataset")
def test_captioned_image(tmpdir_factory, benchmark):
    write_dataset_benchmark(benchmark, tmpdir_factory, create_captioned_image_data)


@pytest.mark.benchmark(group="create_dataset")
def test_scalar(tmpdir_factory, benchmark):
    write_dataset_benchmark(benchmark, tmpdir_factory, create_scalar_data)
