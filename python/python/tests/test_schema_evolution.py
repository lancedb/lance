#  Copyright (c) 2024. Lance Developers
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

import os
import uuid
from pathlib import Path

import lance
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest


def test_drop_columns(tmp_path: Path):
    dims = 32
    nrows = 512
    values = pc.random(nrows * dims).cast("float32")
    table = pa.table({
        "a": pa.FixedSizeListArray.from_arrays(values, dims),
        "b": range(nrows),
        "c": range(nrows),
    })
    dataset = lance.write_dataset(table, tmp_path)
    dataset.create_index("a", "IVF_PQ", num_partitions=2, num_sub_vectors=1)

    # Drop a column, index is kept
    dataset.drop_columns(["b"])
    assert dataset.schema == pa.schema({
        "a": pa.list_(pa.float32(), dims),
        "c": pa.int64(),
    })
    assert len(dataset.list_indices()) == 1

    # Drop vector column, index is dropped
    dataset.drop_columns(["a"])
    assert dataset.schema == pa.schema({"c": pa.int64()})
    assert len(dataset.list_indices()) == 0

    # Can't drop all columns
    with pytest.raises(ValueError):
        dataset.drop_columns(["c"])


def test_add_columns_udf(tmp_path):
    tab = pa.table({"a": range(100), "b": range(100)})
    dataset = lance.write_dataset(tab, tmp_path, max_rows_per_file=25)

    @lance.batch_udf(
        output_schema=pa.schema([pa.field("double_a", pa.int64())]),
    )
    def double_a(batch):
        assert batch.schema.names == ["a"]
        return pa.record_batch(
            [pa.array([2 * x.as_py() for x in batch["a"]])], ["double_a"]
        )

    dataset.add_columns(double_a, read_columns=["a"])

    expected = tab.append_column("double_a", pa.array([2 * x for x in range(100)]))
    assert expected == dataset.to_table()

    # Check: errors if produces inconsistent schema
    @lance.batch_udf()
    def make_new_col(batch):
        col_name = str(uuid.uuid4())
        return pa.record_batch([batch["a"]], [col_name])

    with pytest.raises(
        Exception, match="Output schema of function does not match the expected schema"
    ):
        dataset.add_columns(make_new_col)

    # Schema inference and Pandas conversion
    @lance.batch_udf()
    def triple_a(batch):
        return pd.DataFrame({"triple_a": [3 * x.as_py() for x in batch["a"]]})

    dataset.add_columns(triple_a, read_columns=["a"])

    expected = expected.append_column("triple_a", pa.array([3 * x for x in range(100)]))
    assert expected == dataset.to_table()


def test_add_columns_udf_caching(tmp_path):
    tab = pa.table({
        "a": range(100),
        "b": range(100),
    })
    dataset = lance.write_dataset(tab, tmp_path, max_rows_per_file=20)

    @lance.batch_udf(checkpoint_file=tmp_path / "cache.sqlite")
    def double_a(batch):
        if batch["a"][0].as_py() >= 50:
            raise RuntimeError("I failed")
        return pa.record_batch([pc.multiply(batch["a"], pa.scalar(2))], ["a_times_2"])

    with pytest.raises(Exception):
        dataset.add_columns(double_a, read_columns=["a"])

    assert dataset.version == 1
    assert "cache.sqlite" in os.listdir(tmp_path)

    @lance.batch_udf(checkpoint_file=tmp_path / "cache.sqlite")
    def double_a(batch):
        # We should skip these batches if they are cached
        # (It can be zero due to schema inference looking at the first batch.)
        assert batch["a"][0].as_py() == 0 or batch["a"][0].as_py() >= 50
        return pa.record_batch([pc.multiply(batch["a"], pa.scalar(2))], ["a_times_2"])

    dataset.add_columns(double_a, read_columns=["a"])
    assert dataset.schema.names == ["a", "b", "a_times_2"]

    assert "cache.sqlite" not in os.listdir(tmp_path)


def test_add_columns_exprs(tmp_path):
    tab = pa.table({"a": range(100)})
    dataset = lance.write_dataset(tab, tmp_path)
    dataset.add_columns({"b": "a + 1"})
    assert dataset.to_table() == pa.table({"a": range(100), "b": range(1, 101)})


def test_query_after_merge(tmp_path):
    # https://github.com/lancedb/lance/issues/1905
    tab = pa.table({
        "id": range(100),
        "vec": pa.FixedShapeTensorArray.from_numpy_ndarray(
            np.random.rand(100, 128).astype("float32")
        ),
    })
    tab2 = pa.table({
        "id": range(100),
        "data": range(100, 200),
    })
    dataset = lance.write_dataset(tab, tmp_path)

    dataset.merge(tab2, left_on="id")

    dataset.to_table(
        nearest=dict(column="vec", k=10, q=np.random.rand(128).astype("float32"))
    )


def test_alter_columns(tmp_path: Path):
    schema = pa.schema([
        pa.field("a", pa.int64(), nullable=False),
        pa.field("b", pa.string(), nullable=False),
    ])
    tab = pa.table(
        {"a": pa.array([1, 2, 3]), "b": pa.array(["a", "b", "c"])}, schema=schema
    )

    dataset = lance.write_dataset(tab, tmp_path)

    dataset.alter_columns(
        {"path": "a", "name": "x", "nullable": True},
        {"path": "b", "name": "y"},
    )

    expected_schema = pa.schema([
        pa.field("x", pa.int64()),
        pa.field("y", pa.string(), nullable=False),
    ])
    assert dataset.schema == expected_schema

    expected_tab = pa.table(
        {"x": pa.array([1, 2, 3]), "y": pa.array(["a", "b", "c"])},
        schema=expected_schema,
    )
    assert dataset.to_table() == expected_tab
