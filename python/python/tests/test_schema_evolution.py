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
from pathlib import Path

import lance
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
