# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import pickle
from pathlib import Path

import lance
import pyarrow as pa
from lance.schema import LanceSchema


def test_lance_schema(tmp_path: Path):
    # Include nested fields to test the reconstruction of the schema
    data = pa.table(
        {
            "x": range(2),
            "s": [{"a": 1, "b": "hello"}, {"a": 2, "b": "world"}],
            "y": [[1.0, 2.0], [3.0, 4.0]],
        }
    )
    dataset = lance.write_dataset(data, tmp_path)

    schema = dataset.lance_schema

    assert repr(schema).startswith("Schema {")

    dumped = pickle.dumps(schema)
    loaded = pickle.loads(dumped)
    assert schema == loaded

    assert schema.to_pyarrow() == data.schema
    assert LanceSchema.from_pyarrow(data.schema) == schema
