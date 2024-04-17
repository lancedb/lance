# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import pickle
from pathlib import Path

import lance
import pyarrow as pa
from lance.schema import LanceSchema


def test_lance_schema(tmp_path: Path):
    data = pa.table({"x": range(2)})
    dataset = lance.write_dataset(data, tmp_path)

    schema = dataset.lance_schema

    assert repr(schema).startswith("Schema {")

    dumped = pickle.dumps(schema)
    loaded = pickle.loads(dumped)
    assert schema == loaded

    assert schema.to_pyarrow() == data.schema
    assert LanceSchema.from_pyarrow(data.schema) == schema
