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

    fields = schema.fields()
    assert len(fields) == 3
    assert fields[0].name() == "x"
    assert fields[0].id() == 0
    assert fields[1].name() == "s"
    assert fields[1].id() == 1

    s_children = fields[1].children()
    assert len(s_children) == 2
    assert s_children[0].name() == "a"
    assert s_children[0].id() == 2
    assert s_children[1].name() == "b"
    assert s_children[1].id() == 3

    assert fields[2].name() == "y"
    assert fields[2].id() == 4

    l_children = fields[2].children()
    assert len(l_children) == 1
    assert l_children[0].name() == "item"
    assert l_children[0].id() == 5

    # Changing column name does not change the id
    dataset.alter_columns({"path": "s.a", "name": "new_name"})
    schema = dataset.lance_schema
    fields = schema.fields()
    s_fields = fields[1].children()
    assert s_fields[0].name() == "new_name"
    assert s_fields[0].id() == 2
