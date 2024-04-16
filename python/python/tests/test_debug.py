# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from pathlib import Path

import lance
import pyarrow as pa
from lance.debug import (
    format_fragment,
    format_manifest,
    format_schema,
    list_transactions,
)


def test_format_schema(tmp_path: Path):
    schema = pa.schema({
        "a": pa.int64(),
        "b": pa.string(),
        "c": pa.bool_(),
    })
    table = pa.Table.from_batches([], schema)
    dataset = lance.write_dataset(table, tmp_path)

    output = format_schema(dataset)
    assert output.startswith("Schema")
    assert (
        'Field {\n            name: "a",\n            id: 0,\n            parent_id:'
        ' -1,\n            logical_type: LogicalType(\n                "int64"'
        in output
    )
    assert (
        'Field {\n            name: "b",\n            id: 1,\n            parent_id:'
        ' -1,\n            logical_type: LogicalType(\n                "string"'
        in output
    )
    assert (
        'Field {\n            name: "c",\n            id: 2,\n            parent_id:'
        ' -1,\n            logical_type: LogicalType(\n                "bool"' in output
    )


def test_format_manifest(tmp_path: Path):
    table = pa.table({"x": range(10)})
    dataset = lance.write_dataset(table, tmp_path)

    output = format_manifest(dataset)
    assert output.startswith("Manifest {")
    assert "Schema {" in output
    assert (
        'writer_version: Some(\n        WriterVersion {\n            library: "lance"'
        in output
    )
    assert "fragments: [\n        Fragment {" in output


def test_format_fragment(tmp_path: Path):
    table = pa.table({"x": range(10)})
    dataset = lance.write_dataset(table, tmp_path)
    dataset.add_columns({"y": "x + 1", "z": "'hello'"})

    fragment = dataset.get_fragments()[0].metadata

    output = format_fragment(fragment)

    assert output.startswith("PrettyPrintableFragment {")
    assert "files: [" in output
    assert "schema: Schema {" in output


def test_list_transactions(tmp_path: Path):
    table = pa.table({"x": range(10)})
    dataset = lance.write_dataset(table, tmp_path)
    dataset = lance.write_dataset(table, tmp_path, mode="append")
    dataset.delete("x = 5")
    dataset.update({"x": "1"}, where="x = 2")

    results = list_transactions(dataset)
    assert len(results) == 4
    assert all(r.startswith("Transaction {") for r in results)
    assert "operation: Update {" in results[0]
    assert "operation: Delete {" in results[1]
    assert "operation: Append {" in results[2]
    assert "operation: Overwrite {" in results[3]

    results_2 = list_transactions(dataset, max_transactions=2)
    assert len(results_2) == 2
    assert results_2 == results[:2]
