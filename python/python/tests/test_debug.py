# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from pathlib import Path

import lance
import pyarrow as pa
from lance.debug import list_transactions, print_manifest, print_schema


def test_print_schema(capfd, tmp_path: Path):
    schema = pa.schema({
        "a": pa.int64(),
        "b": pa.string(),
        "c": pa.bool_(),
    })
    table = pa.Table.from_batches([], schema)
    dataset = lance.write_dataset(table, tmp_path)

    print_schema(dataset)
    captured = capfd.readouterr()
    # breakpoint()
    assert captured.out.startswith("Schema")
    assert (
        'Field { name: "a", id: 0, parent_id: -1, logical_type: LogicalType("int64")'
        in captured.out
    )
    assert (
        'Field { name: "b", id: 1, parent_id: -1, logical_type: LogicalType("string")'
        in captured.out
    )
    assert (
        'Field { name: "c", id: 2, parent_id: -1, logical_type: LogicalType("bool")'
        in captured.out
    )


def test_print_manifest(capfd, tmp_path: Path):
    table = pa.table({"x": range(10)})
    dataset = lance.write_dataset(table, tmp_path)

    print_manifest(dataset)
    captured = capfd.readouterr()
    assert captured.out.startswith("Manifest {")
    assert "Schema {" in captured.out
    assert 'writer_version: Some(WriterVersion { library: "lance"' in captured.out
    assert "fragments: [Fragment { id: 0," in captured.out


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
