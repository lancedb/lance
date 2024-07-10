# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from lance.file import LanceFileReader, LanceFileWriter


def test_file_writer(tmp_path):
    path = tmp_path / "foo.lance"
    schema = pa.schema([pa.field("a", pa.int64())])
    with LanceFileWriter(str(path), schema) as writer:
        writer.write_batch(pa.table({"a": [1, 2, 3]}))
    reader = LanceFileReader(str(path))
    metadata = reader.metadata()
    assert metadata.num_rows == 3


def test_write_no_schema(tmp_path):
    path = tmp_path / "foo.lance"
    with LanceFileWriter(str(path)) as writer:
        writer.write_batch(pa.table({"a": [1, 2, 3]}))
    reader = LanceFileReader(str(path))
    assert reader.read_all().to_table() == pa.table({"a": [1, 2, 3]})


def test_no_schema_no_data(tmp_path):
    path = tmp_path / "foo.lance"
    with pytest.raises(
        ValueError, match="Schema is unknown and file cannot be created"
    ):
        with LanceFileWriter(str(path)) as _:
            pass


def test_schema_only(tmp_path):
    path = tmp_path / "foo.lance"
    schema = pa.schema([pa.field("a", pa.int64())])
    with LanceFileWriter(str(path), schema=schema) as _:
        pass
    reader = LanceFileReader(str(path))
    assert reader.metadata().schema == schema


def test_aborted_write(tmp_path):
    path = tmp_path / "foo.lance"
    schema = pa.schema([pa.field("a", pa.int64())])
    writer = LanceFileWriter(str(path), schema)
    writer.write_batch(pa.table({"a": [1, 2, 3]}))
    del writer
    assert not path.exists()


def test_multiple_close(tmp_path):
    path = tmp_path / "foo.lance"
    schema = pa.schema([pa.field("a", pa.int64())])
    writer = LanceFileWriter(str(path), schema)
    writer.write_batch(pa.table({"a": [1, 2, 3]}))
    writer.close()
    writer.close()


def test_take(tmp_path):
    path = tmp_path / "foo.lance"
    schema = pa.schema([pa.field("a", pa.int64())])
    writer = LanceFileWriter(str(path), schema)
    writer.write_batch(pa.table({"a": [i for i in range(100)]}))
    writer.close()

    reader = LanceFileReader(str(path))
    # Can't read out of range
    with pytest.raises(ValueError):
        reader.take_rows([0, 100]).to_table()

    table = reader.take_rows([0, 77, 83]).to_table()
    assert table == pa.table({"a": [0, 77, 83]})


def check_round_trip(tmp_path, table):
    path = tmp_path / "foo.lance"
    with LanceFileWriter(str(path), table.schema) as writer:
        for batch in table.to_batches():
            writer.write_batch(pa.table(batch))
    reader = LanceFileReader(str(path))
    result = reader.read_all().to_table()
    assert result == table


def test_different_types(tmp_path):
    check_round_trip(
        tmp_path,
        pa.table(
            {
                "large_string": pa.array(["foo", "bar", "baz"], pa.large_string()),
                "large_binary": pa.array([b"foo", b"bar", b"baz"], pa.large_binary()),
            }
        ),
    )


def test_with_nulls(tmp_path):
    check_round_trip(
        tmp_path,
        pa.table(
            {
                "some_null_1": pa.array([1, 2, None], pa.int64()),
                "some_null_2": pa.array([None, None, 3], pa.int64()),
                "all_null": pa.array([None, None, None], pa.int64()),
                "null_strings": pa.array([None, "foo", None], pa.string()),
            }
        ),
    )


def test_round_trip(tmp_path):
    path = tmp_path / "foo.lance"
    schema = pa.schema([pa.field("a", pa.int64())])
    data = pa.table({"a": [1, 2, 3]})
    with LanceFileWriter(str(path), schema) as writer:
        writer.write_batch(data)
    reader = LanceFileReader(str(path))
    result = reader.read_all().to_table()
    assert result == data

    # TODO: Currently fails, need to fix reader
    # result = reader.read_range(1, 1).to_table()
    # assert result == pa.table({"a": [2]})

    # TODO: Test reading invalid ranges
    # TODO: Test invalid batch sizes


def test_metadata(tmp_path):
    path = tmp_path / "foo.lance"
    schema = pa.schema([pa.field("a", pa.int64())])
    data = pa.table({"a": [1, 2, 3]})
    with LanceFileWriter(str(path), schema) as writer:
        writer.write_batch(data)
    reader = LanceFileReader(str(path))
    metadata = reader.metadata()

    assert metadata.schema == schema
    assert metadata.num_rows == 3
    assert metadata.num_global_buffer_bytes > 0
    assert metadata.num_column_metadata_bytes > 0
    assert metadata.num_data_bytes == 55
    assert len(metadata.columns) == 1

    column = metadata.columns[0]
    assert len(column.column_buffers) == 0
    assert len(column.pages) == 1

    page = column.pages[0]
    assert len(page.buffers) == 1
    assert page.buffers[0].position == 0
    assert page.buffers[0].size == 24

    assert len(page.encoding) > 0


def test_round_trip_parquet(tmp_path):
    pq_path = tmp_path / "foo.parquet"
    table = pa.table({"int": [1, 2], "list_str": [["x", "yz", "abc"], ["foo", "bar"]]})
    pq.write_table(table, str(pq_path))
    table = pq.read_table(str(pq_path))

    lance_path = tmp_path / "foo.lance"
    with LanceFileWriter(str(lance_path)) as writer:
        writer.write_batch(table)

    reader = LanceFileReader(str(lance_path))
    round_tripped = reader.read_all().to_table()
    assert round_tripped == table


def test_list_field_name(tmp_path):
    weird_field = pa.field("why does this name even exist", pa.string())
    weird_string_type = pa.list_(weird_field)
    schema = pa.schema([pa.field("list_str", weird_string_type)])
    table = pa.table({"list_str": [["x", "yz", "abc"], ["foo", "bar"]]}, schema=schema)

    path = tmp_path / "foo.lance"
    with LanceFileWriter(str(path)) as writer:
        writer.write_batch(table)

    reader = LanceFileReader(str(path))
    round_tripped = reader.read_all().to_table()

    assert round_tripped == table
    assert round_tripped.schema.field("list_str").type == weird_string_type
