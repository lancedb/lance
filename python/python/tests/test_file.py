# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import pyarrow as pa
from lance.file import LanceFileReader, LanceFileWriter


def test_file_writer(tmp_path):
    path = tmp_path / "foo.lance"
    schema = pa.schema([pa.field("a", pa.int64())])
    with LanceFileWriter(str(path), schema) as writer:
        writer.write_batch(pa.table({"a": [1, 2, 3]}))
    reader = LanceFileReader(str(path), schema)
    metadata = reader.metadata()
    assert metadata.num_rows == 3


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


def test_different_types(tmp_path):
    path = tmp_path / "foo.lance"
    schema = pa.schema(
        [
            pa.field("large_string", pa.large_string()),
            pa.field("large_binary", pa.large_binary()),
        ]
    )
    writer = LanceFileWriter(str(path), schema)
    data = pa.table(
        {
            "large_string": pa.array(["foo", "bar", "baz"], pa.large_string()),
            "large_binary": pa.array([b"foo", b"bar", b"baz"], pa.large_binary()),
        }
    )
    writer.write_batch(data)
    writer.close()

    reader = LanceFileReader(str(path), schema)
    result = reader.read_all().to_table()
    assert result == data


def test_round_trip(tmp_path):
    path = tmp_path / "foo.lance"
    schema = pa.schema([pa.field("a", pa.int64())])
    data = pa.table({"a": [1, 2, 3]})
    with LanceFileWriter(str(path), schema) as writer:
        writer.write_batch(data)
    reader = LanceFileReader(str(path), schema)
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
    reader = LanceFileReader(str(path), schema)
    metadata = reader.metadata()

    assert metadata.schema == schema
    assert metadata.num_rows == 3
    assert metadata.num_global_buffer_bytes > 0
    assert metadata.num_column_metadata_bytes > 0
    assert metadata.num_data_bytes == 24
    assert len(metadata.columns) == 1

    column = metadata.columns[0]
    assert len(column.column_buffers) == 0
    assert len(column.pages) == 1

    page = column.pages[0]
    assert len(page.buffers) == 1
    assert page.buffers[0].position == 0
    assert page.buffers[0].size == 24

    assert len(page.encoding) > 0
