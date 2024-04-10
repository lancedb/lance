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

import pyarrow as pa
from lance.file import LanceFileReader, LanceFileWriter


def test_file_writer(tmp_path):
    path = tmp_path / "foo.lance"
    schema = pa.schema([pa.field("a", pa.int64())])
    with LanceFileWriter(str(path), schema) as writer:
        writer.write_batch(pa.table({"a": [1, 2, 3]}))
    assert len(path.read_bytes()) > 0


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
