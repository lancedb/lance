# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import os

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


def test_version(tmp_path):
    path = tmp_path / "foo.lance"
    schema = pa.schema([pa.field("a", pa.int64())])

    with LanceFileWriter(str(path), schema) as writer:
        writer.write_batch(pa.table({"a": [1, 2, 3]}))
    reader = LanceFileReader(str(path))
    metadata = reader.metadata()
    assert metadata.major_version == 0
    assert metadata.minor_version == 3

    path = tmp_path / "foo2.lance"
    with LanceFileWriter(str(path), schema, version="2.1") as writer:
        writer.write_batch(pa.table({"a": [1, 2, 3]}))
    reader = LanceFileReader(str(path))
    metadata = reader.metadata()
    assert metadata.major_version == 2
    assert metadata.minor_version == 1


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
    dict_values = pa.array(["foo", "bar", "baz"], pa.string())
    dict_indices = pa.array([2, 1, 0], pa.uint8())

    check_round_trip(
        tmp_path,
        pa.table(
            {
                "large_string": pa.array(["foo", "bar", "baz"], pa.large_string()),
                "large_binary": pa.array([b"foo", b"bar", b"baz"], pa.large_binary()),
                "dict_string": pa.DictionaryArray.from_arrays(
                    dict_indices, dict_values
                ),
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
                "nullable_list": pa.array(
                    [[1, 2], None, [None, 3]], pa.list_(pa.int64())
                ),
                "nullable_fsl": pa.array(
                    [[1, 2], None, [None, 3]], pa.list_(pa.int64(), 2)
                ),
                "all_null": pa.array([None, None, None], pa.int64()),
                "null_strings": pa.array([None, "foo", None], pa.string()),
            }
        ),
    )


def test_batch_sizes(tmp_path):
    # Need a big string so there aren't too many rows per page because we
    # want to test different page sizes:
    #  - batch that spans multiple pages (including more than 2)
    #  - batch that is smaller than a page (including much smaller)
    my_str = b"0" * 299593

    data = [[my_str] for _ in range(1009)]
    tab = pa.table({"val": data})

    path = str(tmp_path / "foo.lance")
    with LanceFileWriter(path) as writer:
        writer.write_batch(tab)

    reader = LanceFileReader(path)

    for batch_size in range(10, 1050, 10):
        reader.read_all(batch_size=batch_size).to_table()


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


def test_field_meta(tmp_path):
    schema = pa.schema(
        [
            pa.field("primitive", pa.int64(), metadata={"foo": "bar"}),
            pa.field(
                "list",
                pa.list_(pa.field("item", pa.int64(), metadata={"list": "yes"})),
                metadata={"foo": "baz"},
            ),
            pa.field(
                "struct",
                pa.struct([pa.field("a", pa.int64(), metadata={"struct": "yes"})]),
                metadata={"foo": "qux"},
            ),
        ]
    )
    table = pa.table(
        {
            "primitive": [1, 2, 3],
            "list": [[1, 2], [3, 4], [5, 6]],
            "struct": [{"a": 1}, {"a": 2}, {"a": 3}],
        },
        schema=schema,
    )

    with LanceFileWriter(str(tmp_path / "foo.lance")) as writer:
        writer.write_batch(table)

    reader = LanceFileReader(str(tmp_path / "foo.lance"))
    round_tripped = reader.read_all().to_table()

    assert round_tripped == table


def test_dictionary(tmp_path):
    # Basic round trip
    dictionary = pa.array(["foo", "bar", "baz"], pa.string())
    indices = pa.array([0, 1, 2, 0, 1, 2], pa.int32())
    dict_arr = pa.DictionaryArray.from_arrays(indices, dictionary)

    def round_trip(arr):
        table = pa.table({"dict": arr})

        path = tmp_path / "foo.lance"
        with LanceFileWriter(str(path)) as writer:
            writer.write_batch(table)

        reader = LanceFileReader(str(path))
        table2 = reader.read_all().to_table()
        return table2.column("dict").chunk(0)

    round_tripped = round_trip(dict_arr)

    assert round_tripped == dict_arr
    assert round_tripped.type == dict_arr.type

    # Dictionary that doesn't use all values
    dictionary = pa.array(["foo", "bar", "baz"], pa.string())
    indices = pa.array([0, 0, 1, 1], pa.int32())
    dict_arr = pa.DictionaryArray.from_arrays(indices, dictionary)

    round_tripped = round_trip(dict_arr)

    assert round_tripped.dictionary == dictionary

    # different indices types
    dictionary = pa.array(["foo", "bar", "baz"], pa.string())
    for data_type in [
        pa.uint8(),
        pa.uint16(),
        pa.uint32(),
        pa.uint64(),
        pa.int8(),
        pa.int16(),
        pa.int32(),
        pa.int64(),
    ]:
        indices = pa.array([0, 1, 2, 0, 1, 2], data_type)
        dict_arr = pa.DictionaryArray.from_arrays(indices, dictionary)
        round_tripped = round_trip(dict_arr)
        assert round_tripped == dict_arr
        assert round_tripped.type == dict_arr.type


def test_write_read_global_buffer(tmp_path):
    table = pa.table({"a": [1, 2, 3]})
    path = tmp_path / "foo.lance"
    global_buffer_text = "hello"
    global_buffer_bytes = bytes(global_buffer_text, "utf-8")
    with LanceFileWriter(str(path)) as writer:
        writer.write_batch(table)
        global_buffer_pos = writer.add_global_buffer(global_buffer_bytes)
    reader = LanceFileReader(str(path))
    assert reader.read_all().to_table() == table
    assert reader.metadata().global_buffers[global_buffer_pos].size == len(
        global_buffer_bytes
    )
    assert (
        bytes(reader.read_global_buffer(global_buffer_pos)).decode()
        == global_buffer_text
    )


def test_write_read_additional_schema_metadata(tmp_path):
    table = pa.table({"a": [1, 2, 3]})
    path = tmp_path / "foo.lance"
    schema_metadata_key = "foo"
    schema_metadata_value = "bar"
    with LanceFileWriter(str(path)) as writer:
        writer.write_batch(table)
        writer.add_schema_metadata(schema_metadata_key, schema_metadata_value)
    reader = LanceFileReader(str(path))
    assert reader.read_all().to_table() == table
    assert (
        reader.metadata().schema.metadata.get(schema_metadata_key.encode()).decode()
        == schema_metadata_value
    )


def test_writer_maintains_order(tmp_path):
    # 100Ki strings, each string is a couple of KiBs
    big_strings = [f"{i}" * 1024 for i in range(100 * 1024)]
    table = pa.table({"big_strings": big_strings})

    for i in range(4):
        path = tmp_path / f"foo-{i}.lance"
        with LanceFileWriter(str(path)) as writer:
            writer.write_batch(table)

        reader = LanceFileReader(str(path))
        result = reader.read_all().to_table()
        assert result == table


def test_compression(tmp_path):
    # 10Ki strings, which should be highly compressible, but not eligible for dictionary
    compressible_strings = [f"compress_me_please-{i}" for i in range(10 * 1024)]
    table_default = pa.table({"compressible_strings": compressible_strings})

    schema_compress = pa.schema(
        [
            pa.field(
                "compressible_strings",
                pa.string(),
                metadata={"lance-encoding:compression": "zstd"},
            )
        ]
    )
    table_compress = pa.table(
        {"compressible_strings": compressible_strings}, schema=schema_compress
    )

    with LanceFileWriter(str(tmp_path / "default.lance")) as writer:
        writer.write_batch(table_default)

    with LanceFileWriter(str(tmp_path / "compress.lance"), schema_compress) as writer:
        writer.write_batch(table_compress)

    size_default = os.path.getsize(tmp_path / "default.lance")
    size_compress = os.path.getsize(tmp_path / "compress.lance")

    assert size_compress < size_default


def test_blob(tmp_path):
    # 100 1MiB values.  If we store as regular large_binary we end up
    # with several pages of values.  If we store as a blob we get a
    # single page
    vals = pa.array([b"0" * (1024 * 1024) for _ in range(100)], pa.large_binary())
    schema_no_blob = pa.schema([pa.field("val", pa.large_binary())])
    schema_blob = pa.schema(
        [pa.field("val", pa.large_binary(), metadata={"lance-encoding:blob": "true"})]
    )

    path = tmp_path / "no_blob.lance"
    with LanceFileWriter(str(path), schema_no_blob) as writer:
        writer.write_batch(pa.table({"val": vals}))

    reader = LanceFileReader(str(path))
    assert len(reader.metadata().columns[0].pages) > 1
    assert reader.read_all().to_table() == pa.table({"val": vals})

    path = tmp_path / "blob.lance"
    with LanceFileWriter(str(path), schema_blob) as writer:
        writer.write_batch(pa.table({"val": vals}))

    reader = LanceFileReader(str(path))
    assert len(reader.metadata().columns[0].pages) == 1
    assert reader.read_all().to_table() == pa.table({"val": vals})
