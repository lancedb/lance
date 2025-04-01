import pathlib

import pyarrow.parquet as pq
import pytest
from lance.file import LanceFileReader, LanceFileWriter


def test_file_integration(
    tmp_path: pathlib.Path, test_data_file: str, test_data_base_uri: str
):
    file_uri = test_data_base_uri + "/" + test_data_file
    print(f"Testing file: {file_uri}")

    full_table = pq.read_table(file_uri)
    full_table = full_table.combine_chunks()
    for column in full_table.column_names:
        if full_table.column(column).num_chunks > 1:
            pytest.fail(
                f"Column {column} has {full_table.column(column).num_chunks} chunks"
            )

    print(f"Read in parquet file with {full_table.num_rows} rows")

    with LanceFileWriter(
        tmp_path / "test.lance", full_table.schema, version="2.1"
    ) as writer:
        writer.write_batch(full_table)

    print(f"Wrote to Lance file to {tmp_path / 'test.lance'}")

    reader = LanceFileReader(tmp_path / "test.lance")
    readback = reader.read_all().to_table()

    print(f"Read back from Lance file with {readback.num_rows} rows")

    readback = readback.combine_chunks()

    assert readback == full_table
