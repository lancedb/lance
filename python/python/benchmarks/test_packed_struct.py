# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from lance.file import LanceFileReader, LanceFileWriter
from lance.tracing import trace_to_chrome

trace_to_chrome(level="debug", file="/tmp/trace.json")

NUM_ROWS = 10_000_000
RANDOM_ACCESS = "indices"
NUM_INDICES = 1000
NUM_ROUNDS = 10
BATCH_SIZE = 16 * 1024

# This file compares benchmarks for reading and writing a StructArray column using
# (i) parquet
# (ii) the lance v2 format with default struct encoding
# (iii) the lance v2 format with a packed struct encoding
# We can test both random access and full scan access performance by
# setting RANDOM_ACCESS to "indices" or "full", respectively


@pytest.fixture(scope="module")
def test_data(tmp_path_factory):
    table = pa.table(
        {
            "struct_col": pa.StructArray.from_arrays(
                [
                    pc.random(NUM_ROWS).cast(pa.float32()),  # f1
                    pc.random(NUM_ROWS).cast(pa.float32()),  # f2
                    pc.random(NUM_ROWS).cast(pa.float32()),  # f3
                    pc.random(NUM_ROWS).cast(pa.float32()),  # f4
                ],
                ["f1", "f2", "f3", "f4"],
            )
        }
    )

    return table


# generate NUM_INDICES random indices between 0 and NUM_ROWS for scanning
@pytest.fixture(scope="module")
def random_indices():
    random_indices = [random.randint(0, NUM_ROWS) for _ in range(NUM_INDICES)]
    random_indices.sort()
    return random_indices


@pytest.mark.benchmark(group="read")
def test_parquet_read(tmp_path: Path, benchmark, test_data, random_indices):
    parquet_path = tmp_path / "data.parquet"
    pq.write_table(test_data, parquet_path)

    def read_parquet():
        parquet_file = pq.ParquetFile(parquet_path)
        batches = parquet_file.iter_batches(batch_size=BATCH_SIZE)
        tab_parquet = pa.Table.from_batches(batches)
        return tab_parquet

    if RANDOM_ACCESS == "indices":
        benchmark.pedantic(
            lambda: pq.read_table(parquet_path).take(random_indices), rounds=5
        )
    elif RANDOM_ACCESS == "full":
        benchmark.pedantic(lambda: read_parquet(), rounds=5)


def read_lance_file_random(lance_path, random_indices):
    for batch in (
        LanceFileReader(lance_path).take_rows(indices=random_indices).to_batches()
    ):
        pass


def read_lance_file_full(lance_path):
    for batch in (
        LanceFileReader(lance_path).read_all(batch_size=BATCH_SIZE).to_batches()
    ):
        pass


@pytest.mark.benchmark(group="read")
def test_lance_read(tmp_path: Path, benchmark, test_data, random_indices):
    lance_path = str(tmp_path) + "/lance_data"

    with LanceFileWriter(lance_path, test_data.schema) as writer:
        for batch in test_data.to_batches():
            writer.write_batch(batch)

    if RANDOM_ACCESS == "indices":
        benchmark.pedantic(
            read_lance_file_random, args=(lance_path, random_indices), rounds=NUM_ROUNDS
        )
    elif RANDOM_ACCESS == "full":
        benchmark.pedantic(read_lance_file_full, args=(lance_path,), rounds=NUM_ROUNDS)


@pytest.mark.benchmark(group="read")
def test_lance_read_packed(tmp_path: Path, benchmark, test_data, random_indices):
    lance_path = str(tmp_path) + "/lance_data"
    field = test_data.schema.field("struct_col")
    metadata = {b"packed": b"true"}
    updated_field = pa.field(field.name, field.type, metadata=metadata)

    updated_schema = pa.schema([updated_field])

    new_table = pa.Table.from_arrays(test_data.columns, schema=updated_schema)

    with LanceFileWriter(lance_path, new_table.schema) as writer:
        for batch in new_table.to_batches():
            writer.write_batch(batch)

    if RANDOM_ACCESS == "indices":
        benchmark.pedantic(
            read_lance_file_random, args=(lance_path, random_indices), rounds=NUM_ROUNDS
        )
    elif RANDOM_ACCESS == "full":
        benchmark.pedantic(read_lance_file_full, args=(lance_path,), rounds=NUM_ROUNDS)


@pytest.mark.benchmark(group="write")
def test_parquet_write(tmp_path: Path, benchmark, test_data):
    parquet_path = tmp_path / "data.parquet"
    benchmark.pedantic(
        pq.write_table, args=(test_data, parquet_path), rounds=NUM_ROUNDS
    )


def write_lance_file(lance_path, test_data):
    with LanceFileWriter(lance_path, test_data.schema, version="2.1") as writer:
        for batch in test_data.to_batches():
            writer.write_batch(batch)


@pytest.mark.benchmark(group="write")
def test_lance_write(tmp_path: Path, benchmark, test_data):
    lance_path = str(tmp_path) + "/lance_data"

    benchmark.pedantic(
        write_lance_file, args=(lance_path, test_data), rounds=NUM_ROUNDS
    )


@pytest.mark.benchmark(group="write")
def test_lance_write_packed(tmp_path: Path, benchmark, test_data):
    lance_path = str(tmp_path) + "/lance_data"

    field = test_data.schema.field("struct_col")
    metadata = {b"packed": b"true"}
    updated_field = pa.field(field.name, field.type, metadata=metadata)
    updated_schema = pa.schema([updated_field])
    new_table = pa.Table.from_arrays(test_data.columns, schema=updated_schema)

    benchmark.pedantic(
        write_lance_file, args=(lance_path, new_table), rounds=NUM_ROUNDS
    )
