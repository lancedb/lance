# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
from datetime import datetime

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.compute as pc
from lance.file import LanceFileReader, LanceFileWriter
from lance.tracing import trace_to_chrome
from pathlib import Path
import pytest
import random

trace_to_chrome(level='debug', file='/tmp/trace.json')

NUM_ROWS = 1_000_000
RANDOM_ACCESS = "full"
NUM_INDICES = 10
NUM_ROUNDS = 5

# This file compares benchmarks for reading and writing a StructArray column using
# (i) parquet
# (ii) the lance v2 format with default struct encoding
# (iii) the lance v2 format with a packed struct encoding
# We can test both random access and full scan access performance by
# setting RANDOM_ACCESS to "indices" or "full", respectively

@pytest.fixture(scope="module")
def test_data(tmp_path_factory):
    table = pa.table({"struct_col": pa.StructArray.from_arrays(
        [
            pc.random(NUM_ROWS).cast(pa.float32()),
            pa.array(range(NUM_ROWS), type=pa.int32()),
            pa.FixedSizeListArray.from_arrays(
                pc.random(NUM_ROWS * 5).cast(pa.float32()), 5
            ),
            # pa.array(range(NUM_ROWS), type=pa.int32()),
            # pa.array(range(NUM_ROWS), type=pa.int32()),
        ],
        ["f", 
         "i", 
         "fsl", 
        #  "i2", 
        #  "i3"
        ],
        )
    })

    return table

# generate NUM_INDICES random indices between 0 and NUM_ROWS for scanning
@pytest.fixture(scope="module")
def random_indices():
    random_indices = [random.randint(0, NUM_ROWS) for _ in range(NUM_INDICES)]
    return random_indices

@pytest.mark.benchmark(group="read")
def test_parquet_read(tmp_path: Path, benchmark, test_data, random_indices):
    # dataset = lance.write_dataset(table, tmp_path)
    parquet_path = tmp_path / "data.parquet"
    pq.write_table(test_data, parquet_path)

    if RANDOM_ACCESS == "indices":
        func = lambda: pq.read_table(parquet_path).take(random_indices)
    elif RANDOM_ACCESS == "full":
        func = lambda: pq.read_table(parquet_path)
    
    parquet_result = benchmark.pedantic(
        func,
        rounds=1
    )

def read_lance_file_random(lance_path, random_indices):
    for batch in LanceFileReader(lance_path).take_rows(indices=random_indices).to_batches():
        pass

def read_lance_file_full(lance_path):
    for batch in LanceFileReader(lance_path).read_all(batch_size=1000).to_batches():
        pass

@pytest.mark.benchmark(group="read")
def test_lance_read(tmp_path: Path, benchmark, test_data, random_indices):
    lance_path = str(tmp_path) + "/lance_data"

    with LanceFileWriter(lance_path, test_data.schema) as writer:
        for batch in test_data.to_batches():
            writer.write_batch(batch)
    
    if RANDOM_ACCESS == "indices":
        func = lambda: read_lance_file_random(lance_path, random_indices)
    elif RANDOM_ACCESS == "full":
        func = lambda: read_lance_file_full(lance_path)
    
    lance_result = benchmark.pedantic(
        func,
        rounds=NUM_ROUNDS
    )

@pytest.mark.benchmark(group="read")
def test_lance_read_packed_encoding(tmp_path: Path, benchmark, test_data, random_indices):
    lance_path = str(tmp_path) + "/lance_data"
    field = test_data.schema.field("struct_col")
    metadata = {b'packed': b'true'}
    updated_field = pa.field(field.name, field.type, metadata=metadata)

    updated_schema = pa.schema([updated_field])

    new_table = pa.Table.from_arrays(test_data.columns, schema=updated_schema)
    # print(new_table.schema)
    print(new_table.schema.field("struct_col").metadata)

    with LanceFileWriter(lance_path, new_table.schema) as writer:
        for batch in new_table.to_batches():
            writer.write_batch(batch)

    if RANDOM_ACCESS == "indices":
        func = lambda: read_lance_file_random(lance_path, random_indices)
    elif RANDOM_ACCESS == "full":
        func = lambda: read_lance_file_full(lance_path)
    
    lance_result = benchmark.pedantic(
        func,
        rounds=NUM_ROUNDS
    )

@pytest.mark.benchmark(group="read")
def test_parquet_write(tmp_path: Path, benchmark, test_data):
    parquet_path = tmp_path / "data.parquet"
    benchmark.pedantic(
        pq.write_table,
        args=(test_data, parquet_path),
        rounds=NUM_ROUNDS
    )

def write_lance_file(lance_path, test_data):
    with LanceFileWriter(lance_path, test_data.schema) as writer:
        for batch in test_data.to_batches():
            writer.write_batch(batch)

@pytest.mark.benchmark(group="write")
def test_lance_write(tmp_path: Path, benchmark, test_data):
    lance_path = str(tmp_path) + "/lance_data"
  
    benchmark.pedantic(
        write_lance_file,
        args=(lance_path, test_data),
        rounds=NUM_ROUNDS
    )

@pytest.mark.benchmark(group="write")
def test_lance_write_packed_encoding(tmp_path: Path, benchmark, test_data):
    lance_path = str(tmp_path) + "/lance_data"

    field = test_data.schema.field("struct_col")
    metadata = {b'packed': b'true'}
    updated_field = pa.field(field.name, field.type, metadata=metadata)

    updated_schema = pa.schema([updated_field])
  
    benchmark.pedantic(
        write_lance_file,
        args=(lance_path, test_data),
        rounds=NUM_ROUNDS
    )