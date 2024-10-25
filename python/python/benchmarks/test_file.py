# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
from pathlib import Path

import pyarrow as pa
import pytest
from lance.file import LanceFileReader, LanceFileWriter
from lance.tracing import trace_to_chrome

trace_to_chrome(level="debug", file="/tmp/trace.json")

NUM_ROWS = 10_000_000
ROWS_TO_SAMPLE = 10


@pytest.mark.parametrize(
    "version",
    ["2.0", "2.1"],
    ids=["2_0", "2_1"],
)
@pytest.mark.benchmark(group="scan_single_column")
def test_scan_integer(tmp_path: Path, benchmark, version):
    schema = pa.schema([pa.field("values", pa.uint64(), True)])

    def gen_data():
        remaining = NUM_ROWS
        offset = 0
        while remaining > 0:
            to_take = min(remaining, 10000)
            values = pa.array(range(offset, offset + to_take))
            batch = pa.table({"values": values}).to_batches()[0]
            yield batch
            remaining -= to_take
            offset += to_take

    with LanceFileWriter(
        str(tmp_path / "file.lance"), schema, version=version
    ) as writer:
        for batch in gen_data():
            writer.write_batch(batch)

    def read_all():
        reader = LanceFileReader(str(tmp_path / "file.lance"))
        return reader.read_all(batch_size=16 * 1024).to_table()

    result = benchmark.pedantic(read_all, rounds=1, iterations=1)

    assert result.num_rows == NUM_ROWS


@pytest.mark.parametrize(
    "version",
    ["2.0", "2.1"],
    ids=["2_0", "2_1"],
)
@pytest.mark.benchmark(group="scan_single_column")
def test_scan_nullable_integer(tmp_path: Path, benchmark, version):
    schema = pa.schema([pa.field("values", pa.uint64(), True)])

    def gen_data():
        remaining = NUM_ROWS
        offset = 0
        while remaining > 0:
            to_take = min(remaining, 10000)
            values = pa.array(
                [None if i % 2 == 0 else i for i in range(offset, offset + to_take)]
            )
            batch = pa.table({"values": values}).to_batches()[0]
            yield batch
            remaining -= to_take
            offset += to_take

    with LanceFileWriter(
        str(tmp_path / "file.lance"), schema, version=version
    ) as writer:
        for batch in gen_data():
            writer.write_batch(batch)

    def read_all():
        reader = LanceFileReader(str(tmp_path / "file.lance"))
        return reader.read_all(batch_size=16 * 1024).to_table()

    result = benchmark.pedantic(read_all, rounds=1, iterations=1)

    assert result.num_rows == NUM_ROWS


@pytest.mark.benchmark(group="scan_single_column")
def test_scan_nested_integer(tmp_path: Path, benchmark):
    def get_val(i: int):
        if i % 4 == 0:
            return None
        elif i % 4 == 1:
            return {"outer": None}
        elif i % 4 == 2:
            return {"outer": {"inner": None}}
        else:
            return {"outer": {"inner": i}}

    dtype = pa.struct(
        [pa.field("outer", pa.struct([pa.field("inner", pa.uint64(), True)]), True)]
    )
    schema = pa.schema(
        [
            pa.field(
                "values",
                dtype,
                True,
            )
        ]
    )

    def gen_data():
        remaining = NUM_ROWS
        offset = 0
        while remaining > 0:
            to_take = min(remaining, 10000)
            values = pa.array([get_val(i) for i in range(offset, offset + to_take)])
            batch = pa.table({"values": values}).to_batches()[0]
            yield batch
            remaining -= to_take
            offset += to_take

    with LanceFileWriter(str(tmp_path / "file.lance"), schema, version="2.1") as writer:
        for batch in gen_data():
            writer.write_batch(batch)

    def read_all():
        reader = LanceFileReader(str(tmp_path / "file.lance"))
        return reader.read_all(batch_size=16 * 1024).to_table()

    result = benchmark.pedantic(read_all, rounds=1, iterations=1)

    assert result.num_rows == NUM_ROWS


@pytest.mark.parametrize(
    "version",
    ["2.0", "2.1"],
    ids=["2_0", "2_1"],
)
@pytest.mark.benchmark(group="sample_single_column")
def test_sample_integer(tmp_path: Path, benchmark, version):
    schema = pa.schema([pa.field("values", pa.uint64(), True)])

    def gen_data():
        remaining = NUM_ROWS
        offset = 0
        while remaining > 0:
            to_take = min(remaining, 10000)
            values = pa.array(range(offset, offset + to_take))
            batch = pa.table({"values": values}).to_batches()[0]
            yield batch
            remaining -= to_take
            offset += to_take

    with LanceFileWriter(
        str(tmp_path / "file.lance"), schema, version=version
    ) as writer:
        for batch in gen_data():
            writer.write_batch(batch)

    reader = LanceFileReader(str(tmp_path / "file.lance"))
    indices = list(range(0, NUM_ROWS, NUM_ROWS // ROWS_TO_SAMPLE))

    def sample():
        return reader.take_rows(indices).to_table()

    result = benchmark.pedantic(sample, rounds=30, iterations=1)

    assert result.num_rows == NUM_ROWS


@pytest.mark.benchmark(group="sample_single_column")
def test_sample_nested_integer(tmp_path: Path, benchmark):
    def get_val(i: int):
        if i % 4 == 0:
            return None
        elif i % 4 == 1:
            return {"outer": None}
        elif i % 4 == 2:
            return {"outer": {"inner": None}}
        else:
            return {"outer": {"inner": i}}

    dtype = pa.struct(
        [pa.field("outer", pa.struct([pa.field("inner", pa.uint64(), True)]), True)]
    )
    schema = pa.schema(
        [
            pa.field(
                "values",
                dtype,
                True,
            )
        ]
    )

    def gen_data():
        remaining = NUM_ROWS
        offset = 0
        while remaining > 0:
            to_take = min(remaining, 10000)
            values = pa.array([get_val(i) for i in range(offset, offset + to_take)])
            batch = pa.table({"values": values}).to_batches()[0]
            yield batch
            remaining -= to_take
            offset += to_take

    with LanceFileWriter(str(tmp_path / "file.lance"), schema, version="2.1") as writer:
        for batch in gen_data():
            writer.write_batch(batch)

    reader = LanceFileReader(str(tmp_path / "file.lance"))
    indices = list(range(0, NUM_ROWS, NUM_ROWS // ROWS_TO_SAMPLE))

    def sample():
        return reader.take_rows(indices).to_table()

    result = benchmark.pedantic(sample, rounds=30, iterations=1)

    assert result.num_rows == NUM_ROWS
