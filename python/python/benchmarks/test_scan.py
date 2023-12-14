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
import random
from pathlib import Path

import lance
import pyarrow as pa
import pyarrow.compute as pc
import pytest

NUM_ROWS = 10_000


@pytest.mark.parametrize(
    "array_factory",
    [
        lambda: pa.array(range(NUM_ROWS), type=pa.int32()),
        lambda: pc.random(NUM_ROWS),
        lambda: pa.array(
            [random.choice(["hello", "world", "today"]) for _ in range(NUM_ROWS)],
            type=pa.string(),
        ),
        lambda: pa.array(
            [random.choice(["hello", "world", "today"]) for _ in range(NUM_ROWS)],
            type=pa.dictionary(pa.int8(), pa.string()),
        ),
        lambda: pa.FixedSizeListArray.from_arrays(
            pc.random(NUM_ROWS * 128).cast(pa.float32()), 128
        ),
    ],
    ids=["i32", "f64", "string", "dictionary", "vector"],
)
@pytest.mark.benchmark(group="scan_single_column")
def test_scan_integer(tmp_path: Path, benchmark, array_factory):
    values = array_factory()
    table = pa.table({"values": values})
    dataset = lance.write_dataset(table, tmp_path)

    result = benchmark(
        dataset.to_table,
    )

    assert result.num_rows == NUM_ROWS


@pytest.fixture(scope="module")
def sample_dataset(tmpdir_factory):
    tmp_path = Path(tmpdir_factory.mktemp("data"))
    table = pa.table(
        {
            "i": pa.array(range(NUM_ROWS), type=pa.int32()),
            "f": pc.random(NUM_ROWS).cast(pa.float32()),
            "s": pa.array(
                [random.choice(["hello", "world", "today"]) for _ in range(NUM_ROWS)],
                type=pa.string(),
            ),
            "fsl": pa.FixedSizeListArray.from_arrays(
                pc.random(NUM_ROWS * 128).cast(pa.float32()), 128
            ),
            "blob": pa.array(
                [
                    random.choice(
                        [
                            random.randbytes(100 * 1024),
                            random.randbytes(100 * 1024),
                            random.randbytes(100 * 1024),
                        ]
                    )
                    for _ in range(NUM_ROWS)
                ],
                type=pa.binary(),
            ),
        }
    )

    return lance.write_dataset(table, tmp_path)


@pytest.mark.benchmark(group="scan_table")
def test_scan_table_full(benchmark, sample_dataset):
    result = benchmark(
        sample_dataset.to_table,
    )

    assert result.num_rows == NUM_ROWS


@pytest.mark.benchmark(group="scan_table")
def test_scan_table_project(benchmark, sample_dataset):
    result = benchmark(sample_dataset.to_table, columns=["i", "f"])

    assert result.schema.names == ["i", "f"]
    assert result.num_rows == NUM_ROWS


@pytest.mark.parametrize("keep_percent", [0.1, 0.5, 0.9])
@pytest.mark.benchmark(group="scan_table")
def test_scan_table_filter_project(benchmark, sample_dataset, keep_percent):
    result = benchmark(
        sample_dataset.to_table,
        filter=f"f < {keep_percent}",
        columns=["i", "blob"],
    )

    assert result.schema.names == ["i", "blob"]


@pytest.mark.parametrize("keep_percent", [0.1, 0.5, 0.9])
@pytest.mark.benchmark(group="scan_table")
def test_scan_table_filter_full(benchmark, sample_dataset, keep_percent):
    result = benchmark(
        sample_dataset.to_table,
        filter=f"f < {keep_percent}",
    )

    assert result.schema.names == ["i", "f", "s", "fsl", "blob"]


@pytest.mark.benchmark(group="filter_table")
def test_filter_for_range(benchmark, sample_dataset):
    result = benchmark(
        sample_dataset.to_table,
        filter="i > 1000 and i < 5000",
    )

    assert result.schema.names == ["i", "f", "s", "fsl", "blob"]


@pytest.mark.benchmark(group="filter_table")
def test_filter_for_row(benchmark, sample_dataset):
    result = benchmark(
        sample_dataset.to_table,
        filter="i = 4200",
    )

    assert result.num_rows == 1
    assert result.schema.names == ["i", "f", "s", "fsl", "blob"]


@pytest.mark.benchmark(group="filter_table")
def test_filter_multiple(benchmark, sample_dataset):
    result = benchmark(
        sample_dataset.to_table,
        filter="i > 1000 and i < 5000 and s in ('hello', 'world')",
    )

    assert result.schema.names == ["i", "f", "s", "fsl", "blob"]
