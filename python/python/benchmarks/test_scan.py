import random
from pathlib import Path

import lance
import pyarrow as pa
import pyarrow.compute as pc
import pytest

NUM_ROWS = 1_000_000


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


@pytest.mark.benchmark(group="scan_table")
def test_scan_table(tmp_path: Path, benchmark):
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
                    random.choice([b"hello", b"world", b"today"])
                    for _ in range(NUM_ROWS)
                ],
                type=pa.binary(),
            ),
        }
    )

    dataset = lance.write_dataset(table, tmp_path)

    result = benchmark(
        dataset.to_table,
    )

    assert result.num_rows == NUM_ROWS
