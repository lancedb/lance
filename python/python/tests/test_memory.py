# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from pathlib import Path

import lance
import pyarrow as pa
import pytest

memtest = pytest.importorskip(
    "memtest", reason="memtest is not available. Please install from ../memtest"
)


def test_insert_memory(tmp_path: Path):
    def batch_generator():
        # 5MB batches -> 100MB total
        for _ in range(20):
            yield pa.RecordBatch.from_arrays(
                [pa.array([b"x" * 1024 * 1024] * 5)], names=["data"]
            )

    reader = pa.RecordBatchReader.from_batches(
        schema=pa.schema([("data", pa.binary())]),
        batches=batch_generator(),
    )

    with memtest.track() as get_stats:
        lance.write_dataset(
            reader,
            tmp_path / "test.lance",
        )
        stats = get_stats()

    assert stats["peak_bytes"] >= 5 * 1024 * 1024
    assert stats["peak_bytes"] < 30 * 1024 * 1024
