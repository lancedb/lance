# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from pathlib import Path

import lance
import pyarrow as pa
import pytest

memtest = pytest.importorskip(
    "memtest", reason="memtest is not available. Please install from ../memtest"
)


def test_bitmap_index_allocations(tmp_path: Path):
    """Test that bitmap index creation doesn't cause excessive allocations.

    This test creates ~100MB of int64 data and builds a bitmap index that reproduces https://github.com/lancedb/lance/issues/4047.
    """
    # 100MB of int64
    num_values = 100 * 1024 * 1024 // 8
    data = pa.table({"values": pa.array([i % 1000 for i in range(num_values)])})
    dataset = lance.write_dataset(data, tmp_path / "dataset")

    with memtest.track() as get_stats:
        dataset.create_scalar_index("values", index_type="BITMAP")
        stats = get_stats()

    assert stats["total_allocations"] < 500_000, (
        f"Bitmap index creation caused {stats['total_allocations']:,} allocations. "
        "This may indicate a regression in allocation efficiency. "
        "See https://github.com/lancedb/lance/issues/4047"
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
