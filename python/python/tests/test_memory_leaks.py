# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

import gc
import os
from typing import Callable

import lance
import psutil
import pyarrow as pa

MiB = 1024 * 1024


def get_memory_usage() -> int:
    return psutil.Process(os.getpid()).memory_info().rss


def assert_noleaks(
    operation: Callable[[], None],
    *,
    iterations: int = 100,
    warmup_iterations: int = 5,
    threshold_mb: float = 1.0,
    check_interval: int = 10,
    leeway_factor: float = 2.0,  # optional jitter cushion
) -> None:
    """Check if an operation retains memory across repeated executions.

    Args:
        operation: A callable that performs the operation to test
        iterations: Number of times to run the operation
        warmup_iterations: Number of warmup runs before measuring
        threshold_mb: Maximum allowed memory growth in MB
        check_interval: How often to check memory during iterations
        leeway_factor: Factor to multiply threshold for early bailout

    Raises:
        AssertionError: If memory leak is detected
    """
    if iterations <= 0:
        raise ValueError("iterations must be > 0")
    if check_interval <= 0:
        raise ValueError("check_interval must be > 0")

    for _ in range(warmup_iterations):
        operation()
    gc.collect()

    baseline = get_memory_usage()

    for i in range(iterations):
        operation()

        if i > 0 and i % check_interval == 0:
            gc.collect()
            current = get_memory_usage()
            growth_mb = (current - baseline) / MiB
            if growth_mb > threshold_mb * leeway_factor:
                raise AssertionError(
                    f"Possible leak: +{growth_mb:.2f} MiB after {i}/{iterations} "
                    f"(threshold {threshold_mb:.2f} MiB; leeway x{leeway_factor}). "
                    f"rss_base={baseline}, rss_now={current}"
                )

    gc.collect()
    final = get_memory_usage()
    total_mb = (final - baseline) / MiB

    if total_mb > threshold_mb:
        avg = total_mb / iterations
        raise AssertionError(
            f"Memory leak detected: +{total_mb:.2f} MiB over {iterations} iterations "
            f"(threshold {threshold_mb:.2f} MiB; avg {avg:.4f} MiB/iter). "
            f"rss_base={baseline}, rss_final={final}"
        )


class TestMemoryLeaks:
    def test_index_statistics_no_leak(self, tmp_path) -> None:
        dataset_path = str(tmp_path / "dataset")
        data = pa.table({"id": [1]})
        ds = lance.write_dataset(data, dataset_path)
        ds.create_scalar_index("id", index_type="BTREE")

        def access_index_stats() -> None:
            d = lance.dataset(dataset_path)
            for idx in d.list_indices():
                if name := idx.get("name"):
                    d.stats.index_stats(name)

        assert_noleaks(
            access_index_stats, iterations=1000, threshold_mb=2.0, check_interval=25
        )
