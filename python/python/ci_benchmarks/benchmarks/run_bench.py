#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run concurrent write benchmark directly (no pytest-benchmark required).

Usage:
    # Optimistic (default)
    LANCE_COMMIT_STRATEGY=optimistic python run_bench.py

    # Pessimistic
    LANCE_COMMIT_STRATEGY=pessimistic python run_bench.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ci_benchmarks.benchmarks.test_concurrent_write import _run_benchmark

if __name__ == "__main__":
    strategy = os.environ.get("LANCE_COMMIT_STRATEGY", "optimistic")
    print(f"\n{'#' * 60}")
    print(f"# Running benchmark with commit_strategy={strategy}")
    print(f"{'#' * 60}\n")

    scenarios = [
        {
            "name": "append-only (30 writers, 0 deleters, 0 updaters)",
            "kwargs": dict(
                num_writers=30,
                num_deleters=0,
                num_updaters=0,
                num_writes_per_writer=10,
                num_deletes_per_deleter=0,
                num_updates_per_updater=0,
                rows_per_write=100,
            ),
        },
        {
            "name": "mixed (20 writers, 10 deleters, 10 updaters)",
            "kwargs": dict(
                num_writers=20,
                num_deleters=10,
                num_updaters=10,
                num_writes_per_writer=10,
                num_deletes_per_deleter=5,
                num_updates_per_updater=5,
                rows_per_write=100,
            ),
        },
        {
            "name": "high-concurrency (30 writers, 15 deleters, 15 updaters)",
            "kwargs": dict(
                num_writers=30,
                num_deleters=15,
                num_updaters=15,
                num_writes_per_writer=10,
                num_deletes_per_deleter=5,
                num_updates_per_updater=5,
                rows_per_write=100,
            ),
        },
    ]

    for scenario in scenarios:
        print(f"\n--- Scenario: {scenario['name']} ---")
        _run_benchmark(**scenario["kwargs"])

    print(f"\n{'#' * 60}")
    print(f"# All benchmarks complete (commit_strategy={strategy})")
    print(f"{'#' * 60}")
