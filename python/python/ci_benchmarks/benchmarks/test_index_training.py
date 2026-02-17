# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Benchmarks for BTree and Bitmap index training time."""

import tempfile
from pathlib import Path

import lance
import pyarrow as pa
import pytest


def _generate_data(num_rows: int, dtype: str, cardinality: str):
    """Generate test data for index training benchmarks.

    Args:
        num_rows: Total number of rows to generate
        dtype: "float" or "string"
        cardinality: "high" (unique values) or "low" (100 unique values)
    """
    batch_size = 10_000
    num_batches = num_rows // batch_size

    if cardinality == "high":
        # High cardinality: all unique values
        if dtype == "float":
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                values = pa.array(
                    [float(start_idx + i) for i in range(batch_size)], type=pa.float64()
                )
                batch = pa.record_batch([values], names=["value"])
                yield batch
        else:  # string
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                # Zero-padded strings for proper sorting
                values = pa.array(
                    [f"string_{start_idx + i:010d}" for i in range(batch_size)]
                )
                batch = pa.record_batch([values], names=["value"])
                yield batch
    else:
        # Low cardinality: 100 unique values, each repeated multiple times
        num_unique = 100
        rows_per_value = num_rows // num_unique

        if dtype == "float":
            for value_idx in range(num_unique):
                value = float(value_idx)
                rows_generated = 0
                while rows_generated < rows_per_value:
                    current_batch_size = min(
                        batch_size, rows_per_value - rows_generated
                    )
                    values = pa.array([value] * current_batch_size, type=pa.float64())
                    batch = pa.record_batch([values], names=["value"])
                    yield batch
                    rows_generated += current_batch_size
        else:  # string
            for value_idx in range(num_unique):
                value = f"value_{value_idx:03d}"
                rows_generated = 0
                while rows_generated < rows_per_value:
                    current_batch_size = min(
                        batch_size, rows_per_value - rows_generated
                    )
                    values = pa.array([value] * current_batch_size)
                    batch = pa.record_batch([values], names=["value"])
                    yield batch
                    rows_generated += current_batch_size


# Test parameters
NUM_ROWS = [1_000_000, 5_000_000, 10_000_000]
NUM_ROWS_LABELS = ["1M", "5M", "10M"]
INDEX_TYPES = ["BTREE", "BITMAP"]
DTYPES = ["float", "string"]
CARDINALITIES = ["high", "low"]


@pytest.mark.parametrize("num_rows", NUM_ROWS, ids=NUM_ROWS_LABELS)
@pytest.mark.parametrize("index_type", INDEX_TYPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("cardinality", CARDINALITIES)
def test_index_training(benchmark, num_rows, index_type, dtype, cardinality):
    """Benchmark index training time for different configurations.

    Tests both BTree and Bitmap indices with:
    - Different row counts (1M, 5M, 10M)
    - Different data types (float, string)
    - Different cardinalities (high=unique, low=100 values)
    """
    # Set iterations based on dataset size
    iterations = 3 if num_rows == 1_000_000 else 1

    def bench():
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_uri = str(Path(tmpdir) / "test_dataset.lance")

            # Determine schema based on dtype
            if dtype == "float":
                schema = pa.schema([("value", pa.float64())])
            else:
                schema = pa.schema([("value", pa.string())])

            # Create dataset with generated data
            data = _generate_data(num_rows, dtype, cardinality)
            ds = lance.write_dataset(
                data,
                dataset_uri,
                schema=schema,
                mode="create",
            )

            # Train the index (this is what we're benchmarking)
            ds.create_scalar_index("value", index_type)

    # Run benchmark with appropriate iterations
    benchmark.pedantic(bench, rounds=1, iterations=iterations)
