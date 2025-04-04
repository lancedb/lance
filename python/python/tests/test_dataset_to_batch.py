# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import time

import lance
import numpy as np  # Added missing import
import pyarrow as pa
import pyarrow.compute as pc
from lance import LanceDataset

dataset_path = "s3://ceshi/test_lance_dataset"
storage_options = {
    "access_key_id": "xx",
    "secret_access_key": "xx==",
    "aws_endpoint": "https://xx.com",
    "virtual_hosted_style_request": "true",
}


@staticmethod
def generate_test_dataset(dataset_path: str, num_rows: int = 10_000_000):
    """Generate test dataset supporting multi-value queries"""
    rng = np.random.default_rng(42)
    unique_vals = [f"ID_{i:08d}" for i in range(100000)]

    data = pa.Table.from_arrays(
        [
            pa.array(range(num_rows)),
            pa.array(rng.choice(unique_vals, num_rows)),
            pa.array(rng.integers(0, 100, num_rows)),
            pa.array(rng.choice(["A", "B", "C", "D"], num_rows)),
            pa.array(rng.random(num_rows)),
        ],
        names=["id", "uid", "range_col", "category", "value"],
    )

    lance.write_dataset(data, dataset_path, storage_options=storage_options)


class EnhancedBenchmark:
    def __init__(self, dataset_path: str):
        self.dataset = LanceDataset(dataset_path, storage_options=storage_options)
        self._warmup()

    def _warmup(self):
        """Warm up dataset by loading metadata"""
        _ = list(self.dataset.to_batches(batch_size=1024, prefetch_batches=0))

    def test_combinations(self, batch_sizes: list[int] = [1024, 8192, 32768]) -> dict:
        """Test performance of parameter combinations across different batch sizes"""
        results = {}

        # Modified parameter name to batch_readahead
        test_configs = [
            (
                "No prefetching/No batch readahead",
                {"prefetch_batches": 0, "batch_readahead": 0},
            ),
            (
                "With prefetching/No batch readahead",
                {"prefetch_batches": 4, "batch_readahead": 0},
            ),
            (
                "No prefetching/With batch readahead",
                {"prefetch_batches": 0, "batch_readahead": 4},
            ),
            (
                "With prefetching/With batch readahead",
                {"prefetch_batches": 4, "batch_readahead": 4},
            ),
        ]

        for batch_size in batch_sizes:
            params = {
                "filter": pc.field("uid").isin(
                    [f"ID_{i:08d}" for i in range(50, 1100)]
                ),
                "columns": ["uid", "category"],
                "batch_size": batch_size,
            }

            batch_results = {}
            base_time = None

            for config_name, config in test_configs:
                start = time.perf_counter()
                sum(
                    batch.num_rows
                    for batch in self.dataset.to_batches(**params, **config)
                )
                elapsed = time.perf_counter() - start

                if base_time is None:
                    base_time = elapsed
                    improvement = 0.0
                else:
                    improvement = (base_time / elapsed - 1) * 100

                batch_results[config_name] = {
                    "time": elapsed,
                    "improvement": improvement,
                    "throughput": (self.dataset.__len__() / elapsed),
                }

            results[batch_size] = batch_results

        return results


if __name__ == "__main__":
    # On the first run, please comment out generate_test_dataset
    # generate_test_dataset(dataset_path, num_rows=10_000_000)
    benchmark = EnhancedBenchmark(dataset_path)
    results = benchmark.test_combinations(batch_sizes=[1024, 8192, 32768])

    print("\nPerformance Comparison Across Batch Sizes:")
    for batch_size, batch_data in results.items():
        print(f"\nBatch Size: {batch_size}")
        for config_name, data in batch_data.items():
            print(f"{config_name}:")
            print(f"  Time: {data['time']:.2f}s")
            print(f"  Throughput: {data['throughput'] / 1e6:.2f}M rows/s")
            print(f"  Improvement: {data['improvement']:+.1f}%")
            print("-" * 40)
