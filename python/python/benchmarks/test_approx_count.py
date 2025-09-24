# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import logging
import os
import subprocess
import sys
import tempfile
import time
import uuid

import numpy as np
import pyarrow as pa
import pytest

# Add the project path to sys.path to import lance modules correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lance
from lance.dataset import LanceDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test parameters
DEFAULT_BATCH_SIZE = 1024
NUM_FRAGMENTS = 5
ROWS_PER_FRAGMENT = 5000


def create_dataset(
    path: str,
    data_storage_version,
    num_batches: int,
    file_size: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    compression: str = None,
    num_fragments: int = NUM_FRAGMENTS,
) -> LanceDataset:
    metadata = {}
    if compression:
        metadata["lance-encoding:compression"] = compression

    schema = pa.schema(
        [
            pa.field("i", pa.int32(), nullable=False, metadata=metadata),
            pa.field("f", pa.float32(), nullable=False, metadata=metadata),
            pa.field("s", pa.string(), nullable=False, metadata=metadata),
            pa.field("blob", pa.binary(), nullable=False, metadata=metadata),
        ]
    )

    # Create RecordBatches
    batches = []
    for i in range(num_batches):
        batch = pa.RecordBatch.from_arrays(
            [
                pa.array(range(i * batch_size, (i + 1) * batch_size), type=pa.int32()),
                pa.array(np.random.randn(batch_size).astype(np.float32)),
                pa.array([f"item_{j}" for j in range(batch_size)]),
                pa.array([b"blob" for _ in range(batch_size)]),
            ],
            schema=schema,
        )
        batches.append(batch)

    table = pa.Table.from_batches(batches, schema=schema)

    # Set storage options for non-memory paths
    storage_options = {}
    if not path.startswith("memory://") and not path.startswith("file://"):
        storage_options.update(
            {
                "access_key_id": os.environ.get("ENV_OBJECT_STORAGE_ACCESS_KEY_ID", ""),
                "secret_access_key": os.environ.get(
                    "ENV_OBJECT_STORAGE_SECRET_ACCESS_KEY", ""
                ),
                "allow_http": os.environ.get("ENV_OBJECT_STORAGE_ALLOW_HTTP", "true"),
                "skip_signature": os.environ.get(
                    "ENV_OBJECT_STORAGE_SKIP_SIGNATURE", "true"
                ),
            },
        )

    ds = lance.write_dataset(
        table,
        path,
        max_rows_per_file=file_size,
        max_rows_per_group=batch_size,
        data_storage_version=data_storage_version,
        storage_options=storage_options,
    )

    # If fragment count is specified and greater than 1, create more fragments
    # by appending data
    if num_fragments > 1:
        rows_per_fragment = (num_batches * batch_size) // num_fragments
        for f in range(1, num_fragments):
            # Create new data batches
            start_idx = f * rows_per_fragment
            end_idx = (f + 1) * rows_per_fragment
            new_batches = []
            for i in range(start_idx // batch_size, end_idx // batch_size):
                batch_start = max(i * batch_size, start_idx)
                batch_end = min((i + 1) * batch_size, end_idx)
                batch_size_actual = batch_end - batch_start
                if batch_size_actual > 0:
                    batch = pa.RecordBatch.from_arrays(
                        [
                            pa.array(range(batch_start, batch_end), type=pa.int32()),
                            pa.array(
                                np.random.randn(batch_size_actual).astype(np.float32)
                            ),
                            pa.array([f"item_{j}" for j in range(batch_size_actual)]),
                            pa.array([b"blob" for _ in range(batch_size_actual)]),
                        ],
                        schema=schema,
                    )
                    new_batches.append(batch)

            if new_batches:
                new_table = pa.Table.from_batches(new_batches, schema=schema)
                ds = lance.write_dataset(new_table, path, mode="append")

    return ds


def get_path_prefixes():
    temp_dir = tempfile.mkdtemp()
    prefixes = ["memory://", f"file://{temp_dir}"]
    object_store_path_prefix = os.getenv(
        "ENV_OBJECT_STORAGE_TEST_DATASET_URI_PREFIX", ""
    )
    if object_store_path_prefix:
        prefixes.append(object_store_path_prefix)
    return prefixes


def get_scheme_from_path(path: str) -> str:
    if "://" in path:
        return path.split("://")[0]
    return "file"


def clear_page_cache():
    """Clear page cache to ensure fair benchmarking."""
    try:
        subprocess.run(
            ["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If we can't clear cache, that's fine
        pass


@pytest.mark.parametrize("method", ["exact", "approx"])
@pytest.mark.parametrize("index_type", [None, "BTREE", "ZONEMAP"])
@pytest.mark.parametrize("num_fragments", [1, 5, 100])
@pytest.mark.parametrize(
    "filter_condition",
    [
        None,
        "i = 50",
        "i > 50",
        "i < 50",
        "i >= 50 AND i <= 100",
        "i > 9000",
        "i < 1000",
    ],
)
def test_count_comparison(
    benchmark, method, index_type, num_fragments, filter_condition
):
    """Test for comparing exact and approximate counting methods."""
    path = f"memory://count-comparison-{uuid.uuid4()}.lance/"

    # Adjust dataset size based on number of fragments
    if num_fragments == 100:
        num_batches = 100
        batch_size = 1000
    elif num_fragments == 5:
        num_batches = 50
        batch_size = 1000
    else:  # 1 fragment
        num_batches = 10
        batch_size = 1000

    file_size = 1024 * 1024
    ds = create_dataset(
        path, "2.0", num_batches, file_size, batch_size, num_fragments=num_fragments
    )

    # Create index based on parameter
    if index_type is not None:
        ds.create_scalar_index("i", index_type)

    def exact_count_function():
        if filter_condition is None:
            fragments = ds.get_fragments()
            count = 0
            for fragment in fragments:
                count += fragment.count_rows()
            return count
        else:
            # For filtered scans, we need to use the scanner to get matching fragments
            scanner = ds.scanner(filter=filter_condition, columns=[], with_row_id=True)
            # We'll use the original approach for filtered scans
            return scanner.count_rows()

    def approx_count_function():
        fragments = ds.get_fragments()
        count = 0
        for fragment in fragments:
            count += fragment.approx_count_rows()
        return count

    # Set group name based on parameters
    benchmark.group = (
        f"Count Comparison - Index: {index_type}, Fragments: {num_fragments}, "
        f"Filter: {filter_condition}"
    )

    # Run appropriate benchmark based on method
    if method == "exact":
        result = benchmark.pedantic(
            exact_count_function, setup=clear_page_cache, rounds=5, iterations=1
        )
        logger.info("Exact count result: %s", result)
    else:  # method == "approx"
        result = benchmark.pedantic(
            approx_count_function, setup=clear_page_cache, rounds=5, iterations=1
        )
        logger.info("Approx count result: %s", result)


def test_zonemap_approx_count_advantage():
    """Test the advantage of zonemap index in approximate counting"""
    path = f"memory://zonemap-advantage-{uuid.uuid4()}.lance/"

    # Create a dataset with more fragments to demonstrate the advantage of zonemap index
    num_batches = 50
    batch_size = 1000
    file_size = 5000  # Reduce file size to create more fragments
    ds = create_dataset(
        path, "2.0", num_batches, file_size, batch_size, num_fragments=10
    )

    # Create zonemap index
    ds.create_scalar_index("i", "ZONEMAP")

    # Get fragments
    fragments = ds.get_fragments()
    logger.info("Dataset contains %s fragments", len(fragments))

    # Test filter conditions, zonemap index should significantly reduce the
    # number of fragments that need to be scanned
    test_filters = [
        "i > 45000",  # High value filter, should skip most fragments
        "i < 5000",  # Low value filter, should skip most fragments
        "i = 25000",  # Exact value filter
    ]

    for filter_condition in test_filters:
        logger.info("\nTesting filter condition: %s", filter_condition)

        # Exact count
        start_time = time.perf_counter()
        exact_result = ds.scanner(
            filter=filter_condition, columns=[], with_row_id=True
        ).count_rows()
        exact_time = time.perf_counter() - start_time

        # Approximate count
        start_time = time.perf_counter()
        approx_result = 0
        scanned_fragments = 0
        for fragment in fragments:
            frag_approx = fragment.approx_count_rows(filter=filter_condition)
            approx_result += frag_approx
            # If the fragment returns a non-zero count, consider it scanned
            if frag_approx > 0:
                scanned_fragments += 1
        approx_time = time.perf_counter() - start_time

        logger.info("  Exact count result: %s (Time: %.6fs)", exact_result, exact_time)
        logger.info(
            "  Approximate count result: %s (Time: %.6fs)", approx_result, approx_time
        )
        logger.info("  Scanned fragments: %s/%s", scanned_fragments, len(fragments))

        # Verify that approximate count is at least equal to exact count
        # (zonemap index characteristic)
        assert approx_result >= exact_result, (
            "Approximate count should be greater than or equal to exact count: "
            f"{approx_result} >= {exact_result}"
        )

        # If the number of scanned fragments is less than the total number of fragments,
        # the zonemap index is working
        if scanned_fragments < len(fragments):
            logger.info(
                "  [Advantage] Zonemap index skipped %s fragments",
                len(fragments) - scanned_fragments,
            )
        else:
            logger.info("  [Note] Zonemap index did not skip any fragments")
