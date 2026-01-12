# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Benchmarks for IVF_PQ vector search performance."""

import math
import tempfile
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest
from ci_benchmarks.utils import wipe_os_cache
from lance.tracing import trace_to_chrome

trace_to_chrome(file="/tmp/trace.json")


# Test parameters
DATASET_SIZES = [100_000, 1_000_000]
DATASET_SIZE_LABELS = ["100K", "1M"]
VECTOR_DIM = 1024

# Number of partitions to search (nprobes)
NPROBES = [10, 50]
NPROBES_LABELS = ["10probes", "50probes"]

# Refine factor for vector search
REFINE_FACTORS = [None, 1]
REFINE_FACTOR_LABELS = ["no_refine", "refine_1x"]

# Number of results to return (k)
K_VALUES = [10, 100]
K_LABELS = ["k10", "k100"]


# Global cache for datasets, keyed by (num_rows, dim)
_DATASET_CACHE = {}


def _generate_vector_dataset(num_rows: int, dim: int = 1024):
    """Generate random vector dataset for IVF_PQ search benchmarks.

    Args:
        num_rows: Number of vectors to generate
        dim: Dimensionality of vectors (default: 1024)

    Yields:
        PyArrow RecordBatch with random float32 vectors
    """
    batch_size = 10_000
    num_batches = num_rows // batch_size

    for batch_idx in range(num_batches):
        # Generate random vectors with 32-bit floats
        vectors = np.random.randn(batch_size, dim).astype(np.float32)

        # Convert to PyArrow fixed_size_list
        vector_array = pa.FixedSizeListArray.from_arrays(
            pa.array(vectors.flatten(), type=pa.float32()), list_size=dim
        )

        # Add an ID column for reference
        ids = pa.array(
            range(batch_idx * batch_size, (batch_idx + 1) * batch_size), type=pa.int64()
        )

        batch = pa.record_batch([vector_array, ids], names=["vector", "id"])
        yield batch


def _get_or_create_dataset(num_rows: int, dim: int = 1024) -> str:
    """Get or create a dataset with the specified parameters.

    Datasets are cached globally per process to avoid expensive recreation.
    Returns the URI to the dataset.
    """
    cache_key = (num_rows, dim)

    if cache_key not in _DATASET_CACHE:
        # Create a persistent temporary directory for this dataset
        tmpdir = tempfile.mkdtemp(prefix=f"lance_bench_{num_rows}_{dim}_")
        dataset_uri = str(Path(tmpdir) / "vector_dataset.lance")

        # Create schema
        schema = pa.schema(
            [
                pa.field("vector", pa.list_(pa.float32(), dim)),
                pa.field("id", pa.int64()),
            ]
        )

        # Generate and write dataset
        data = _generate_vector_dataset(num_rows, dim)
        ds = lance.write_dataset(
            data,
            dataset_uri,
            schema=schema,
            mode="create",
        )

        num_partitions = min(num_rows // 4000, int(math.sqrt(num_rows)))

        # Create IVF_PQ index
        ds.create_index(
            "vector",
            index_type="IVF_PQ",
            num_partitions=num_partitions,
            num_sub_vectors=dim // 16,
        )

        _DATASET_CACHE[cache_key] = dataset_uri

    return _DATASET_CACHE[cache_key]


@pytest.mark.parametrize("num_rows", DATASET_SIZES, ids=DATASET_SIZE_LABELS)
@pytest.mark.parametrize("nprobes", NPROBES, ids=NPROBES_LABELS)
@pytest.mark.parametrize("refine_factor", REFINE_FACTORS, ids=REFINE_FACTOR_LABELS)
@pytest.mark.parametrize("k", K_VALUES, ids=K_LABELS)
@pytest.mark.parametrize("use_cache", [True, False], ids=["cache", "no_cache"])
def test_ivf_pq_search(
    benchmark,
    num_rows: int,
    nprobes: int,
    refine_factor: int | None,
    k: int,
    use_cache: bool,
):
    """Benchmark IVF_PQ vector search with different configurations.

    Tests vector search performance with:
    - Different dataset sizes (100K, 1M vectors)
    - Different numbers of partitions searched (10, 50 nprobes)
    - Different refine factors (None, 1x)
    - Different result counts (k=10, k=100)
    - Cached vs uncached index performance

    Uses 1024-dimensional float32 vectors with IVF_PQ index.
    """
    # Get or create the dataset (cached globally per process)
    dataset_uri = _get_or_create_dataset(num_rows, dim=VECTOR_DIM)
    ds = lance.dataset(dataset_uri)

    # Generate query vector
    query_vector = np.random.randn(VECTOR_DIM).astype(np.float32)

    # Setup function to clear OS cache if needed
    def clear_cache():
        if not use_cache:
            wipe_os_cache(dataset_uri)

    def bench():
        # Reload dataset if not using cache
        search_ds = ds if use_cache else lance.dataset(dataset_uri)

        # Build search parameters
        search_params = {
            "column": "vector",
            "q": query_vector,
            "k": k,
            "nprobes": nprobes,
        }
        if refine_factor is not None:
            search_params["refine_factor"] = refine_factor

        # Perform vector search
        search_ds.to_table(
            nearest=search_params,
            columns=["id"],
        )

    if use_cache:
        setup = None
        warmup_rounds = 1
    else:
        setup = clear_cache
        warmup_rounds = 0

    benchmark.pedantic(
        bench,
        warmup_rounds=warmup_rounds,
        rounds=100,
        setup=setup,
    )


@pytest.mark.parametrize("num_rows", DATASET_SIZES, ids=DATASET_SIZE_LABELS)
@pytest.mark.parametrize("nprobes", NPROBES, ids=NPROBES_LABELS)
@pytest.mark.parametrize("refine_factor", REFINE_FACTORS, ids=REFINE_FACTOR_LABELS)
@pytest.mark.parametrize("k", K_VALUES, ids=K_LABELS)
@pytest.mark.parametrize("use_cache", [True, False], ids=["cache", "no_cache"])
def test_ivf_pq_search_with_payload(
    benchmark,
    num_rows: int,
    nprobes: int,
    refine_factor: int | None,
    k: int,
    use_cache: bool,
):
    """Benchmark IVF_PQ vector search with payload columns.

    Similar to test_ivf_pq_search but includes retrieving vector data
    along with results, which tests data loading performance.
    """
    # Get or create the dataset (cached globally per process)
    dataset_uri = _get_or_create_dataset(num_rows, dim=VECTOR_DIM)
    ds = lance.dataset(dataset_uri)

    # Generate query vector
    query_vector = np.random.randn(VECTOR_DIM).astype(np.float32)

    def clear_cache():
        if not use_cache:
            wipe_os_cache(dataset_uri)

    def bench():
        search_ds = ds if use_cache else lance.dataset(dataset_uri)

        # Build search parameters
        search_params = {
            "column": "vector",
            "q": query_vector,
            "k": k,
            "nprobes": nprobes,
        }
        if refine_factor is not None:
            search_params["refine_factor"] = refine_factor

        # Search and retrieve both vector and id columns
        search_ds.to_table(
            nearest=search_params,
            columns=["vector", "id"],
        )

    if use_cache:
        setup = None
        warmup_rounds = 1
    else:
        setup = clear_cache
        warmup_rounds = 0

    benchmark.pedantic(
        bench,
        warmup_rounds=warmup_rounds,
        rounds=100,
        iterations=1,
        setup=setup,
    )
