# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
Vector index compatibility tests for Lance.

Tests that vector indices (IVF_PQ, etc.) created with one version of Lance
can be read and written by other versions.
"""

import shutil
from pathlib import Path

import lance
import pyarrow as pa
import pyarrow.compute as pc

from .compat_decorator import (
    UpgradeDowngradeTest,
    compat_test,
)


@compat_test(min_version="0.29.1.beta2")
class PqVectorIndex(UpgradeDowngradeTest):
    """Test PQ (Product Quantization) vector index compatibility."""

    def __init__(self, path: Path):
        self.path = path

    def create(self):
        """Create dataset with PQ vector index."""
        shutil.rmtree(self.path, ignore_errors=True)
        ndims = 32
        nvecs = 512

        data = pa.table(
            {
                "id": pa.array(range(nvecs)),
                "vec": pa.FixedSizeListArray.from_arrays(
                    pc.random(ndims * nvecs).cast(pa.float32()), ndims
                ),
            }
        )

        dataset = lance.write_dataset(data, self.path)
        dataset.create_index(
            "vec",
            "IVF_PQ",
            num_partitions=1,
            num_sub_vectors=4,
        )

    def check_read(self):
        """Verify PQ index can be queried."""
        ds = lance.dataset(self.path)
        # Query with random vector
        q = pc.random(32).cast(pa.float32())
        result = ds.to_table(
            nearest={
                "q": q,
                "k": 4,
                "column": "vec",
            }
        )
        assert result.num_rows == 4

    def check_write(self):
        """Verify can insert vectors and rebuild index."""
        ds = lance.dataset(self.path)
        # Add new vectors
        data = pa.table(
            {
                "id": pa.array([1000]),
                "vec": pa.FixedSizeListArray.from_arrays(
                    pc.random(32).cast(pa.float32()), 32
                ),
            }
        )
        ds.insert(data)
        ds.optimize.optimize_indices()
        ds.optimize.compact_files()


@compat_test(min_version="0.39.0")
class HnswPqVectorIndex(UpgradeDowngradeTest):
    """Test IVF_HNSW_PQ vector index compatibility.

    Note: Only tests versions >= 0.39.0 because earlier versions don't support
    remapping for IVF_HNSW_PQ indices, which is required for optimize operations.
    """

    def __init__(self, path: Path):
        self.path = path

    def create(self):
        """Create dataset with IVF_HNSW_PQ vector index."""
        shutil.rmtree(self.path, ignore_errors=True)
        ndims = 32
        nvecs = 512

        data = pa.table(
            {
                "id": pa.array(range(nvecs)),
                "vec": pa.FixedSizeListArray.from_arrays(
                    pc.random(ndims * nvecs).cast(pa.float32()), ndims
                ),
            }
        )

        dataset = lance.write_dataset(data, self.path)
        dataset.create_index(
            "vec",
            "IVF_HNSW_PQ",
            num_partitions=4,
            num_sub_vectors=4,
        )

    def check_read(self):
        """Verify IVF_HNSW_PQ index can be queried."""
        ds = lance.dataset(self.path)
        # Query with random vector
        q = pc.random(32).cast(pa.float32())
        result = ds.to_table(
            nearest={
                "q": q,
                "k": 4,
                "column": "vec",
            }
        )
        assert result.num_rows == 4

    def check_write(self):
        """Verify can insert vectors and rebuild index."""
        ds = lance.dataset(self.path)
        # Add new vectors
        data = pa.table(
            {
                "id": pa.array([1000]),
                "vec": pa.FixedSizeListArray.from_arrays(
                    pc.random(32).cast(pa.float32()), 32
                ),
            }
        )
        ds.insert(data)
        ds.optimize.optimize_indices()
        ds.optimize.compact_files()


@compat_test(min_version="0.39.0")
class HnswSqVectorIndex(UpgradeDowngradeTest):
    """Test IVF_HNSW_SQ vector index compatibility.

    Note: Only tests versions >= 0.39.0 because earlier versions don't support
    remapping for IVF_HNSW_SQ indices, which is required for optimize operations.
    """

    def __init__(self, path: Path):
        self.path = path

    def create(self):
        """Create dataset with IVF_HNSW_SQ vector index."""
        shutil.rmtree(self.path, ignore_errors=True)
        ndims = 32
        nvecs = 512

        data = pa.table(
            {
                "id": pa.array(range(nvecs)),
                "vec": pa.FixedSizeListArray.from_arrays(
                    pc.random(ndims * nvecs).cast(pa.float32()), ndims
                ),
            }
        )

        dataset = lance.write_dataset(data, self.path)
        dataset.create_index(
            "vec",
            "IVF_HNSW_SQ",
            num_partitions=4,
            num_sub_vectors=4,
        )

    def check_read(self):
        """Verify IVF_HNSW_SQ index can be queried."""
        ds = lance.dataset(self.path)
        # Query with random vector
        q = pc.random(32).cast(pa.float32())
        result = ds.to_table(
            nearest={
                "q": q,
                "k": 4,
                "column": "vec",
            }
        )
        assert result.num_rows == 4

    def check_write(self):
        """Verify can insert vectors and rebuild index."""
        ds = lance.dataset(self.path)
        # Add new vectors
        data = pa.table(
            {
                "id": pa.array([1000]),
                "vec": pa.FixedSizeListArray.from_arrays(
                    pc.random(32).cast(pa.float32()), 32
                ),
            }
        )
        ds.insert(data)
        ds.optimize.optimize_indices()
        ds.optimize.compact_files()
