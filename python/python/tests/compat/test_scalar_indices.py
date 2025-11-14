# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
Scalar index compatibility tests for Lance.

Tests that scalar indices (BTREE, BITMAP, LABEL_LIST, NGRAM, ZONEMAP,
BLOOMFILTER, JSON, FTS) created with one version of Lance can be read
and written by other versions.
"""

import shutil
from pathlib import Path

import lance
import pyarrow as pa

from .compat_decorator import (
    UpgradeDowngradeTest,
    compat_test,
)


@compat_test(min_version="0.30.0")
class BTreeIndex(UpgradeDowngradeTest):
    """Test BTREE scalar index compatibility (introduced in 0.20.0).

    Started fully working in 0.30.0 with various fixes.
    """

    def __init__(self, path: Path):
        self.path = path

    def create(self):
        """Create dataset with BTREE index."""
        shutil.rmtree(self.path, ignore_errors=True)
        data = pa.table(
            {
                "idx": pa.array(range(1000)),
                "btree": pa.array(range(1000)),
            }
        )
        dataset = lance.write_dataset(data, self.path, max_rows_per_file=100)
        dataset.create_scalar_index("btree", "BTREE")

    def check_read(self):
        """Verify BTREE index can be queried."""
        ds = lance.dataset(self.path)
        table = ds.to_table(filter="btree == 7")
        assert table.num_rows == 1
        assert table.column("idx").to_pylist() == [7]

        # Verify index is used
        explain = ds.scanner(filter="btree == 7").explain_plan()
        assert "ScalarIndexQuery" in explain or "MaterializeIndex" in explain

    def check_write(self):
        """Verify can insert data and optimize BTREE index."""
        ds = lance.dataset(self.path)
        data = pa.table(
            {
                "idx": pa.array([1000]),
                "btree": pa.array([1000]),
            }
        )
        ds.insert(data)
        ds.optimize.optimize_indices()
        ds.optimize.compact_files()

        # Verify new data is queryable
        table = ds.to_table(filter="btree == 1000")
        assert table.num_rows >= 1


@compat_test(min_version="0.22.0")
class BitmapLabelListIndex(UpgradeDowngradeTest):
    """Test BITMAP and LABEL_LIST scalar index compatibility (introduced in 0.20.0).

    Started fully working in 0.22.0 with fixes to LABEL_LIST index.
    """

    def __init__(self, path: Path):
        self.path = path

    def create(self):
        """Create dataset with BITMAP and LABEL_LIST indices."""
        shutil.rmtree(self.path, ignore_errors=True)
        data = pa.table(
            {
                "idx": pa.array(range(1000)),
                "bitmap": pa.array(range(1000)),
                "label_list": pa.array([[f"label{i}"] for i in range(1000)]),
            }
        )
        dataset = lance.write_dataset(data, self.path, max_rows_per_file=100)
        dataset.create_scalar_index("bitmap", "BITMAP")
        dataset.create_scalar_index("label_list", "LABEL_LIST")

    def check_read(self):
        """Verify BITMAP and LABEL_LIST indices can be queried."""
        ds = lance.dataset(self.path)

        # Test BITMAP index
        table = ds.to_table(filter="bitmap == 7")
        assert table.num_rows == 1
        assert table.column("idx").to_pylist() == [7]

        # Test LABEL_LIST index
        table = ds.to_table(filter="array_has_any(label_list, ['label7'])")
        assert table.num_rows == 1
        assert table.column("idx").to_pylist() == [7]

    def check_write(self):
        """Verify can insert data and optimize indices."""
        ds = lance.dataset(self.path)
        data = pa.table(
            {
                "idx": pa.array([1000]),
                "bitmap": pa.array([1000]),
                "label_list": pa.array([["label1000"]]),
            }
        )
        ds.insert(data)
        ds.optimize.optimize_indices()
        ds.optimize.compact_files()


@compat_test(min_version="0.36.0")
class NgramIndex(UpgradeDowngradeTest):
    """Test NGRAM index compatibility (introduced in 0.36.0)."""

    def __init__(self, path: Path):
        self.path = path

    def create(self):
        """Create dataset with NGRAM index."""
        shutil.rmtree(self.path, ignore_errors=True)
        data = pa.table(
            {
                "idx": pa.array(range(1000)),
                "ngram": pa.array([f"word{i}" for i in range(1000)]),
            }
        )
        dataset = lance.write_dataset(data, self.path, max_rows_per_file=100)
        dataset.create_scalar_index("ngram", "NGRAM")

    def check_read(self):
        """Verify NGRAM index can be queried."""
        ds = lance.dataset(self.path)
        table = ds.to_table(filter="contains(ngram, 'word7')")
        # word7, word70-79, word700-799 = 111 results
        assert table.num_rows == 111

        # Verify index is used
        explain = ds.scanner(filter="contains(ngram, 'word7')").explain_plan()
        assert "ScalarIndexQuery" in explain

    def check_write(self):
        """Verify can insert data and optimize NGRAM index."""
        ds = lance.dataset(self.path)
        data = pa.table(
            {
                "idx": pa.array([1000]),
                "ngram": pa.array(["word1000"]),
            }
        )
        ds.insert(data)
        ds.optimize.optimize_indices()
        ds.optimize.compact_files()


@compat_test(min_version="0.36.0")
class ZonemapBloomfilterIndex(UpgradeDowngradeTest):
    """Test ZONEMAP and BLOOMFILTER index compatibility (introduced in 0.36.0)."""

    def __init__(self, path: Path):
        self.path = path

    def create(self):
        """Create dataset with ZONEMAP and BLOOMFILTER indices."""
        shutil.rmtree(self.path, ignore_errors=True)
        data = pa.table(
            {
                "idx": pa.array(range(1000)),
                "zonemap": pa.array(range(1000)),
                "bloomfilter": pa.array(range(1000)),
            }
        )
        dataset = lance.write_dataset(data, self.path, max_rows_per_file=100)
        dataset.create_scalar_index("zonemap", "ZONEMAP")
        dataset.create_scalar_index("bloomfilter", "BLOOMFILTER")

    def check_read(self):
        """Verify ZONEMAP and BLOOMFILTER indices can be queried."""
        ds = lance.dataset(self.path)

        # Test ZONEMAP
        table = ds.to_table(filter="zonemap == 7")
        assert table.num_rows == 1
        assert table.column("idx").to_pylist() == [7]

        # Test BLOOMFILTER
        table = ds.to_table(filter="bloomfilter == 7")
        assert table.num_rows == 1
        assert table.column("idx").to_pylist() == [7]

    def check_write(self):
        """Verify can insert data and optimize indices."""
        ds = lance.dataset(self.path)
        data = pa.table(
            {
                "idx": pa.array([1000]),
                "zonemap": pa.array([1000]),
                "bloomfilter": pa.array([1000]),
            }
        )
        ds.insert(data)
        ds.optimize.optimize_indices()
        ds.optimize.compact_files()


@compat_test(min_version="0.36.0")
class JsonIndex(UpgradeDowngradeTest):
    """Test JSON index compatibility (introduced in 0.36.0)."""

    def __init__(self, path: Path):
        self.path = path

    def create(self):
        """Create dataset with JSON index."""
        from lance.indices import IndexConfig

        shutil.rmtree(self.path, ignore_errors=True)
        data = pa.table(
            {
                "idx": pa.array(range(1000)),
                "json": pa.array([f'{{"val": {i}}}' for i in range(1000)], pa.json_()),
            }
        )
        dataset = lance.write_dataset(data, self.path, max_rows_per_file=100)
        dataset.create_scalar_index(
            "json",
            IndexConfig(
                index_type="json",
                parameters={"target_index_type": "btree", "path": "val"},
            ),
        )

    def check_read(self):
        """Verify JSON index can be queried."""
        ds = lance.dataset(self.path)
        table = ds.to_table(filter="json_get_int(json, 'val') == 7")
        assert table.num_rows == 1
        assert table.column("idx").to_pylist() == [7]

        # Verify index is used
        explain = ds.scanner(filter="json_get_int(json, 'val') == 7").explain_plan()
        assert "ScalarIndexQuery" in explain

    def check_write(self):
        """Verify can insert data with JSON index."""
        ds = lance.dataset(self.path)
        data = pa.table(
            {
                "idx": pa.array([1000]),
                "json": pa.array(['{"val": 1000}'], pa.json_()),
            }
        )
        ds.insert(data)
        ds.optimize.compact_files()


@compat_test(min_version="0.36.0")
class FtsIndex(UpgradeDowngradeTest):
    """Test FTS (full-text search) index compatibility (introduced in 0.36.0)."""

    def __init__(self, path: Path):
        self.path = path

    def create(self):
        """Create dataset with FTS index."""
        shutil.rmtree(self.path, ignore_errors=True)
        data = pa.table(
            {
                "idx": pa.array(range(1000)),
                "text": pa.array(
                    [f"document with words {i} and more text" for i in range(1000)]
                ),
            }
        )
        dataset = lance.write_dataset(data, self.path, max_rows_per_file=100)
        dataset.create_scalar_index("text", "INVERTED")

    def check_read(self):
        """Verify FTS index can be queried."""
        ds = lance.dataset(self.path)
        # Search for documents containing "words" and "7"
        # Note: Actual FTS query syntax may vary
        table = ds.to_table(filter="text LIKE '%words 7 %'")
        assert table.num_rows > 0

    def check_write(self):
        """Verify can insert data with FTS index."""
        # Dataset::load_manifest does not do retain_supported_indices
        # so this can only work with no cache
        session = lance.Session(index_cache_size_bytes=0, metadata_cache_size_bytes=0)
        ds = lance.dataset(self.path, session=session)
        data = pa.table(
            {
                "idx": pa.array([1000]),
                "text": pa.array(["new document to index"]),
            }
        )
        ds.insert(data)
        ds.optimize.compact_files()
