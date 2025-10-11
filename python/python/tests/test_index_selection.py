# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from pathlib import Path

import lance
import pyarrow as pa
import pytest


def test_index_selection_options(tmp_path: Path):
    """Test the IndexSelectionOptions functionality"""
    # Create a test dataset
    table = pa.table(
        {
            "id": range(1000),
            "value": [f"value_{i}" for i in range(1000)],
            "price": [float(i % 100) for i in range(1000)],
        }
    )

    base_dir = tmp_path / "test"
    dataset = lance.write_dataset(table, base_dir)

    # Create scalar indices
    dataset.create_scalar_index("id", "BTREE")
    dataset.create_scalar_index("value", "INVERTED")
    dataset.create_scalar_index("price", "BTREE")

    # Test basic index selection with hints
    index_selection = {
        "hints": ["btree", "inverted"],
        "column_hints": {"id": ["btree"], "value": ["inverted"], "price": ["btree"]},
    }

    # Test scanner with index selection options
    from lance.dataset import ScannerBuilder

    scanner_builder = ScannerBuilder(dataset)
    scanner_builder = scanner_builder.filter("id > 500 AND value LIKE '%5%'")
    scanner = scanner_builder.index_selection_options(index_selection).to_scanner()

    # Verify the scanner works correctly
    result = scanner.to_table()
    assert len(result) > 0

    # Test with different strategy
    index_selection_best_effort = {
        "hints": ["btree"],
        "column_hints": {"id": ["btree"], "price": ["btree"]},
    }

    from lance.dataset import ScannerBuilder

    scanner_builder_best_effort = ScannerBuilder(dataset)
    scanner_builder_best_effort = scanner_builder_best_effort.filter("price < 50")
    scanner_best_effort = scanner_builder_best_effort.index_selection_options(
        index_selection_best_effort
    ).to_scanner()

    result_best_effort = scanner_best_effort.to_table()
    assert len(result_best_effort) > 0

    # Test with explain enabled
    index_selection_explain = {
        "hints": ["btree"],
        "column_hints": {"id": ["btree"]},
    }

    from lance.dataset import ScannerBuilder

    scanner_builder_explain = ScannerBuilder(dataset)
    scanner_builder_explain = scanner_builder_explain.filter("id < 100")
    scanner_explain = scanner_builder_explain.index_selection_options(
        index_selection_explain
    ).to_scanner()

    # Should work even with explain enabled
    result_explain = scanner_explain.to_table()
    assert len(result_explain) > 0


def test_index_selection_with_unknown_types(tmp_path: Path):
    """Test index selection with unknown index types"""
    # Create a test dataset
    table = pa.table({"id": range(100), "value": [f"value_{i}" for i in range(100)]})

    base_dir = tmp_path / "test_unknown"
    dataset = lance.write_dataset(table, base_dir)

    # Create a scalar index
    dataset.create_scalar_index("id", "BTREE")

    # Test with unknown index type in hints
    index_selection = {
        "hints": ["unknown_type", "btree"],
        "column_hints": {"id": ["unknown_type", "btree"]},
    }

    from lance.dataset import ScannerBuilder

    scanner_builder = ScannerBuilder(dataset)
    scanner_builder = scanner_builder.filter("id > 50")
    scanner = scanner_builder.index_selection_options(index_selection).to_scanner()

    # Should still work with unknown types (they'll be treated as
    # IndexTypeHint::Unknown)
    result = scanner.to_table()
    assert len(result) > 0


def test_index_selection_without_indices(tmp_path: Path):
    """Test index selection when no indices exist"""
    # Create a test dataset without indices
    table = pa.table({"id": range(100), "value": [f"value_{i}" for i in range(100)]})

    base_dir = tmp_path / "test_no_indices"
    dataset = lance.write_dataset(table, base_dir)

    # Test with index selection options but no actual indices
    index_selection = {
        "hints": ["btree"],
        "column_hints": {"id": ["btree"]},
    }

    from lance.dataset import ScannerBuilder

    scanner_builder = ScannerBuilder(dataset)
    scanner_builder = scanner_builder.filter("id > 50")
    scanner = scanner_builder.index_selection_options(index_selection).to_scanner()

    # Should fall back to regular scan
    result = scanner.to_table()
    assert len(result) == 49  # id > 50 for range(100) gives us 49 records (51-99)


if __name__ == "__main__":
    pytest.main([__file__])
