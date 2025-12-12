# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
Tests for column name handling with mixed case and special characters.

These tests verify that Lance properly handles column names that:
1. Use mixed case (e.g., "userId", "OrderId") - common in TypeScript/JavaScript
2. Contain special characters (e.g., "user-id", "order:id")

See: https://github.com/lancedb/lance/issues/3424
"""

from pathlib import Path

import lance
import pyarrow as pa
import pytest
from lance.dataset import ColumnOrdering


class TestMixedCaseColumnNames:
    """
    Test that mixed-case column names work without requiring backtick quoting.

    Users coming from TypeScript/JavaScript commonly use camelCase column names.
    These should work in filter expressions, order by, scalar indices, etc.
    without requiring backtick escaping.

    Current behavior:
    - Table creation: WORKS
    - Filter with SQL string: FAILS (column names are lowercased)
    - Order by: WORKS (uses column name directly, not SQL parsing)
    - Scalar index: FAILS (column name is lowercased during validation)
    - Alter column: WORKS
    - Drop column: WORKS
    - Merge insert: FAILS (column names lowercased in join expressions)
    """

    @pytest.fixture
    def mixed_case_table(self):
        """Create a table with mixed-case column names."""
        return pa.table(
            {
                "userId": range(100),
                "OrderId": range(100, 200),
                "itemName": [f"item_{i}" for i in range(100)],
            }
        )

    @pytest.fixture
    def mixed_case_dataset(self, tmp_path: Path, mixed_case_table):
        """Create a dataset with mixed-case column names."""
        return lance.write_dataset(mixed_case_table, tmp_path / "mixed_case")

    def test_create_table_with_mixed_case(self, mixed_case_dataset):
        """Verify table creation with mixed-case columns works."""
        # Table creation preserves column names - this works
        assert "userId" in [f.name for f in mixed_case_dataset.schema]
        assert "OrderId" in [f.name for f in mixed_case_dataset.schema]
        assert "itemName" in [f.name for f in mixed_case_dataset.schema]

    def test_filter_with_mixed_case(self, mixed_case_dataset):
        """Filter expressions should work with mixed-case column names."""
        # This should work without backticks
        result = mixed_case_dataset.to_table(filter="userId > 50")
        assert result.num_rows == 49

        # Also test with the other mixed-case columns
        result = mixed_case_dataset.to_table(filter="OrderId >= 150")
        assert result.num_rows == 50

        result = mixed_case_dataset.to_table(filter="itemName = 'item_25'")
        assert result.num_rows == 1

    def test_order_by_with_mixed_case(self, mixed_case_dataset):
        """Order by works with mixed-case column names when using proper API."""
        # order_by takes a list of column names or ColumnOrdering objects
        # This does NOT go through SQL parsing, so it preserves case
        ordering = ColumnOrdering("userId", ascending=False)
        scanner = mixed_case_dataset.scanner(order_by=[ordering])
        result = scanner.to_table()
        assert result.num_rows == 100
        assert result["userId"][0].as_py() == 99

        # Also test ordering by OrderId
        ordering = ColumnOrdering("OrderId", ascending=True)
        scanner = mixed_case_dataset.scanner(order_by=[ordering])
        result = scanner.to_table()
        assert result["OrderId"][0].as_py() == 100

    def test_scalar_index_with_mixed_case(self, mixed_case_dataset):
        """Scalar index creation should work with mixed-case column names."""
        mixed_case_dataset.create_scalar_index("userId", index_type="BTREE")

        indices = mixed_case_dataset.list_indices()
        assert len(indices) == 1
        assert indices[0]["fields"] == ["userId"]

        # Query using the indexed column (would also fail due to filter issue)
        result = mixed_case_dataset.to_table(filter="userId = 50")
        assert result.num_rows == 1

    def test_alter_column_with_mixed_case(self, mixed_case_dataset):
        """Altering columns works with mixed-case column names."""
        # alter_columns uses direct schema lookup, not SQL parsing
        mixed_case_dataset.alter_columns({"path": "userId", "name": "user_id"})

        assert "user_id" in [f.name for f in mixed_case_dataset.schema]
        assert "userId" not in [f.name for f in mixed_case_dataset.schema]

    def test_drop_column_with_mixed_case(self, tmp_path: Path, mixed_case_table):
        """Dropping columns works with mixed-case column names."""
        # drop_columns uses direct schema lookup, not SQL parsing
        dataset = lance.write_dataset(mixed_case_table, tmp_path / "drop_test")

        dataset.drop_columns(["OrderId"])

        assert "OrderId" not in [f.name for f in dataset.schema]
        assert "userId" in [f.name for f in dataset.schema]

    def test_merge_insert_with_mixed_case_key(self, tmp_path: Path, mixed_case_table):
        """Merge insert should work with mixed-case column as the key."""
        dataset = lance.write_dataset(mixed_case_table, tmp_path / "merge_test")

        new_data = pa.table(
            {
                "userId": range(50, 150),
                "OrderId": range(1000, 1100),
                "itemName": [f"new_item_{i}" for i in range(100)],
            }
        )

        dataset.merge_insert(
            "userId"
        ).when_matched_update_all().when_not_matched_insert_all().execute(new_data)

        result = dataset.to_table()
        assert result.num_rows == 150


class TestSpecialCharacterColumnNames:
    """
    Test that column names with special characters work properly.

    Users may have column names with dashes, colons, or other special
    characters. These should work in filter expressions, order by,
    scalar indices, etc.

    Note: Column names with `.` are NOT allowed at the top level since `.` is
    used for nested field paths. This test uses `-` and `:` instead.

    Current behavior:
    - Table creation: WORKS
    - Filter with backticks: WORKS
    - Order by: WORKS (uses column name directly, not SQL parsing)
    - Scalar index: WORKS
    - Alter column: WORKS
    - Drop column: WORKS
    - Merge insert: FAILS (column name causes parsing issues in join expressions)
    """

    @pytest.fixture
    def special_char_table(self):
        """Create a table with special character column names."""
        return pa.table(
            {
                "user-id": range(100),
                "order:id": range(100, 200),
                "item_name": [f"item_{i}" for i in range(100)],
            }
        )

    @pytest.fixture
    def special_char_dataset(self, tmp_path: Path, special_char_table):
        """Create a dataset with special character column names."""
        return lance.write_dataset(special_char_table, tmp_path / "special_char")

    def test_create_table_with_special_chars(self, special_char_dataset):
        """Verify table creation with special character columns works."""
        # Table creation preserves column names - this works
        assert "user-id" in [f.name for f in special_char_dataset.schema]
        assert "order:id" in [f.name for f in special_char_dataset.schema]
        assert "item_name" in [f.name for f in special_char_dataset.schema]

    def test_filter_with_special_chars_using_backticks(self, special_char_dataset):
        """Filter expressions work with special char columns when using backticks."""
        # Backticks work for escaping special characters in SQL
        result = special_char_dataset.to_table(filter="`user-id` > 50")
        assert result.num_rows == 49

        result = special_char_dataset.to_table(filter="`order:id` >= 150")
        assert result.num_rows == 50

        # Regular column for comparison
        result = special_char_dataset.to_table(filter="item_name = 'item_25'")
        assert result.num_rows == 1

    def test_order_by_with_special_chars(self, special_char_dataset):
        """Order by works with special character column names."""
        # order_by uses column name directly, not SQL parsing
        ordering = ColumnOrdering("user-id", ascending=False)
        scanner = special_char_dataset.scanner(order_by=[ordering])
        result = scanner.to_table()
        assert result.num_rows == 100
        assert result["user-id"][0].as_py() == 99

        ordering = ColumnOrdering("order:id", ascending=True)
        scanner = special_char_dataset.scanner(order_by=[ordering])
        result = scanner.to_table()
        assert result["order:id"][0].as_py() == 100

    def test_scalar_index_with_special_chars(self, special_char_dataset):
        """Scalar index creation works with special character column names."""
        # Column name is used directly without SQL parsing
        special_char_dataset.create_scalar_index("user-id", index_type="BTREE")

        indices = special_char_dataset.list_indices()
        assert len(indices) == 1
        assert indices[0]["fields"] == ["user-id"]

        # Query using the indexed column (requires backticks in filter)
        result = special_char_dataset.to_table(filter="`user-id` = 50")
        assert result.num_rows == 1

    def test_alter_column_with_special_chars(self, special_char_dataset):
        """Altering columns works with special character column names."""
        # alter_columns uses direct schema lookup
        special_char_dataset.alter_columns({"path": "user-id", "name": "user_id"})

        assert "user_id" in [f.name for f in special_char_dataset.schema]
        assert "user-id" not in [f.name for f in special_char_dataset.schema]

    def test_drop_column_with_special_chars(self, tmp_path: Path, special_char_table):
        """Dropping columns works with special character column names."""
        # drop_columns uses direct schema lookup
        dataset = lance.write_dataset(special_char_table, tmp_path / "drop_test")

        dataset.drop_columns(["order:id"])

        assert "order:id" not in [f.name for f in dataset.schema]
        assert "user-id" in [f.name for f in dataset.schema]

    def test_merge_insert_with_special_char_key(
        self, tmp_path: Path, special_char_table
    ):
        """Merge insert should work with special character column as the key."""
        dataset = lance.write_dataset(special_char_table, tmp_path / "merge_test")

        new_data = pa.table(
            {
                "user-id": range(50, 150),
                "order:id": range(1000, 1100),
                "item_name": [f"new_item_{i}" for i in range(100)],
            }
        )

        dataset.merge_insert(
            "user-id"
        ).when_matched_update_all().when_not_matched_insert_all().execute(new_data)

        result = dataset.to_table()
        assert result.num_rows == 150


class TestNestedFieldColumnNames:
    """
    Test that column names with mixed case and special characters work
    properly within nested (struct) fields.

    This tests nested field paths like:
    - metadata.userId (mixed case in nested field)
    - metadata.user-id (special chars in nested field)
    """

    @pytest.fixture
    def nested_mixed_case_table(self):
        """Create a table with mixed-case nested column names."""
        return pa.table(
            {
                "id": range(100),
                "metadata": [{"userId": i, "itemCount": i * 10} for i in range(100)],
            }
        )

    @pytest.fixture
    def nested_mixed_case_dataset(self, tmp_path: Path, nested_mixed_case_table):
        """Create a dataset with mixed-case nested column names."""
        return lance.write_dataset(
            nested_mixed_case_table, tmp_path / "nested_mixed_case"
        )

    def test_create_table_with_nested_mixed_case(self, nested_mixed_case_dataset):
        """Verify table creation with nested mixed-case columns preserves names."""
        schema = nested_mixed_case_dataset.schema
        assert "metadata" in [f.name for f in schema]
        metadata_field = schema.field("metadata")
        nested_names = [f.name for f in metadata_field.type]
        assert "userId" in nested_names
        assert "itemCount" in nested_names

    def test_filter_with_nested_mixed_case(self, nested_mixed_case_dataset):
        """Filter expressions should work with mixed-case nested column names."""
        result = nested_mixed_case_dataset.to_table(filter="metadata.userId > 50")
        assert result.num_rows == 49

        result = nested_mixed_case_dataset.to_table(filter="metadata.itemCount >= 500")
        assert result.num_rows == 50

    def test_scalar_index_with_nested_mixed_case(self, nested_mixed_case_dataset):
        """Scalar index creation should work with mixed-case nested column names."""
        nested_mixed_case_dataset.create_scalar_index(
            "metadata.userId", index_type="BTREE"
        )

        indices = nested_mixed_case_dataset.list_indices()
        assert len(indices) == 1
        assert indices[0]["fields"] == ["metadata.userId"]

        # Query using the indexed column
        result = nested_mixed_case_dataset.to_table(filter="metadata.userId = 50")
        assert result.num_rows == 1

    @pytest.fixture
    def nested_special_char_table(self):
        """Create a table with special character nested column names."""
        return pa.table(
            {
                "id": range(100),
                "meta-data": [{"user-id": i, "item:count": i * 10} for i in range(100)],
            }
        )

    @pytest.fixture
    def nested_special_char_dataset(self, tmp_path: Path, nested_special_char_table):
        """Create a dataset with special character nested column names."""
        return lance.write_dataset(
            nested_special_char_table, tmp_path / "nested_special_char"
        )

    def test_create_table_with_nested_special_chars(self, nested_special_char_dataset):
        """Verify table creation with nested special char columns preserves names."""
        schema = nested_special_char_dataset.schema
        assert "meta-data" in [f.name for f in schema]
        metadata_field = schema.field("meta-data")
        nested_names = [f.name for f in metadata_field.type]
        assert "user-id" in nested_names
        assert "item:count" in nested_names

    def test_filter_with_nested_special_chars(self, nested_special_char_dataset):
        """Filter expressions work with special char nested columns using backticks."""
        # Both the parent and child need backticks when they contain special chars
        result = nested_special_char_dataset.to_table(
            filter="`meta-data`.`user-id` > 50"
        )
        assert result.num_rows == 49

        result = nested_special_char_dataset.to_table(
            filter="`meta-data`.`item:count` >= 500"
        )
        assert result.num_rows == 50

    def test_scalar_index_with_nested_special_chars(self, nested_special_char_dataset):
        """Scalar index creation should work with special char nested column names."""
        # Use backtick syntax for nested field path with special chars
        nested_special_char_dataset.create_scalar_index(
            "`meta-data`.`user-id`", index_type="BTREE"
        )

        indices = nested_special_char_dataset.list_indices()
        assert len(indices) == 1
        # Backticks are stripped when storing the field path
        assert indices[0]["fields"] == ["meta-data.user-id"]

        # Query using the indexed column (backticks required in filter)
        result = nested_special_char_dataset.to_table(
            filter="`meta-data`.`user-id` = 50"
        )
        assert result.num_rows == 1
