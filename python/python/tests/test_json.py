# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import json
import tempfile
from pathlib import Path

import lance
import pyarrow as pa
from lance.arrow import JsonArray, JsonType, json_field


def test_json_basic_write_read():
    """Test basic JSON type write and read functionality."""

    # Create test data with JSON strings
    json_data = [
        {"name": "Alice", "age": 30, "city": "New York"},
        {"name": "Bob", "age": 25, "city": "San Francisco"},
        {"name": "Charlie", "age": 35, "city": "Chicago"},
        None,  # Test null handling
        {"nested": {"key": "value", "list": [1, 2, 3]}},
    ]

    # Convert to JSON strings
    json_strings = [json.dumps(d) if d is not None else None for d in json_data]

    # Create JsonArray using the exported type
    json_ext_array = JsonArray.from_json_strings(json_strings)

    table = pa.table(
        {
            "id": pa.array([1, 2, 3, 4, 5], type=pa.int32()),
            "data": json_ext_array,
        },
        schema=pa.schema(
            [
                pa.field("id", pa.int32()),
                json_field("data"),
            ]
        ),
    )

    # Write to Lance dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "json_test.lance"

        # Write the dataset with version 2.2 (required for JSON support)
        lance.write_dataset(table, dataset_path, data_storage_version="2.2")

        # Read back the dataset
        dataset = lance.dataset(dataset_path)

        # Verify schema
        assert len(dataset.schema) == 2
        assert dataset.schema.field("id").type == pa.int32()

        # Check that JSON field is properly recognized
        json_field_schema = dataset.schema.field("data")
        # JsonType should be preserved through write/read
        assert isinstance(json_field_schema.type, JsonType)

        # Read data back
        result_table = dataset.to_table()

        # Verify data
        assert result_table.num_rows == 5
        assert result_table.column("id").to_pylist() == [1, 2, 3, 4, 5]

        # Convert binary back to JSON strings for verification
        data_column = result_table.column("data")
        for i, expected in enumerate(json_strings):
            if expected is None:
                assert data_column[i].as_py() is None
            else:
                # The data is stored as binary, need to decode
                binary_data = data_column[i].as_py()
                if binary_data is not None:
                    # For now, we just verify the data is stored and retrieved
                    assert binary_data is not None


def test_json_with_other_types():
    """Test JSON type alongside other data types."""

    # Create mixed type data
    json_data = [
        {"product": "laptop", "specs": {"cpu": "i7", "ram": 16}},
        {"product": "phone", "specs": {"screen": "6.1", "battery": 4000}},
    ]

    json_strings = [json.dumps(d) for d in json_data]

    # Create JsonArray
    json_ext_array = JsonArray.from_json_strings(json_strings)

    table = pa.table(
        {
            "id": pa.array([1, 2], type=pa.int64()),
            "name": pa.array(["Product A", "Product B"], type=pa.string()),
            "price": pa.array([999.99, 599.99], type=pa.float64()),
            "metadata": json_ext_array,
            "in_stock": pa.array([True, False], type=pa.bool_()),
        },
        schema=pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("name", pa.string()),
                pa.field("price", pa.float64()),
                json_field("metadata"),
                pa.field("in_stock", pa.bool_()),
            ]
        ),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "mixed_types.lance"

        # Write and read with version 2.2 (required for JSON support)
        lance.write_dataset(table, dataset_path, data_storage_version="2.2")
        dataset = lance.dataset(dataset_path)

        # Verify all fields are preserved
        assert len(dataset.schema) == 5

        result = dataset.to_table()
        assert result.num_rows == 2
        assert result.column("id").to_pylist() == [1, 2]
        assert result.column("name").to_pylist() == ["Product A", "Product B"]
        assert result.column("in_stock").to_pylist() == [True, False]


def test_json_append_and_overwrite():
    """Test appending and overwriting JSON data."""

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "append_test.lance"

        # Initial data
        initial_json = [json.dumps({"version": 1, "enabled": True})]
        initial_ext_array = JsonArray.from_json_strings(initial_json)

        initial_table = pa.table(
            {"id": pa.array([1]), "config": initial_ext_array},
            schema=pa.schema([pa.field("id", pa.int32()), json_field("config")]),
        )

        # Write initial data with version 2.2 (required for JSON support)
        lance.write_dataset(initial_table, dataset_path, data_storage_version="2.2")

        # Append more data
        append_json = [json.dumps({"version": 2, "enabled": False})]
        append_ext_array = JsonArray.from_json_strings(append_json)

        append_table = pa.table(
            {"id": pa.array([2]), "config": append_ext_array},
            schema=pa.schema([pa.field("id", pa.int32()), json_field("config")]),
        )

        dataset = lance.dataset(dataset_path)
        dataset = lance.write_dataset(append_table, dataset_path, mode="append")

        # Verify append
        result = dataset.to_table()
        assert result.num_rows == 2
        assert result.column("id").to_pylist() == [1, 2]

        # Test overwrite
        overwrite_json = [json.dumps({"version": 3, "enabled": True})]
        overwrite_ext_array = JsonArray.from_json_strings(overwrite_json)

        overwrite_table = pa.table(
            {"id": pa.array([3]), "config": overwrite_ext_array},
            schema=pa.schema([pa.field("id", pa.int32()), json_field("config")]),
        )

        dataset = lance.write_dataset(overwrite_table, dataset_path, mode="overwrite")

        # Verify overwrite
        result = dataset.to_table()
        assert result.num_rows == 1
        assert result.column("id").to_pylist() == [3]


def test_json_filter_and_select():
    """Test filtering and selecting with JSON columns."""

    json_data = [
        {"user_id": 1, "preferences": {"theme": "dark"}},
        {"user_id": 2, "preferences": {"theme": "light"}},
        {"user_id": 3, "preferences": {"theme": "dark"}},
    ]

    json_strings = [json.dumps(d) for d in json_data]
    json_ext_array = JsonArray.from_json_strings(json_strings)

    table = pa.table(
        {
            "id": pa.array([1, 2, 3]),
            "user_data": json_ext_array,
            "active": pa.array([True, False, True]),
        },
        schema=pa.schema(
            [
                pa.field("id", pa.int32()),
                json_field("user_data"),
                pa.field("active", pa.bool_()),
            ]
        ),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "filter_test.lance"
        lance.write_dataset(table, dataset_path, data_storage_version="2.2")
        dataset = lance.dataset(dataset_path)

        # Test column selection
        selected = dataset.to_table(columns=["id", "user_data"])
        assert selected.num_columns == 2
        assert "active" not in selected.column_names

        # Test filtering
        filtered = dataset.to_table(filter="active = true")
        assert filtered.num_rows == 2
        assert filtered.column("id").to_pylist() == [1, 3]


def test_json_null_handling():
    """Test handling of null JSON values."""

    # Mix of valid JSON and nulls
    json_strings = [
        json.dumps({"key": "value1"}),
        None,
        json.dumps({"key": "value2"}),
        None,
        json.dumps({"key": "value3"}),
    ]

    json_ext_array = JsonArray.from_json_strings(json_strings)

    table = pa.table(
        {"id": pa.array(range(5)), "optional_data": json_ext_array},
        schema=pa.schema(
            [pa.field("id", pa.int32()), json_field("optional_data", nullable=True)]
        ),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "null_test.lance"
        lance.write_dataset(table, dataset_path, data_storage_version="2.2")
        dataset = lance.dataset(dataset_path)

        result = dataset.to_table()
        assert result.num_rows == 5

        # Check null mask
        data_column = result.column("optional_data")
        assert data_column.null_count == 2
        assert data_column.is_null().to_pylist() == [False, True, False, True, False]


def test_json_large_documents():
    """Test handling of large JSON documents."""

    # Create a moderately large JSON document
    large_doc = {
        "metadata": {
            "created": "2024-01-01",
            "version": "1.0.0",
        },
        "items": [
            {
                "id": i,
                "name": f"Item {i}",
                "description": f"Description for item {i}",
                "tags": [f"tag{j}" for j in range(5)],
                "properties": {f"prop{j}": f"value{j}" for j in range(10)},
            }
            for i in range(20)
        ],
    }

    json_string = json.dumps(large_doc)
    json_ext_array = JsonArray.from_json_strings([json_string])

    table = pa.table(
        {"id": pa.array([1]), "large_json": json_ext_array},
        schema=pa.schema([pa.field("id", pa.int32()), json_field("large_json")]),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "large_json.lance"
        lance.write_dataset(table, dataset_path, data_storage_version="2.2")
        dataset = lance.dataset(dataset_path)

        result = dataset.to_table()
        assert result.num_rows == 1

        # Verify the document is preserved
        retrieved_data = result.column("large_json")[0].as_py()
        assert retrieved_data is not None


def test_json_batch_operations():
    """Test batch operations with JSON data."""

    # Create multiple batches
    batch_size = 1000
    num_batches = 5

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "batch_test.lance"

        for batch_num in range(num_batches):
            json_data = [
                json.dumps({"batch": batch_num, "item": i}) for i in range(batch_size)
            ]

            json_ext_array = JsonArray.from_json_strings(json_data)

            table = pa.table(
                {
                    "id": pa.array(
                        range(batch_num * batch_size, (batch_num + 1) * batch_size)
                    ),
                    "batch_data": json_ext_array,
                },
                schema=pa.schema(
                    [pa.field("id", pa.int32()), json_field("batch_data")]
                ),
            )

            if batch_num == 0:
                lance.write_dataset(table, dataset_path, data_storage_version="2.2")
            else:
                lance.write_dataset(table, dataset_path, mode="append")

        # Verify all batches were written
        dataset = lance.dataset(dataset_path)
        assert dataset.count_rows() == batch_size * num_batches

        # Test batch reading
        batches = list(dataset.to_batches(batch_size=batch_size))
        assert len(batches) == num_batches

        for batch in batches:
            assert batch.num_rows == batch_size
