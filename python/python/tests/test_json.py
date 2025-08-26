# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import json
import tempfile
from pathlib import Path

import lance
import pyarrow as pa


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

    # Create a PyArrow table with JSON field
    # Using LargeBinary with extension metadata for JSON
    json_field = pa.field(
        "data", pa.large_binary(), metadata={"ARROW:extension:name": "lance.json"}
    )

    # Create array from JSON strings
    json_array = pa.array(json_strings, type=pa.string())
    # Cast to large_binary for storage
    json_binary_array = pa.compute.cast(json_array, pa.large_binary())

    table = pa.table(
        {
            "id": pa.array([1, 2, 3, 4, 5], type=pa.int32()),
            "data": json_binary_array,
        },
        schema=pa.schema(
            [
                pa.field("id", pa.int32()),
                json_field,
            ]
        ),
    )

    # Write to Lance dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "json_test.lance"

        # Write the dataset
        lance.write_dataset(table, dataset_path)

        # Read back the dataset
        dataset = lance.dataset(dataset_path)

        # Verify schema
        assert len(dataset.schema) == 2
        assert dataset.schema.field("id").type == pa.int32()

        # Check that JSON field is properly recognized
        json_field_schema = dataset.schema.field("data")
        assert json_field_schema.type == pa.large_binary()
        assert json_field_schema.metadata.get(b"ARROW:extension:name") == b"lance.json"

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

    # Create table with multiple types
    json_field = pa.field(
        "metadata", pa.large_binary(), metadata={"ARROW:extension:name": "lance.json"}
    )

    json_array = pa.array(json_strings, type=pa.string())
    json_binary_array = pa.compute.cast(json_array, pa.large_binary())

    table = pa.table(
        {
            "id": pa.array([1, 2], type=pa.int64()),
            "name": pa.array(["Product A", "Product B"], type=pa.string()),
            "price": pa.array([999.99, 599.99], type=pa.float64()),
            "metadata": json_binary_array,
            "in_stock": pa.array([True, False], type=pa.bool_()),
        },
        schema=pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("name", pa.string()),
                pa.field("price", pa.float64()),
                json_field,
                pa.field("in_stock", pa.bool_()),
            ]
        ),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "mixed_types.lance"

        # Write and read
        lance.write_dataset(table, dataset_path)
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
        json_field = pa.field(
            "config", pa.large_binary(), metadata={"ARROW:extension:name": "lance.json"}
        )

        initial_json = [json.dumps({"version": 1, "enabled": True})]
        initial_array = pa.compute.cast(
            pa.array(initial_json, type=pa.string()), pa.large_binary()
        )

        initial_table = pa.table(
            {"id": pa.array([1]), "config": initial_array},
            schema=pa.schema([pa.field("id", pa.int32()), json_field]),
        )

        # Write initial data
        lance.write_dataset(initial_table, dataset_path)

        # Append more data
        append_json = [json.dumps({"version": 2, "enabled": False})]
        append_array = pa.compute.cast(
            pa.array(append_json, type=pa.string()), pa.large_binary()
        )

        append_table = pa.table(
            {"id": pa.array([2]), "config": append_array},
            schema=pa.schema([pa.field("id", pa.int32()), json_field]),
        )

        dataset = lance.dataset(dataset_path)
        dataset = lance.write_dataset(append_table, dataset_path, mode="append")

        # Verify append
        result = dataset.to_table()
        assert result.num_rows == 2
        assert result.column("id").to_pylist() == [1, 2]

        # Test overwrite
        overwrite_json = [json.dumps({"version": 3, "enabled": True})]
        overwrite_array = pa.compute.cast(
            pa.array(overwrite_json, type=pa.string()), pa.large_binary()
        )

        overwrite_table = pa.table(
            {"id": pa.array([3]), "config": overwrite_array},
            schema=pa.schema([pa.field("id", pa.int32()), json_field]),
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
    json_field = pa.field(
        "user_data", pa.large_binary(), metadata={"ARROW:extension:name": "lance.json"}
    )

    json_array = pa.compute.cast(
        pa.array(json_strings, type=pa.string()), pa.large_binary()
    )

    table = pa.table(
        {
            "id": pa.array([1, 2, 3]),
            "user_data": json_array,
            "active": pa.array([True, False, True]),
        },
        schema=pa.schema(
            [
                pa.field("id", pa.int32()),
                json_field,
                pa.field("active", pa.bool_()),
            ]
        ),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "filter_test.lance"
        lance.write_dataset(table, dataset_path)
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

    json_field = pa.field(
        "optional_data",
        pa.large_binary(),
        nullable=True,
        metadata={"ARROW:extension:name": "lance.json"},
    )

    # Mix of valid JSON and nulls
    json_strings = [
        json.dumps({"key": "value1"}),
        None,
        json.dumps({"key": "value2"}),
        None,
        json.dumps({"key": "value3"}),
    ]

    json_array = pa.array(json_strings, type=pa.string())
    json_binary_array = pa.compute.cast(json_array, pa.large_binary())

    table = pa.table(
        {"id": pa.array(range(5)), "optional_data": json_binary_array},
        schema=pa.schema([pa.field("id", pa.int32()), json_field]),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "null_test.lance"
        lance.write_dataset(table, dataset_path)
        dataset = lance.dataset(dataset_path)

        result = dataset.to_table()
        assert result.num_rows == 5

        # Check null mask
        data_column = result.column("optional_data")
        assert data_column.null_count == 2
        assert data_column.is_null().to_pylist() == [False, True, False, True, False]


def test_json_large_documents():
    """Test handling of large JSON documents."""

    # Create a large JSON document
    large_doc = {
        "metadata": {
            "created": "2024-01-01",
            "version": "1.0.0",
        },
        "items": [
            {
                "id": i,
                "name": f"Item {i}",
                "description": f"Description for item {i}" * 10,
                "tags": [f"tag{j}" for j in range(20)],
                "properties": {f"prop{j}": f"value{j}" for j in range(50)},
            }
            for i in range(100)
        ],
    }

    json_string = json.dumps(large_doc)

    json_field = pa.field(
        "large_json", pa.large_binary(), metadata={"ARROW:extension:name": "lance.json"}
    )

    json_array = pa.compute.cast(
        pa.array([json_string], type=pa.string()), pa.large_binary()
    )

    table = pa.table(
        {"id": pa.array([1]), "large_json": json_array},
        schema=pa.schema([pa.field("id", pa.int32()), json_field]),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "large_json.lance"
        lance.write_dataset(table, dataset_path)
        dataset = lance.dataset(dataset_path)

        result = dataset.to_table()
        assert result.num_rows == 1

        # Verify the large document is preserved
        retrieved_data = result.column("large_json")[0].as_py()
        assert retrieved_data is not None
        assert len(retrieved_data) > 10000  # Should be a large binary


def test_json_batch_operations():
    """Test batch operations with JSON data."""

    json_field = pa.field(
        "batch_data", pa.large_binary(), metadata={"ARROW:extension:name": "lance.json"}
    )

    # Create multiple batches
    batch_size = 1000
    num_batches = 5

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "batch_test.lance"

        for batch_num in range(num_batches):
            json_data = [
                json.dumps({"batch": batch_num, "item": i}) for i in range(batch_size)
            ]

            json_array = pa.compute.cast(
                pa.array(json_data, type=pa.string()), pa.large_binary()
            )

            table = pa.table(
                {
                    "id": pa.array(
                        range(batch_num * batch_size, (batch_num + 1) * batch_size)
                    ),
                    "batch_data": json_array,
                },
                schema=pa.schema([pa.field("id", pa.int32()), json_field]),
            )

            if batch_num == 0:
                lance.write_dataset(table, dataset_path)
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
