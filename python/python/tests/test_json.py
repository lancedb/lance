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

    json_strings = [json.dumps(d) if d is not None else None for d in json_data]
    json_arr = pa.array(json_strings, type=pa.json_())
    table = pa.table(
        {
            "id": pa.array([1, 2, 3, 4, 5], type=pa.int32()),
            "data": json_arr,
        }
    )

    # Write to Lance dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "json_test.lance"

        # Write the dataset
        lance.write_dataset(table, dataset_path)

        # Read back the dataset
        dataset = lance.dataset(dataset_path)

        # Verify logical schema exposed to users
        logical_schema = dataset.schema
        assert len(logical_schema) == 2
        assert logical_schema.field("id").type == pa.int32()
        logical_field = logical_schema.field("data")
        assert (
            str(logical_field.type) == "extension<arrow.json>"
            or logical_field.type == pa.utf8()
        )

        # Read data back
        result_table = dataset.to_table()

        # Check that data is returned as Arrow JSON for Python
        result_field = result_table.schema.field("data")
        # PyArrow extension types print as extension<arrow.json> but
        # the storage type is utf8
        assert (
            str(result_field.type) == "extension<arrow.json>"
            or result_field.type == pa.utf8()
        )

        # Verify data
        assert result_table.num_rows == 5
        assert result_table.column("id").to_pylist() == [1, 2, 3, 4, 5]


def test_json_with_other_types():
    """Test JSON type alongside other data types."""

    # Create mixed type data
    json_data = [
        {"product": "laptop", "specs": {"cpu": "i7", "ram": 16}},
        {"product": "phone", "specs": {"screen": "6.1", "battery": 4000}},
    ]

    json_strings = [json.dumps(d) for d in json_data]

    # Create JSON array using PyArrow's JSON type
    json_arr = pa.array(json_strings, type=pa.json_())

    table = pa.table(
        {
            "id": pa.array([1, 2], type=pa.int64()),
            "name": pa.array(["Product A", "Product B"], type=pa.string()),
            "price": pa.array([999.99, 599.99], type=pa.float64()),
            "metadata": json_arr,
            "in_stock": pa.array([True, False], type=pa.bool_()),
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "mixed_types.lance"

        # Write and read the dataset
        lance.write_dataset(table, dataset_path)
        dataset = lance.dataset(dataset_path)

        # Verify all fields are preserved
        assert len(dataset.schema) == 5

        result = dataset.to_table()
        assert result.num_rows == 2
        assert result.column("id").to_pylist() == [1, 2]
        assert result.column("name").to_pylist() == ["Product A", "Product B"]
        assert result.column("in_stock").to_pylist() == [True, False]


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

    # Create JSON array with nulls using PyArrow's JSON type
    json_arr = pa.array(json_strings, type=pa.json_())

    table = pa.table({"id": pa.array(range(5)), "optional_data": json_arr})

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

            # Create JSON array using PyArrow's JSON type
            json_arr = pa.array(json_data, type=pa.json_())

            table = pa.table(
                {
                    "id": pa.array(
                        range(batch_num * batch_size, (batch_num + 1) * batch_size)
                    ),
                    "batch_data": json_arr,
                }
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


def test_json_path_queries():
    """Test JSON path queries using json_extract and json_exists."""
    # Create test data with JSON columns
    json_data = [
        {"user": {"name": "Alice", "age": 30}, "tags": ["python", "ml"]},
        {"user": {"name": "Bob", "age": 25}, "tags": ["rust", "db"]},
        {"user": {"name": "Charlie"}, "tags": []},
        None,
    ]

    json_strings = [json.dumps(d) if d is not None else None for d in json_data]
    json_arr = pa.array(json_strings, type=pa.json_())

    # Create a Lance dataset with JSON data
    table = pa.table(
        {
            "id": [1, 2, 3, 4],
            "data": json_arr,
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ds_path = Path(tmpdir) / "json_test.lance"
        lance.write_dataset(table, ds_path)
        dataset = lance.dataset(ds_path)

        # Test json_extract
        result = dataset.to_table(
            filter="json_extract(data, '$.user.name') = '\"Alice\"'"
        )
        assert result.num_rows == 1
        assert result["id"][0].as_py() == 1

        # Test json_exists
        result = dataset.to_table(filter="json_exists(data, '$.user.age')")
        assert result.num_rows == 2  # Alice and Bob have age field

        # Test json_array_contains
        result = dataset.to_table(
            filter="json_array_contains(data, '$.tags', 'python')"
        )
        assert result.num_rows == 1
        assert result["id"][0].as_py() == 1


def test_json_get_functions():
    """Test json_get_* functions for type-safe value extraction."""
    # Create test data with various JSON types
    json_data = [
        {"name": "Alice", "age": 30, "active": True, "score": 95.5},
        {"name": "Bob", "age": 25, "active": False, "score": 87.3},
        {"name": "Charlie", "age": "35", "active": "true", "score": "92"},
        {"name": "David"},  # Missing fields
    ]

    json_strings = [json.dumps(d) for d in json_data]
    json_arr = pa.array(json_strings, type=pa.json_())

    table = pa.table(
        {
            "id": [1, 2, 3, 4],
            "data": json_arr,
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ds_path = Path(tmpdir) / "json_get_test.lance"
        lance.write_dataset(table, ds_path)
        dataset = lance.dataset(ds_path)

        # Test json_get_string
        result = dataset.to_table(filter="json_get_string(data, 'name') = 'Alice'")
        assert result.num_rows == 1
        assert result["id"][0].as_py() == 1

        # Test json_get_int with type coercion
        result = dataset.to_table(filter="json_get_int(data, 'age') > 28")
        assert result.num_rows == 2  # Alice (30) and Charlie ("35" -> 35)

        # Test json_get_bool with type coercion
        result = dataset.to_table(filter="json_get_bool(data, 'active') = true")
        assert result.num_rows == 2  # Alice (true) and Charlie ("true" -> true)

        # Test json_get_float
        result = dataset.to_table(filter="json_get_float(data, 'score') > 90")
        assert result.num_rows == 2  # Alice (95.5) and Charlie ("92" -> 92.0)


def test_nested_json_access():
    """Test accessing nested JSON structures."""
    json_data = [
        {"user": {"profile": {"name": "Alice", "settings": {"theme": "dark"}}}},
        {"user": {"profile": {"name": "Bob", "settings": {"theme": "light"}}}},
    ]

    json_strings = [json.dumps(d) for d in json_data]
    json_arr = pa.array(json_strings, type=pa.json_())

    table = pa.table(
        {
            "id": [1, 2],
            "data": json_arr,
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ds_path = Path(tmpdir) / "nested_json_test.lance"
        lance.write_dataset(table, ds_path)
        dataset = lance.dataset(ds_path)

        # Access nested fields using json_get recursively
        # First get user, then profile, then name
        result = dataset.to_table(
            filter="""
                json_get_string(
                    json_get(
                        json_get(data, 'user'),
                        'profile'),
                    'name')
                = 'Alice'"""
        )
        assert result.num_rows == 1
        assert result["id"][0].as_py() == 1

        # Or use JSONPath for deep access
        result = dataset.to_table(
            filter="json_extract(data, '$.user.profile.settings.theme') = '\"dark\"'"
        )
        assert result.num_rows == 1
        assert result["id"][0].as_py() == 1


def test_json_array_operations():
    """Test JSON array operations."""
    json_data = [
        {"items": ["apple", "banana", "orange"], "counts": [1, 2, 3, 4, 5]},
        {"items": ["grape", "melon"], "counts": [10, 20]},
        {"items": [], "counts": []},
    ]

    json_strings = [json.dumps(d) for d in json_data]
    json_arr = pa.array(json_strings, type=pa.json_())

    table = pa.table(
        {
            "id": [1, 2, 3],
            "data": json_arr,
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ds_path = Path(tmpdir) / "array_json_test.lance"
        lance.write_dataset(table, ds_path)
        dataset = lance.dataset(ds_path)

        # Test array contains
        result = dataset.to_table(
            filter="json_array_contains(data, '$.items', 'apple')"
        )
        assert result.num_rows == 1
        assert result["id"][0].as_py() == 1

        # Test array length
        result = dataset.to_table(filter="json_array_length(data, '$.counts') > 3")
        assert result.num_rows == 1
        assert result["id"][0].as_py() == 1

        # Test empty array
        result = dataset.to_table(filter="json_array_length(data, '$.items') = 0")
        assert result.num_rows == 1
        assert result["id"][0].as_py() == 3


def test_json_filter_append_missing_json_cast(tmp_path: Path):
    """Ensure appending via dataset.schema keeps JSON columns valid."""

    dataset_path = tmp_path / "json_append_issue.lance"

    initial_table = pa.table(
        {
            "article_metadata": pa.array(
                [json.dumps({"article_journal": "Cell"})], type=pa.json_()
            ),
            "article_journal": pa.array(["Cell"], type=pa.string()),
        }
    )

    lance.write_dataset(initial_table, dataset_path)
    dataset = lance.dataset(dataset_path)
    schema = dataset.schema
    field = schema.field("article_metadata")
    assert str(field.type) == "extension<arrow.json>" or field.type == pa.utf8()

    append_table = pa.table(
        {
            "article_metadata": pa.array(
                [
                    json.dumps({"article_journal": "PLoS One"}),
                    json.dumps({"article_journal": "Nature"}),
                ],
                type=pa.json_(),
            ),
            "article_journal": pa.array(["PLoS One", "Nature"], type=pa.string()),
        }
    )

    append_cast = append_table.cast(schema)
    first_value = append_cast.column("article_metadata").to_pylist()[0]
    assert isinstance(first_value, str)

    lance.write_dataset(append_cast, dataset_path, mode="append")
    dataset = lance.dataset(dataset_path)
    assert dataset.count_rows() == 3

    result = dataset.to_table(
        filter="json_get(article_metadata, 'article_journal') IS NOT NULL"
    )

    assert result.num_rows == 3
    assert result.column("article_journal").to_pylist() == [
        "Cell",
        "PLoS One",
        "Nature",
    ]
