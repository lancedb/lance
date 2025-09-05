# JSON Support

Lance provides comprehensive support for storing and querying JSON data, enabling you to work with semi-structured data efficiently. This guide covers how to store JSON data in Lance datasets and use JSON functions to query and filter your data.

## Getting Started

```python
import lance
import pyarrow as pa
import json

# Create a table with JSON data
json_data = {"name": "Alice", "age": 30, "city": "New York"}
json_arr = pa.array([json.dumps(json_data)], type=pa.json_())
table = pa.table({"id": [1], "data": json_arr})

# Write the dataset
lance.write_dataset(table, "dataset.lance")
```

## Storage Format

Lance stores JSON data internally as JSONB (binary JSON) using the `lance.json` extension type. This provides:

- Efficient storage through binary encoding
- Fast query performance for nested field access
- Compatibility with Apache Arrow's JSON type

When you read JSON data back from Lance, it's automatically converted to Arrow's JSON type for seamless integration with your data processing pipelines.

## JSON Functions

Lance provides a comprehensive set of JSON functions for querying and filtering JSON data. These functions can be used in filter expressions with methods like `to_table()`, `scanner()`, and SQL queries through DataFusion integration.

### Data Access Functions

#### json_extract

Extracts a value from JSON using JSONPath syntax.

**Syntax:** `json_extract(json_column, json_path)`

**Returns:** JSON-formatted string representation of the extracted value

**Example:**
```python
# Sample data: {"user": {"name": "Alice", "age": 30}}
result = dataset.to_table(
    filter="json_extract(data, '$.user.name') = '\"Alice\"'"
)
# Returns: "\"Alice\"" for strings, "30" for numbers, "true" for booleans
```

!!! note
    `json_extract` returns values in JSON format. String values include quotes (e.g., `"Alice"`), 
    numbers are returned as-is (e.g., `30`), and booleans as `true`/`false`.

#### json_get

Retrieves a field or array element from JSON, returning it as JSONB for further processing.

**Syntax:** `json_get(json_column, key_or_index)`

**Parameters:**
- `key_or_index`: Field name (string) or array index (numeric string like "0", "1")

**Returns:** JSONB binary value (can be used for nested access)

**Example:**
```python
# Access nested JSON by chaining json_get calls
# Sample data: {"user": {"profile": {"name": "Alice"}}}
result = dataset.to_table(
    filter="json_get_string(json_get(json_get(data, 'user'), 'profile'), 'name') = 'Alice'"
)

# Access array elements by index
# Sample data: ["first", "second", "third"]
result = dataset.to_table(
    filter="json_get_string(data, '0') = 'first'"  # Gets first array element
)
```

### Type-Safe Value Extraction

These functions extract values with strict type conversion. The conversion uses JSONB's built-in strict mode, which requires values to be of compatible types:

#### json_get_string

Extracts a string value from JSON.

**Syntax:** `json_get_string(json_column, key_or_index)`

**Parameters:**
- `key_or_index`: Field name or array index (as string)

**Returns:** String value (without JSON quotes), null if conversion fails

**Type Conversion:** Uses strict conversion - numbers and booleans are converted to their string representation

**Example:**
```python
result = dataset.to_table(
    filter="json_get_string(data, 'name') = 'Alice'"
)

# Array access example
# Sample data: ["first", "second"]
result = dataset.to_table(
    filter="json_get_string(data, '1') = 'second'"  # Gets second array element
)
```

#### json_get_int

Extracts an integer value with strict type conversion.

**Syntax:** `json_get_int(json_column, key_or_index)`

**Returns:** 64-bit integer, null if conversion fails

**Type Conversion:** Uses JSONB's strict `to_i64()` conversion:
- Numbers are truncated to integers
- Strings must be parseable as numbers
- Booleans: true → 1, false → 0

**Example:**
```python
# {"age": 30} works, {"age": "30"} may work if JSONB allows string parsing
result = dataset.to_table(
    filter="json_get_int(data, 'age') > 25"
)
```

#### json_get_float

Extracts a floating-point value with strict type conversion.

**Syntax:** `json_get_float(json_column, key_or_index)`

**Returns:** 64-bit float, null if conversion fails

**Type Conversion:** Uses JSONB's strict `to_f64()` conversion:
- Integers are converted to floats
- Strings must be parseable as numbers
- Booleans: true → 1.0, false → 0.0

**Example:**
```python
result = dataset.to_table(
    filter="json_get_float(data, 'score') >= 90.5"
)
```

#### json_get_bool

Extracts a boolean value with strict type conversion.

**Syntax:** `json_get_bool(json_column, key_or_index)`

**Returns:** Boolean, null if conversion fails

**Type Conversion:** Uses JSONB's strict `to_bool()` conversion:
- Numbers: 0 → false, non-zero → true
- Strings: "true" → true, "false" → false (exact match required)
- Other values may fail conversion

**Example:**
```python
result = dataset.to_table(
    filter="json_get_bool(data, 'active') = true"
)
```

### Existence and Array Functions

#### json_exists

Checks if a JSONPath exists in the JSON data.

**Syntax:** `json_exists(json_column, json_path)`

**Returns:** Boolean

**Example:**
```python
# Find records that have an age field
result = dataset.to_table(
    filter="json_exists(data, '$.user.age')"
)
```

#### json_array_contains

Checks if a JSON array contains a specific value.

**Syntax:** `json_array_contains(json_column, json_path, value)`

**Returns:** Boolean

**Comparison Logic:** 
- Compares array elements as JSON strings
- For string matching, tries both with and without quotes
- Example: searching for 'python' matches both `"python"` and `python` in the array

**Example:**
```python
# Sample data: {"tags": ["python", "ml", "data"]}
result = dataset.to_table(
    filter="json_array_contains(data, '$.tags', 'python')"
)
```

#### json_array_length

Returns the length of a JSON array.

**Syntax:** `json_array_length(json_column, json_path)`

**Returns:** 
- Integer: length of the array
- null: if path doesn't exist
- Error: if path points to a non-array value

**Example:**
```python
# Find records with more than 3 tags
result = dataset.to_table(
    filter="json_array_length(data, '$.tags') > 3"
)

# Empty arrays return 0
result = dataset.to_table(
    filter="json_array_length(data, '$.empty_array') = 0"
)
```

## Usage Examples

### Working with Nested JSON

```python
import lance
import pyarrow as pa
import json

# Create nested JSON data
data = [
    {
        "id": 1,
        "user": {
            "profile": {
                "name": "Alice",
                "settings": {
                    "theme": "dark",
                    "notifications": True
                }
            },
            "scores": [95, 87, 92]
        }
    },
    {
        "id": 2,
        "user": {
            "profile": {
                "name": "Bob",
                "settings": {
                    "theme": "light",
                    "notifications": False
                }
            },
            "scores": [88, 91, 85]
        }
    }
]

# Convert to Lance dataset
json_strings = [json.dumps(d) for d in data]
table = pa.table({
    "data": pa.array(json_strings, type=pa.json_())
})

lance.write_dataset(table, "nested.lance")
dataset = lance.dataset("nested.lance")

# Query nested fields using JSONPath
dark_theme_users = dataset.to_table(
    filter="json_extract(data, '$.user.profile.settings.theme') = '\"dark\"'"
)

# Or using chained json_get
high_scorers = dataset.to_table(
    filter="json_array_length(data, '$.user.scores') >= 3"
)
```

### Combining JSON with Other Data Types

```python
# Create mixed-type table with JSON metadata
products = pa.table({
    "id": [1, 2, 3],
    "name": ["Laptop", "Phone", "Tablet"],
    "price": [999.99, 599.99, 399.99],
    "specs": pa.array([
        json.dumps({"cpu": "i7", "ram": 16, "storage": 512}),
        json.dumps({"screen": 6.1, "battery": 4000, "5g": True}),
        json.dumps({"screen": 10.5, "battery": 7000, "stylus": True})
    ], type=pa.json_())
})

lance.write_dataset(products, "products.lance")
dataset = lance.dataset("products.lance")

# Find products with specific specs
result = dataset.to_table(
    filter="price < 600 AND json_get_bool(specs, '5g') = true"
)
```

### Handling Arrays in JSON

```python
# Create data with JSON arrays
records = pa.table({
    "id": [1, 2, 3],
    "data": pa.array([
        json.dumps({"name": "Project A", "tags": ["python", "ml", "production"]}),
        json.dumps({"name": "Project B", "tags": ["rust", "systems"]}),
        json.dumps({"name": "Project C", "tags": ["python", "web", "api", "production"]})
    ], type=pa.json_())
})

lance.write_dataset(records, "projects.lance")
dataset = lance.dataset("projects.lance")

# Find projects with Python
python_projects = dataset.to_table(
    filter="json_array_contains(data, '$.tags', 'python')"
)

# Find projects with more than 3 tags
complex_projects = dataset.to_table(
    filter="json_array_length(data, '$.tags') > 3"
)
```

## Performance Considerations

1. **Choose the right function**: Use `json_get_*` functions for direct field access and type conversion; use `json_extract` for complex JSONPath queries.
2. **Index frequently queried paths**: Consider creating computed columns for frequently accessed JSON paths to improve query performance.
3. **Minimize deep nesting**: While Lance supports arbitrary nesting, flatter structures generally perform better.
4. **Understand type conversion**: The `json_get_*` functions use strict type conversion, which may fail if types don't match. Plan your schema accordingly.
5. **Array access**: When working with JSON arrays, you can access elements by index using numeric strings (e.g., "0", "1") with `json_get` functions.

## Integration with DataFusion

All JSON functions are available when using Lance with Apache DataFusion for SQL queries. See the [DataFusion Integration](../integrations/datafusion.md#json-functions) guide for more details on using JSON functions in SQL contexts.

## Limitations

- JSONPath support follows standard JSONPath syntax but may not support all advanced features
- Large JSON documents may impact query performance
- JSON functions are currently only available for filtering, not for projection in query results
