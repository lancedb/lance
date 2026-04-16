# JSON 支持

Lance 全面支持存储和查询 JSON 数据，使你能够高效地处理半结构化数据。本指南介绍如何在 Lance 数据集中存储 JSON 数据，以及如何使用 JSON 函数来查询和过滤数据。

## 入门

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

## 存储格式

Lance 内部使用 JSONB（二进制 JSON）格式存储 JSON 数据，采用 `lance.json` 扩展类型。这提供了：

- 通过二进制编码实现高效存储
- 嵌套字段访问的快速查询性能
- 与 Apache Arrow JSON 类型的兼容性

当你从 Lance 读回 JSON 数据时，它会自动转换为 Arrow 的 JSON 类型，与你的数据处理管道无缝集成。

## JSON 函数

Lance 提供了一组全面的 JSON 函数，用于查询和过滤 JSON 数据。这些函数可用于 `to_table()`、`scanner()` 等方法的过滤表达式中，以及通过 DataFusion 集成的 SQL 查询中。

### 数据访问函数

#### json_extract

使用 JSONPath 语法从 JSON 中提取值。

**语法：** `json_extract(json_column, json_path)`

**返回：** 提取值的 JSON 格式字符串表示

**示例：**
```python
# Sample data: {"user": {"name": "Alice", "age": 30}}
result = dataset.to_table(
    filter="json_extract(data, '$.user.name') = '\"Alice\"'"
)
# Returns: "\"Alice\"" for strings, "30" for numbers, "true" for booleans
```

!!! note
    `json_extract` 以 JSON 格式返回值。字符串值包含引号（例如 `"Alice"`），
    数字按原样返回（例如 `30`），布尔值为 `true`/`false`。

#### json_get

从 JSON 中检索字段或数组元素，以 JSONB 格式返回用于进一步处理。

**语法：** `json_get(json_column, key_or_index)`

**参数：**
- `key_or_index`：字段名（字符串）或数组索引（数字字符串如 "0"、"1"）

**返回：** JSONB 二进制值（可用于嵌套访问）

**示例：**
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

### 类型安全的值提取

这些函数以严格类型转换提取值。转换使用 JSONB 内置的严格模式，要求值必须是兼容类型：

#### json_get_string

从 JSON 中提取字符串值。

**语法：** `json_get_string(json_column, key_or_index)`

**参数：**
- `key_or_index`：字段名或数组索引（字符串形式）

**返回：** 字符串值（不含 JSON 引号），转换失败时返回 null

**类型转换：** 使用严格转换 - 数字和布尔值会被转换为其字符串表示

**示例：**
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

以严格类型转换提取整数值。

**语法：** `json_get_int(json_column, key_or_index)`

**返回：** 64 位整数，转换失败时返回 null

**类型转换：** 使用 JSONB 的严格 `to_i64()` 转换：
- 数字会被截断为整数
- 字符串必须可解析为数字
- 布尔值：true → 1，false → 0

**示例：**
```python
# {"age": 30} works, {"age": "30"} may work if JSONB allows string parsing
result = dataset.to_table(
    filter="json_get_int(data, 'age') > 25"
)
```

#### json_get_float

以严格类型转换提取浮点值。

**语法：** `json_get_float(json_column, key_or_index)`

**返回：** 64 位浮点数，转换失败时返回 null

**类型转换：** 使用 JSONB 的严格 `to_f64()` 转换：
- 整数会被转换为浮点数
- 字符串必须可解析为数字
- 布尔值：true → 1.0，false → 0.0

**示例：**
```python
result = dataset.to_table(
    filter="json_get_float(data, 'score') >= 90.5"
)
```

#### json_get_bool

以严格类型转换提取布尔值。

**语法：** `json_get_bool(json_column, key_or_index)`

**返回：** 布尔值，转换失败时返回 null

**类型转换：** 使用 JSONB 的严格 `to_bool()` 转换：
- 数字：0 → false，非零 → true
- 字符串："true" → true，"false" → false（需精确匹配）
- 其他值可能转换失败

**示例：**
```python
result = dataset.to_table(
    filter="json_get_bool(data, 'active') = true"
)
```

### 存在性和数组函数

#### json_exists

检查 JSONPath 是否存在于 JSON 数据中。

**语法：** `json_exists(json_column, json_path)`

**返回：** 布尔值

**示例：**
```python
# Find records that have an age field
result = dataset.to_table(
    filter="json_exists(data, '$.user.age')"
)
```

#### json_array_contains

检查 JSON 数组是否包含特定值。

**语法：** `json_array_contains(json_column, json_path, value)`

**返回：** 布尔值

**比较逻辑：**
- 将数组元素作为 JSON 字符串进行比较
- 对于字符串匹配，会尝试带引号和不带引号两种方式
- 示例：搜索 'python' 可以匹配数组中的 `"python"` 和 `python`

**示例：**
```python
# Sample data: {"tags": ["python", "ml", "data"]}
result = dataset.to_table(
    filter="json_array_contains(data, '$.tags', 'python')"
)
```

#### json_array_length

返回 JSON 数组的长度。

**语法：** `json_array_length(json_column, json_path)`

**返回：**
- 整数：数组的长度
- null：如果路径不存在
- 错误：如果路径指向非数组值

**示例：**
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

## JSON 索引

Lance 支持对 JSON 列建立索引，以加速对频繁查询路径的过滤。

### JSON 路径上的标量索引

对于 `pa.json_()` 列，使用 `IndexConfig` 创建标量索引并指定要索引的 JSON 路径。查询应使用与索引相同的路径字面量。

```python
import json
import lance
import pyarrow as pa
from lance.indices import IndexConfig

table = pa.table({
    "id": [1, 2, 3, 4],
    "data": pa.array([
        json.dumps({"x": 7, "y": 10}),
        json.dumps({"x": 11, "y": 22}),
        json.dumps({"y": 0}),
        json.dumps({"x": 10}),
    ], type=pa.json_()),
})

lance.write_dataset(table, "json-index.lance")
dataset = lance.dataset("json-index.lance")

dataset.create_scalar_index(
    "data",
    IndexConfig(
        index_type="json",
        parameters={
            "target_index_type": "btree",
            "path": "x",
        },
    ),
)

result = dataset.to_table(filter="json_get_int(data, 'x') = 10")
```

!!! note
    JSON 索引按路径字面量匹配查询。例如，如果索引使用 `path="x"` 构建，
    则过滤器也应使用 `"x"` 配合 `json_get_int(data, 'x')` 等函数。如果索引使用
    `path="$.user.name"` 构建，则过滤器应使用 `json_extract(data, '$.user.name')`。

### JSON 文档的全文搜索

如果你想对 JSON 文档的内容进行文本搜索而非对单个路径进行标量过滤，请在 JSON 列上创建 `INVERTED` 索引。

```python
dataset.create_scalar_index(
    "data",
    index_type="INVERTED",
    base_tokenizer="simple",
    lower_case=True,
    stem=True,
    remove_stop_words=True,
)
```

!!! note
    JSON 列和嵌套 struct 列的索引方式不同。对于嵌套 struct 字段，使用点号表示法如
    `meta.lang`。对于 `pa.json_()` 列，使用上面展示的 JSON 索引并用 `json_get_*`
    或 `json_extract` 查询。

## 使用示例

### 处理嵌套 JSON

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

### 组合 JSON 与其他数据类型

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

### 处理 JSON 中的数组

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

## 性能考量

1. **选择合适的函数**：使用 `json_get_*` 函数进行直接字段访问和类型转换；使用 `json_extract` 进行复杂的 JSONPath 查询。
2. **为频繁查询的路径建立索引**：在为相同字段创建计算列之前，先对频繁过滤的路径使用 JSON 标量索引。
3. **减少深层嵌套**：虽然 Lance 支持任意嵌套，但扁平化的结构通常性能更好。
4. **理解类型转换**：`json_get_*` 函数使用严格类型转换，如果类型不匹配可能会失败。请相应地规划你的 Schema。
5. **数组访问**：处理 JSON 数组时，你可以使用数字字符串（例如 "0"、"1"）通过 `json_get` 函数按索引访问元素。

## 与 DataFusion 集成

所有 JSON 函数在使用 Lance 与 Apache DataFusion 进行 SQL 查询时均可用。更多关于在 SQL 上下文中使用 JSON 函数的详细信息，请参阅 [DataFusion 集成](../integrations/datafusion.md#json-functions)指南。

## 限制

- JSONPath 支持遵循标准 JSONPath 语法，但可能不支持所有高级功能
- 大型 JSON 文档可能影响查询性能
- JSON 函数目前仅可用于过滤，不能用于查询结果的投影
