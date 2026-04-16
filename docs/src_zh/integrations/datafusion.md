# Apache DataFusion 集成

Lance 数据集可以使用 [Apache DataFusion](https://datafusion.apache.org/) 进行查询，
这是一个用 Rust 编写的可扩展查询引擎，使用 Apache Arrow 作为其内存格式。
这意味着你可以编写复杂的 SQL 查询来分析 Lance 中的数据。

该集成允许用户将列选择和基本过滤条件下推到 Lance，
从而在执行查询时减少扫描的数据量。
此外，该集成允许从 Lance 数据集流式传输数据，
使用户能够进行超出内存限制的聚合操作。

## Rust

Lance 包含一个 DataFusion 表提供器（Table Provider）`lance::datafusion::LanceTableProvider`。
用户可以在 DataFusion 中将 Lance 数据集注册为表并运行 SQL 查询：

### 简单 SQL

```rust
use datafusion::prelude::SessionContext;
use lance::datafusion::LanceTableProvider;

let ctx = SessionContext::new();

ctx.register_table("dataset",
    Arc::new(LanceTableProvider::new(
    Arc::new(dataset.clone()),
    /* with_row_id */ false,
    /* with_row_addr */ false,
    )))?;

let df = ctx.sql("SELECT * FROM dataset LIMIT 10").await?;
let result = df.collect().await?;
```

### 连接两个表

```rust
use datafusion::prelude::SessionContext;
use lance::datafusion::LanceTableProvider;

let ctx = SessionContext::new();

ctx.register_table("orders",
    Arc::new(LanceTableProvider::new(
    Arc::new(orders_dataset.clone()),
    /* with_row_id */ false,
    /* with_row_addr */ false,
    )))?;

ctx.register_table("customers",
    Arc::new(LanceTableProvider::new(
    Arc::new(customers_dataset.clone()),
    /* with_row_id */ false,
    /* with_row_addr */ false,
    )))?;

let df = ctx.sql("
    SELECT o.order_id, o.amount, c.customer_name 
    FROM orders o 
    JOIN customers c ON o.customer_id = c.customer_id
    LIMIT 10
").await?;

let result = df.collect().await?;
```

### 注册 UDF
Lance 提供了一些内置的 UDF（用户自定义函数），用户可以手动注册并在查询中使用。
以下示例演示如何注册和使用 ```contains_tokens```。

```rust
use datafusion::prelude::SessionContext;
use lance::datafusion::LanceTableProvider;
use lance_datafusion::udf::register_functions;

let ctx = SessionContext::new();

// Register built-in UDFs
register_functions(&ctx);

ctx.register_table("dataset",
    Arc::new(LanceTableProvider::new(
    Arc::new(dataset.clone()),
    /* with_row_id */ false,
    /* with_row_addr */ false,
    )))?;

let df = ctx.sql("SELECT * FROM dataset WHERE contains_tokens(text, 'cat')").await?;
let result = df.collect().await?;
```

### JSON 函数 { #json-functions }

Lance 通过一组内置 UDF 提供了全面的 JSON 支持，当你使用 `register_functions()` 时会自动注册。这些函数使你能够高效地查询和过滤 JSON 数据。

完整的 JSON 函数指南包括：
- `json_extract` - 使用 JSONPath 提取值
- `json_get`、`json_get_string`、`json_get_int`、`json_get_float`、`json_get_bool` - 类型安全的值提取
- `json_exists` - 检查路径是否存在
- `json_array_contains`、`json_array_length` - 数组操作

详细文档和示例请参阅 [JSON 支持指南](../guide/json.md)。

**示例：在 SQL 中查询 JSON**
```rust
// After registering functions as shown above
let df = ctx.sql("
    SELECT * FROM dataset 
    WHERE json_get_string(metadata, 'category') = 'electronics'
    AND json_array_contains(metadata, '$.tags', 'featured')
").await?;
```

## Python

在 Python 中，该集成通过 [Datafusion FFI](https://docs.rs/datafusion-ffi/latest/datafusion_ffi/) 实现。
`pylance` 中包含了一个 FFI 表提供器 `FFILanceTableProvider`。
例如，如果你想查询 `my_lance_dataset`：

```python
from datafusion import SessionContext # pip install datafusion
from lance import FFILanceTableProvider

ctx = SessionContext()

table1 = FFILanceTableProvider(
    my_lance_dataset, with_row_id=True, with_row_addr=True
)
ctx.register_table("table1", table1)
ctx.table("table1")
ctx.sql("SELECT * FROM table1 LIMIT 10")
```
