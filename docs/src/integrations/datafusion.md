# Apache DataFusion Integration

Lance datasets can be queried with [Apache Datafusion](https://datafusion.apache.org/), 
an extensible query engine written in Rust that uses Apache Arrow as its in-memory format. 
This means you can write complex SQL queries to analyze your data in Lance.

The integration allows users to pass down column selections and basic filters to Lance, 
reducing the amount of scanned data when executing your query. 
Additionally, the integration allows streaming data from Lance datasets,
which allows users to do aggregation larger-than-memory.

## Rust

Lance includes a DataFusion table provider `lance::datafusion::LanceTableProvider`.
Users can register a Lance dataset as a table in DataFusion and run SQL with it:

### Simple SQL

```rust
use datafusion::prelude::SessionContext;
use crate::datafusion::LanceTableProvider;

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

### Join 2 Tables

```rust
use datafusion::prelude::SessionContext;
use crate::datafusion::LanceTableProvider;

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

## Python

In Python, this integration is done via [Datafusion FFI](https://docs.rs/datafusion-ffi/latest/datafusion_ffi/).
An FFI table provider `FFILanceTableProvider` is included in `pylance`.
For example, if I want to query `my_lance_dataset`:

```python
from datafusion import SessionContext # pip install datafusion
from lance import FFILanceTableProvider

ctx = SessionContext()

table1 = FFILanceTableProvider(
    my_lance_dataset, with_row_id=True, with_row_addr=True
)
ctx.register_table_provider("table1", table1)
ctx.table("table1")
ctx.sql("SELECT * FROM table1 LIMIT 10")
```