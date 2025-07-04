# Apache DataFusion Integration

Lance includes a DataFusion table provider `lance::datafusion::LanceTableProvider`.
Users can register a Lance dataset as a table in DataFusion and run SQL with it:

## Simple SQL

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

## Join 2 Tables

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