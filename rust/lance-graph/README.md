# Lance Graph Query Engine

A graph query engine for Lance datasets with Cypher syntax support. This crate enables querying Lance's columnar datasets using familiar graph query patterns, interpreting tabular data as property graphs.

## Features

- Cypher query parsing and AST construction
- Graph configuration for mapping Lance tables to nodes and relationships
- Semantic validation with typed `GraphError` diagnostics
- Translation to DataFusion SQL with a direct-execution fast path for simple patterns
- Async query execution that returns Arrow `RecordBatch` results
- JSON-serializable parameter binding for reusable query templates

## Quick Start

```rust
use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance_graph::{CypherQuery, GraphConfig};

let config = GraphConfig::builder()
    .with_node_label("Person", "person_id")
    .with_relationship("KNOWS", "src_person_id", "dst_person_id")
    .build()?;

let schema = Arc::new(Schema::new(vec![
    Field::new("person_id", DataType::Int32, false),
    Field::new("name", DataType::Utf8, false),
    Field::new("age", DataType::Int32, false),
]));
let batch = RecordBatch::try_new(
    schema,
    vec![
        Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
        Arc::new(StringArray::from(vec!["Alice", "Bob"])) as ArrayRef,
        Arc::new(Int32Array::from(vec![29, 35])) as ArrayRef,
    ],
)?;

let mut tables = HashMap::new();
tables.insert("Person".to_string(), batch);

let query = CypherQuery::new("MATCH (p:Person) WHERE p.age > $min RETURN p.name")?
    .with_config(config)
    .with_parameter("min", 30);

let runtime = tokio::runtime::Runtime::new()?;
let result = runtime.block_on(query.execute(tables))?;
```

The query expects a `HashMap<String, RecordBatch>` keyed by the labels and relationship types referenced in the Cypher text. Each record batch should expose the columns configured through `GraphConfig` (ID fields, property fields, etc.). Relationship mappings also expect a batch keyed by the relationship type (for example `KNOWS`) that contains the configured source/target ID columns and any optional property columns.

## Configuring Graph Mappings

Graph mappings are declared with `GraphConfig::builder()`:

```rust
use lance_graph::{GraphConfig, NodeMapping, RelationshipMapping};

let config = GraphConfig::builder()
    .with_node_label("Person", "person_id")
    .with_relationship("KNOWS", "src_person_id", "dst_person_id")
    .build()?;
```

For finer control, build `NodeMapping` and `RelationshipMapping` instances explicitly:

```rust
let person = NodeMapping::new("Person", "person_id")
    .with_properties(vec!["name".into(), "age".into()])
    .with_filter("kind = 'person'");

let knows = RelationshipMapping::new("KNOWS", "src_person_id", "dst_person_id")
    .with_properties(vec!["since".into()]);

let config = GraphConfig::builder()
    .with_node_mapping(person)
    .with_relationship_mapping(knows)
    .build()?;
```

## Executing Cypher Queries

- `CypherQuery::new` parses Cypher text into the internal AST.
- `with_config` attaches the graph configuration used for validation and execution.
- `with_parameter` / `with_parameters` bind JSON-serializable values that can be referenced as `$param` in the Cypher text.
- `execute` is asynchronous and returns an Arrow `RecordBatch`.

Queries with a single `MATCH` clause containing a path pattern are planned as joins using the provided mappings. Other queries fall back to a single-table projection/filter pipeline on the first registered dataset.

A builder (`CypherQueryBuilder`) is also available for constructing queries programmatically without parsing text.

## Supported Cypher Surface

- Node patterns `(:Label)` with optional variables.
- Relationship patterns with fixed direction and type, including multi-hop paths.
- Property comparisons against literal values with `AND`/`OR`/`NOT`/`EXISTS`.
- RETURN lists of property accesses, optional `DISTINCT`, and `LIMIT`.
- Positional and named parameters (e.g. `$min_age`).

Features such as ORDER BY, aggregations, optional matches, and subqueries are parsed but not executed yet.

## Crate Layout

- `ast` – Cypher AST definitions.
- `parser` – Nom-based Cypher parser.
- `semantic` – Lightweight semantic checks on the AST.
- `logical_plan` – Builders for DataFusion logical plans.
- `datafusion_planner` and `query_processor` – Execution planning utilities.
- `config` – Graph configuration types and builders.
- `query` – High level `CypherQuery` API and runtime.
- `error` – `GraphError` and result helpers.
- `source_catalog` – Helpers for looking up table metadata.

## Error Handling

Most APIs return `Result<T, GraphError>`. Errors include parsing failures, missing mappings, and execution issues surfaced from DataFusion.

## Testing

```bash
cargo test -p lance-graph
```

## Python Bindings

Python bindings for this crate live under `python/src/graph.rs` and expose the same configuration and query APIs via PyO3.

## License

Apache-2.0. See the top-level LICENSE file for details.
