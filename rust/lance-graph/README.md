# Lance Graph Query Engine

A graph query engine for Lance datasets with Cypher syntax support. This crate enables querying Lance's columnar datasets using familiar graph query patterns, interpreting tabular data as property graphs.

## Features

- **Cypher Query Support**: Parse and execute Cypher queries against Lance datasets
- **Property Graph Model**: Interpret columnar data as nodes and relationships  
- **SQL Translation**: Convert Cypher queries to optimized DataFusion SQL
- **Flexible Configuration**: Map dataset columns to graph elements
- **Type Safety**: Full Rust type safety with comprehensive error handling
- **Performance**: Leverage Lance's columnar storage and DataFusion's query optimization

## Quick Start

### Basic Usage

```rust
use lance_graph::{CypherQuery, GraphConfig};

// Configure how your dataset maps to a property graph
let config = GraphConfig::builder()
    .with_node_label("Person", "person_id")
    .with_relationship("KNOWS", "src_person_id", "dst_person_id")
    .build()?;

// Parse and configure a Cypher query
let query = CypherQuery::new("MATCH (p:Person) WHERE p.age > 30 RETURN p.name")?
    .with_config(config);

// Convert to SQL for execution
let sql = query.to_sql()?;
println!("Generated SQL: {}", sql);
```

### Query Examples

#### Simple Node Queries
```cypher
-- Find all persons
MATCH (p:Person) RETURN p.name, p.age

-- Filter by properties
MATCH (p:Person) WHERE p.age > 30 RETURN p.name

-- Node with property constraints
MATCH (p:Person {city: "New York"}) RETURN p.name
```

#### Relationship Queries
```cypher
-- Find relationships
MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a.name, b.name

-- Multi-hop paths
MATCH (a:Person)-[:KNOWS]->(b:Person)-[:WORKS_WITH]->(c:Person)
RETURN a.name, b.name, c.name

-- Relationship properties
MATCH (a:Person)-[r:KNOWS {since: 2020}]->(b:Person) RETURN a.name, b.name
```

#### Advanced Queries
```cypher
-- Parameterized queries
MATCH (p:Person) WHERE p.age > $minAge RETURN p.name

-- Aggregations
MATCH (p:Person) RETURN p.city, COUNT(p) AS population

-- Sorting and limiting
MATCH (p:Person) RETURN p.name ORDER BY p.age DESC LIMIT 10
```

## Architecture

### Components

1. **AST (`ast.rs`)**: Abstract syntax tree representation of Cypher queries
2. **Parser (`parser.rs`)**: Nom-based parser for Cypher syntax
3. **Config (`config.rs`)**: Graph configuration for dataset mapping
4. **Planner (`planner.rs`)**: Query planner for Cypher-to-SQL translation
5. **Query (`query.rs`)**: High-level query interface and validation
6. **Error (`error.rs`)**: Comprehensive error handling

### Query Flow

```
Cypher Query → Parser → AST → Planner → SQL → DataFusion → Results
                ↓
          Configuration
```

## Configuration

### Node Mappings

Configure how dataset rows become graph nodes:

```rust
let config = GraphConfig::builder()
    .with_node_label("Person", "person_id")        // Label and ID field
    .with_node_label("Company", "company_id")
    .build()?;
```

### Relationship Mappings

Configure how relationships are represented:

```rust
let config = GraphConfig::builder()
    .with_relationship("WORKS_FOR", "person_id", "company_id")
    .with_relationship("KNOWS", "person_id", "friend_id")
    .build()?;
```

### Advanced Configuration

```rust
let node_mapping = NodeMapping::new("person_id")
    .with_filter("type = 'person'")                // Filter conditions
    .with_property_mapping("name", "full_name");   // Property mappings

let rel_mapping = RelationshipMapping::new("src_id", "dst_id")
    .with_filter("relationship_type = 'knows'")
    .with_property_mapping("strength", "weight");
```

## Data Model

### Unified Table Approach

Store both nodes and relationships in a single table:

```sql
CREATE TABLE graph_data (
    id BIGINT,
    type VARCHAR,           -- 'person', 'company', 'relationship'
    name VARCHAR,           -- Node properties
    age INT,
    source_id BIGINT,       -- Relationship properties  
    target_id BIGINT,
    relationship_type VARCHAR
);
```

### Separate Tables Approach

Use separate tables for nodes and relationships:

```sql
-- Nodes table
CREATE TABLE persons (
    person_id BIGINT,
    name VARCHAR,
    age INT,
    city VARCHAR
);

-- Relationships table  
CREATE TABLE relationships (
    id BIGINT,
    source_person_id BIGINT,
    target_person_id BIGINT,
    relationship_type VARCHAR,
    since_year INT
);
```

## Supported Cypher Features

### Currently Supported

- ✅ Node patterns: `(n:Label)`, `(n:Label {prop: value})`
- ✅ Relationship patterns: `-[r:TYPE]->`, `<-[r:TYPE]-`, `-[r:TYPE]-`
- ✅ WHERE clauses with comparisons (`=`, `<>`, `<`, `>`, `<=`, `>=`)
- ✅ RETURN clauses with properties and aliases
- ✅ ORDER BY and LIMIT clauses
- ✅ Query parameters (`$param`)
- ✅ Variable references in RETURN

### Planned Features

- 🔄 Boolean operators in WHERE (`AND`, `OR`, `NOT`)
- 🔄 Pattern expressions (`EXISTS`, `OPTIONAL MATCH`)
- 🔄 Aggregation functions (`COUNT`, `SUM`, `AVG`, etc.)
- 🔄 Path expressions and variable-length paths
- 🔄 UNION and subqueries
- 🔄 CREATE, UPDATE, DELETE operations
- 🔄 Two-hop and multi-hop expansions in DataFusion execution (e.g., `(p)-[:R1]->(m)-[:R2]->(q)`), including small bounded variable-length unrolling (e.g., `*1..2`).

## Error Handling

The crate provides comprehensive error handling:

```rust
use lance_graph::{GraphError, Result};

match CypherQuery::new("INVALID QUERY") {
    Ok(query) => println!("Query parsed successfully"),
    Err(GraphError::ParseError { message, position, .. }) => {
        println!("Parse error at position {}: {}", position, message);
    }
    Err(e) => println!("Other error: {}", e),
}
```

## Performance Considerations

### Query Optimization

- **Predicate Pushdown**: WHERE clauses are pushed down to the dataset scan
- **Join Optimization**: DataFusion optimizes relationship traversals
- **Column Pruning**: Only required columns are read from storage
- **Index Usage**: Leverage Lance's scalar and vector indices

### Best Practices

1. **Use Selective Filters**: Add WHERE clauses to reduce data scanned
2. **Limit Results**: Use LIMIT for large result sets
3. **Index Key Fields**: Create indices on ID fields used in joins
4. **Batch Queries**: Process multiple queries together when possible

## Testing

Run the test suite:

```bash
cargo test -p lance-graph
```

Run specific tests:

```bash
cargo test -p lance-graph parser::tests::test_parse_simple_node_query
```

## Examples

See the `tests/` directory for comprehensive examples:

- `test_parser.rs`: Cypher parsing examples
- `test_planner.rs`: SQL translation examples  
- `test_query.rs`: End-to-end query examples
- `test_config.rs`: Configuration examples

## Integration

### With Lance Datasets

```rust
// This would be the integration point with Lance datasets
// (Implementation depends on Lance's dataset APIs)

let dataset = lance::Dataset::open("path/to/dataset")?;
let config = GraphConfig::builder()...;
let query = CypherQuery::new("MATCH (n) RETURN n")?;

// Execute via DataFusion
let ctx = SessionContext::new();
let sql = query.to_sql()?;
let df = ctx.sql(&sql).await?;
let results = df.collect().await?;
```

### With Python

Python bindings would provide a clean interface:

```python
import lance
from lance.graph import GraphConfig, CypherQuery

# Configure graph interpretation
config = GraphConfig.builder() \
    .with_node_label("Person", "person_id") \
    .build()

# Execute query
query = CypherQuery("MATCH (p:Person) RETURN p.name") \
    .with_config(config)

dataset = lance.dataset("path/to/data")
result = query.execute(dataset)
```

## Contributing

1. Follow Rust conventions and the existing code style
2. Add tests for new features
3. Update documentation for API changes
4. Run `cargo fmt` and `cargo clippy` before submitting

## License

Apache-2.0 License - see LICENSE file for details.
