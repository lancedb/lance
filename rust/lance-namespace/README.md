# lance-namespace

Lance Namespace Core APIs for managing namespaces and tables.

## Overview

This crate provides the core APIs for Lance namespaces, including:

- `LanceNamespace` trait - The main interface for namespace operations
- `RestNamespace` - REST API implementation of the namespace trait
- Schema conversion utilities for Arrow schemas
- Models and APIs for namespace operations

## Features

The namespace API supports:

- Creating and managing namespaces
- Creating and managing tables within namespaces
- Listing namespaces and tables
- Schema management
- Multiple backend implementations (REST, directory-based, etc.)

## Usage

```rust
use lance_namespace::{LanceNamespace, RestNamespace};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a REST namespace client
    let mut properties = HashMap::new();
    properties.insert("uri".to_string(), "http://localhost:8080".to_string());

    let namespace = RestNamespace::new(properties);

    // List tables in the namespace
    let tables = namespace.list_tables(Default::default()).await?;

    Ok(())
}
```

## Documentation

For more information about Lance and its namespace system, see the [Lance Namespace documentation](https://lancedb.github.io/lance/format/namespace).
