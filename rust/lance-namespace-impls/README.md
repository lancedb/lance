# lance-namespace-impls

Lance Namespace implementation backends.

## Overview

This crate provides concrete implementations of the Lance namespace trait:

- Unified connection interface for all implementations
- **REST Namespace** - REST API client for remote Lance namespace servers (feature: `rest`)
- **Directory Namespace** - File system-based namespace that stores tables as Lance datasets (feature: `dir`)

## Features

### REST Namespace (feature: `rest`)

The REST namespace implementation provides a client for connecting to remote Lance namespace servers via REST API.

### Directory Namespace (feature: `dir`)

The directory namespace implementation stores tables as Lance datasets in a directory structure on local or cloud storage.

Supported storage backends:
- Local filesystem
- AWS S3
- Google Cloud Storage (GCS)
- Azure Blob Storage

## Usage

### Connecting to a Namespace

```rust
use lance_namespace_impls::connect;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connect to REST implementation
    let mut props = HashMap::new();
    props.insert("uri".to_string(), "http://localhost:8080".to_string());
    let namespace = connect("rest", props).connect().await?;

    Ok(())
}
```

### Using Directory Namespace

```rust
use lance_namespace_impls::connect;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connect to local directory
    let mut props = HashMap::new();
    props.insert("root".to_string(), "/path/to/data".to_string());
    let namespace = connect("dir", props).connect().await?;

    // Connect to S3
    let mut props = HashMap::new();
    props.insert("root".to_string(), "s3://my-bucket/path".to_string());
    props.insert("storage.region".to_string(), "us-west-2".to_string());
    let namespace = connect("dir", props).connect().await?;

    Ok(())
}
```

### Using a Shared Session

For the directory namespace, you can optionally provide a shared `Session` to reuse object store connections and caches across multiple namespace instances:

```rust
use lance_namespace_impls::connect;
use lance::session::Session;
use std::collections::HashMap;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a shared session
    let session = Arc::new(Session::default());

    // Connect multiple namespaces with the same session
    let mut props1 = HashMap::new();
    props1.insert("root".to_string(), "/path/to/data1".to_string());
    let namespace1 = connect("dir", props1)
        .with_session(session.clone())
        .connect()
        .await?;

    let mut props2 = HashMap::new();
    props2.insert("root".to_string(), "/path/to/data2".to_string());
    let namespace2 = connect("dir", props2)
        .with_session(session.clone())
        .connect()
        .await?;

    Ok(())
}
```

## Configuration

### Directory Namespace Properties

- `root` - Root path for the namespace (local path or cloud storage URI)
- `storage.*` - Storage-specific options (e.g., `storage.region`, `storage.access_key_id`)

## Documentation

For more information about Lance and its namespace system, see the [Lance Namespace documentation](https://lancedb.github.io/lance/format/namespace).
