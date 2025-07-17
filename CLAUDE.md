# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Lance is a modern columnar data format optimized for ML workflows and datasets. It provides:
- 100x faster random access than Parquet
- Native vector search capabilities (IVF, HNSW, PQ)
- Zero-copy automatic versioning
- Deep integration with Apache Arrow, Pandas, Polars, DuckDB, Ray
- Cloud-native with S3, Azure, GCS support

## Project Vision

The de facto standard columnar data format for machine learning and large language models.

## Project Requirements

- Always use English in code, examples, and comments.
- Features should be implemented concisely, maintainably, and efficiently.
- Code is not just for execution, but also for readability.
- Only add meaingful comments and tests.

## Architecture

The project is organized as a Rust workspace with Python and Java bindings:

- `rust/lance/` - Main Lance library implementing the columnar format
- `rust/lance-arrow/` - Apache Arrow integration layer
- `rust/lance-core/` - Core types, traits, and utilities
- `rust/lance-encoding/` - Data encoding and compression algorithms
- `rust/lance-file/` - File format reading/writing
- `rust/lance-index/` - Vector and scalar indexing implementations
- `rust/lance-io/` - I/O operations and object store integration
- `rust/lance-linalg/` - Linear algebra operations for vector search
- `rust/lance-table/` - Table format and operations
- `rust/lance-datafusion/` - DataFusion query engine integration
- `python/` - Python bindings using PyO3/maturin
- `java/` - Java bindings using JNI

## Common Development Commands

### Rust Development

```bash
# Building
cargo build                          # Debug build
cargo build --release               # Release build
cargo build --features <features>   # Build with specific features

# Testing
cargo test                          # Run all tests
cargo test <test_name>             # Run specific test
cargo test --package <package>     # Test specific crate
cargo test -- --nocapture          # Show println! output

# Formatting & Linting
cargo fmt                          # Format code
cargo fmt -- --check              # Check formatting
cargo clippy -- -D warnings       # Run linter

# Dependency Checking
cargo deny check                   # Audit dependencies
```

### Python Development

```bash
# Building Extension
maturin develop                    # Build and install in dev mode
maturin develop --release          # Build optimized version
maturin develop --profile release-with-debug  # Release with debug symbols

# Testing
pip install -e python[tests]       # Install test dependencies
pytest python/tests               # Run all tests
pytest -vvv -s python/tests      # Verbose output
pytest --doctest-modules python/lance  # Run doctests
pytest -k "test_pattern"         # Run tests matching pattern
pytest --run-integration python/tests/test_s3_ddb.py  # Integration tests

# Formatting & Linting
ruff format python               # Format Python code
ruff check python                # Run linter
ruff check --fix python          # Auto-fix issues
pyright                          # Type checking

# Benchmarking
pip install -e python[benchmarks]
pytest python/benchmarks         # Run quick benchmarks
pytest python/benchmarks -m "not slow"  # Skip slow benchmarks
```

### Building Wheels

```bash
# Linux x86_64 (requires Docker)
docker run -v $(pwd):/io ghcr.io/pyo3/maturin:latest build --release -m python/Cargo.toml --features <features>

# macOS
maturin build --release -m python/Cargo.toml
```

### Integration Testing

```bash
# Start required services
cd test_data && docker compose up -d

# Run S3/DynamoDB tests
AWS_DEFAULT_REGION=us-east-1 pytest --run-integration python/tests/test_s3_ddb.py

# Performance profiling
maturin develop --release -m python/Cargo.toml -E benchmarks
python python/benchmarks/test_knn.py --iterations 100
```

## Key Technical Details

1. **Async-first Architecture**: Heavy use of tokio and async/await throughout Rust codebase
2. **Arrow-native**: All data operations work directly with Apache Arrow arrays
3. **Version Control**: Every write creates a new version with manifest tracking
4. **Indexing**: Supports both vector indices (for similarity search) and scalar indices (BTree, inverted)
5. **Encoding**: Custom encodings optimized for ML data patterns
6. **Object Store**: Unified interface for local, S3, Azure, GCS storage

## Development Notes

- All public APIs should have comprehensive documentation with examples
- Performance-critical code uses SIMD optimizations where available
- Always rebuild Python extension after Rust changes using `maturin develop`
- Integration tests require Docker for local S3/DynamoDB emulation
- Use feature flags to control dependencies (e.g., `datafusion` for SQL support)
