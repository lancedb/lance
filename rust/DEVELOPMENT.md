# Developing Lance

Lance in one diagram

![img.png](img.png)

The write path is analogous but there's no need for io planning. Instead,
we need to version data correctly and create the right manifest data to
be written

## Data format spec

TODO: release data format spec and add link here

## Rust codebase guide

| Directory | Description                                  |
|-----------|----------------------------------------------|
| dataset   | dataset-level scanner and writer abstraction |
| encodings | base level file serde                        |
| format    | structs to match the format spec             |
| index     | vector indices right now                     |
| io        | readers, writers, and IO planning            |
| utils     | distance metrics and testing utilities       |
| arrow     | extensions for arrow-rs                      |
| bin       | lq utility cli                               |

## Rust Roadmap

### Predicate support

Support for predicate pushdown needs to be ported over from C++

### NA support

For fixed stride data (plain encoder) we currently do not handle NA's.

### Versioning

Appends, overwrites, and adding columns need to be ported from C++.
Deletes and updates need to be implemented.

### Type support

See `Arrow types` section below for what's missing. 

### Non-python duckdb reader

Integrate DuckDB-rs and provide TableFunction for lance with replacement scan

## Python

1. Need to expose things like row count from Rust
2. Need to support `filter=` for predicate pushdown (and how to pass filters to Rust)
3. Convenience functions for checkout and tags to better support experimentation

## Development

### Build

1. Rust

Under /rust directory, run `cargo build` or `cargo build --release` for optimized package.

2. Python

Under /rust/pylance directory, run `maturin develop` or `maturin develop --release`.

### Testing

1. Rust

Under /rust, run `cargo test`

2. Python

Under /rust/pylance, run `pytest python/tests`

### Encodings

|                   | Read | Write | Null |
|-------------------|------|-------|------|
| Plain             | Yes  | Yes   | No   |
| Var-length Binary | Yes  | Yes   | Yes  |
| Dictionary        | Yes  | Yes   | No   |
| RLE               | No   | No    | No   |


### Arrow types

|                         | Read | Write | Null |
|-------------------------|------|-------|------|
| NA                      | NO   | NO    | NO   |
| Bool                    | Yes  | Yes   | No   |
| UINT8                   | Yes  | Yes   | No   |
| INT8                    | Yes  | Yes   | No   |
| UINT16                  | Yes  | Yes   | No   |
| INT16                   | Yes  | Yes   | No   |
| UINT32                  | Yes  | Yes   | No   |
| INT32                   | Yes  | Yes   | No   |
| UINT64                  | Yes  | Yes   | No   |
| INT64                   | Yes  | Yes   | No   |
| HALF_FLOAT              | Yes  | Yes   | No   |
| FLOAT                   | Yes  | Yes   | No   |
| DOUBLE                  | Yes  | Yes   | No   |
| DECIMAL                 | No   | No    | No   |
| DECIMAL128              | No   | No    | No   |
| DECIMAL256              | No   | No    | No   |
| STRING                  | Yes  | Yes   | Yes  |
| LARGE_STRING            | Yes  | Yes   | Yes  |
| BINARY                  | Yes  | Yes   | Yes  |
| LARGE_BINARY            | Yes  | Yes   | Yes  |
| FIXED_SIZE_LIST         | Yes  | Yes   | No   |
| FIXED_SIZE_BINARY       | Yes  | Yes   | No   |
| DATE32                  | Yes  | Yes   | No   |
| DATE64                  | Yes  | Yes   | No   |
| TIMESTAMP               | Yes  | Yes   | No   |
| TIME32                  | Yes  | Yes   | No   |
| TIME64                  | Yes  | Yes   | No   |
| INTERVAL_MONTHS         | No   | No    | No   |
| INTERVAL_DAY_TIME       | No   | No    | No   |
| INTERVAL_MONTH_DAY_NANO | No   | No    | No   |
| DURATION                | No   | No    | No   |
| LIST                    | Yes  | Yes   | No   |
| LARGE_LIST              | Yes  | Yes   | No   |
| STRUCT                  | Yes  | Yes   | No   |
| DICTIONARY              | Yes  | Yes   | No   |
| MAP                     | No   | No    | No   |
| DENSE_UNION             | No   | No    | No   |
| SPARSE_UNION            | No   | No    | No   |


## Extension types

We want to add extension types for ML:
- Images, Videos, PDF, Document
- Boxes, Labels, Polygon, etc

Currently arrow-rs doesn't have an extension mechanism nor does it talk to the
C++ extension registry.