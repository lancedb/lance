# Lance Rust Workspace

Where core rust code lance lives.

## Architecture

The Lance project is organized as a Rust workspace with multiple crates that work together to provide the columnar data format implementation.

### Dependency Graph

<!-- BEGIN_CARGO_DIAGRAM -->
<!-- This section is auto-generated. Run `python ci/update_rust_readme.py` to update. -->
```mermaid
graph TD
    fsst["fsst"]
    lance["lance"]
    lance-arrow["arrow"]
    lance-bitpacking["bitpacking"]
    lance-core["core"]
    lance-datafusion["datafusion"]
    lance-datagen["datagen"]
    lance-encoding["encoding"]
    lance-examples["examples"]
    lance-file["file"]
    lance-index["index"]
    lance-io["io"]
    lance-linalg["linalg"]
    lance-namespace["namespace"]
    lance-namespace-impls["namespace-impls"]
    lance-table["table"]
    lance-test-macros["test-macros"]
    lance-testing["testing"]
    lance-tools["tools"]

    fsst --> lance-datagen
    lance --> lance-arrow
    lance --> lance-core
    lance --> lance-datafusion
    lance --> lance-datagen
    lance --> lance-encoding
    lance --> lance-file
    lance --> lance-index
    lance --> lance-io
    lance --> lance-linalg
    lance --> lance-namespace
    lance --> lance-table
    lance --> lance-test-macros
    lance --> lance-testing
    lance-core --> lance-arrow
    lance-core --> lance-testing
    lance-datafusion --> lance-arrow
    lance-datafusion --> lance-core
    lance-datafusion --> lance-datagen
    lance-encoding --> fsst
    lance-encoding --> lance-arrow
    lance-encoding --> lance-bitpacking
    lance-encoding --> lance-core
    lance-encoding --> lance-datagen
    lance-encoding --> lance-testing
    lance-examples --> lance
    lance-examples --> lance-core
    lance-examples --> lance-datagen
    lance-examples --> lance-index
    lance-examples --> lance-linalg
    lance-file --> lance-arrow
    lance-file --> lance-core
    lance-file --> lance-datagen
    lance-file --> lance-encoding
    lance-file --> lance-io
    lance-file --> lance-testing
    lance-index --> lance-arrow
    lance-index --> lance-core
    lance-index --> lance-datafusion
    lance-index --> lance-datagen
    lance-index --> lance-encoding
    lance-index --> lance-file
    lance-index --> lance-io
    lance-index --> lance-linalg
    lance-index --> lance-table
    lance-index --> lance-testing
    lance-io --> lance-arrow
    lance-io --> lance-core
    lance-io --> lance-namespace
    lance-linalg --> lance-arrow
    lance-linalg --> lance-core
    lance-linalg --> lance-testing
    lance-namespace --> lance-core
    lance-namespace-impls --> lance
    lance-namespace-impls --> lance-core
    lance-namespace-impls --> lance-namespace
    lance-table --> lance-arrow
    lance-table --> lance-core
    lance-table --> lance-datagen
    lance-table --> lance-file
    lance-table --> lance-io
    lance-testing --> lance-arrow
    lance-tools --> lance-core
    lance-tools --> lance-encoding
    lance-tools --> lance-file
    lance-tools --> lance-io
```
<!-- END_CARGO_DIAGRAM -->

<!-- BEGIN_CARGO_CRATE_LIST -->
<!-- This section is auto-generated. Run `python ci/update_rust_readme.py` to update. -->
## Workspace Crates

- **fsst** (`rust/compression/fsst/`) - FSST string compression for Lance
- **lance** (`rust/lance/`) - A columnar data format that is 100x faster than Parquet for random access.
- **lance-arrow** (`rust/lance-arrow/`) - Arrow Extension for Lance
- **lance-bitpacking** (`rust/compression/bitpacking/`) - Vendored copy of https://github.com/spiraldb/fastlanes for use in Lance
- **lance-core** (`rust/lance-core/`) - Lance Columnar Format -- Core Library
- **lance-datafusion** (`rust/lance-datafusion/`) - Internal utilities used by other lance modules to simplify working with datafusion
- **lance-datagen** (`rust/lance-datagen/`) - A columnar data format that is 100x faster than Parquet for random access.
- **lance-encoding** (`rust/lance-encoding/`) - Encoders and decoders for the Lance file format
- **lance-examples** (`rust/examples/`) - Lance examples in Rust
- **lance-file** (`rust/lance-file/`) - Utilities for the Lance file format
- **lance-index** (`rust/lance-index/`) - Lance indices implementation
- **lance-io** (`rust/lance-io/`) - I/O utilities for Lance
- **lance-linalg** (`rust/lance-linalg/`) - A columnar data format that is 100x faster than Parquet for random access.
- **lance-namespace** (`rust/lance-namespace/`) - Lance Namespace Core APIs
- **lance-namespace-impls** (`rust/lance-namespace-impls/`) - Lance Namespace Implementations
- **lance-table** (`rust/lance-table/`) - Utilities for the Lance table format
- **lance-test-macros** (`rust/lance-test-macros/`) - A columnar data format that is 100x faster than Parquet for random access.
- **lance-testing** (`rust/lance-testing/`) - A columnar data format that is 100x faster than Parquet for random access.
- **lance-tools** (`rust/lance-tools/`) - Tools for interacting with Lance files and tables
<!-- END_CARGO_CRATE_LIST -->

## Updating This README

To update the dependency graph and crate list, run:

```bash
python scripts/update_rust_readme.py
```
