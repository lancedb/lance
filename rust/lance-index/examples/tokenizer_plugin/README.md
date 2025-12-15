# Tokenizer Plugin Example

This directory contains an example implementation of a Lance tokenizer plugin.

## Overview

The `simple_tokenizer.rs` file demonstrates how to implement a tokenizer plugin
that can be dynamically loaded by Lance at runtime. This example implements a
simple whitespace tokenizer with optional lowercase conversion.

## Building a Plugin

To build a tokenizer plugin as a shared library:

1. Create a new Rust library crate:

```bash
cargo new --lib my_tokenizer
cd my_tokenizer
```

2. Update `Cargo.toml` to build a C-compatible dynamic library:

```toml
[package]
name = "my_tokenizer"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]
```

3. Copy the `simple_tokenizer.rs` content to `src/lib.rs` and modify as needed.

4. Build the library:

```bash
cargo build --release
```

The resulting library will be in `target/release/`:
- Linux: `libmy_tokenizer.so`
- macOS: `libmy_tokenizer.dylib`
- Windows: `my_tokenizer.dll`

## Using a Plugin

```rust
use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;

let params = InvertedIndexParams::default()
    .plugin(
        "/path/to/libmy_tokenizer.so".to_string(),
        Some(r#"{"lowercase": true}"#.to_string()),
    );

// Use params when creating an inverted index
dataset.create_index(
    &["content"],
    IndexType::Inverted,
    Some(params.into()),
).await?;
```

## Plugin Interface

Your plugin must implement all functions defined in `include/lance_tokenizer_plugin.h`.
The key requirements are:

1. **Entry Point**: Export a function named `lance_tokenizer_get_plugin` that returns
   a pointer to a `LanceTokenizerPlugin` struct.

2. **API Version**: The `api_version` function must return `1` (current API version).

3. **Lifecycle Functions**:
   - `create_factory`: Initialize shared resources (dictionaries, models)
   - `create_tokenizer`: Create a tokenizer instance from a factory
   - `create_stream`: Create a token stream for a piece of text
   - `next_token`: Iterate through tokens
   - `destroy_*`: Clean up resources

4. **Thread Safety**: Each factory/tokenizer/stream should be independent.
   The plugin library itself should be thread-safe.

## Error Handling

- Return `NULL` from `create_*` functions on error
- Set an error message using the factory's error storage
- Users can retrieve the error via `get_error`

## Testing

The example includes unit tests that can be run with:

```bash
# From this directory
rustc --test simple_tokenizer.rs -o test_tokenizer && ./test_tokenizer
```
