# Lance Tokenizer Plugin — Reference Implementation

This crate is the canonical sample plugin for the Lance tokenizer plugin C ABI
defined in `include/lance_tokenizer_plugin.h`. It is built and loaded by the
`tokenizer_plugin_integration` integration tests in the parent crate, so it
exercises the same code paths a real user-authored plugin would.

If you want to write your own tokenizer plugin, start by copying the structure
of `src/lib.rs` into a new Rust crate.

## Crate setup

```toml
[package]
name = "my_tokenizer"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]
```

Build the shared library:

```bash
cargo build --release
```

The output lives under `target/release/`:

- Linux: `libmy_tokenizer.so`
- macOS: `libmy_tokenizer.dylib`
- Windows: `my_tokenizer.dll`

## Implementing the plugin

Replace the `Factory` / `Tokenizer` / `TokenStream` types in `src/lib.rs` with
your own logic. The C ABI requires the following:

1. **Entry point** — export `lance_tokenizer_get_plugin` returning a
   `*const LanceTokenizerPlugin` vtable.
2. **API version** — `api_version` must return `1`.
3. **Lifecycle callbacks** — every field of `LanceTokenizerPlugin` must be
   non-NULL: `create_factory` / `create_tokenizer` / `create_stream` /
   `next_token` and the matching `destroy_*` callbacks, plus `name` / `version`.
4. **Token text encoding** — token text passed back to the host through
   `LanceStringRef.text` must be valid UTF-8. Lance copies the bytes immediately
   after each `next_token` call, so the plugin may reuse internal scratch
   buffers between tokens.

`Factory`, `Tokenizer`, and `TokenStream` are not required to be thread-safe
individually — Lance creates them per worker. The plugin library itself must be
safe to load and call from multiple threads.

## Using the plugin from Lance

```rust
use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;

let params = InvertedIndexParams::default()
    .plugin(
        "/path/to/libmy_tokenizer.so".to_string(),
        r#"{"lowercase": true}"#.to_string(),
    );
```

Then pass `params` when creating the inverted index. The path is absolutized
when the params are persisted, so a relative path is fine at construction time.

## Error reporting

`create_factory` / `create_tokenizer` / `create_stream` may return `NULL` to
signal an error. Set `LanceError.message` to a UTF-8 `LanceStringRef` pointing
into plugin-owned memory that remains valid until the next call on the same
object — Lance copies the message into a Rust `String` immediately. `next_token`
returns a negative integer on error and uses the same message contract.

## Reference behavior in this crate

`src/lib.rs` accepts a JSON-ish config string with two debug knobs used by the
host integration tests:

- `"lowercase": true` — apply ASCII lowercasing inside the plugin.
- `"error_after_n_tokens": N` — emit `N` tokens, then return a negative status
  with an error message.
- `"emit_invalid_utf8": true` — emit one token whose text bytes are not valid
  UTF-8, used to verify that the host rejects invalid plugin output.

These knobs are illustrative; real plugins are free to define any config
format they want.
