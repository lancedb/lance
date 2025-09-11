# Contributing to Rust

To format and lint Rust code:

```bash
cargo fmt --all
cargo clippy --all-features --tests --benches
```

## Core Format

The core format is implemented in Rust under the `rust` directory. Once you've setup Rust you can build the core format with:

```bash
cargo build
```

This builds the debug build. For the optimized release build:

```bash
cargo build -r
```

To run the Rust unit tests:

```bash
cargo test
```

If you're working on a performance related feature, benchmarks can be run via:

```bash
cargo bench
```
