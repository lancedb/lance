# Guide for New Contributors

This is a guide for new contributors to the Lance project.
Even if you have no previous experience with python, rust, and open source, you can still make an non-trivial
impact by helping us improve documentation, examples, and more.
For experienced developers, the issues you can work on run the gamut from warm-ups to serious challenges in python and rust.

If you have any questions, please join our [Discord](https://discord.gg/zMM32dvNtd) for real-time support. Your feedback is always welcome!

## Getting Started

1. Join our Discord and say hi
2. Setup your development environment
3. Pick an issue to work on. See https://github.com/lancedb/lance/contribute for good first issues.
4. Have fun!

## Development Environment

Currently Lance is implemented in Rust and comes with a Python wrapper. So you'll want to make sure you setup both.

1. Install Rust: https://www.rust-lang.org/tools/install
2. Install Python 3.9+: https://www.python.org/downloads/
3. Install protoctol buffers: https://grpc.io/docs/protoc-installation/ (make sure you have version 3.20 or higher)
4. Install commit hooks:
    a. Install pre-commit: https://pre-commit.com/#install
    b. Run `pre-commit install` in the root of the repo

## Sample Workflow

1. Fork the repo
2. Pick [Github issue](https://github.com/lancedb/lance/issues)
3. Create a branch for the issue
4. Make your changes
5. Create a pull request from your fork to lancedb/lance
6. Get feedback and iterate
7. Merge!
8. Go back to step 2

## Python Development

The python integration is done via pyo3 + custom python code:

1. The Rust code that directly supports the Python bindings are under `python/src` while the pure Python code lives under `python/python`.
2. We make wrapper classes in Rust for Dataset/Scanner/RecordBatchReader that's exposed to python.
3. These are then used by LanceDataset / LanceScanner implementations that extend pyarrow Dataset/Scanner for duckdb compat.
4. Data is delivered via the Arrow C Data Interface

To build the Python bindings, first install requirements:

```bash
pip install maturin
```

To make a dev install:

```bash
cd python
maturin develop
```

After installing, you can run `import lance` in a Python shell within the virtual environment.

To run tests and integration tests:
```bash
make test
make integtest
```

## Rust Development

To format and lint Rust code:

```bash
cargo fmt --all
cargo clippy --all-features --tests --benches
```

### Core Format

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


## Documentation

### Main website

The main documentation website is built using [mkdocs-material](https://squidfunk.github.io/mkdocs-material/).
To build the docs, first install requirements:

```bash
pip install -r docs/requirements.txt
```

Then build and start the docs server:

```bash
cd docs
mkdocs serve
```

### Python Generated Doc

Python code documentation is built using Sphinx in [lance-python-doc](https://github.com/lancedb/lance-python-doc),
and published through [Github Pages](https://lancedb.github.io/lance-python-doc/) in ReadTheDocs style.

### Rust Generated Doc

Rust code documentation is built and published to the [Rust official docs website](https://docs.rs/lance/latest/lance/)
as a part of the release process.


## Example Notebooks

Example notebooks are under `examples`. 
These are standalone notebooks you should be able to download and run.

## Benchmarks

Our Rust benchmarks are run multiple times a day and the history can be found [here](https://github.com/lancedb/lance-benchmark-results).

Separately, we have vector index benchmarks that test against the sift1m dataset, as well as benchmarks for tpch.
These live under `benchmarks`.

## Code of Conduct

We follow the Code of Conduct of [Python Foundation](https://www.python.org/psf/conduct/) and 
[Rust Foundation](https://www.rust-lang.org/policies/code-of-conduct). 