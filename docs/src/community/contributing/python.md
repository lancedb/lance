# Contributing to Python

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

To run the tests on OS X, you may need to increase the default limit on the number of open files:
`ulimit -n 2048`