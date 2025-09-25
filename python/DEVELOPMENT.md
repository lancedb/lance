# Development

## Building the project

This project is built with [maturin](https://github.com/PyO3/maturin).

It can be built in development mode with:

```shell
maturin develop
```

This builds the Rust native module in place. You will need to re-run this
whenever you change the Rust code. But changing the Python code doesn't require
re-building.

## Running tests

To run the tests, first install the test packages:

```shell
pip install '.[tests]'
```

then:

```shell
make test
```

To check the documentation examples, use

```shell
make doctest
```

## Formatting and linting

To run formatters, run:

```shell
make format
```

(To run for just Python or just Rust, use `make format-python` or `cargo fmt`.)

To run format checker and linters, run:

```shell
make lint
```

(To run for just Python or just Rust, use `make lint-python` or `make lint-rust`.)

### Format and lint on commit

If you would like to run the formatters and linters when you commit your code
then you can use the pre-commit tool. The project includes a pre-commit config
file already. First, install the pre-commit tool:

```shell
pip install pre-commit
```

Then install the hooks:

```shell
pre-commit install
```

From now any, any attempt to commit, will first run the linters against the
modified files:

```shell
$ git commit -m"Changed some python files"
black....................................................................Passed
isort (python)...........................................................Passed
ruff.....................................................................Passed
[main daf91ed] Changed some python files
 1 file changed, 1 insertion(+), 1 deletion(-)
```

## Cleaning build artifacts
To clean up build artifacts, run:
```shell
make clean
```

## Benchmarks

The benchmarks in `python/benchmarks` can be used to identify and diagnose
performance issues. They are run with [pytest-benchmark](https://pytest-benchmark.readthedocs.io/en/latest/).
These benchmarks aren't mean to showcase performance on full-scale real world
datasets; rather they are meant to be useful for developers to iterate on
performance improvements and to catch performance regressions. Therefore, any
benchmarks added there should run in less than 5 seconds.

Before running benchmarks, you should build pylance in release mode:

```shell
maturin develop --profile release-with-debug --extras benchmarks --features datagen
```

(You can also use `--release` or `--profile release`, but `--profile release-with-debug`
will provide debug symbols for profiling.)

Then you can run the benchmarks with

```shell
pytest python/benchmarks -m "not slow"
```

Note: the first time you run the benchmarks, they may take a while, since they
will write out test datasets and build vector indices. Once these are built,
they are re-used between benchmark runs.

Some benchmarks are especially slow, so they are skipped `-m "not slow"`. To run
the slow benchmarks, use:

```shell
pytest python/benchmarks
```

### Run a particular benchmark

To filter benchmarks by name, use the usual pytest `-k` flag (this can be a
substring match, so you don't need to type the full name):

```shell
pytest python/benchmarks -k test_ivf_pq_index_search
```

### Profile a benchmark

If you have [cargo-flamegraph](https://github.com/flamegraph-rs/flamegraph)
installed, you can create a flamegraph of a benchmark by running:

```shell
flamegraph -F 100 --no-inline -- $(which python) \
    -m pytest python/benchmarks \
    --benchmark-min-time=2 \
    -k test_ivf_pq_index_search
```

Note the parameter `--benchmark-min-time`: this controls how many seconds to run
the benchmark in each round (default 5 rounds). The default is very low but you
can increase this so that the profile gets more samples.

You can drop the `--no-inline` to have the program try to identify which functions
were inlined to get more detail, though this will make the processing take
considerably longer.

This will only work on Linux.

Note that you'll want to run the benchmarks once prior to profiling, so that
the setup is complete and not captured as part of profiling.

### Compare benchmarks against previous version

You can easily compare the performance of the current version against `main`.
Checkout `main` branch, run the benchmarks, and save
the output using `--benchmark-save`. Then install the current version and run
the benchmarks again with `--benchmark-compare`.

```shell
CURRENT_BRANCH=$(git branch --show-current)
git checkout main
maturin develop --profile release-with-debug  --features datagen
pytest --benchmark-save=baseline python/benchmarks -m "not slow"
COMPARE_ID=$(ls .benchmarks/*/ | tail -1 | cut -c1-4)
git checkout $CURRENT_BRANCH
maturin develop --profile release-with-debug  --features datagen
pytest --benchmark-compare=$COMPARE_ID python/benchmarks -m "not slow"
```

## Tracing a python script

If you would like to trace a python script (application, benchmark, test) then you can easily
do so using the lance.tracing module. Simply call:

```python
from lance.tracing import trace_to_chrome

trace_to_chrome(level="debug")

# rest of script
```

A single .json trace file will be generated after python has exited.

You can use the `trace_to_chrome` function within the benchmarks, but for
sensible results you'll want to force the benchmark to just run only once.
To do this, rewrite the benchmark using the pedantic API:

```python
def run():
    "Put code to benchmark here"
    ...
benchmark.pedantic(run, iterations=1, rounds=1)
```

### Trace visualization limitations

The current tracing implementation is slightly flawed when it comes to async
operations that run in parallel. The rust tracing-chrome library emits
trace events into the chrome trace events JSON format. This format is not
sophisticated enough to represent asynchronous parallel work.

As a result, a single instrumented async method may appear as many different
spans in the UI.

## Running S3 Integration tests

The integration tests run against local minio and local dynamodb. To start the
services, run

```shell
docker compose up
```

Then you can run the tests with

```shell
pytest --run-integration python/tests/test_s3_ddb.py
```

## Building wheels locally

### Linux

On Mac or Linux, you can build manylinux wheels locally for Linux. The easiest
way to do this is to use `zig` with `maturin build`. Before you do this, you'll
need to make you (1) [install zig](https://github.com/ziglang/zig/wiki/Install-Zig-from-a-Package-Manager)
and (2) install the toolchains:

```shell
rustup target add x86_64-unknown-linux-gnu
rustup target add aarch64-unknown-linux-gnu
```

For x86 Linux:

```shell
maturin build --release --zig \
    --target x86_64-unknown-linux-gnu \
    --compatibility manylinux2014 \
    --out wheels
```

For ARM / aarch64 Linux:

```shell
maturin build --release --zig \
    --target aarch_64-unknown-linux-gnu \
    --compatibility manylinux2014 \
    --out wheels
```

### MacOS

On a Mac, you can build wheels locally for MacOS:

```shell
maturin build --release \
    --target aarch64-apple-darwin \
    --out wheels
```

```shell
maturin build --release \
    --target x86_64-apple-darwin \
    --out wheels
```
