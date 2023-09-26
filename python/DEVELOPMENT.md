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

```shell
pytest python/tests
```

To check the documentation examples, use

```shell
pytest --doctest-modules python/lance
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

### Format and linting on commit

If you would like to run the formatters and linters when you commit your code
then you can use the pre-commit tool.  The project includes a pre-commit config
file already.  First, install the pre-commit tool:

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

## Benchmarks

The benchmarks in `python/benchmarks` can be used to identify and diagnose
performance issues. They are run with [pytest-benchmark](https://pytest-benchmark.readthedocs.io/en/latest/).
These benchmarks aren't mean to showcase performance on full-scale real world
datasets; rather they are meant to be useful for developers to iterate on
performance improvements and to catch performance regressions. Therefore, any
benchmarks added there should run in less than 5 seconds.

Before running benchmarks, you should build pylance in release mode:

```shell
maturin develop --profile release-with-debug --extras benchmarks
```

(You can also use `--release` or `--profile release`, but `--profile release-with-debug`
will provide debug symbols for profiling.)

Then you can run the benchmarks with

```shell
pytest python/benchmarks
```

Note: the first time you run the benchmarks, they may take a while, since they
will write out test datasets and build vector indices. Once these are built,
they are re-used between benchmark runs.

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

You can easily compare the performance of the current version against a previous
version of pylance. Install the previous version, run the benchmarks, and save
the output using `--benchmark-save`. Then install the current version and run
the benchmarks again with `--benchmark-compare`.

```shell
pip uninstall -y pylance
pip install pylance==0.4.18
pytest --benchmark-save=baseline python/benchmarks
COMPARE_ID=$(ls .benchmarks/*/ | tail -1 | cut -c1-4)
maturin develop --profile release-with-debug
pytest --benchmark-compare=$COMPARE_ID python/benchmarks
```
