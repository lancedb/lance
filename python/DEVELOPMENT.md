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

## Formatting and linting Python

To check the formatting, use

```shell
black --check python
isort --check-only python
ruff check python
```

To automatically fix, use

```shell
black python
isort python
ruff python --fix
```

## Formatting and linting Rust

```shell
cargo fmt
cargo clippy --all-features
```

Some lints can be fixed automatically:

```shell
cargo clippy --all-features --fix
```
