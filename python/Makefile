ifeq ($(CI), true)
    PYTEST_ARGS = -vvv -s --durations=30 -m "not recurring"
else
    PYTEST_ARGS = -vvv -s -m "not recurring"
endif

test:
	uv run pytest $(PYTEST_ARGS) python/tests
.PHONY: test

integtest:
	uv run pytest --run-integration $(PYTEST_ARGS) python/tests/test_s3_ddb.py
.PHONY: integtest

doctest:
	uv run pytest --doctest-modules $(PYTEST_ARGS) python/lance
.PHONY: doctest

TORCH_TESTS = python/tests/torch_tests

test-torch:
	uv run --package pylance-torch-tests pytest $(PYTEST_ARGS) $(TORCH_TESTS)
.PHONY: test-torch

format: format-python
	cargo fmt
.PHONY: format

build:
	uv run maturin develop
.PHONY: build

clean:
	rm -rf ./target
.PHONY: clean

format-python:
	uv run ruff format python
	uv run ruff check --fix python
.PHONY: format-python

lint: lint-python lint-rust
.PHONY: lint

lint-python:
	uv run ruff format --check --diff python
	uv run ruff check python
	uv run pyright
.PHONY: lint-python

lint-rust:
	cargo fmt -- --check
	cargo clippy -- -D warnings
.PHONY: lint-rust

clean:
	cargo clean
	rm -rf target
.PHONY: clean
