ifeq ($(CI), true)
    PYTEST_ARGS = -vvv -s --durations=30
else
    PYTEST_ARGS = -vvv -s
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
	uv run ruff format --check python
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
