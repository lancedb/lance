ifeq ($(CI), true)
    PYTEST_ARGS = -vvv -s --durations=30 -m "not recurring"
else
    PYTEST_ARGS = -vvv -s -m "not recurring"
endif

PYTEST_MARK_EXPR ?=
ifneq ($(strip $(PYTEST_MARK_EXPR)),)
    PYTEST_MARK_ARGS = -m "$(PYTEST_MARK_EXPR)"
endif

test:
	uv run pytest $(PYTEST_MARK_ARGS) $(PYTEST_ARGS) python/tests
.PHONY: test

integtest:
	uv run pytest --run-integration $(PYTEST_ARGS) python/tests/test_s3_ddb.py
.PHONY: integtest

doctest:
	uv run pytest --doctest-modules $(PYTEST_ARGS) python/lance
.PHONY: doctest

test-torch:
	cd python/tests/torch_tests && uv run pytest $(PYTEST_ARGS) .
.PHONY: test-torch

test-pandas:
	cd python/tests/pandas_tests && uv run pytest $(PYTEST_ARGS)
.PHONY: test-pandas

test-tf:
	cd python/tests/tf_tests && uv run pytest $(PYTEST_ARGS)
.PHONY: test-tf

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
