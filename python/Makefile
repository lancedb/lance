ifeq ($(CI), true)
    PYTEST_ARGS = -vvv -s --durations=30
else
    PYTEST_ARGS = -vvv -s
endif

test:
	pytest $(PYTEST_ARGS) python/tests
.PHONY: test

integtest:
	pytest --run-integration $(PYTEST_ARGS) python/tests/test_s3_ddb.py
.PHONY: integtest

doctest:
	pytest --doctest-modules $(PYTEST_ARGS) python/lance
.PHONY: doctest

format: format-python
	cargo fmt
.PHONY: format

format-python:
	ruff format python
	ruff check --fix python
.PHONY: format-python

lint: lint-python lint-rust
.PHONY: lint

lint-python:
	ruff format --check python
	ruff check python
	pyright
.PHONY: lint-python

lint-rust:
	cargo fmt -- --check
	cargo clippy -- -D warnings
.PHONY: lint-rust
