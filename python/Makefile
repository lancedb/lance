ifeq ($(CI), true)
    PYTEST_ARGS = -vvv -s --durations=30 -m "not recurring"
else
    PYTEST_ARGS = -vvv -s -m "not recurring"
endif

test:
	pytest $(PYTEST_ARGS) python/tests
.PHONY: test

integtest:
	pytest --run-integration $(PYTEST_ARGS) python/tests/test_s3_ddb.py python/tests/test_namespace_integration.py
.PHONY: integtest

doctest:
	pytest --doctest-modules $(PYTEST_ARGS) python/lance
.PHONY: doctest

compattest:
	pytest --run-compat $(PYTEST_ARGS)  python/tests/compat
.PHONY: compattest

format: format-python
	cargo fmt
.PHONY: format

build:
	maturin develop
.PHONY: build

format-python:
	ruff format python
	ruff check --fix python
.PHONY: format-python

lint: lint-python lint-rust
.PHONY: lint

lint-python:
	ruff format --check --diff python
	ruff check python
	pyright
.PHONY: lint-python

lint-rust:
	cargo fmt -- --check
	cargo clippy -- -D warnings
.PHONY: lint-rust

clean:
	cargo clean
	rm -rf target
.PHONY: clean