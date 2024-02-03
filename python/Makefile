ifeq ($(CI), true)
    PYTEST_ARGS = -vvv -s -W error::DeprecationWarning -W error::FutureWarning --durations=30
else
    PYTEST_ARGS = -vvv -s -W error::DeprecationWarning -W error::FutureWarning
endif

test:
	pytest $(PYTEST_ARGS) python/tests
.PHONY: test

integtest:
	pytest --run-integration $(PYTEST_ARGS) python/tests
.PHONY: integtest

format: format-python
	cargo fmt
.PHONY: format

format-python:
	ruff format --preview python
	ruff --preview --fix python
.PHONY: format-python

lint: lint-python lint-rust
.PHONY: lint

lint-python:
	ruff format --preview --check python
	ruff --preview python
.PHONY: lint-python

lint-rust:
	cargo fmt -- --check
	cargo clippy -- -D warnings
.PHONY: lint-rust
