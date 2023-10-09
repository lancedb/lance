ifeq ($(CI), true)
    PYTEST_ARGS = -v -W error::FutureWarning --durations=30
else
    PYTEST_ARGS = -v -W error::FutureWarning
endif

test:
	pytest $(PYTEST_ARGS) python/tests
.PHONY: test

integtest:
	pytest --run-integration $(PYTEST_ARGS) python/tests
.PHONY: integtest

format:
	make format-python && cargo fmt
.PHONY: format

format-python:
	black python && isort python
.PHONY: format-python

lint:
	make lint-python && make lint-rust
.PHONY: lint

lint-python:
	black --check python && isort --check-only python && ruff python
.PHONY: lint-python

lint-rust:
	cargo fmt -- --check && cargo clippy -- -D warnings
.PHONY: lint-rust
