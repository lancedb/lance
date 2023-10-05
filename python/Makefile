ifeq ($(CI), true)
    PYTEST_ARGS = -v --durations=30
else
    PYTEST_ARGS = -v
endif

test:
	pytest -m 'not integration' -m 'not cuda' $(PYTEST_ARGS) python/tests
.PHONY: test

integtest:
	pytest -m 'integration' $(PYTEST_ARGS) python/tests
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
