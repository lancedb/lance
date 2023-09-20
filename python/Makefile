ifeq ($(CI), true)
    PYTEST_ARGS = -v --durations=30
else
    PYTEST_ARGS = -v
endif

test:
	pytest -m 'not integration' $(PYTEST_ARGS) python/tests
.PHONY: test

integtest:
	pytest -m 'integration' $(PYTEST_ARGS) python/tests
.PHONY: integtest

format:
	black python && isort python && cargo fmt
.PHONY: format

lint:
	black --check python && isort --check-only python && ruff python && \
	cargo fmt -- --check && cargo clippy -- -D warnings
.PHONY: lint