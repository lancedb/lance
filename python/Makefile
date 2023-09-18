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
