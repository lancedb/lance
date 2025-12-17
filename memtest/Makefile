.PHONY: build test lint format clean

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
LIB_FILE := libmemtest.dylib
PRELOAD_ENV := DYLD_INSERT_LIBRARIES
else
LIB_FILE := libmemtest.so
PRELOAD_ENV := LD_PRELOAD
endif

build:
	cargo build
	cp target/debug/$(LIB_FILE) python/memtest/
	pip install -e .

build-release:
	cargo build --release
	cp target/release/libmemtest.so python/memtest/
	pip install -e .

test:
	$(PRELOAD_ENV)=./python/memtest/$(LIB_FILE) pytest python/tests/ -v

lint:
	cargo clippy -- -D warnings
	ruff check python/

format:
	cargo fmt
	ruff format python/

clean:
	cargo clean
	rm -rf target/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.so" -delete
	find . -type f -name "*.dylib" -delete
