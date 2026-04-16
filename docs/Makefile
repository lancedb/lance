.PHONY: help serve build clean install dev-install check-links sync serve-zh build-zh build-all

help:
	@echo "Available commands:"
	@echo "  make serve        - Serve English documentation locally"
	@echo "  make serve-zh     - Serve Chinese documentation locally"
	@echo "  make build        - Build English documentation"
	@echo "  make build-zh     - Build Chinese documentation"
	@echo "  make build-all    - Build both English and Chinese documentation"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make check-links  - Check for broken links"
	@echo "  make sync         - Sync dependencies with pyproject.toml"

sync:
	uv sync --all-extras

serve:
	uv run mkdocs serve

serve-zh:
	uv run mkdocs serve -f mkdocs_zh.yml

build:
	uv run mkdocs build

build-zh:
	uv run mkdocs build -f mkdocs_zh.yml -d site/zh

build-all: build build-zh

clean:
	rm -rf site/
	rm -rf .cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +

check-links:
	uv run mkdocs-linkcheck src
