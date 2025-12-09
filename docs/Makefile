.PHONY: help serve build clean install dev-install check-links sync

help:
	@echo "Available commands:"
	@echo "  make serve        - Serve documentation locally (auto-installs deps)"
	@echo "  make build        - Build documentation (auto-installs deps)"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make check-links  - Check for broken links"
	@echo "  make sync         - Sync dependencies with pyproject.toml"

sync:
	uv sync --all-extras

serve:
	uv run mkdocs serve

build:
	uv run mkdocs build

clean:
	rm -rf site/
	rm -rf .cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +

check-links:
	uv run mkdocs-linkcheck src
