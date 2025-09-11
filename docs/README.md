# Lance Documentation

This directory contains the documentation for Lance, built with MkDocs and Material theme.

## Getting Started with uv

### Setup

1. Install uv if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. That's it! No manual dependency installation needed.

### Building Documentation

To build and serve the documentation locally:

```bash
make serve
```

The documentation will be available at http://localhost:8000

### Building for Production

```bash
make build
```

This will create a `site/` directory with the built documentation.

### Managing Dependencies

#### Adding Dependencies

```bash
# Add a dependency and update pyproject.toml
uv add <package>

# Add a dev dependency
uv add --dev <package>
```

#### Manual Sync (if needed)

```bash
make sync
```

#### Upgrading Dependencies

```bash
# Upgrade a specific package
uv add <package>@latest

# Upgrade all dependencies
uv sync --upgrade
```

## Project Structure

- `src/` - Source markdown files for documentation
- `mkdocs.yml` - MkDocs configuration
- `pyproject.toml` - Python project configuration (uv compatible)
