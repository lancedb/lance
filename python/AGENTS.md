# Python Guidelines

Also see [root AGENTS.md](../AGENTS.md) for cross-language standards.

## Commands

* Build: `maturin develop` (required after Rust changes)
* Test: `make test`
* Run single test: `pytest python/tests/<test_file>.py::<test_name>`
* Doctest: `make doctest`
* Lint: `make lint`
* Format: `make format`

## API Design

- Keep bindings as thin wrappers — centralize validation and logic in Rust core.
- Extend existing methods with named arguments instead of adding new methods that accept policy/config objects — the Python API should feel Pythonic (e.g., `cleanup_old_versions(..., retain_versions=N)`), not mirror Rust builder patterns.
- Pass all fields to Python dataclass constructors via PyO3, converting Rust `None` to `py.None()` instead of omitting args — dataclass constructors require all positional params.
- Use parameterized type hints (e.g., `list[DatasetBasePath]`, `Optional[Dict[str, str]]`) — never bare generics. Keep docstring type descriptions in sync with hints.

## Testing

- Use `@pytest.mark.parametrize` for tests that differ only in inputs — extract shared setup into helpers.
- Add tests to existing `test_{module}.py` files rather than creating new test files for the same module.
- Replace `print()` in tests with `assert` statements.
