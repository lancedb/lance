# AGENTS.md

Lance is a modern columnar data format optimized for ML workflows and datasets, providing high-performance random access, vector search, zero-copy automatic versioning, and ecosystem integrations. The vision is to become the de facto standard columnar data format for machine learning and large language models.

Also see directory-specific guidelines: [rust/](rust/AGENTS.md) | [python/](python/AGENTS.md) | [java/](java/AGENTS.md) | [protos/](protos/AGENTS.md) | [docs/src/format/](docs/src/format/AGENTS.md)

## Architecture

Rust workspace with Python and Java bindings:

- `rust/lance/` - Main library implementing the columnar format
- `rust/lance-core/` - Core types, traits, and utilities
- `rust/lance-arrow/` - Apache Arrow integration layer
- `rust/lance-encoding/` - Data encoding and compression algorithms
- `rust/lance-file/` - File format reading/writing
- `rust/lance-index/` - Vector and scalar indexing
- `rust/lance-io/` - I/O operations and object store integration
- `rust/lance-linalg/` - Linear algebra for vector search
- `rust/lance-table/` - Table format and operations
- `rust/lance-geo/` - Geospatial data support
- `rust/lance-datagen/` - Data generation for tests and benchmarks
- `rust/lance-namespace/` / `rust/lance-namespace-impls/` - Namespace/catalog interfaces
- `rust/lance-test-macros/` / `rust/lance-testing/` - Test infrastructure
- `rust/lance-tools/` - CLI and developer tooling
- `rust/examples/` - Sample binaries and demonstrations
- `rust/compression/bitpacking/` / `rust/compression/fsst/` - Compression codecs
- `rust/lance-datafusion/` - DataFusion integration (built separately)
- `python/` - Python bindings (PyO3/maturin)
- `java/` - Java bindings (JNI)

Key technical traits: async-first (tokio), Arrow-native, versioned writes with manifest tracking, custom ML-optimized encodings, unified object store interface (local/S3/Azure/GCS).

## Development Commands

### Rust

* Check: `cargo check --workspace --tests --benches`
* Test: `cargo test --workspace` or `cargo test -p <package> <test_name>`
* Lint: `cargo clippy --all --tests --benches -- -D warnings`
* Format: `cargo fmt --all`
* Coverage: `cargo +nightly llvm-cov -q -p <crate> --branch`
* Coverage HTML: `cargo +nightly llvm-cov -q -p <crate> --branch --html`
* Coverage for file: `python ci/coverage.py -p <crate> -f <file_path>`

### Python / Java

See [python/AGENTS.md](python/AGENTS.md) and [java/AGENTS.md](java/AGENTS.md).

## Language-Specific Environment Contract

- For language-specific tasks, always follow the environment and command rules in the corresponding subdirectory guide before running build, test, lint, format, or tooling commands.
- Do not substitute a different environment manager or toolchain just because a command appears missing, unavailable, or slow.
- If a language-specific command fails outside the documented workflow, treat that as an environment usage mistake first. Fix the environment usage, rerun with the prescribed commands, and only then conclude that a dependency or tool is unavailable.

### Integration Testing

```bash
cd test_data && docker compose up -d
AWS_DEFAULT_REGION=us-east-1 pytest --run-integration python/tests/test_s3_ddb.py
```

## Coding Standards

### General

- Always use English in code, examples, and comments.
- Code is for readability, not just execution. Only add meaningful comments and tests.
- Comments should explain non-obvious "why" reasoning, not restate what the code does.
- Remove debug prints (`println!`, `dbg!`, `print()`) before merging — use `tracing` or logging frameworks.
- Extract logic repeated in 2+ places into a shared helper; inline single-use logic at its call site.
- Think carefully before adding a helper: only introduce one when it materially reduces cognitive load or eliminates substantial duplication, and do not add thin wrappers that only rename or forward existing calls.
- Keep PRs focused — no drive-by refactors, reformatting, or cosmetic changes.
- Be mindful of memory use: avoid collecting streams of `RecordBatch` into memory; use `RoaringBitmap` instead of `HashSet<u32>`.

### Cross-Language Bindings

- Keep Python and Java bindings as thin wrappers — centralize validation and logic in the Rust core.
- Keep parameter names consistent across all bindings (Rust, Python, Java) — rename everywhere or nowhere.
- Never break public API signatures — deprecate with `#[deprecated]`/`@deprecated` and add a new method.
- Replace mutually exclusive boolean flags with a single enum/mode parameter.

### Naming

- Name variables after what the value *is* (e.g., `partition_id` not `mask`) — precise names act as inline docs.
- Drop redundant prefixes when the struct/module already implies the domain.
- Use `indices` (not `indexes`) consistently in all APIs and docs.
- Use storage-agnostic terms in API names (e.g., `base` not `bucket`).
- When renaming a type/struct/enum, update all references (methods, fields, variables, test names).

### Error Handling

- Validate inputs and reject invalid values with descriptive errors at API boundaries — never silently clamp or adjust.
- Validate mutually exclusive options in builders/configs — throw a clear error if both are set.
- Include full context in error messages: variable names, values, sizes, types.

### Dependencies

- Prefer implementing functionality with the standard library or existing workspace dependencies before adding new external crates.
- Keep `Cargo.lock` changes intentional; revert unrelated dependency bumps. Pin broken deps with a comment linking the upstream issue.
- Gate optional/domain-specific deps behind Cargo feature flags. Prefer separate crates for domain functionality (geo, NLP).

## Testing Standards

- **All bugfixes and features must have corresponding tests. We do not merge code without tests.**
- Use `rstest` (Rust) or `@pytest.mark.parametrize` (Python) for tests that differ only in inputs. Use `#[case::{name}(...)]` for readable case names.
- Replace `print()` in tests with `assert` — prints don't catch regressions.
- Extend existing tests instead of adding overlapping new ones. Add to existing test files.
- Link a GitHub issue when skipping a test — never bare `@pytest.mark.skip` or `@Ignore` without a tracking URL.
- Include multi-fragment scenarios for dataset operations (reads, indexes, scans).
- Cover NULL edge cases in index tests: null items, all-null collections, empty collections, null columns.
- Vector index tests must assert recall metrics (>=0.5 threshold), not just verify creation succeeds.
- For backwards compatibility, use the `test_data` directory with checked-in datasets from older versions. Include a `datagen.py` that asserts the Lance version used. Use `copy_test_data_to_tmp` to read this data.
- Avoid `ignore` in doctests — write Rust doctests that compile a function instead:
  ```
  /// ```
  /// # use lance::{Dataset, Result};
  /// # async fn test(dataset: &Dataset) -> Result<()> {
  /// dataset.delete("id = 25").await?;
  /// # Ok(())
  /// # }
  /// ```
  ```
- Skip coverage for test utilities using `#[cfg_attr(coverage, coverage(off))]`.

## Documentation Standards

- All public APIs must have documentation with examples. Link to relevant structs and methods.
- Use ASCII tree diagrams for hierarchical structures (encoding layers, file formats, storage layouts).
- Keep doc examples in sync with actual API signatures — update when refactoring.
- Indent content under MkDocs admonition directives (`!!! note`, etc.) with 4 spaces.
- Proofread comments and docs for typos before committing.

## Review Guidelines

Contributor and maintainer attention is the most valuable resource. Less is more.

- Be concise and clear. Focus on P0/P1 issues: severe bugs, performance degradation, security concerns.
- Do not reiterate detailed changes or repeat what's already well done.
- Check naming consistency, error handling patterns, and test coverage.
