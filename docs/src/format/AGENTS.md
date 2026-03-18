# Format Documentation Guidelines

Also see [root AGENTS.md](../../../AGENTS.md) for cross-language standards.

## Style

- Keep format docs as concise, text-only reference — no code examples (put those in user guide sections).
- Express file schemas as `pyarrow` schema definitions, not markdown tables or informal text — pyarrow schemas are unambiguous and executable.
- Use language-agnostic definitions (JSON Schema, protobuf) — not language-specific code like Rust structs.

## Content

- Explain schema/data evolution with concrete mechanics (field IDs, tombstones, data rewrites) — don't just name operations or defer to external specs.
- Describe all algorithms with full detail: parameters, precision, ordering, normalization bounds, and implementation steps — never reference an algorithm by name alone.
- Index docs must include explicit file schemas and describe reader navigation (page type distinction, root/entry point location) — follow the pattern in `table/index/scalar/bitmap.md`.
