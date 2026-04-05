# Protobuf Guidelines

Also see [root AGENTS.md](../AGENTS.md) for cross-language standards.

## Compatibility

- All changes must be backwards compatible. Never re-use or change field numbers of existing fields.

## Schema Design

- Use `optional` when you need to distinguish "not set" from "zero value" — `optional` enables presence tracking (`has_*` methods) and maps to `Option<T>` in Rust. Bare proto3 fields have no presence semantics: they always hold a value (defaulting to zero), so you cannot tell if the sender explicitly set them.
- Use structured message types (e.g., `BasePath`) instead of plain scalars, and scope fields to operation-specific messages (e.g., `InsertTransaction`) rather than generic top-level ones.
- Don't duplicate data across messages — store each fact once and derive relationships. Prefer parallel sequences over maps when keys already exist in another field.

## Documentation

- Document the semantic meaning of both present and absent states for `optional` fields — explain when each case applies.
- Use precise domain terminology in field descriptions — avoid ambiguous abbreviations or terms that collide with domain concepts.
