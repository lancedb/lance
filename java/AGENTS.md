# Java Guidelines

Also see [root AGENTS.md](../AGENTS.md) for cross-language standards.

## Commands

Use `./mvnw` instead of `mvn` to ensure the correct version of Maven is used.

* Format: `./mvnw spotless:apply && cargo fmt --manifest-path ./lance-jni/Cargo.toml --all`
* Format (check only): `./mvnw spotless:check`
* Lint Rust: `cargo clippy --tests --manifest-path ./lance-jni/Cargo.toml`
* Compile: `./mvnw compile`
* Test: `./mvnw test`

JDK: pom.xml targets Java 11 (`maven.compiler.release` 11); align Rust toolchain with repository `rust-toolchain.toml`.

## Structure

* Java code: `java/src/main/java`
* Rust JNI bindings: `java/lance-jni`

## API Design

- Encapsulate related/optional params in `XxxOptions`/`XxxParams` objects or use the builder pattern instead of growing argument lists.
- Use strongly-typed enums (not raw `String`s) for version/config values. Serialize via explicit fields (e.g., `toRustString()`) matching Rust-expected formats, not Java `SCREAMING_SNAKE_CASE` `toString()`. Test all enum values across the JNI boundary.
- Return protobuf-backed opaque handles (e.g., serialized `bytes`) from public APIs and IPC interfaces instead of exposing internal data structures.

## Code Style

- Prefer top-level imports over fully qualified class names — only use fully qualified names to resolve ambiguity.
- Use JavaBean-style `getXXX()` for getter methods, not bare accessor style — serialization frameworks and IDE tooling rely on this convention.

## Documentation

- Copy Rust docs (defaults, constraints, invariants) into Javadoc for binding classes — users shouldn't need to read Rust source to understand API behavior.
