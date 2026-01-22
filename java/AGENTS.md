## Structure
Java code: `java/src/main/java`
Rust JNI bindings: `java/lance-jni`

## Commands
Use `./mvnw` instead of `mvn` to ensure the correct version of Maven is used.
format: `./mvnw spotless:apply && cargo fmt --manifest-path ./lance-jni/Cargo.toml --all`
format (check only): `./mvnw spotless:check`
lint rust: `cargo clippy --tests --manifest-path ./lance-jni/Cargo.toml`
compile: `./mvnw compile`
test: `./mvnw test`

JDK: pom.xml targets Java 11 (`maven.compiler.release` 11); align Rust toolchain with repository `rust-toolchain.toml`.
