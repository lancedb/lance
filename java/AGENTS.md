## Structure
Java code: `core/src`
Rust JNI bindings: `lance-jni`

## Commands
Use `./mvnw` instead of `mvn` to ensure the correct version of Maven is used.
format: `./mvnw spotless:apply && cargo fmt --manifest-path ./lance-jni/Cargo.toml --all`
lint rust: `cargo clippy --tests --manifest-path ./lance-jni/Cargo.toml`
compile: `./mvnw compile`
test: `./mvnw test`
