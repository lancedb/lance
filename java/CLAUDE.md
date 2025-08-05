## Structure
Java code: `core/src`
Rust JNI bindings: `core/lance-jni`

## Commands
Use `./mvnw` instead of `mvn` to ensure the correct version of Maven is used.
format: `./mvnw spotless:apply && cargo fmt --all`
lint rust: `cargo clippy --tests`
compile: `./mvnw compile`
test: `./mvnw test`
