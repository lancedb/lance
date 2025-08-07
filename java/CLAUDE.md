## Structure
Java code: `core/src`
Rust JNI bindings: `core/lance-jni`

## Commands
Use `./mvnw` instead of `mvn` to ensure the correct version of Maven is used.
format: `./mvnw spotless:apply && cd core/lance-jni && cargo fmt --all && cd ../..`
lint rust: `cargo clippy --tests`
compile: `./mvnw compile`
test: `./mvnw test`
