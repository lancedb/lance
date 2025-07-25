## Structure
Java code: `core/src`
Rust JNI bindings: `core/lance-jni`

## Commands
format: `mvn spotless:apply && cargo fmt --all`
lint rust: `cargo clippy --tests`
compile: `mvn compile`
test: `mvn test`
