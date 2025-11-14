// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#[test]
fn test_settings() {
    let can_run = std::env::var("NEXTEST_RUN_ID").is_ok()
        || std::env::var("CARGO_TEST_THREADS")
            .map(|v| v == "1")
            .unwrap_or(false);
    assert!(
        can_run,
        "Memory tests require single-threaded execution. \
            Please run with `CARGO_TEST_THREADS=1` or use `cargo nextest`."
    );
}

mod resource_test;
