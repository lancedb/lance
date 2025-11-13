// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

// If NEXTEST_RUN_ID is set, or --test-threads=1 is passed, then we are running in single-threaded mode.
// Otherwise, we should skip these tests to avoid interference.

use std::sync::LazyLock;

#[test]
fn test_settings() {
    let can_run = std::env::var("NEXTEST_RUN_ID").is_ok()
        || std::env::var("CARGO_TEST_THREADS")
            .map(|v| v == "1")
            .unwrap_or(false);
    assert!(
        can_run,
        "Memory tests require single-threaded execution. \
            Please run with `-- --test-threads=1` or use `cargo nextest`."
    );
}

mod resource_test;
