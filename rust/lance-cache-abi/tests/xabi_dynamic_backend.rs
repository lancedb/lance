// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use lance_cache_abi::{CacheKey128, CacheMeasure, load_backend_named};

static FIXTURE_LIBRARY: OnceLock<PathBuf> = OnceLock::new();

#[test]
fn xabi_dynamic_backend_round_trips_through_real_library() {
    futures::executor::block_on(async {
        let library_path = fixture_library_path();
        let backend = unsafe { load_backend_named(library_path, "memory") }
            .expect("fixture backend loads through xabi");
        let key = CacheKey128::from_bytes([
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xfe, 0xed,
            0xbe, 0xef,
        ]);

        assert_eq!(backend.name().expect("name is available"), "memory-fixture");
        assert_eq!(backend.get(key).await.expect("miss is returned"), None);

        backend
            .insert(key, b"from-plugin", 128)
            .await
            .expect("insert succeeds through dynamic xabi handle");

        let hit = backend
            .get(key)
            .await
            .expect("get succeeds through dynamic xabi handle")
            .expect("inserted key is present");
        assert_eq!(hit.bytes, b"from-plugin");
        assert_eq!(hit.size_bytes, 128);

        assert_eq!(
            backend.measure().await.expect("measure succeeds"),
            CacheMeasure::new(1, 128)
        );

        backend.clear().await.expect("clear succeeds");
        assert_eq!(backend.get(key).await.expect("cleared key misses"), None);
    });
}

#[test]
fn xabi_dynamic_backend_transports_typed_errors() {
    futures::executor::block_on(async {
        let library_path = fixture_library_path();
        let backend = unsafe { load_backend_named(library_path, "memory") }
            .expect("fixture backend loads through xabi");
        let err = backend
            .get(CacheKey128::from_bytes([13; 16]))
            .await
            .expect_err("fixture returns a typed export error");

        assert_eq!(err.to_string(), "fixture rejected key 13");
    });
}

fn fixture_library_path() -> &'static Path {
    FIXTURE_LIBRARY.get_or_init(build_fixture_library).as_path()
}

fn build_fixture_library() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_dir = manifest_dir
        .ancestors()
        .nth(2)
        .expect("lance-cache-abi lives under rust/")
        .to_path_buf();
    let target_dir = workspace_dir.join("target").join("xabi-fixtures");
    let status = Command::new("cargo")
        .args(["build", "-p", "lance-cache-xabi-fixture", "--target-dir"])
        .arg(&target_dir)
        .args(["--message-format", "short"])
        .current_dir(&workspace_dir)
        .env_remove("RUSTC_WRAPPER")
        .env_remove("CARGO_TARGET_DIR")
        .status()
        .expect("cargo build can be launched");
    assert!(status.success(), "fixture cdylib build failed");

    let profile_dir = target_dir.join("debug");
    let library_path = profile_dir.join(dynamic_library_name("lance_cache_xabi_fixture"));
    assert!(
        library_path.exists(),
        "fixture cdylib was not built at {}",
        library_path.display()
    );
    library_path
}

fn dynamic_library_name(stem: &str) -> String {
    if cfg!(target_os = "macos") {
        format!("lib{stem}.dylib")
    } else if cfg!(target_os = "windows") {
        format!("{stem}.dll")
    } else {
        format!("lib{stem}.so")
    }
}
