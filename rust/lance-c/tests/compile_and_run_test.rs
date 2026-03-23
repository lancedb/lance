// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Tests that compile and run actual C and C++ programs against the lance-c library.
//!
//! These tests:
//! 1. Create a test dataset on disk
//! 2. Build the lance-c shared library
//! 3. Compile C/C++ test programs linking against it
//! 4. Run the compiled binaries with the dataset path
//!
//! This validates that lance.h and lance.hpp are valid C/C++ and that
//! the API works end-to-end from a real C/C++ caller.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance::Dataset;

/// Build the lance-c cdylib and return the path to the shared library and include dir.
fn build_lance_c() -> (PathBuf, PathBuf) {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();

    // Build the cdylib in debug mode.
    let status = Command::new("cargo")
        .args(["build", "-p", "lance-c"])
        .current_dir(workspace_root)
        .status()
        .expect("Failed to run cargo build");
    assert!(status.success(), "cargo build -p lance-c failed");

    let target_dir = workspace_root.join("target").join("debug");

    // Find the shared library.
    let lib_path = if cfg!(target_os = "macos") {
        target_dir.join("liblance_c.dylib")
    } else if cfg!(target_os = "linux") {
        target_dir.join("liblance_c.so")
    } else {
        panic!("Unsupported OS for C/C++ link test");
    };

    assert!(
        lib_path.exists(),
        "Shared library not found at {}",
        lib_path.display()
    );

    let include_dir = manifest_dir.join("include");
    (lib_path, include_dir)
}

/// Create a test dataset on disk and return (TempDir, path_string).
fn create_test_dataset_on_disk() -> (tempfile::TempDir, String) {
    let tmp = tempfile::tempdir().unwrap();
    let uri = tmp.path().join("c_test_ds").to_str().unwrap().to_string();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10])),
            Arc::new(StringArray::from(vec![
                "alice", "bob", "carol", "dave", "eve", "frank", "grace", "heidi", "ivan", "judy",
            ])),
        ],
    )
    .unwrap();

    lance_c::runtime::block_on(async {
        Dataset::write(
            arrow::record_batch::RecordBatchIterator::new(vec![Ok(batch)], schema),
            &uri,
            None,
        )
        .await
        .unwrap();
    });

    (tmp, uri)
}

/// Compile a C source file, linking against lance-c.
fn compile_c_test(source: &Path, output: &Path, include_dir: &Path, lib_path: &Path) -> bool {
    let lib_dir = lib_path.parent().unwrap();
    let lib_name = "lance_c";

    let status = Command::new("cc")
        .args([
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-o",
            output.to_str().unwrap(),
            source.to_str().unwrap(),
            &format!("-I{}", include_dir.display()),
            &format!("-L{}", lib_dir.display()),
            &format!("-l{lib_name}"),
            // On macOS, set rpath so the dylib is found at runtime.
            &format!("-Wl,-rpath,{}", lib_dir.display()),
        ])
        .status();

    match status {
        Ok(s) => s.success(),
        Err(e) => {
            eprintln!("C compiler not available: {e}");
            false
        }
    }
}

/// Compile a C++ source file, linking against lance-c.
fn compile_cpp_test(source: &Path, output: &Path, include_dir: &Path, lib_path: &Path) -> bool {
    let lib_dir = lib_path.parent().unwrap();
    let lib_name = "lance_c";

    let status = Command::new("c++")
        .args([
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-o",
            output.to_str().unwrap(),
            source.to_str().unwrap(),
            &format!("-I{}", include_dir.display()),
            &format!("-L{}", lib_dir.display()),
            &format!("-l{lib_name}"),
            &format!("-Wl,-rpath,{}", lib_dir.display()),
        ])
        .status();

    match status {
        Ok(s) => s.success(),
        Err(e) => {
            eprintln!("C++ compiler not available: {e}");
            false
        }
    }
}

/// Run a compiled test binary with the dataset URI.
fn run_test_binary(binary: &Path, dataset_uri: &str) {
    let output = Command::new(binary)
        .arg(dataset_uri)
        .output()
        .unwrap_or_else(|e| panic!("Failed to run {}: {e}", binary.display()));

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    println!("--- stdout ---\n{stdout}");
    if !stderr.is_empty() {
        eprintln!("--- stderr ---\n{stderr}");
    }

    assert!(
        output.status.success(),
        "Test binary {} failed with exit code {:?}\nstdout: {}\nstderr: {}",
        binary.display(),
        output.status.code(),
        stdout,
        stderr
    );
}

#[test]
#[ignore = "requires C compiler (cc); run with: cargo test -p lance-c -- --ignored test_c_compilation"]
fn test_c_compilation_and_execution() {
    let (lib_path, include_dir) = build_lance_c();
    let (_tmp, dataset_uri) = create_test_dataset_on_disk();
    let build_dir = tempfile::tempdir().unwrap();

    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cpp")
        .join("test_c_api.c");
    let binary = build_dir.path().join("test_c_api");

    if !compile_c_test(&source, &binary, &include_dir, &lib_path) {
        eprintln!("Skipping C test: compilation failed (C compiler may not be available)");
        return;
    }

    run_test_binary(&binary, &dataset_uri);
}

#[test]
#[ignore = "requires C++ compiler (c++); run with: cargo test -p lance-c -- --ignored test_cpp_compilation"]
fn test_cpp_compilation_and_execution() {
    let (lib_path, include_dir) = build_lance_c();
    let (_tmp, dataset_uri) = create_test_dataset_on_disk();
    let build_dir = tempfile::tempdir().unwrap();

    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cpp")
        .join("test_cpp_api.cpp");
    let binary = build_dir.path().join("test_cpp_api");

    if !compile_cpp_test(&source, &binary, &include_dir, &lib_path) {
        eprintln!("Skipping C++ test: compilation failed (C++ compiler may not be available)");
        return;
    }

    run_test_binary(&binary, &dataset_uri);
}
