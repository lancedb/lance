// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#![cfg(feature = "tokenizer-plugin")]

use std::env;
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use lance_index::pbold::InvertedIndexDetails;
use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;
use lance_index::scalar::inverted::tokenizer::document_tokenizer::LanceTokenizer;
use lance_tokenizer::TokenStream;
use rstest::rstest;
use serial_test::serial;

static PLUGIN_PATH: OnceLock<PathBuf> = OnceLock::new();

fn get_plugin_path() -> PathBuf {
    PLUGIN_PATH
        .get_or_init(|| {
            let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
            let plugin_dir = PathBuf::from(&manifest_dir).join("tests/test_plugin");

            let output = Command::new("cargo")
                .args(["build", "--release"])
                .current_dir(&plugin_dir)
                .output()
                .expect("Failed to build test plugin");

            if !output.status.success() {
                panic!(
                    "Failed to build test plugin:\nstdout: {}\nstderr: {}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
            }

            let lib_name = if cfg!(target_os = "macos") {
                "liblance_test_tokenizer_plugin.dylib"
            } else if cfg!(target_os = "windows") {
                "lance_test_tokenizer_plugin.dll"
            } else {
                "liblance_test_tokenizer_plugin.so"
            };

            plugin_dir.join("target/release").join(lib_name)
        })
        .clone()
}

fn tokenize(tokenizer: &mut Box<dyn LanceTokenizer>, text: &str) -> Vec<String> {
    let mut stream = tokenizer.token_stream_for_doc(text);
    let mut tokens = Vec::new();
    while stream.advance() {
        tokens.push(stream.token().text.clone());
    }
    tokens
}

/// Plugin config must survive a proto roundtrip and produce identical tokens
/// after reopen. The `lower_case` filter case asserts plugin output flows
/// through the standard chain; the `preserve_case` case disables both the
/// plugin's own lowercase and the chain so case is preserved end-to-end —
/// any divergence between writer and reader would silently break search.
#[rstest]
#[case::lower_case_via_plugin(
    r#"{"lowercase": true}"#,
    true,
    "HELLO WORLD",
    &["hello", "world"],
)]
#[case::preserve_case_end_to_end(
    r#"{"lowercase": false}"#,
    false,
    "Hello World",
    &["Hello", "World"],
)]
#[serial(plugin_tests)]
fn test_plugin_config_survives_proto_roundtrip(
    #[case] plugin_config: &str,
    #[case] enable_filter_chain: bool,
    #[case] input: &str,
    #[case] expected: &[&str],
) {
    let plugin_path = get_plugin_path();
    let mut original_params = InvertedIndexParams::default().plugin(
        plugin_path.to_string_lossy().to_string(),
        plugin_config.to_string(),
    );
    if !enable_filter_chain {
        original_params = original_params
            .lower_case(false)
            .stem(false)
            .remove_stop_words(false)
            .ascii_folding(false);
    }

    let mut original_tokenizer = original_params.build().expect("build original");
    let original_tokens = tokenize(&mut original_tokenizer, input);
    assert_eq!(original_tokens, expected);

    let proto: InvertedIndexDetails = (&original_params).try_into().expect("to proto");
    assert_eq!(
        proto.tokenizer_plugin_library.as_deref(),
        Some(plugin_path.to_string_lossy().as_ref())
    );
    assert_eq!(
        proto.tokenizer_plugin_config.as_deref(),
        Some(plugin_config)
    );

    let restored_params: InvertedIndexParams = (&proto).try_into().expect("from proto");
    let mut restored_tokenizer = restored_params.build().expect("build restored");
    assert_eq!(tokenize(&mut restored_tokenizer, input), original_tokens);
}

/// Old indexes saved before the plugin fields existed (or any current index
/// whose `base_tokenizer != "plugin"`) must still load and build cleanly.
/// Regression guard for the manifest-compat path.
#[test]
#[serial(plugin_tests)]
fn test_proto_without_plugin_fields_still_builds() {
    let proto = InvertedIndexDetails {
        base_tokenizer: Some("simple".to_string()),
        language: r#""English""#.to_string(),
        tokenizer_plugin_library: None,
        tokenizer_plugin_config: None,
        ..Default::default()
    };
    let params: InvertedIndexParams = (&proto).try_into().expect("from proto");
    let mut tokenizer = params.build().expect("build");
    assert!(!tokenize(&mut tokenizer, "Hello World").is_empty());
}

/// A proto with `base_tokenizer = "plugin"` and a missing `tokenizer_plugin_config`
/// must surface as `Err` from `build()`, not a panic later.
#[test]
#[serial(plugin_tests)]
fn test_plugin_proto_without_config_fails_build() {
    let plugin_path = get_plugin_path();

    let proto = InvertedIndexDetails {
        base_tokenizer: Some("plugin".to_string()),
        language: r#""English""#.to_string(),
        tokenizer_plugin_library: Some(plugin_path.to_string_lossy().to_string()),
        tokenizer_plugin_config: None,
        ..Default::default()
    };
    let params: InvertedIndexParams = (&proto).try_into().expect("from proto");
    let err = params.build().expect_err("build must fail without config");
    assert!(
        err.to_string()
            .contains("tokenizer_plugin_config is not set"),
        "got: {}",
        err
    );
}
