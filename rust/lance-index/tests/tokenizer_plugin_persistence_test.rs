// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#![cfg(feature = "tokenizer-plugin")]

use std::env;
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;
use lance_index::scalar::inverted::tokenizer::document_tokenizer::LanceTokenizer;
use lance_tokenizer::TokenStream;
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

/// Helper to tokenize text and return token strings
fn tokenize(tokenizer: &mut Box<dyn LanceTokenizer>, text: &str) -> Vec<String> {
    let mut stream = tokenizer.token_stream_for_doc(text);
    let mut tokens = Vec::new();
    while stream.advance() {
        tokens.push(stream.token().text.clone());
    }
    tokens
}

/// Test that plugin configuration survives protobuf roundtrip.
///
/// This verifies that:
/// 1. Plugin settings are serialized to protobuf
/// 2. Plugin settings are deserialized from protobuf
/// 3. The restored tokenizer behaves identically to the original
#[test]
#[serial(plugin_tests)]
fn test_tokenizer_plugin_config_preserved_after_protobuf_roundtrip() {
    use lance_index::pbold::InvertedIndexDetails;

    let plugin_path = get_plugin_path();

    // Step 1: Create params with plugin (lowercase enabled)
    let original_params = InvertedIndexParams::default().plugin(
        plugin_path.to_string_lossy().to_string(),
        r#"{"lowercase": true}"#.to_string(),
    );

    // Verify original params work correctly with plugin
    let mut original_tokenizer = original_params
        .build()
        .expect("Failed to build original tokenizer");
    let original_tokens = tokenize(&mut original_tokenizer, "HELLO WORLD");
    assert_eq!(
        original_tokens,
        vec!["hello", "world"],
        "Plugin tokenizer should lowercase tokens"
    );

    // Step 2: Serialize to protobuf (simulating index save)
    let proto: InvertedIndexDetails = (&original_params)
        .try_into()
        .expect("Failed to convert to protobuf");

    // Verify protobuf contains plugin fields
    assert_eq!(
        proto.tokenizer_plugin_library,
        Some(plugin_path.to_string_lossy().to_string()),
        "Protobuf should contain tokenizer_plugin_library"
    );
    assert_eq!(
        proto.tokenizer_plugin_config,
        Some(r#"{"lowercase": true}"#.to_string()),
        "Protobuf should contain tokenizer_plugin_config"
    );

    // Step 3: Deserialize from protobuf (simulating index reopen)
    let restored_params: InvertedIndexParams = (&proto)
        .try_into()
        .expect("Failed to convert from protobuf");

    // Step 4: Build tokenizer from restored params - should use the plugin
    let mut restored_tokenizer = restored_params
        .build()
        .expect("Failed to build restored tokenizer");

    // Step 5: Verify restored tokenizer behaves identically to original
    let restored_tokens = tokenize(&mut restored_tokenizer, "HELLO WORLD");
    assert_eq!(
        restored_tokens, original_tokens,
        "Restored tokenizer should produce identical tokens to original"
    );

    // Verify both JSON representations contain the plugin path
    let original_json = serde_json::to_string(&original_params).unwrap();
    let restored_json = serde_json::to_string(&restored_params).unwrap();
    assert!(
        original_json.contains(&plugin_path.to_string_lossy().to_string()),
        "Original JSON should contain plugin path"
    );
    assert!(
        restored_json.contains(&plugin_path.to_string_lossy().to_string()),
        "Restored JSON should contain plugin path (persistence working)"
    );
}

/// Test that tokens match after index reopen with plugin.
///
/// This verifies that search will work correctly after reopening an index
/// created with a plugin tokenizer.
#[test]
#[serial(plugin_tests)]
fn test_tokens_match_after_reopen_with_plugin() {
    use lance_index::pbold::InvertedIndexDetails;

    let plugin_path = get_plugin_path();

    // Create params with plugin that does NOT lowercase, and disable the
    // InvertedIndexParams filter chain so that case is preserved end-to-end.
    let original_params = InvertedIndexParams::default()
        .plugin(
            plugin_path.to_string_lossy().to_string(),
            r#"{"lowercase": false}"#.to_string(),
        )
        .lower_case(false)
        .stem(false)
        .remove_stop_words(false)
        .ascii_folding(false);

    // Index document with plugin tokenizer (preserves case)
    let mut index_tokenizer = original_params
        .build()
        .expect("Failed to build index tokenizer");
    let indexed_tokens = tokenize(&mut index_tokenizer, "Hello World");
    assert_eq!(
        indexed_tokens,
        vec!["Hello", "World"],
        "Plugin should preserve case when lowercase=false and filters are disabled"
    );

    // Simulate index save/reopen via protobuf
    let proto: InvertedIndexDetails = (&original_params)
        .try_into()
        .expect("Failed to convert to protobuf");
    let restored_params: InvertedIndexParams = (&proto)
        .try_into()
        .expect("Failed to convert from protobuf");

    // After reopen, should still use plugin with same config
    let mut search_tokenizer = restored_params
        .build()
        .expect("Failed to build search tokenizer");
    let search_tokens = tokenize(&mut search_tokenizer, "Hello World");

    // Tokens should match - search will work correctly
    assert_eq!(
        indexed_tokens, search_tokens,
        "Indexed tokens and search tokens should match after index reopen"
    );
}

/// Verify that protobuf schema has fields for plugin configuration.
#[test]
#[serial(plugin_tests)]
fn test_protobuf_schema_has_plugin_fields() {
    use lance_index::pbold::InvertedIndexDetails;

    let plugin_path = get_plugin_path();

    // Create params with plugin
    let params = InvertedIndexParams::default().plugin(
        plugin_path.to_string_lossy().to_string(),
        r#"{"mode": "test"}"#.to_string(),
    );

    // Convert to protobuf
    let proto: InvertedIndexDetails = (&params).try_into().expect("Failed to convert to protobuf");

    // Verify plugin fields are present in protobuf
    assert!(
        proto.tokenizer_plugin_library.is_some(),
        "Protobuf should have tokenizer_plugin_library field"
    );
    assert!(
        proto.tokenizer_plugin_config.is_some(),
        "Protobuf should have tokenizer_plugin_config field"
    );
    assert_eq!(
        proto.tokenizer_plugin_library.as_ref().unwrap(),
        &plugin_path.to_string_lossy().to_string()
    );
    assert_eq!(
        proto.tokenizer_plugin_config.as_ref().unwrap(),
        r#"{"mode": "test"}"#
    );
}

/// Test backward compatibility: old indexes without plugin fields should work.
#[test]
#[serial(plugin_tests)]
fn test_backward_compatibility_no_plugin_fields() {
    use lance_index::pbold::InvertedIndexDetails;

    // Simulate an old protobuf message without plugin fields
    let proto = InvertedIndexDetails {
        base_tokenizer: Some("simple".to_string()),
        language: "\"English\"".to_string(),
        with_position: false,
        max_token_length: Some(40),
        lower_case: true,
        stem: true,
        remove_stop_words: true,
        ascii_folding: true,
        min_ngram_length: 3,
        max_ngram_length: 3,
        prefix_only: false,
        tokenizer_plugin_library: None, // No plugin in old index
        tokenizer_plugin_config: None,
    };

    // Should successfully convert to params
    let params: InvertedIndexParams = (&proto)
        .try_into()
        .expect("Failed to convert from protobuf");

    // Should build default tokenizer without error
    let mut tokenizer = params.build().expect("Failed to build tokenizer");

    // Default tokenizer should work
    let tokens = tokenize(&mut tokenizer, "Hello World");
    assert!(
        !tokens.is_empty(),
        "Default tokenizer should produce tokens"
    );
}

/// Test that missing plugin config causes an error.
#[test]
#[serial(plugin_tests)]
fn test_plugin_requires_config() {
    use lance_index::pbold::InvertedIndexDetails;

    let plugin_path = get_plugin_path();

    // Create params with plugin and explicit empty JSON config
    let original_params = InvertedIndexParams::default()
        .plugin(plugin_path.to_string_lossy().to_string(), "{}".to_string());

    // Roundtrip through protobuf
    let proto: InvertedIndexDetails = (&original_params)
        .try_into()
        .expect("Failed to convert to protobuf");

    assert!(proto.tokenizer_plugin_library.is_some());
    assert_eq!(proto.tokenizer_plugin_config, Some("{}".to_string()));

    let restored_params: InvertedIndexParams = (&proto)
        .try_into()
        .expect("Failed to convert from protobuf");

    // Should work with explicit empty JSON config. Default params apply
    // LowerCaser, so the expected tokens are lowercased.
    let mut tokenizer = restored_params.build().expect("Failed to build tokenizer");
    let tokens = tokenize(&mut tokenizer, "Hello World");
    assert_eq!(tokens, vec!["hello", "world"]);

    // Test that missing config causes an error (construct via proto)
    let proto_without_config = InvertedIndexDetails {
        base_tokenizer: Some("plugin".to_string()),
        language: r#""English""#.to_string(), // Required field
        tokenizer_plugin_library: Some(plugin_path.to_string_lossy().to_string()),
        tokenizer_plugin_config: None,
        ..Default::default()
    };
    let params_without_config: InvertedIndexParams = (&proto_without_config)
        .try_into()
        .expect("Failed to convert from protobuf");
    let result = params_without_config.build();
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(
        err.to_string()
            .contains("tokenizer_plugin_config is not set"),
        "Expected error about missing config, got: {}",
        err
    );
}
