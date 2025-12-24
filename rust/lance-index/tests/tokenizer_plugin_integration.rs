// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#![cfg(feature = "tokenizer-plugin")]

use std::env;
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use lance_index::scalar::inverted::tokenizer::lance_tokenizer::LanceTokenizer;
use lance_index::scalar::inverted::tokenizer::plugin::{PluginTokenizer, TokenizerPluginLibrary};
use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;
use tantivy::tokenizer::TokenStream;

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

            let lib_path = plugin_dir.join("target/release").join(lib_name);

            if !lib_path.exists() {
                panic!("Plugin library not found at {:?}", lib_path);
            }

            lib_path
        })
        .clone()
}

#[test]
fn test_plugin_library_load() {
    let plugin_path = get_plugin_path();

    let library = TokenizerPluginLibrary::load(&plugin_path).expect("Failed to load plugin");

    assert_eq!(library.name(), "test_whitespace_tokenizer");
    assert_eq!(library.version(), "0.1.0");
}

#[test]
fn test_plugin_tokenizer_basic() {
    let plugin_path = get_plugin_path();

    let mut tokenizer =
        PluginTokenizer::new(&plugin_path, "{}").expect("Failed to create tokenizer");

    assert_eq!(tokenizer.plugin_name(), "test_whitespace_tokenizer");
    assert_eq!(tokenizer.plugin_version(), "0.1.0");

    // Test tokenization
    let mut stream = tokenizer.token_stream_for_doc("Hello World");
    let mut tokens = Vec::new();
    while stream.advance() {
        tokens.push((
            stream.token().text.clone(),
            stream.token().position,
            stream.token().offset_from,
            stream.token().offset_to,
        ));
    }

    assert_eq!(tokens.len(), 2);
    assert_eq!(tokens[0].0, "Hello");
    assert_eq!(tokens[0].1, 0);
    assert_eq!(tokens[0].2, 0);
    assert_eq!(tokens[0].3, 5);
    assert_eq!(tokens[1].0, "World");
    assert_eq!(tokens[1].1, 1);
    assert_eq!(tokens[1].2, 6);
    assert_eq!(tokens[1].3, 11);
}

#[test]
fn test_plugin_tokenizer_lowercase() {
    let plugin_path = get_plugin_path();

    let mut tokenizer = PluginTokenizer::new(&plugin_path, r#"{"lowercase": true}"#)
        .expect("Failed to create tokenizer");

    let mut stream = tokenizer.token_stream_for_doc("Hello WORLD");
    let mut tokens = Vec::new();
    while stream.advance() {
        tokens.push(stream.token().text.clone());
    }

    assert_eq!(tokens.len(), 2);
    assert_eq!(tokens[0], "hello");
    assert_eq!(tokens[1], "world");
}

#[test]
fn test_plugin_tokenizer_empty_text() {
    let plugin_path = get_plugin_path();

    let mut tokenizer =
        PluginTokenizer::new(&plugin_path, "{}").expect("Failed to create tokenizer");

    let mut stream = tokenizer.token_stream_for_doc("");
    let mut count = 0;
    while stream.advance() {
        count += 1;
    }

    assert_eq!(count, 0);
}

#[test]
fn test_plugin_tokenizer_multiple_spaces() {
    let plugin_path = get_plugin_path();

    let mut tokenizer =
        PluginTokenizer::new(&plugin_path, "{}").expect("Failed to create tokenizer");

    let mut stream = tokenizer.token_stream_for_doc("  Hello   World  ");
    let mut tokens = Vec::new();
    while stream.advance() {
        tokens.push((stream.token().text.clone(), stream.token().offset_from));
    }

    assert_eq!(tokens.len(), 2);
    assert_eq!(tokens[0].0, "Hello");
    assert_eq!(tokens[0].1, 2);
    assert_eq!(tokens[1].0, "World");
    assert_eq!(tokens[1].1, 10);
}

#[test]
fn test_plugin_tokenizer_search_stream() {
    let plugin_path = get_plugin_path();

    let mut tokenizer =
        PluginTokenizer::new(&plugin_path, "{}").expect("Failed to create tokenizer");

    let mut stream = tokenizer.token_stream_for_search("query terms");
    let mut tokens = Vec::new();
    while stream.advance() {
        tokens.push(stream.token().text.clone());
    }

    assert_eq!(tokens.len(), 2);
    assert_eq!(tokens[0], "query");
    assert_eq!(tokens[1], "terms");
}

#[test]
fn test_plugin_tokenizer_clone() {
    let plugin_path = get_plugin_path();

    let tokenizer = PluginTokenizer::new(&plugin_path, r#"{"lowercase": true}"#)
        .expect("Failed to create tokenizer");

    let mut cloned = tokenizer.box_clone();

    let mut stream = cloned.token_stream_for_doc("HELLO");
    let mut tokens = Vec::new();
    while stream.advance() {
        tokens.push(stream.token().text.clone());
    }

    assert_eq!(tokens.len(), 1);
    assert_eq!(tokens[0], "hello");
}

#[test]
fn test_inverted_index_params_with_plugin() {
    let plugin_path = get_plugin_path();

    let params = InvertedIndexParams::default().plugin(
        plugin_path.to_string_lossy().to_string(),
        Some(r#"{"lowercase": true}"#.to_string()),
    );

    let mut tokenizer = params
        .build()
        .expect("Failed to build tokenizer from params");

    let mut stream = tokenizer.token_stream_for_doc("Test Document");
    let mut tokens = Vec::new();
    while stream.advance() {
        tokens.push(stream.token().text.clone());
    }

    assert_eq!(tokens.len(), 2);
    assert_eq!(tokens[0], "test");
    assert_eq!(tokens[1], "document");
}

#[test]
fn test_plugin_load_nonexistent() {
    let result = TokenizerPluginLibrary::load("/nonexistent/path/to/plugin.so");
    assert!(result.is_err());
}

#[test]
fn test_plugin_unicode_text() {
    let plugin_path = get_plugin_path();

    let mut tokenizer =
        PluginTokenizer::new(&plugin_path, "{}").expect("Failed to create tokenizer");

    let mut stream = tokenizer.token_stream_for_doc("Hello 世界 Rust");
    let mut tokens = Vec::new();
    while stream.advance() {
        tokens.push((
            stream.token().text.clone(),
            stream.token().offset_from,
            stream.token().offset_to,
        ));
    }

    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[0].0, "Hello");
    assert_eq!(tokens[1].0, "世界");
    assert_eq!(tokens[2].0, "Rust");
}

#[test]
fn test_plugin_long_text() {
    let plugin_path = get_plugin_path();

    let mut tokenizer =
        PluginTokenizer::new(&plugin_path, "{}").expect("Failed to create tokenizer");

    let words: Vec<&str> = (0..100).map(|_| "word").collect();
    let long_text = words.join(" ");

    let mut stream = tokenizer.token_stream_for_doc(&long_text);
    let mut count = 0;
    while stream.advance() {
        count += 1;
    }

    assert_eq!(count, 100);
}

#[test]
fn test_plugin_error_after_zero_tokens() {
    let plugin_path = get_plugin_path();

    // Configure to error immediately (after 0 tokens)
    let mut tokenizer = PluginTokenizer::new(&plugin_path, r#"{"error_after_n_tokens": 0}"#)
        .expect("Failed to create tokenizer");

    // Should produce no tokens due to error
    let mut stream = tokenizer.token_stream_for_doc("Hello World");
    let mut count = 0;
    while stream.advance() {
        count += 1;
    }

    assert_eq!(
        count, 0,
        "Should produce no tokens when error occurs at start"
    );
}

#[test]
fn test_plugin_error_after_one_token() {
    let plugin_path = get_plugin_path();

    // Configure to error after 1 token
    let mut tokenizer = PluginTokenizer::new(&plugin_path, r#"{"error_after_n_tokens": 1}"#)
        .expect("Failed to create tokenizer");

    // Error discards all tokens
    let mut stream = tokenizer.token_stream_for_doc("Hello World Test");
    let mut tokens = Vec::new();
    while stream.advance() {
        tokens.push(stream.token().text.clone());
    }

    assert_eq!(
        tokens.len(),
        0,
        "Should produce no tokens when error occurs"
    );
}

#[test]
fn test_plugin_error_after_two_tokens() {
    let plugin_path = get_plugin_path();

    // Configure to error after 2 tokens
    let mut tokenizer = PluginTokenizer::new(&plugin_path, r#"{"error_after_n_tokens": 2}"#)
        .expect("Failed to create tokenizer");

    // Error discards all tokens
    let mut stream = tokenizer.token_stream_for_doc("one two three four five");
    let mut tokens = Vec::new();
    while stream.advance() {
        tokens.push(stream.token().text.clone());
    }

    assert_eq!(
        tokens.len(),
        0,
        "Should produce no tokens when error occurs"
    );
}

#[test]
fn test_plugin_error_not_triggered_when_fewer_tokens() {
    let plugin_path = get_plugin_path();

    // Configure to error after 10 tokens, but input has only 2
    let mut tokenizer = PluginTokenizer::new(&plugin_path, r#"{"error_after_n_tokens": 10}"#)
        .expect("Failed to create tokenizer");

    // Should produce all tokens since error threshold not reached
    let mut stream = tokenizer.token_stream_for_doc("Hello World");
    let mut tokens = Vec::new();
    while stream.advance() {
        tokens.push(stream.token().text.clone());
    }

    assert_eq!(
        tokens.len(),
        2,
        "Should produce all tokens when error threshold not reached"
    );
    assert_eq!(tokens[0], "Hello");
    assert_eq!(tokens[1], "World");
}

#[test]
fn test_plugin_error_with_inverted_index_params() {
    let plugin_path = get_plugin_path();

    // Test error propagation through InvertedIndexParams interface
    let params = InvertedIndexParams::default().plugin(
        plugin_path.to_string_lossy().to_string(),
        Some(r#"{"error_after_n_tokens": 1}"#.to_string()),
    );

    let mut tokenizer = params
        .build()
        .expect("Failed to build tokenizer from params");

    let mut stream = tokenizer.token_stream_for_doc("one two three");
    let mut tokens = Vec::new();
    while stream.advance() {
        tokens.push(stream.token().text.clone());
    }

    assert_eq!(
        tokens.len(),
        0,
        "Error should be propagated through InvertedIndexParams"
    );
}
