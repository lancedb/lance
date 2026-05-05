// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#![cfg(feature = "tokenizer-plugin")]

use std::env;
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;
use lance_index::scalar::inverted::tokenizer::document_tokenizer::LanceTokenizer;
use lance_index::scalar::inverted::tokenizer::plugin::{PluginTokenizer, TokenizerPluginLibrary};
use lance_tokenizer::TokenStream;
use libloading::Library;
use serial_test::serial;

static PLUGIN_PATH: OnceLock<PathBuf> = OnceLock::new();
static PLUGIN_LIBRARY: OnceLock<Library> = OnceLock::new();

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

/// Get a reference to the loaded plugin library for calling test helper functions.
fn get_plugin_library() -> &'static Library {
    PLUGIN_LIBRARY.get_or_init(|| {
        let plugin_path = get_plugin_path();
        unsafe { Library::new(&plugin_path).expect("Failed to load plugin library") }
    })
}

/// Get the number of times create_factory has been called in the test plugin.
fn get_factory_create_count() -> u32 {
    let library = get_plugin_library();
    unsafe {
        let func: libloading::Symbol<extern "C" fn() -> u32> = library
            .get(b"lance_test_get_factory_create_count")
            .expect("Failed to get lance_test_get_factory_create_count");
        func()
    }
}

/// Reset the factory create counter to zero.
fn reset_factory_create_count() {
    let library = get_plugin_library();
    unsafe {
        let func: libloading::Symbol<extern "C" fn()> = library
            .get(b"lance_test_reset_factory_create_count")
            .expect("Failed to get lance_test_reset_factory_create_count");
        func()
    }
}

#[test]
#[serial(plugin_tests)]
fn test_plugin_library_load() {
    let plugin_path = get_plugin_path();

    let library = TokenizerPluginLibrary::load(&plugin_path).expect("Failed to load plugin");

    assert_eq!(library.name(), "test_whitespace_tokenizer");
    assert_eq!(library.version(), "0.1.0");
}

#[test]
#[serial(plugin_tests)]
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
#[serial(plugin_tests)]
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
#[serial(plugin_tests)]
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
#[serial(plugin_tests)]
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
#[serial(plugin_tests)]
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
#[serial(plugin_tests)]
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
#[serial(plugin_tests)]
fn test_inverted_index_params_with_plugin() {
    let plugin_path = get_plugin_path();

    let params = InvertedIndexParams::default().plugin(
        plugin_path.to_string_lossy().to_string(),
        r#"{"lowercase": true}"#.to_string(),
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
#[serial(plugin_tests)]
fn test_plugin_output_flows_through_filter_chain() {
    // Regression test for the issue where plugin tokenizers bypassed the
    // InvertedIndexParams filter chain (lower_case / stem / remove_stop_words /
    // ascii_folding / max_token_length) and the doc-type wrapping. Plugin
    // tokenizers must produce the same final output as Lindera/Jieba when the
    // same params are configured. The plugin is invoked with `{}` so it returns
    // raw whitespace tokens with no normalization applied; all normalization
    // here must come from InvertedIndexParams filters.
    let plugin_path = get_plugin_path();

    let params = InvertedIndexParams::default()
        .plugin(plugin_path.to_string_lossy().to_string(), "{}".to_string())
        .lower_case(true)
        .stem(true)
        .remove_stop_words(true)
        .ascii_folding(true);

    let mut tokenizer = params
        .build()
        .expect("Failed to build tokenizer from params");

    let mut stream = tokenizer.token_stream_for_doc("The Quick Brown Foxes");
    let mut tokens = Vec::new();
    while stream.advance() {
        tokens.push(stream.token().text.clone());
    }

    // Expected pipeline:
    //   plugin (raw whitespace) -> ["The", "Quick", "Brown", "Foxes"]
    //   LowerCaser              -> ["the", "quick", "brown", "foxes"]
    //   Stemmer(English)        -> ["the", "quick", "brown", "fox"]
    //   StopWordFilter(English) -> ["quick", "brown", "fox"]    ("the" removed)
    assert_eq!(tokens, vec!["quick", "brown", "fox"]);
}

#[test]
#[serial(plugin_tests)]
fn test_plugin_max_token_length_applied() {
    // Regression test: max_token_length must be applied to plugin output.
    let plugin_path = get_plugin_path();

    let params = InvertedIndexParams::default()
        .plugin(plugin_path.to_string_lossy().to_string(), "{}".to_string())
        .lower_case(false)
        .stem(false)
        .remove_stop_words(false)
        .ascii_folding(false)
        .max_token_length(Some(5));

    let mut tokenizer = params
        .build()
        .expect("Failed to build tokenizer from params");

    let mut stream = tokenizer.token_stream_for_doc("a abcdef ab abcdefghi");
    let mut tokens = Vec::new();
    while stream.advance() {
        tokens.push(stream.token().text.clone());
    }

    // RemoveLongFilter removes tokens longer than 5 chars.
    assert_eq!(tokens, vec!["a", "ab"]);
}

#[test]
#[serial(plugin_tests)]
fn test_plugin_load_nonexistent() {
    let result = TokenizerPluginLibrary::load("/nonexistent/path/to/plugin.so");
    assert!(result.is_err());
}

#[test]
#[serial(plugin_tests)]
fn test_plugin_constructor_rejects_invalid_config() {
    // A malformed config must surface as a regular `Err` from
    // `PluginTokenizer::new`, not as a panic on first tokenization. The test
    // plugin returns a NULL factory + error message when the config contains
    // "reject_config": true.
    let plugin_path = get_plugin_path();
    let result = PluginTokenizer::new(&plugin_path, r#"{"reject_config": true}"#);
    let err = result.expect_err("constructor must reject invalid config");
    let msg = err.to_string();
    assert!(
        msg.contains("simulated config rejection"),
        "error must propagate the plugin's message, got: {}",
        msg
    );
}

#[test]
#[serial(plugin_tests)]
fn test_inverted_index_params_build_rejects_invalid_plugin_config() {
    // Same contract going through the public `InvertedIndexParams::build()`
    // path: the user must see the error here, not later during tokenization.
    let plugin_path = get_plugin_path();
    let params = InvertedIndexParams::default().plugin(
        plugin_path.to_string_lossy().to_string(),
        r#"{"reject_config": true}"#.to_string(),
    );
    let err = params
        .build()
        .expect_err("build must reject invalid config");
    assert!(
        err.to_string().contains("simulated config rejection"),
        "error must propagate the plugin's message, got: {}",
        err
    );
}

#[test]
#[serial(plugin_tests)]
#[should_panic(expected = "invalid UTF-8")]
fn test_plugin_invalid_utf8_token_is_rejected() {
    // The plugin ABI requires UTF-8 token text. A buggy plugin that returns
    // non-UTF-8 bytes must be rejected loudly, otherwise corrupted FTS terms
    // would be silently persisted with replacement characters substituted.
    let plugin_path = get_plugin_path();
    let mut tokenizer = PluginTokenizer::new(&plugin_path, r#"{"emit_invalid_utf8": true}"#)
        .expect("Failed to create tokenizer");
    let mut stream = tokenizer.token_stream_for_doc("hello");
    while stream.advance() {}
}

#[test]
#[serial(plugin_tests)]
fn test_plugin_relative_path_is_absolutized_before_persist() {
    // A relative `tokenizer_plugin_library` would be re-interpreted against
    // the CWD of whichever process later reopens the index, which can fail
    // to find the plugin or load a wrong file. The persisted (proto) form
    // must be absolute regardless of what the caller passed in.
    use lance_index::pbold::InvertedIndexDetails;

    let params = InvertedIndexParams::default()
        .plugin("relative/path/to/plugin.so".to_string(), "{}".to_string());
    let proto: InvertedIndexDetails = (&params).try_into().expect("proto conversion failed");
    let proto_path = proto
        .tokenizer_plugin_library
        .as_ref()
        .expect("proto plugin path must be set");
    assert!(
        std::path::Path::new(proto_path).is_absolute(),
        "proto-persisted plugin path must be absolute; got {:?}",
        proto_path
    );
    // Round-trip back into params and confirm the path stays absolute (so a
    // second save doesn't reintroduce relativity).
    let restored: InvertedIndexParams = (&proto).try_into().expect("proto -> params failed");
    let proto2: InvertedIndexDetails = (&restored).try_into().expect("re-serialize failed");
    let proto2_path = proto2
        .tokenizer_plugin_library
        .as_ref()
        .expect("re-serialized plugin path must be set");
    assert!(
        std::path::Path::new(proto2_path).is_absolute(),
        "round-tripped plugin path must remain absolute; got {:?}",
        proto2_path
    );
}

#[test]
#[serial(plugin_tests)]
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
#[serial(plugin_tests)]
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

/// Drive the token stream until either it ends or the plugin panics. Returns
/// the tokens collected before the panic, plus whether a panic occurred. This
/// proves both:
///   1. tokenization is streamed (we observe partial results before the panic
///      reaches us, so the adapter is not buffering the whole document); and
///   2. plugin errors propagate as panics, so a misbehaving plugin cannot
///      silently truncate index/search input.
fn collect_until_panic(tokenizer: &mut PluginTokenizer, text: &str) -> (Vec<String>, bool) {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::Mutex;

    let collected: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut stream = tokenizer.token_stream_for_doc(text);
        while stream.advance() {
            collected.lock().unwrap().push(stream.token().text.clone());
        }
    }));
    (collected.into_inner().unwrap(), result.is_err())
}

#[test]
#[serial(plugin_tests)]
fn test_plugin_error_after_zero_tokens() {
    let plugin_path = get_plugin_path();
    let mut tokenizer = PluginTokenizer::new(&plugin_path, r#"{"error_after_n_tokens": 0}"#)
        .expect("Failed to create tokenizer");

    let (tokens, panicked) = collect_until_panic(&mut tokenizer, "Hello World");

    assert!(
        panicked,
        "Plugin error must surface as a panic so the FTS pipeline can detect it",
    );
    assert!(
        tokens.is_empty(),
        "no tokens should be observed before the immediate error"
    );
}

#[test]
#[serial(plugin_tests)]
fn test_plugin_error_after_one_token() {
    let plugin_path = get_plugin_path();
    let mut tokenizer = PluginTokenizer::new(&plugin_path, r#"{"error_after_n_tokens": 1}"#)
        .expect("Failed to create tokenizer");

    let (tokens, panicked) = collect_until_panic(&mut tokenizer, "Hello World Test");

    assert!(panicked, "Plugin error must surface as a panic");
    // The first token must be observable to the caller before the panic — this
    // is the streaming contract. Buffered (collect-then-return) implementations
    // would have lost this token because the error follows it in the stream.
    assert_eq!(tokens, vec!["Hello"]);
}

#[test]
#[serial(plugin_tests)]
fn test_plugin_error_after_two_tokens() {
    let plugin_path = get_plugin_path();
    let mut tokenizer = PluginTokenizer::new(&plugin_path, r#"{"error_after_n_tokens": 2}"#)
        .expect("Failed to create tokenizer");

    let (tokens, panicked) = collect_until_panic(&mut tokenizer, "one two three four five");

    assert!(panicked, "Plugin error must surface as a panic");
    assert_eq!(tokens, vec!["one", "two"]);
}

#[test]
#[serial(plugin_tests)]
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
#[serial(plugin_tests)]
#[should_panic(expected = "Plugin tokenizer error during tokenization")]
fn test_plugin_error_with_inverted_index_params() {
    // The same fail-loud contract must hold when the plugin is wrapped by the
    // full InvertedIndexParams pipeline (filter chain + TextTokenizer).
    let plugin_path = get_plugin_path();
    let params = InvertedIndexParams::default().plugin(
        plugin_path.to_string_lossy().to_string(),
        r#"{"error_after_n_tokens": 1}"#.to_string(),
    );
    let mut tokenizer = params
        .build()
        .expect("Failed to build tokenizer from params");
    let mut stream = tokenizer.token_stream_for_doc("one two three");
    while stream.advance() {}
}

#[test]
#[serial(plugin_tests)]
fn test_plugin_multithread_with_clones() {
    use std::sync::Arc;
    use std::thread;

    let plugin_path = get_plugin_path();

    // Create the original tokenizer
    let tokenizer = PluginTokenizer::new(&plugin_path, "{}").expect("Failed to create tokenizer");

    // Wrap in Arc for sharing (we'll clone the tokenizer, not share the Arc)
    let tokenizer = Arc::new(tokenizer);

    let handles: Vec<_> = (0..4)
        .map(|i| {
            let mut thread_tokenizer = (*tokenizer).clone();
            thread::spawn(move || {
                let text = format!("thread {} test data", i);
                let mut stream = thread_tokenizer.token_stream_for_doc(&text);
                let mut tokens = Vec::new();
                while stream.advance() {
                    tokens.push(stream.token().text.clone());
                }
                (i, tokens)
            })
        })
        .collect();

    for handle in handles {
        let (thread_id, tokens) = handle.join().expect("Thread panicked");
        assert_eq!(
            tokens.len(),
            4,
            "Thread {} should produce 4 tokens",
            thread_id
        );
        assert_eq!(tokens[0], "thread");
        assert_eq!(tokens[1], thread_id.to_string());
        assert_eq!(tokens[2], "test");
        assert_eq!(tokens[3], "data");
    }
}

#[test]
#[serial(plugin_tests)]
fn test_plugin_multithread_repeated_tokenization() {
    use std::thread;

    let plugin_path = get_plugin_path();

    // Test that the cached factory is reused correctly within each thread
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let path = plugin_path.clone();
            thread::spawn(move || {
                let mut tokenizer =
                    PluginTokenizer::new(&path, "{}").expect("Failed to create tokenizer");

                // Tokenize multiple times to test factory caching
                let mut all_results = Vec::new();
                for j in 0..10 {
                    let text = format!("iteration {}", j);
                    let mut stream = tokenizer.token_stream_for_doc(&text);
                    let mut tokens = Vec::new();
                    while stream.advance() {
                        tokens.push(stream.token().text.clone());
                    }
                    all_results.push(tokens);
                }

                (i, all_results)
            })
        })
        .collect();

    for handle in handles {
        let (thread_id, results) = handle.join().expect("Thread panicked");
        assert_eq!(
            results.len(),
            10,
            "Thread {} should have 10 iterations",
            thread_id
        );
        for (j, tokens) in results.iter().enumerate() {
            assert_eq!(
                tokens.len(),
                2,
                "Thread {} iteration {} should have 2 tokens",
                thread_id,
                j
            );
            assert_eq!(tokens[0], "iteration");
            assert_eq!(tokens[1], j.to_string());
        }
    }
}

#[test]
#[serial(plugin_tests)]
fn test_plugin_multithread_with_lance_tokenizer_trait() {
    use std::thread;

    let plugin_path = get_plugin_path();

    // Test using Box<dyn LanceTokenizer> which is the typical usage pattern
    let tokenizer: Box<dyn LanceTokenizer> =
        Box::new(PluginTokenizer::new(&plugin_path, r#"{"lowercase": true}"#).unwrap());

    let handles: Vec<_> = (0..4)
        .map(|_| {
            // Clone using the trait method
            let mut thread_tokenizer = tokenizer.box_clone();
            thread::spawn(move || {
                let mut stream = thread_tokenizer.token_stream_for_doc("HELLO World");
                let mut tokens = Vec::new();
                while stream.advance() {
                    tokens.push(stream.token().text.clone());
                }
                tokens
            })
        })
        .collect();

    for handle in handles {
        let tokens = handle.join().expect("Thread panicked");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0], "hello");
        assert_eq!(tokens[1], "world");
    }
}

#[test]
#[serial(plugin_tests)]
fn test_factory_caching_behavior() {
    // Reset counter, then construct the tokenizer. Factory creation now
    // happens eagerly in `PluginTokenizer::new` so a malformed config can
    // surface as a regular `Err` instead of a panic during tokenization.
    reset_factory_create_count();

    let plugin_path = get_plugin_path();

    let count_before_new = get_factory_create_count();
    let mut tokenizer =
        PluginTokenizer::new(&plugin_path, "{}").expect("Failed to create tokenizer");
    let count_after_new = get_factory_create_count();
    assert_eq!(
        count_after_new - count_before_new,
        1,
        "Factory should be created eagerly during PluginTokenizer::new"
    );

    // Subsequent tokenizations must reuse the cached factory (no new
    // create_factory calls).
    let count_before = get_factory_create_count();
    {
        let mut stream = tokenizer.token_stream_for_doc("hello world");
        let mut tokens = Vec::new();
        while stream.advance() {
            tokens.push(stream.token().text.clone());
        }
        assert_eq!(tokens.len(), 2);
    }
    {
        let mut stream = tokenizer.token_stream_for_doc("foo bar baz");
        let mut tokens = Vec::new();
        while stream.advance() {
            tokens.push(stream.token().text.clone());
        }
        assert_eq!(tokens.len(), 3);
    }
    let count_after = get_factory_create_count();
    assert_eq!(
        count_after - count_before,
        0,
        "Factory should be cached across tokenizations (no new factory created)"
    );
}

/// Cloning a `PluginTokenizer` must share the eagerly-built factory rather
/// than recreating it on the clone's first tokenization.
///
/// Why this matters:
///
/// 1. `PluginTokenizer::new` validates the user's config eagerly so a bad
///    config surfaces as an `Err` from `params.build()` rather than a panic
///    deep inside an FTS worker. If `clone()` reset the cache to `None`,
///    every clone would re-run `create_factory` on first use — and any
///    failure there can only be reported as a panic from the tokenization
///    adapter, defeating the eager-validation guarantee.
/// 2. `OwnedPluginFactory` is `Sync` and serializes `create_tokenizer`
///    through an internal `Mutex`, so multiple clones can safely share one
///    factory instance.
/// 3. FTS build creates one tokenizer per worker via `Clone`. Recreating the
///    plugin's (often heavyweight) factory N times per build burns CPU for
///    no benefit.
#[test]
#[serial(plugin_tests)]
fn test_clone_shares_cached_factory() {
    reset_factory_create_count();

    let plugin_path = get_plugin_path();

    // `PluginTokenizer::new` creates the factory eagerly (1 call so far).
    let count_before_new = get_factory_create_count();
    let mut tokenizer =
        PluginTokenizer::new(&plugin_path, "{}").expect("Failed to create tokenizer");
    let count_after_new = get_factory_create_count();
    assert_eq!(
        count_after_new - count_before_new,
        1,
        "Original tokenizer should create one factory eagerly during construction"
    );

    // Tokenizing through the original must reuse the cached factory.
    let count_before = get_factory_create_count();
    {
        let mut stream = tokenizer.token_stream_for_doc("hello");
        while stream.advance() {}
    }
    assert_eq!(
        get_factory_create_count() - count_before,
        0,
        "Original tokenizer should reuse its cached factory across tokenizations"
    );

    // Cloning must Arc-share the cached factory, not reset it.
    let mut cloned = tokenizer.clone();
    let count_before = get_factory_create_count();
    {
        let mut stream = cloned.token_stream_for_doc("world");
        while stream.advance() {}
    }
    assert_eq!(
        get_factory_create_count() - count_before,
        0,
        "Cloned tokenizer must share the original's factory; create_factory \
         should not be called again on first use"
    );

    // The original is unaffected by the clone using the shared factory.
    let count_before = get_factory_create_count();
    {
        let mut stream = tokenizer.token_stream_for_doc("test");
        while stream.advance() {}
    }
    assert_eq!(
        get_factory_create_count() - count_before,
        0,
        "Original tokenizer should still use the shared cached factory"
    );
}

/// `OwnedPluginFactory` is `pub` and `Sync`, so callers may legitimately
/// share an `Arc<OwnedPluginFactory>` across threads. The plugin C ABI
/// does not require `create_tokenizer` to be safe under concurrent calls
/// against the same factory handle, so the type must serialize access
/// internally. This test hammers a single shared factory from multiple
/// threads and verifies that every produced tokenizer is usable.
#[test]
#[serial(plugin_tests)]
fn test_concurrent_create_tokenizer_through_arc_factory() {
    use std::sync::Arc;
    use std::thread;

    use lance_index::scalar::inverted::tokenizer::plugin::ffi::CToken;
    use lance_index::scalar::inverted::tokenizer::plugin::loader::{
        NextTokenResult, OwnedPluginFactory,
    };

    let library = TokenizerPluginLibrary::load(get_plugin_path()).expect("load plugin");
    let factory = Arc::new(OwnedPluginFactory::new(library, "{}").expect("create factory"));

    const THREADS: usize = 8;
    const ITERATIONS_PER_THREAD: usize = 200;

    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let factory = Arc::clone(&factory);
            thread::spawn(move || {
                for _ in 0..ITERATIONS_PER_THREAD {
                    let instance = factory
                        .create_tokenizer()
                        .expect("create_tokenizer must succeed under concurrent access");
                    let mut stream = instance
                        .create_stream("hello world")
                        .expect("create_stream must succeed");
                    let mut tok = CToken::default();
                    let mut count = 0usize;
                    loop {
                        match stream.next_token(&mut tok) {
                            NextTokenResult::Token => count += 1,
                            NextTokenResult::EndOfStream => break,
                            NextTokenResult::Error(code, msg) => {
                                panic!(
                                    "unexpected plugin error during concurrent stress \
                                     (code={}): {}",
                                    code, msg
                                );
                            }
                        }
                    }
                    assert_eq!(count, 2, "each iteration should yield 2 tokens");
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("worker thread panicked");
    }
}

/// The owned plugin objects must form a parent-keeps-alive chain so that a
/// caller can drop a parent (factory or instance) without invalidating the
/// children. This prevents a use-after-free where the plugin's
/// `destroy_factory` would run before tokenization finishes.
#[test]
#[serial(plugin_tests)]
fn test_owned_objects_outlive_dropped_parents() {
    use std::sync::Arc;

    use lance_index::scalar::inverted::tokenizer::plugin::ffi::CToken;
    use lance_index::scalar::inverted::tokenizer::plugin::loader::{
        NextTokenResult, OwnedPluginFactory,
    };

    let library = TokenizerPluginLibrary::load(get_plugin_path()).expect("load plugin");
    let factory = Arc::new(OwnedPluginFactory::new(library, "{}").expect("create factory"));
    let instance = factory.create_tokenizer().expect("create_tokenizer");

    // Drop the only outer reference to the factory. The instance keeps an
    // Arc<OwnedPluginFactory> internally, so the C-side factory must remain
    // alive — otherwise `create_stream` below dereferences freed state.
    drop(factory);

    // The stream must also be usable after dropping the only outer reference
    // to the instance, because OwnedPluginTokenStream owns the parent
    // instance via Arc and the input text via String. The temporary
    // `format!(...)` would be a use-after-free under a borrowing API.
    let mut stream = instance
        .create_stream(format!("hello {}", "world"))
        .expect("create_stream after dropping factory");
    drop(instance);

    let mut tok = CToken::default();
    let mut tokens = Vec::new();
    loop {
        match stream.next_token(&mut tok) {
            NextTokenResult::Token => unsafe {
                let slice = std::slice::from_raw_parts(
                    tok.text.data as *const u8,
                    tok.text.length as usize,
                );
                tokens.push(std::str::from_utf8(slice).unwrap().to_string());
            },
            NextTokenResult::EndOfStream => break,
            NextTokenResult::Error(code, msg) => {
                panic!("unexpected plugin error (code={}): {}", code, msg);
            }
        }
    }
    assert_eq!(tokens, vec!["hello", "world"]);
}

/// The plugin C ABI requires "the previous stream must be destroyed before
/// creating another from the same tokenizer." A plugin is allowed to share
/// scratch state between a tokenizer and its single live stream (lazy
/// buffers, zero-copy slices into tokenizer state, etc.), so two
/// concurrently-live streams from one tokenizer would data-race or
/// use-after-free on the plugin side. The safe wrapper must reject the
/// second `create_stream` call rather than silently producing an unsafe
/// configuration.
#[test]
#[serial(plugin_tests)]
fn test_create_stream_rejects_overlapping_streams_from_same_instance() {
    use std::sync::Arc;

    use lance_index::scalar::inverted::tokenizer::plugin::loader::OwnedPluginFactory;

    let library = TokenizerPluginLibrary::load(get_plugin_path()).expect("load plugin");
    let factory = Arc::new(OwnedPluginFactory::new(library, "{}").expect("create factory"));
    let instance = factory.create_tokenizer().expect("create instance");

    let _first = instance
        .create_stream("hello world")
        .expect("first create_stream must succeed");

    // Second call against the same instance, while `_first` is still alive,
    // must fail with a clear error rather than handing the plugin two
    // overlapping streams. `OwnedPluginTokenStream` does not implement
    // `Debug`, so we cannot use `expect_err` directly.
    match instance.create_stream("foo bar") {
        Ok(_) => panic!("second create_stream while first is alive must fail"),
        Err(err) => {
            let msg = err.to_string();
            assert!(
                msg.contains("already has a live token stream"),
                "error must explain the ABI contract, got: {}",
                msg
            );
        }
    }

    // After dropping the first stream, the slot is released and a new
    // stream from the same instance must succeed.
    drop(_first);
    let _second = instance
        .create_stream("foo bar")
        .expect("create_stream must succeed once the previous stream is dropped");
}

/// Distinct tokenizer instances from the same factory have independent
/// "active stream" slots — one instance's live stream must not block
/// another instance from creating its own. This is what FTS workers rely
/// on: each worker holds its own `OwnedPluginTokenizerInstance`.
#[test]
#[serial(plugin_tests)]
fn test_create_stream_independence_across_instances() {
    use std::sync::Arc;

    use lance_index::scalar::inverted::tokenizer::plugin::loader::OwnedPluginFactory;

    let library = TokenizerPluginLibrary::load(get_plugin_path()).expect("load plugin");
    let factory = Arc::new(OwnedPluginFactory::new(library, "{}").expect("create factory"));
    let inst_a = factory.create_tokenizer().expect("instance a");
    let inst_b = factory.create_tokenizer().expect("instance b");

    let _stream_a = inst_a
        .create_stream("hello world")
        .expect("instance a stream");
    // `inst_b` is a separate instance, so it must accept a new stream even
    // while `inst_a` still has one live.
    let _stream_b = inst_b
        .create_stream("foo bar")
        .expect("instance b stream must be independent of instance a's slot");
}

/// Multiple threads racing to call `create_stream` against the same
/// `Arc<OwnedPluginTokenizerInstance>` must see exactly one winner per
/// "stream slot": the others must observe the explicit "already has a live
/// token stream" error, not an unsafe overlap. This is the multi-threaded
/// counterpart of `test_create_stream_rejects_overlapping_streams_*`, and
/// it exercises the `compare_exchange` fence directly.
#[test]
#[serial(plugin_tests)]
fn test_create_stream_serialization_under_concurrent_callers() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Barrier, Mutex};
    use std::thread;

    use lance_index::scalar::inverted::tokenizer::plugin::loader::OwnedPluginFactory;

    let library = TokenizerPluginLibrary::load(get_plugin_path()).expect("load plugin");
    let factory = Arc::new(OwnedPluginFactory::new(library, "{}").expect("create factory"));
    let instance = factory.create_tokenizer().expect("create instance");

    const ROUNDS: usize = 64;
    const THREADS: usize = 8;

    let success_count = Arc::new(AtomicUsize::new(0));
    let reject_count = Arc::new(AtomicUsize::new(0));
    let other_error_messages: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    for _ in 0..ROUNDS {
        let barrier = Arc::new(Barrier::new(THREADS));
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let instance = Arc::clone(&instance);
                let barrier = Arc::clone(&barrier);
                let success_count = Arc::clone(&success_count);
                let reject_count = Arc::clone(&reject_count);
                let other_error_messages = Arc::clone(&other_error_messages);
                thread::spawn(move || {
                    barrier.wait();
                    match instance.create_stream("hello world") {
                        Ok(stream) => {
                            success_count.fetch_add(1, Ordering::Relaxed);
                            // Hold the stream briefly so the other threads
                            // are guaranteed to race against an active slot
                            // rather than picking it up after we've already
                            // released.
                            drop(stream);
                        }
                        Err(err) => {
                            let msg = err.to_string();
                            if msg.contains("already has a live token stream") {
                                reject_count.fetch_add(1, Ordering::Relaxed);
                            } else {
                                other_error_messages.lock().unwrap().push(msg);
                            }
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("thread panicked");
        }
    }

    let other_errors = other_error_messages.lock().unwrap();
    assert!(
        other_errors.is_empty(),
        "every concurrent create_stream must either succeed or be \
         rejected with the exclusion error, got unexpected errors: {:?}",
        *other_errors
    );

    let total = success_count.load(Ordering::Relaxed) + reject_count.load(Ordering::Relaxed);
    assert_eq!(
        total,
        ROUNDS * THREADS,
        "every attempt must be accounted for as success or reject"
    );
    // At least one rejection must have happened across the whole run —
    // otherwise the concurrent reservation is not actually being exercised.
    assert!(
        reject_count.load(Ordering::Relaxed) > 0,
        "no rejection observed across {} rounds of {} threads; the \
         concurrent reservation is not being exercised",
        ROUNDS,
        THREADS
    );
}
