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
use rstest::rstest;
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

fn get_plugin_library() -> &'static Library {
    PLUGIN_LIBRARY.get_or_init(|| {
        let plugin_path = get_plugin_path();
        unsafe { Library::new(&plugin_path).expect("Failed to load plugin library") }
    })
}

fn get_factory_create_count() -> u32 {
    let library = get_plugin_library();
    unsafe {
        let func: libloading::Symbol<extern "C" fn() -> u32> = library
            .get(b"lance_test_get_factory_create_count")
            .expect("Failed to get lance_test_get_factory_create_count");
        func()
    }
}

fn reset_factory_create_count() {
    let library = get_plugin_library();
    unsafe {
        let func: libloading::Symbol<extern "C" fn()> = library
            .get(b"lance_test_reset_factory_create_count")
            .expect("Failed to get lance_test_reset_factory_create_count");
        func()
    }
}

fn collect_tokens(stream: &mut dyn TokenStream) -> Vec<String> {
    let mut tokens = Vec::new();
    while stream.advance() {
        tokens.push(stream.token().text.clone());
    }
    tokens
}

fn tokenize_doc(tokenizer: &mut PluginTokenizer, text: &str) -> Vec<String> {
    let mut stream = tokenizer.token_stream_for_doc(text);
    collect_tokens(&mut stream)
}

#[test]
#[serial(plugin_tests)]
fn test_plugin_library_load() {
    let plugin_path = get_plugin_path();
    let library = TokenizerPluginLibrary::load(&plugin_path).expect("Failed to load plugin");
    assert_eq!(library.name(), "test_whitespace_tokenizer");
    assert_eq!(library.version(), "0.1.0");
}

#[rstest]
#[case::basic("{}", "Hello World", &["Hello", "World"])]
#[case::lowercase(r#"{"lowercase": true}"#, "Hello WORLD", &["hello", "world"])]
#[case::empty("{}", "", &[])]
#[case::multiple_spaces("{}", "  Hello   World  ", &["Hello", "World"])]
#[serial(plugin_tests)]
fn test_plugin_tokenizer_doc(#[case] config: &str, #[case] text: &str, #[case] expected: &[&str]) {
    let plugin_path = get_plugin_path();
    let mut tokenizer = PluginTokenizer::new(&plugin_path, config).expect("create tokenizer");
    assert_eq!(tokenize_doc(&mut tokenizer, text), expected);
}

#[test]
#[serial(plugin_tests)]
fn test_plugin_tokenizer_offsets_and_position() {
    let plugin_path = get_plugin_path();
    let mut tokenizer = PluginTokenizer::new(&plugin_path, "{}").expect("create tokenizer");
    let mut stream = tokenizer.token_stream_for_doc("Hello World");
    let mut tokens = Vec::new();
    while stream.advance() {
        let t = stream.token();
        tokens.push((t.text.clone(), t.position, t.offset_from, t.offset_to));
    }
    assert_eq!(
        tokens,
        vec![
            ("Hello".to_string(), 0, 0, 5),
            ("World".to_string(), 1, 6, 11),
        ]
    );
}

#[test]
#[serial(plugin_tests)]
fn test_plugin_tokenizer_search_stream() {
    let plugin_path = get_plugin_path();
    let mut tokenizer = PluginTokenizer::new(&plugin_path, "{}").expect("create tokenizer");
    let mut stream = tokenizer.token_stream_for_search("query terms");
    assert_eq!(collect_tokens(&mut stream), vec!["query", "terms"]);
}

#[test]
#[serial(plugin_tests)]
fn test_plugin_tokenizer_clone() {
    let plugin_path = get_plugin_path();
    let tokenizer = PluginTokenizer::new(&plugin_path, r#"{"lowercase": true}"#).unwrap();
    let mut cloned = tokenizer.box_clone();
    let mut stream = cloned.token_stream_for_doc("HELLO");
    assert_eq!(collect_tokens(&mut stream), vec!["hello"]);
}

#[test]
#[serial(plugin_tests)]
fn test_inverted_index_params_with_plugin() {
    let plugin_path = get_plugin_path();
    let params = InvertedIndexParams::default().plugin(
        plugin_path.to_string_lossy().to_string(),
        r#"{"lowercase": true}"#.to_string(),
    );
    let mut tokenizer = params.build().expect("build");
    let mut stream = tokenizer.token_stream_for_doc("Test Document");
    assert_eq!(collect_tokens(&mut stream), vec!["test", "document"]);
}

/// Regression: plugin tokens must flow through the standard filter chain
/// (lower_case / stem / remove_stop_words / ascii_folding / max_token_length).
/// Plugin output here is raw whitespace-split; all normalization comes from
/// `InvertedIndexParams` filters.
#[test]
#[serial(plugin_tests)]
fn test_plugin_output_flows_through_filter_chain() {
    let plugin_path = get_plugin_path();
    let params = InvertedIndexParams::default()
        .plugin(plugin_path.to_string_lossy().to_string(), "{}".to_string())
        .lower_case(true)
        .stem(true)
        .remove_stop_words(true)
        .ascii_folding(true);
    let mut tokenizer = params.build().expect("build");
    let mut stream = tokenizer.token_stream_for_doc("The Quick Brown Foxes");
    // plugin → ["The", "Quick", "Brown", "Foxes"]
    // LowerCaser → ["the", "quick", "brown", "foxes"]
    // Stemmer → ["the", "quick", "brown", "fox"]
    // StopWordFilter → ["quick", "brown", "fox"]
    assert_eq!(collect_tokens(&mut stream), vec!["quick", "brown", "fox"]);
}

#[test]
#[serial(plugin_tests)]
fn test_plugin_max_token_length_applied() {
    let plugin_path = get_plugin_path();
    let params = InvertedIndexParams::default()
        .plugin(plugin_path.to_string_lossy().to_string(), "{}".to_string())
        .lower_case(false)
        .stem(false)
        .remove_stop_words(false)
        .ascii_folding(false)
        .max_token_length(Some(5));
    let mut tokenizer = params.build().expect("build");
    let mut stream = tokenizer.token_stream_for_doc("a abcdef ab abcdefghi");
    assert_eq!(collect_tokens(&mut stream), vec!["a", "ab"]);
}

/// A malformed config must surface as a regular `Err` from
/// `PluginTokenizer::new`, not a panic on first tokenization.
#[test]
#[serial(plugin_tests)]
fn test_plugin_constructor_rejects_invalid_config() {
    let plugin_path = get_plugin_path();
    let err = PluginTokenizer::new(&plugin_path, r#"{"reject_config": true}"#)
        .expect_err("constructor must reject invalid config");
    assert!(
        err.to_string().contains("simulated config rejection"),
        "got: {}",
        err
    );
}

/// Same contract via the public `build()` path.
#[test]
#[serial(plugin_tests)]
fn test_inverted_index_params_build_rejects_invalid_plugin_config() {
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
        "got: {}",
        err
    );
}

#[test]
#[serial(plugin_tests)]
#[should_panic(expected = "invalid UTF-8")]
fn test_plugin_invalid_utf8_token_is_rejected() {
    let plugin_path = get_plugin_path();
    let mut tokenizer =
        PluginTokenizer::new(&plugin_path, r#"{"emit_invalid_utf8": true}"#).unwrap();
    let mut stream = tokenizer.token_stream_for_doc("hello");
    while stream.advance() {}
}

#[test]
#[serial(plugin_tests)]
#[should_panic(expected = "NULL text pointer but non-zero length")]
fn test_plugin_null_text_with_nonzero_length_is_rejected() {
    let plugin_path = get_plugin_path();
    let mut tokenizer =
        PluginTokenizer::new(&plugin_path, r#"{"emit_null_with_length": true}"#).unwrap();
    let mut stream = tokenizer.token_stream_for_doc("hello");
    while stream.advance() {}
}

/// Catch a relative path at the persistence boundary even if it slipped past
/// `build()` (hand-built params serialized directly to proto).
#[test]
#[serial(plugin_tests)]
fn test_plugin_relative_path_is_rejected_at_persistence_boundary() {
    use lance_index::pbold::InvertedIndexDetails;

    let params = InvertedIndexParams::default()
        .plugin("relative/path/to/plugin.so".to_string(), "{}".to_string());
    let err = InvertedIndexDetails::try_from(&params)
        .expect_err("proto conversion must reject a relative plugin path");
    assert!(err.to_string().contains("absolute"), "got: {}", err);
}

#[test]
#[serial(plugin_tests)]
fn test_plugin_unicode_text() {
    let plugin_path = get_plugin_path();
    let mut tokenizer = PluginTokenizer::new(&plugin_path, "{}").unwrap();
    assert_eq!(
        tokenize_doc(&mut tokenizer, "Hello 世界 Rust"),
        vec!["Hello", "世界", "Rust"]
    );
}

#[test]
#[serial(plugin_tests)]
fn test_plugin_long_text() {
    let plugin_path = get_plugin_path();
    let mut tokenizer = PluginTokenizer::new(&plugin_path, "{}").unwrap();
    let long_text = vec!["word"; 100].join(" ");
    let mut stream = tokenizer.token_stream_for_doc(&long_text);
    let mut count = 0;
    while stream.advance() {
        count += 1;
    }
    assert_eq!(count, 100);
}

/// Drive the token stream until either it ends or the plugin panics. Returns
/// the tokens collected before the panic, plus whether a panic occurred —
/// proves that tokenization is streamed (partial results observable before
/// the panic) and that plugin errors propagate as panics rather than silently
/// truncating the stream.
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

#[rstest]
#[case::after_zero(0, "Hello World", &[])]
#[case::after_one(1, "Hello World Test", &["Hello"])]
#[case::after_two(2, "one two three four five", &["one", "two"])]
#[serial(plugin_tests)]
fn test_plugin_error_surfaces_as_panic_after_n_tokens(
    #[case] error_after: u32,
    #[case] text: &str,
    #[case] expected_before_panic: &[&str],
) {
    let plugin_path = get_plugin_path();
    let config = format!(r#"{{"error_after_n_tokens": {}}}"#, error_after);
    let mut tokenizer = PluginTokenizer::new(&plugin_path, &config).unwrap();
    let (tokens, panicked) = collect_until_panic(&mut tokenizer, text);
    assert!(panicked, "plugin error must surface as a panic");
    assert_eq!(tokens, expected_before_panic);
}

#[test]
#[serial(plugin_tests)]
fn test_plugin_error_not_triggered_when_fewer_tokens() {
    let plugin_path = get_plugin_path();
    let mut tokenizer =
        PluginTokenizer::new(&plugin_path, r#"{"error_after_n_tokens": 10}"#).unwrap();
    assert_eq!(
        tokenize_doc(&mut tokenizer, "Hello World"),
        vec!["Hello", "World"]
    );
}

#[test]
#[serial(plugin_tests)]
#[should_panic(expected = "Plugin tokenizer error during tokenization")]
fn test_plugin_error_with_inverted_index_params() {
    let plugin_path = get_plugin_path();
    let params = InvertedIndexParams::default().plugin(
        plugin_path.to_string_lossy().to_string(),
        r#"{"error_after_n_tokens": 1}"#.to_string(),
    );
    let mut tokenizer = params.build().expect("build");
    let mut stream = tokenizer.token_stream_for_doc("one two three");
    while stream.advance() {}
}

#[test]
#[serial(plugin_tests)]
fn test_plugin_multithread_with_clones() {
    use std::sync::Arc;
    use std::thread;

    let plugin_path = get_plugin_path();
    let tokenizer = Arc::new(PluginTokenizer::new(&plugin_path, "{}").unwrap());

    let handles: Vec<_> = (0..4)
        .map(|i| {
            let mut thread_tokenizer = (*tokenizer).clone();
            thread::spawn(move || {
                let text = format!("thread {} test data", i);
                let tokens = tokenize_doc(&mut thread_tokenizer, &text);
                (i, tokens)
            })
        })
        .collect();

    for handle in handles {
        let (thread_id, tokens) = handle.join().expect("Thread panicked");
        assert_eq!(
            tokens,
            vec!["thread", &thread_id.to_string(), "test", "data"]
        );
    }
}

#[test]
#[serial(plugin_tests)]
fn test_plugin_multithread_repeated_tokenization() {
    use std::thread;

    let plugin_path = get_plugin_path();
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let path = plugin_path.clone();
            thread::spawn(move || {
                let mut tokenizer = PluginTokenizer::new(&path, "{}").unwrap();
                let mut all_results = Vec::new();
                for j in 0..10 {
                    let text = format!("iteration {}", j);
                    all_results.push(tokenize_doc(&mut tokenizer, &text));
                }
                (i, all_results)
            })
        })
        .collect();

    for handle in handles {
        let (_, results) = handle.join().expect("Thread panicked");
        assert_eq!(results.len(), 10);
        for (j, tokens) in results.iter().enumerate() {
            assert_eq!(tokens, &vec!["iteration".to_string(), j.to_string()]);
        }
    }
}

#[test]
#[serial(plugin_tests)]
fn test_plugin_multithread_with_lance_tokenizer_trait() {
    use std::thread;

    let plugin_path = get_plugin_path();
    let tokenizer: Box<dyn LanceTokenizer> =
        Box::new(PluginTokenizer::new(&plugin_path, r#"{"lowercase": true}"#).unwrap());

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let mut thread_tokenizer = tokenizer.box_clone();
            thread::spawn(move || {
                let mut stream = thread_tokenizer.token_stream_for_doc("HELLO World");
                collect_tokens(&mut stream)
            })
        })
        .collect();

    for handle in handles {
        assert_eq!(
            handle.join().expect("Thread panicked"),
            vec!["hello", "world"]
        );
    }
}

/// Factory creation must be eager (in `PluginTokenizer::new`) and cached
/// across tokenizations — a malformed config has to surface as `Err` from
/// `build()`, not a panic during indexing or search.
#[test]
#[serial(plugin_tests)]
fn test_factory_caching_behavior() {
    reset_factory_create_count();
    let plugin_path = get_plugin_path();

    let count_before = get_factory_create_count();
    let mut tokenizer = PluginTokenizer::new(&plugin_path, "{}").unwrap();
    assert_eq!(
        get_factory_create_count() - count_before,
        1,
        "factory built eagerly"
    );

    let count_before = get_factory_create_count();
    assert_eq!(tokenize_doc(&mut tokenizer, "hello world").len(), 2);
    assert_eq!(tokenize_doc(&mut tokenizer, "foo bar baz").len(), 3);
    assert_eq!(
        get_factory_create_count() - count_before,
        0,
        "factory cached"
    );
}

/// Cloning shares the eagerly-built factory: skipping cache-share would force
/// every clone to re-run `create_factory` on first use, and any failure there
/// can only surface as a panic from the tokenization adapter — defeating the
/// eager-validation guarantee from `PluginTokenizer::new`. FTS build creates
/// one tokenizer per worker via `Clone`, so this also avoids re-running the
/// (often heavyweight) plugin factory N times per build.
#[test]
#[serial(plugin_tests)]
fn test_clone_shares_cached_factory() {
    reset_factory_create_count();
    let plugin_path = get_plugin_path();

    let mut tokenizer = PluginTokenizer::new(&plugin_path, "{}").unwrap();
    assert_eq!(
        get_factory_create_count(),
        1,
        "eager build creates one factory"
    );

    let mut cloned = tokenizer.clone();
    let mut stream = cloned.token_stream_for_doc("world");
    while stream.advance() {}
    assert_eq!(
        get_factory_create_count(),
        1,
        "clone shares the cached factory"
    );

    let mut stream = tokenizer.token_stream_for_doc("test");
    while stream.advance() {}
    assert_eq!(
        get_factory_create_count(),
        1,
        "original still uses shared factory"
    );
}

/// The C ABI does not require `create_tokenizer` to be safe under concurrent
/// calls against the same factory handle, so `OwnedPluginFactory` must
/// serialize internally. This hammers a single shared factory from multiple
/// threads.
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
                    let instance = factory.create_tokenizer().expect("create_tokenizer");
                    let mut stream = instance
                        .create_stream("hello world")
                        .expect("create_stream");
                    let mut tok = CToken::default();
                    let mut count = 0usize;
                    loop {
                        match stream.next_token(&mut tok) {
                            NextTokenResult::Token => count += 1,
                            NextTokenResult::EndOfStream => break,
                            NextTokenResult::Error(code, msg) => {
                                panic!("unexpected plugin error (code={}): {}", code, msg);
                            }
                        }
                    }
                    assert_eq!(count, 2);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("worker thread panicked");
    }
}

/// Owned plugin objects form a parent-keeps-alive chain: dropping a parent
/// (factory or instance) must not invalidate its children, otherwise
/// `destroy_factory` could run before tokenization finishes — use-after-free.
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

    drop(factory);

    // Stream must also be usable after dropping the only outer reference to
    // the instance, and after passing a temporary `format!(...)` as input.
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

/// Plugins may share scratch state between a tokenizer and its single live
/// stream, so two overlapping streams from one tokenizer would data-race or
/// use-after-free on the plugin side. The wrapper must reject the second
/// `create_stream` call rather than silently producing an unsafe configuration.
#[test]
#[serial(plugin_tests)]
fn test_create_stream_rejects_overlapping_streams_from_same_instance() {
    use std::sync::Arc;

    use lance_index::scalar::inverted::tokenizer::plugin::loader::OwnedPluginFactory;

    let library = TokenizerPluginLibrary::load(get_plugin_path()).expect("load plugin");
    let factory = Arc::new(OwnedPluginFactory::new(library, "{}").expect("create factory"));
    let instance = factory.create_tokenizer().expect("create instance");

    let _first = instance.create_stream("hello world").expect("first stream");

    match instance.create_stream("foo bar") {
        Ok(_) => panic!("second create_stream while first is alive must fail"),
        Err(err) => assert!(
            err.to_string().contains("already has a live token stream"),
            "got: {}",
            err
        ),
    }

    drop(_first);
    let _second = instance
        .create_stream("foo bar")
        .expect("after first dropped");
}

/// Distinct instances from the same factory have independent active-stream
/// slots; FTS workers rely on this (each worker holds its own instance).
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
    let _stream_b = inst_b.create_stream("foo bar").expect("instance b stream");
}

/// Multiple threads racing on the same `Arc<OwnedPluginTokenizerInstance>`
/// must see exactly one winner per stream slot; the others must observe the
/// explicit "already has a live token stream" error, not an unsafe overlap.
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
        "unexpected errors: {:?}",
        *other_errors
    );

    let total = success_count.load(Ordering::Relaxed) + reject_count.load(Ordering::Relaxed);
    assert_eq!(total, ROUNDS * THREADS);
    assert!(
        reject_count.load(Ordering::Relaxed) > 0,
        "no rejection observed across {} rounds of {} threads — concurrent reservation not exercised",
        ROUNDS,
        THREADS
    );
}
