// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::path::Path;
use std::sync::Arc;

use lance_core::Result;
use lance_tokenizer::{BoxTokenStream, Token, TokenStream, Tokenizer};

use super::ffi::CToken;
use super::loader::{
    NextTokenResult, OwnedPluginFactory, OwnedPluginTokenStream, TokenizerPluginLibrary,
};
use crate::scalar::inverted::tokenizer::document_tokenizer::{DocType, LanceTokenizer};

/// PluginTokenizer loads a shared library at runtime and uses its tokenization
/// functions through the C ABI interface.
pub struct PluginTokenizer {
    /// The loaded plugin library (shared across clones)
    library: Arc<TokenizerPluginLibrary>,

    /// Configuration string for creating factories (format defined by plugin).
    /// Retained alongside `factory` for diagnostics (`Debug`) and so that a
    /// future API for re-binding the factory has the original input on hand.
    config: String,

    /// Eagerly-built factory shared across `Clone`s.
    ///
    /// `PluginTokenizer::new` constructs the factory up front so a malformed
    /// config surfaces as `Err` from `params.build()` rather than a panic
    /// from the first `token_stream_*` call. Wrapped in `Arc` because every
    /// `OwnedPluginTokenizerInstance` we hand out keeps an
    /// `Arc<OwnedPluginFactory>` back-reference to prevent the C-side
    /// factory from being destroyed while a tokenizer/stream derived from
    /// it is still alive, and `Clone` shares this same `Arc` so cloned
    /// tokenizers (one per FTS worker) reuse the same C-side factory.
    factory: Arc<OwnedPluginFactory>,
}

impl PluginTokenizer {
    pub fn new(library_path: impl AsRef<Path>, config: impl Into<String>) -> Result<Self> {
        let library = TokenizerPluginLibrary::load(library_path)?;
        let config = config.into();
        // Eagerly create the factory so a malformed config surfaces here as a
        // regular `Err` from `PluginTokenizer::new` / `params.build()`. Without
        // this, validation is deferred to the first `token_stream_*()` call,
        // where the failure can only be reported as a panic — turning a
        // user-input error into a task/process crash during query or index
        // build. The factory is also cached for reuse across tokenizations.
        let factory = OwnedPluginFactory::new(Arc::clone(&library), &config)?;
        Ok(Self {
            library,
            config,
            factory: Arc::new(factory),
        })
    }

    pub fn plugin_name(&self) -> &str {
        self.library.name()
    }

    pub fn plugin_version(&self) -> &str {
        self.library.version()
    }

    fn create_stream<'a>(&'a mut self, text: &'a str) -> BoxTokenStream<'a> {
        let stream = PluginTokenStreamAdapter::new(&self.factory, text);
        BoxTokenStream::new(stream)
    }
}

impl Clone for PluginTokenizer {
    fn clone(&self) -> Self {
        // Share the cached factory across clones rather than recreating it
        // lazily on first use. `OwnedPluginFactory` is `Sync` and serializes
        // `create_tokenizer` calls through an internal `Mutex`, so multiple
        // clones may safely tokenize concurrently against the same factory.
        // Cloning the `Arc` also preserves the early-failure guarantee from
        // `PluginTokenizer::new`: a clone derived from a successfully built
        // tokenizer cannot panic later from a deferred factory rebuild that
        // fails.
        Self {
            library: Arc::clone(&self.library),
            config: self.config.clone(),
            factory: Arc::clone(&self.factory),
        }
    }
}

impl std::fmt::Debug for PluginTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PluginTokenizer")
            .field("plugin", &self.library.name())
            .field("version", &self.library.version())
            .field("config", &self.config)
            .finish()
    }
}

impl LanceTokenizer for PluginTokenizer {
    fn token_stream_for_search<'a>(&'a mut self, query_text: &'a str) -> BoxTokenStream<'a> {
        self.create_stream(query_text)
    }

    fn token_stream_for_doc<'a>(&'a mut self, text: &'a str) -> BoxTokenStream<'a> {
        self.create_stream(text)
    }

    fn box_clone(&self) -> Box<dyn LanceTokenizer> {
        Box::new(self.clone())
    }

    fn doc_type(&self) -> DocType {
        // Plugin tokenizers are currently text-based only
        DocType::Text
    }
}

impl Tokenizer for PluginTokenizer {
    type TokenStream<'a> = BoxTokenStream<'a>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> BoxTokenStream<'a> {
        self.create_stream(text)
    }
}

/// Adapter that implements tantivy's `TokenStream` trait by pulling one token
/// at a time from a plugin's `next_token` C callback.
///
/// The stream is consumed lazily so we never buffer an entire document in
/// memory — the FTS worker `worker_memory_limit` budget assumes streamed
/// tokenization, and built-in tokenizers behave the same way.
///
/// # Panic policy
///
/// Plugin failures (factory/instance/stream creation, or `next_token`
/// returning a negative error code) panic out of the adapter rather than
/// silently truncating the stream. A truncated stream would index/search a
/// document as if it had only the tokens emitted before the error, which
/// silently corrupts FTS results.
///
/// The same adapter backs both indexing (`token_stream_for_doc`) and search
/// (`token_stream_for_search`), so the panic policy applies to **both**
/// paths:
///
/// - During an index build, the panic surfaces as an FTS worker task
///   failure and aborts the build job. This is preferable to writing an
///   index whose contents disagree with what the user configured.
/// - During a search, the panic surfaces inside the query worker handling
///   the request. The query fails (rather than returning silently
///   truncated results) and the worker recovers per the host's task-panic
///   policy.
///
/// In both cases the trade-off is "fail loud" over "silently produce
/// inconsistent results". The host `LanceTokenizer` trait does not expose a
/// `Result`-returning token-stream API today, so adapters cannot propagate
/// these errors as `Err` — see `document_tokenizer::LanceTokenizer`.
///
/// `OwnedPluginTokenStream` itself owns its parent tokenizer instance (via
/// `Arc`) and the input text (as `String`), so we don't need to keep them
/// alive separately here.
struct PluginTokenStreamAdapter {
    current_token: Token,
    eof: bool,
    stream: OwnedPluginTokenStream,
}

impl PluginTokenStreamAdapter {
    fn new(factory: &Arc<OwnedPluginFactory>, text: &str) -> Self {
        // The factory is built eagerly in `PluginTokenizer::new`, so the only
        // failure modes left here come from `create_tokenizer` /
        // `create_stream`. The adapter is reached from both
        // `token_stream_for_doc` (indexing) and `token_stream_for_search`
        // (queries), so any panic below aborts the in-flight task in either
        // path. See the type-level docs above for the rationale.
        let instance = factory.create_tokenizer().unwrap_or_else(|e| {
            panic!(
                "failed to create plugin tokenizer instance: {} \
                 (the in-flight indexing or search task is aborted)",
                e
            )
        });
        let stream = instance
            .create_stream(text.to_string())
            .unwrap_or_else(|e| {
                panic!(
                    "failed to create plugin token stream: {} \
                 (the in-flight indexing or search task is aborted)",
                    e
                )
            });

        Self {
            current_token: Token::default(),
            eof: false,
            stream,
        }
    }
}

impl TokenStream for PluginTokenStreamAdapter {
    fn advance(&mut self) -> bool {
        if self.eof {
            return false;
        }
        let mut c_token = CToken::default();
        match self.stream.next_token(&mut c_token) {
            NextTokenResult::Token => {
                let text = if c_token.text.data.is_null() {
                    // A NULL data pointer is only valid when length is zero.
                    // Plugins that return `data == NULL` with `length > 0`
                    // are violating the C ABI; treating the result as an
                    // empty string would silently swap the real token for
                    // "" and corrupt FTS terms — the same failure mode the
                    // invalid-UTF-8 branch fails loud about. Mirror that
                    // policy here.
                    if c_token.text.length != 0 {
                        panic!(
                            "Plugin returned token with NULL text pointer but \
                             non-zero length {} (the plugin ABI requires the \
                             pointer to be non-null whenever length is greater \
                             than zero; the in-flight indexing or search task \
                             is aborted)",
                            c_token.text.length
                        );
                    }
                    String::new()
                } else {
                    let slice = unsafe {
                        std::slice::from_raw_parts(
                            c_token.text.data as *const u8,
                            c_token.text.length as usize,
                        )
                    };
                    // The plugin ABI requires UTF-8 token text. Lossy
                    // conversion would silently substitute U+FFFD for invalid
                    // bytes, which corrupts FTS terms and is hard to diagnose
                    // after the index has been written. Fail loud instead, in
                    // line with how every other plugin contract violation is
                    // handled.
                    std::str::from_utf8(slice)
                        .unwrap_or_else(|e| {
                            panic!(
                                "Plugin returned token with invalid UTF-8 text: {} \
                                 (the plugin ABI requires UTF-8; \
                                 the in-flight indexing or search task is aborted)",
                                e
                            )
                        })
                        .to_string()
                };
                self.current_token = Token {
                    offset_from: c_token.offset_from as usize,
                    offset_to: c_token.offset_to as usize,
                    position: c_token.position as usize,
                    text,
                    position_length: c_token.position_length as usize,
                };
                true
            }
            NextTokenResult::EndOfStream => {
                self.eof = true;
                false
            }
            NextTokenResult::Error(code, msg) => {
                // Fail loud: a partial token stream would corrupt FTS results
                // (some tokens indexed, others silently dropped). The
                // in-flight indexing or search task aborts, which is
                // preferable to silently emitting tokens that disagree with
                // the configured tokenizer.
                panic!(
                    "Plugin tokenizer error during tokenization (code: {}): {} \
                     (the in-flight indexing or search task is aborted)",
                    code, msg
                );
            }
        }
    }

    fn token(&self) -> &Token {
        &self.current_token
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.current_token
    }
}
