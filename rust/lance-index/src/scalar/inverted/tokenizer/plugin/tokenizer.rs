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
    library: Arc<TokenizerPluginLibrary>,
    config: String,
    /// Eagerly built so a malformed config surfaces as `Err` from `build()`
    /// rather than as a panic from the first `token_stream_*` call. Shared
    /// across `Clone`s so cloned tokenizers (one per FTS worker) reuse the
    /// same C-side factory.
    factory: Arc<OwnedPluginFactory>,
}

impl PluginTokenizer {
    pub fn new(library_path: impl AsRef<Path>, config: impl Into<String>) -> Result<Self> {
        let library = TokenizerPluginLibrary::load(library_path)?;
        let config = config.into();
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
        BoxTokenStream::new(PluginTokenStreamAdapter::new(&self.factory, text))
    }
}

impl Clone for PluginTokenizer {
    fn clone(&self) -> Self {
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
/// # Panic policy
///
/// Plugin contract violations (factory/instance/stream creation failure,
/// `next_token` returning a negative code, NULL token text with non-zero
/// length, non-UTF-8 token text) panic out of the adapter rather than
/// silently truncating or substituting tokens. A truncated or replaced stream
/// would index/search a document as if it had different terms, silently
/// corrupting FTS results. The host `LanceTokenizer` trait does not expose a
/// `Result`-returning token-stream API today, so adapters cannot propagate
/// these errors as `Err` — see `document_tokenizer::LanceTokenizer`. The
/// in-flight indexing or search task aborts and the worker recovers per the
/// host's task-panic policy.
struct PluginTokenStreamAdapter {
    current_token: Token,
    eof: bool,
    stream: OwnedPluginTokenStream,
}

impl PluginTokenStreamAdapter {
    fn new(factory: &Arc<OwnedPluginFactory>, text: &str) -> Self {
        let instance = factory
            .create_tokenizer()
            .unwrap_or_else(|e| panic!("failed to create plugin tokenizer instance: {}", e));
        let stream = instance
            .create_stream(text.to_string())
            .unwrap_or_else(|e| panic!("failed to create plugin token stream: {}", e));

        Self {
            current_token: Token::default(),
            eof: false,
            stream,
        }
    }
}

/// Extract UTF-8 token text from a `CToken`. Panics on contract violations
/// (NULL data with non-zero length, invalid UTF-8) per the adapter's
/// fail-loud policy.
fn extract_token_text(c_token: &CToken) -> String {
    if c_token.text.data.is_null() {
        if c_token.text.length != 0 {
            panic!(
                "Plugin returned token with NULL text pointer but non-zero length {} \
                 (the plugin ABI requires a non-null pointer when length > 0)",
                c_token.text.length
            );
        }
        return String::new();
    }
    let slice = unsafe {
        std::slice::from_raw_parts(c_token.text.data as *const u8, c_token.text.length as usize)
    };
    std::str::from_utf8(slice)
        .unwrap_or_else(|e| panic!("Plugin returned token with invalid UTF-8 text: {}", e))
        .to_string()
}

impl TokenStream for PluginTokenStreamAdapter {
    fn advance(&mut self) -> bool {
        if self.eof {
            return false;
        }
        let mut c_token = CToken::default();
        match self.stream.next_token(&mut c_token) {
            NextTokenResult::Token => {
                self.current_token = Token {
                    offset_from: c_token.offset_from as usize,
                    offset_to: c_token.offset_to as usize,
                    position: c_token.position as usize,
                    text: extract_token_text(&c_token),
                    position_length: c_token.position_length as usize,
                };
                true
            }
            NextTokenResult::EndOfStream => {
                self.eof = true;
                false
            }
            NextTokenResult::Error(code, msg) => {
                panic!(
                    "Plugin tokenizer error during tokenization (code: {}): {}",
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
