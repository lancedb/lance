// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::path::Path;
use std::sync::Arc;

use lance_core::Result;
use lance_tokenizer::{BoxTokenStream, Token, TokenStream, Tokenizer};

use super::ffi::CToken;
use super::loader::{
    NextTokenResult, OwnedPluginFactory, OwnedPluginTokenStream, OwnedPluginTokenizerInstance,
    TokenizerPluginLibrary,
};
use crate::scalar::inverted::tokenizer::document_tokenizer::{DocType, LanceTokenizer};

/// PluginTokenizer loads a shared library at runtime and uses its tokenization
/// functions through the C ABI interface.
pub struct PluginTokenizer {
    /// The loaded plugin library (shared across clones)
    library: Arc<TokenizerPluginLibrary>,

    /// Configuration string for creating factories (format defined by plugin)
    config: String,

    /// Cached factory for repeated tokenization (lazily initialized)
    cached_factory: Option<OwnedPluginFactory>,
}

impl PluginTokenizer {
    pub fn new(library_path: impl AsRef<Path>, config: impl Into<String>) -> Result<Self> {
        let library = TokenizerPluginLibrary::load(library_path)?;
        Ok(Self {
            library,
            config: config.into(),
            cached_factory: None,
        })
    }

    pub fn from_library(library: Arc<TokenizerPluginLibrary>, config: impl Into<String>) -> Self {
        Self {
            library,
            config: config.into(),
            cached_factory: None,
        }
    }

    pub fn plugin_name(&self) -> &str {
        self.library.name()
    }

    pub fn plugin_version(&self) -> &str {
        self.library.version()
    }

    fn create_stream<'a>(&'a mut self, text: &'a str) -> BoxTokenStream<'a> {
        let stream = PluginTokenStreamAdapter::new(
            &mut self.cached_factory,
            &self.library,
            &self.config,
            text,
        );
        BoxTokenStream::new(stream)
    }
}

impl Clone for PluginTokenizer {
    fn clone(&self) -> Self {
        Self {
            library: Arc::clone(&self.library),
            config: self.config.clone(),
            cached_factory: None,
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
/// Plugin failures (factory/instance/stream creation, or `next_token`
/// returning a negative error code) panic out of the adapter rather than
/// silently truncating the stream. A truncated stream would index/search the
/// document as if it had only the tokens emitted before the error, which
/// silently corrupts FTS results. Panicking is fail-loud: it propagates up the
/// FTS worker as a task failure that the build pipeline must surface, so a
/// misbehaving plugin cannot persist incorrect index contents.
///
/// Field declaration order matters for `Drop`: `stream` is dropped before
/// `_instance`, because the C-side stream pointer aliases memory owned by the
/// tokenizer instance.
struct PluginTokenStreamAdapter {
    current_token: Token,
    eof: bool,
    stream: OwnedPluginTokenStream,
    _instance: OwnedPluginTokenizerInstance,
}

impl PluginTokenStreamAdapter {
    fn new(
        cached_factory: &mut Option<OwnedPluginFactory>,
        library: &Arc<TokenizerPluginLibrary>,
        config: &str,
        text: &str,
    ) -> Self {
        // Lazily create and cache the factory. A failure here is treated as a
        // hard error: the plugin is misconfigured (bad config string, etc.)
        // and silently returning an empty stream would mask the misconfig.
        if cached_factory.is_none() {
            let factory = OwnedPluginFactory::new(Arc::clone(library), config)
                .unwrap_or_else(|e| panic!("failed to create plugin factory: {}", e));
            *cached_factory = Some(factory);
        }
        let factory = cached_factory.as_ref().unwrap();

        let instance = factory
            .create_tokenizer()
            .unwrap_or_else(|e| panic!("failed to create plugin tokenizer instance: {}", e));
        let stream = instance
            .create_stream(text)
            .unwrap_or_else(|e| panic!("failed to create plugin token stream: {}", e));

        Self {
            current_token: Token::default(),
            eof: false,
            stream,
            _instance: instance,
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
                    String::new()
                } else {
                    unsafe {
                        let slice = std::slice::from_raw_parts(
                            c_token.text.data as *const u8,
                            c_token.text.length as usize,
                        );
                        String::from_utf8_lossy(slice).into_owned()
                    }
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
                // (some tokens indexed, others silently dropped).
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
