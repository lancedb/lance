// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Plugin-based tokenizer implementation.

use std::path::Path;
use std::sync::Arc;

use lance_core::Result;
use tantivy::tokenizer::{BoxTokenStream, Token, TokenStream};

use super::ffi::CToken;
use super::loader::TokenizerPluginLibrary;
use crate::scalar::inverted::tokenizer::document_tokenizer::{DocType, LanceTokenizer};

/// A tokenizer that uses a dynamically loaded plugin.
///
/// This tokenizer loads a shared library at runtime and uses its tokenization
/// functions through the C ABI interface.
pub struct PluginTokenizer {
    /// The loaded plugin library (shared across clones)
    library: Arc<TokenizerPluginLibrary>,

    /// JSON configuration for creating factories
    config: String,
}

impl PluginTokenizer {
    /// Create a new plugin tokenizer.
    ///
    /// # Arguments
    ///
    /// * `library_path` - Path to the plugin shared library
    /// * `config` - JSON configuration string for the tokenizer
    ///
    /// # Errors
    ///
    /// Returns an error if the library cannot be loaded or is invalid.
    pub fn new(library_path: impl AsRef<Path>, config: impl Into<String>) -> Result<Self> {
        let library = TokenizerPluginLibrary::load(library_path)?;
        Ok(Self {
            library,
            config: config.into(),
        })
    }

    /// Create a plugin tokenizer from an already loaded library.
    pub fn from_library(library: Arc<TokenizerPluginLibrary>, config: impl Into<String>) -> Self {
        Self {
            library,
            config: config.into(),
        }
    }

    /// Get the plugin name.
    pub fn plugin_name(&self) -> &str {
        self.library.name()
    }

    /// Get the plugin version.
    pub fn plugin_version(&self) -> &str {
        self.library.version()
    }

    fn create_stream<'a>(&'a mut self, text: &'a str) -> BoxTokenStream<'a> {
        // Create factory and tokenizer for this stream
        // Note: This is not the most efficient approach for repeated tokenization,
        // but it ensures thread safety and simplifies lifetime management.
        // For production use, consider caching the factory/tokenizer.
        let stream = PluginTokenStreamAdapter::new(Arc::clone(&self.library), &self.config, text);
        BoxTokenStream::new(stream)
    }
}

impl Clone for PluginTokenizer {
    fn clone(&self) -> Self {
        Self {
            library: Arc::clone(&self.library),
            config: self.config.clone(),
        }
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
        // Plugin tokenizers are text-based by default
        DocType::Text
    }
}

/// Adapter that implements tantivy's TokenStream trait for plugin streams.
///
/// This struct owns all the resources needed for tokenization and manages
/// their lifetimes appropriately.
struct PluginTokenStreamAdapter {
    /// Current token (reused across calls)
    current_token: Token,

    /// Collected tokens from the plugin stream
    tokens: Vec<Token>,

    /// Current index in tokens
    index: usize,

    /// Whether initialization failed
    error: bool,
}

impl PluginTokenStreamAdapter {
    fn new(library: Arc<TokenizerPluginLibrary>, config: &str, text: &str) -> Self {
        let mut tokens = Vec::new();
        let mut error = false;

        // Create factory, tokenizer, and stream, then collect all tokens
        match library.create_factory(config) {
            Ok(factory) => match factory.create_tokenizer() {
                Ok(tokenizer_instance) => match tokenizer_instance.create_stream(text) {
                    Ok(mut stream) => {
                        let mut c_token = CToken::default();
                        while stream.next_token(&mut c_token).is_some() {
                            // Convert CToken to tantivy Token
                            let text = if c_token.text.is_null() {
                                String::new()
                            } else {
                                // SAFETY: Plugin guarantees text is valid UTF-8
                                unsafe {
                                    let slice = std::slice::from_raw_parts(
                                        c_token.text as *const u8,
                                        c_token.text_len as usize,
                                    );
                                    String::from_utf8_lossy(slice).into_owned()
                                }
                            };

                            tokens.push(Token {
                                offset_from: c_token.offset_from as usize,
                                offset_to: c_token.offset_to as usize,
                                position: c_token.position as usize,
                                text,
                                position_length: c_token.position_length as usize,
                            });
                        }
                    }
                    Err(e) => {
                        log::error!("Failed to create plugin token stream: {}", e);
                        error = true;
                    }
                },
                Err(e) => {
                    log::error!("Failed to create plugin tokenizer instance: {}", e);
                    error = true;
                }
            },
            Err(e) => {
                log::error!("Failed to create plugin factory: {}", e);
                error = true;
            }
        }

        Self {
            current_token: Token::default(),
            tokens,
            index: 0,
            error,
        }
    }
}

impl TokenStream for PluginTokenStreamAdapter {
    fn advance(&mut self) -> bool {
        if self.error || self.index >= self.tokens.len() {
            false
        } else {
            self.current_token = self.tokens[self.index].clone();
            self.index += 1;
            true
        }
    }

    fn token(&self) -> &Token {
        &self.current_token
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.current_token
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Integration tests with actual plugins would go in a separate test file
    // These are unit tests for the adapter logic

    #[test]
    fn test_empty_stream() {
        let adapter = PluginTokenStreamAdapter {
            current_token: Token::default(),
            tokens: vec![],
            index: 0,
            error: false,
        };
        assert_eq!(adapter.tokens.len(), 0);
    }

    #[test]
    fn test_error_stream() {
        let mut adapter = PluginTokenStreamAdapter {
            current_token: Token::default(),
            tokens: vec![Token {
                offset_from: 0,
                offset_to: 5,
                position: 0,
                text: "hello".to_string(),
                position_length: 1,
            }],
            index: 0,
            error: true,
        };
        // Error flag prevents advancement
        assert!(!adapter.advance());
    }

    #[test]
    fn test_token_iteration() {
        let mut adapter = PluginTokenStreamAdapter {
            current_token: Token::default(),
            tokens: vec![
                Token {
                    offset_from: 0,
                    offset_to: 5,
                    position: 0,
                    text: "hello".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 6,
                    offset_to: 11,
                    position: 1,
                    text: "world".to_string(),
                    position_length: 1,
                },
            ],
            index: 0,
            error: false,
        };

        assert!(adapter.advance());
        assert_eq!(adapter.token().text, "hello");
        assert_eq!(adapter.token().position, 0);

        assert!(adapter.advance());
        assert_eq!(adapter.token().text, "world");
        assert_eq!(adapter.token().position, 1);

        assert!(!adapter.advance());
    }
}
