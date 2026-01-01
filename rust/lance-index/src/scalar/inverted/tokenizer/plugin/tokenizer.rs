// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::path::Path;
use std::sync::Arc;

use lance_core::Result;
use tantivy::tokenizer::{BoxTokenStream, Token, TokenStream};

use super::ffi::CToken;
use super::loader::{NextTokenResult, TokenizerPluginLibrary};
use crate::scalar::inverted::tokenizer::document_tokenizer::{DocType, LanceTokenizer};

/// PluginTokenizer loads a shared library at runtime and uses its tokenization
/// functions through the C ABI interface.
pub struct PluginTokenizer {
    /// The loaded plugin library (shared across clones)
    library: Arc<TokenizerPluginLibrary>,

    /// Configuration string for creating factories (format defined by plugin)
    config: String,
}

impl PluginTokenizer {
    pub fn new(library_path: impl AsRef<Path>, config: impl Into<String>) -> Result<Self> {
        let library = TokenizerPluginLibrary::load(library_path)?;
        Ok(Self {
            library,
            config: config.into(),
        })
    }

    pub fn from_library(library: Arc<TokenizerPluginLibrary>, config: impl Into<String>) -> Self {
        Self {
            library,
            config: config.into(),
        }
    }

    pub fn plugin_name(&self) -> &str {
        self.library.name()
    }

    pub fn plugin_version(&self) -> &str {
        self.library.version()
    }

    fn create_stream<'a>(&'a mut self, text: &'a str) -> BoxTokenStream<'a> {
        // Note: This is not the most efficient approach for repeated tokenization,
        //       but it ensures thread safety and simplifies lifetime management.
        //       For production use, consider caching the factory/tokenizer.
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
        // Plugin tokenizers are currently text-based only
        DocType::Text
    }
}

/// Adapter that implements tantivy's TokenStream trait for plugin streams.
struct PluginTokenStreamAdapter {
    current_token: Token,
    tokens: Vec<Token>,
    index: usize,
    has_error: bool,
}

impl PluginTokenStreamAdapter {
    fn new(library: Arc<TokenizerPluginLibrary>, config: &str, text: &str) -> Self {
        let mut tokens = Vec::new();
        let mut has_error = false;

        // Create factory, tokenizer, and stream, then collect all tokens
        match library.create_factory(config) {
            Ok(factory) => match factory.create_tokenizer() {
                Ok(tokenizer_instance) => match tokenizer_instance.create_stream(text) {
                    Ok(mut stream) => {
                        let mut c_token = CToken::default();
                        loop {
                            match stream.next_token(&mut c_token) {
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

                                    tokens.push(Token {
                                        offset_from: c_token.offset_from as usize,
                                        offset_to: c_token.offset_to as usize,
                                        position: c_token.position as usize,
                                        text,
                                        position_length: c_token.position_length as usize,
                                    });
                                }
                                NextTokenResult::EndOfStream => {
                                    break;
                                }
                                NextTokenResult::Error(code, msg) => {
                                    log::error!(
                                        "Plugin tokenizer error during tokenization (code: {}): {}",
                                        code,
                                        msg
                                    );
                                    has_error = true;
                                    tokens.clear();
                                    break;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        log::error!("Failed to create plugin token stream: {}", e);
                        has_error = true;
                    }
                },
                Err(e) => {
                    log::error!("Failed to create plugin tokenizer instance: {}", e);
                    has_error = true;
                }
            },
            Err(e) => {
                log::error!("Failed to create plugin factory: {}", e);
                has_error = true;
            }
        }

        Self {
            current_token: Token::default(),
            tokens,
            index: 0,
            has_error,
        }
    }
}

impl TokenStream for PluginTokenStreamAdapter {
    fn advance(&mut self) -> bool {
        if self.has_error || self.index >= self.tokens.len() {
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

    #[test]
    fn test_empty_stream() {
        let mut adapter = PluginTokenStreamAdapter {
            current_token: Token::default(),
            tokens: vec![],
            index: 0,
            has_error: false,
        };
        assert!(!adapter.advance());
    }

    #[test]
    fn test_error_flag_prevents_advance() {
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
            has_error: true,
        };
        assert!(!adapter.advance());
    }

    #[test]
    fn test_stream_exhausted() {
        let mut adapter = PluginTokenStreamAdapter {
            current_token: Token::default(),
            tokens: vec![Token {
                offset_from: 0,
                offset_to: 5,
                position: 0,
                text: "hello".to_string(),
                position_length: 1,
            }],
            index: 1,
            has_error: false,
        };
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
            has_error: false,
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
