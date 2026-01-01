// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Example tokenizer plugin implementation.
//!
//! This is a simple whitespace tokenizer that demonstrates how to implement
//! a tokenizer plugin for Lance. It can be compiled as a shared library and
//! loaded at runtime.
//!
//! ## Building
//!
//! To build this as a shared library:
//!
//! ```bash
//! # Create a new library crate
//! cargo new --lib my_tokenizer
//! cd my_tokenizer
//!
//! # Add to Cargo.toml:
//! # [lib]
//! # crate-type = ["cdylib"]
//!
//! # Copy this file as src/lib.rs
//! # Build:
//! cargo build --release
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! let params = InvertedIndexParams::default()
//!     .plugin(
//!         "/path/to/libmy_tokenizer.so".to_string(),
//!         Some(r#"{"lowercase": true}"#.to_string()),
//!     );
//! ```

use std::ffi::{c_char, c_void};
use std::ptr;

/// Plugin API version - must match LANCE_TOKENIZER_PLUGIN_API_VERSION
const PLUGIN_API_VERSION: u32 = 1;

/// A reference to a UTF-8 string (not necessarily null-terminated).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LanceStringRef {
    pub data: *const c_char,
    pub length: u32,
}

impl LanceStringRef {
    fn from_str(s: &str) -> Self {
        Self {
            data: s.as_ptr() as *const c_char,
            length: s.len() as u32,
        }
    }

    unsafe fn as_str(&self) -> &str {
        if self.data.is_null() || self.length == 0 {
            ""
        } else {
            let slice = std::slice::from_raw_parts(self.data as *const u8, self.length as usize);
            std::str::from_utf8_unchecked(slice)
        }
    }
}

impl Default for LanceStringRef {
    fn default() -> Self {
        Self {
            data: ptr::null(),
            length: 0,
        }
    }
}

/// Error information returned by plugin functions.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Error {
    pub message: LanceStringRef,
}

/// Token structure matching the C ABI
#[repr(C)]
pub struct LanceToken {
    pub offset_from: u32,
    pub offset_to: u32,
    pub position: u32,
    pub position_length: u32,
    pub text: LanceStringRef,
}

/// Plugin interface structure
#[repr(C)]
pub struct LanceTokenizerPlugin {
    pub api_version: unsafe extern "C" fn() -> u32,
    pub create_factory: unsafe extern "C" fn(LanceStringRef, *mut Error) -> *mut c_void,
    pub destroy_factory: unsafe extern "C" fn(*mut c_void),
    pub create_tokenizer: unsafe extern "C" fn(*mut c_void, *mut Error) -> *mut c_void,
    pub destroy_tokenizer: unsafe extern "C" fn(*mut c_void),
    pub create_stream: unsafe extern "C" fn(*mut c_void, LanceStringRef, *mut Error) -> *mut c_void,
    pub destroy_stream: unsafe extern "C" fn(*mut c_void),
    pub next_token: unsafe extern "C" fn(*mut c_void, *mut LanceToken, *mut Error) -> i32,
    pub name: unsafe extern "C" fn() -> *const c_char,
    pub version: unsafe extern "C" fn() -> *const c_char,
}

/// Configuration for the tokenizer
#[derive(Default)]
struct Config {
    lowercase: bool,
}

/// Factory that creates tokenizers with shared configuration
struct Factory {
    config: Config,
}

/// Tokenizer instance
struct Tokenizer {
    config: Config,
}

/// Token stream for iterating over tokens
struct TokenStream {
    tokens: Vec<(usize, usize, String)>, // (start, end, text)
    index: usize,
    current_token_text: String,
}

impl Factory {
    fn new(config: &str) -> Self {
        let cfg = if config.is_empty() || config == "{}" {
            Config::default()
        } else {
            // Simple config parsing (in this example, JSON format is used)
            let lowercase =
                config.contains("\"lowercase\":true") || config.contains("\"lowercase\": true");
            Config { lowercase }
        };

        Self { config: cfg }
    }
}

impl Tokenizer {
    fn new(config: Config) -> Self {
        Self { config }
    }

    fn tokenize(&self, text: &str) -> Vec<(usize, usize, String)> {
        let mut tokens = Vec::new();
        let mut start = 0;
        let mut in_word = false;

        for (i, c) in text.char_indices() {
            if c.is_whitespace() {
                if in_word {
                    let word = &text[start..i];
                    let token_text = if self.config.lowercase {
                        word.to_lowercase()
                    } else {
                        word.to_string()
                    };
                    tokens.push((start, i, token_text));
                    in_word = false;
                }
            } else if !in_word {
                start = i;
                in_word = true;
            }
        }

        // Handle last word
        if in_word {
            let word = &text[start..];
            let token_text = if self.config.lowercase {
                word.to_lowercase()
            } else {
                word.to_string()
            };
            tokens.push((start, text.len(), token_text));
        }

        tokens
    }
}

impl TokenStream {
    fn new(tokens: Vec<(usize, usize, String)>) -> Self {
        Self {
            tokens,
            index: 0,
            current_token_text: String::new(),
        }
    }

    fn next(&mut self, token: &mut LanceToken) -> bool {
        if self.index >= self.tokens.len() {
            return false;
        }

        let (start, end, ref text) = self.tokens[self.index];
        self.current_token_text = text.clone();

        token.offset_from = start as u32;
        token.offset_to = end as u32;
        token.position = self.index as u32;
        token.position_length = 1;
        token.text = LanceStringRef {
            data: self.current_token_text.as_ptr() as *const c_char,
            length: self.current_token_text.len() as u32,
        };

        self.index += 1;
        true
    }
}

// --- C ABI Functions ---

unsafe extern "C" fn api_version() -> u32 {
    PLUGIN_API_VERSION
}

unsafe extern "C" fn create_factory(config: LanceStringRef, _error: *mut Error) -> *mut c_void {
    let config_str = config.as_str();
    Box::into_raw(Box::new(Factory::new(config_str))) as *mut c_void
}

unsafe extern "C" fn destroy_factory(factory: *mut c_void) {
    if !factory.is_null() {
        drop(Box::from_raw(factory as *mut Factory));
    }
}

unsafe extern "C" fn create_tokenizer(factory: *mut c_void, _error: *mut Error) -> *mut c_void {
    if factory.is_null() {
        return ptr::null_mut();
    }

    let factory = &*(factory as *const Factory);
    let tokenizer = Tokenizer::new(Config {
        lowercase: factory.config.lowercase,
    });
    Box::into_raw(Box::new(tokenizer)) as *mut c_void
}

unsafe extern "C" fn destroy_tokenizer(tokenizer: *mut c_void) {
    if !tokenizer.is_null() {
        drop(Box::from_raw(tokenizer as *mut Tokenizer));
    }
}

unsafe extern "C" fn create_stream(
    tokenizer: *mut c_void,
    text: LanceStringRef,
    _error: *mut Error,
) -> *mut c_void {
    if tokenizer.is_null() {
        return ptr::null_mut();
    }

    let tokenizer = &*(tokenizer as *const Tokenizer);
    let text_str = text.as_str().to_owned();

    let tokens = tokenizer.tokenize(&text_str);
    let stream = TokenStream::new(tokens);

    Box::into_raw(Box::new(stream)) as *mut c_void
}

unsafe extern "C" fn destroy_stream(stream: *mut c_void) {
    if !stream.is_null() {
        drop(Box::from_raw(stream as *mut TokenStream));
    }
}

unsafe extern "C" fn next_token(
    stream: *mut c_void,
    token: *mut LanceToken,
    _error: *mut Error,
) -> i32 {
    if stream.is_null() || token.is_null() {
        return -1;
    }

    let stream = &mut *(stream as *mut TokenStream);
    let token = &mut *token;

    if stream.next(token) {
        1
    } else {
        0
    }
}

unsafe extern "C" fn name() -> *const c_char {
    static NAME: &[u8] = b"simple_whitespace\0";
    NAME.as_ptr() as *const c_char
}

unsafe extern "C" fn version() -> *const c_char {
    static VERSION: &[u8] = b"1.0.0\0";
    VERSION.as_ptr() as *const c_char
}

/// Plugin interface singleton
static PLUGIN: LanceTokenizerPlugin = LanceTokenizerPlugin {
    api_version,
    create_factory,
    destroy_factory,
    create_tokenizer,
    destroy_tokenizer,
    create_stream,
    destroy_stream,
    next_token,
    name,
    version,
};

/// Entry point - must be exported with this exact name
#[no_mangle]
pub extern "C" fn lance_tokenizer_get_plugin() -> *const LanceTokenizerPlugin {
    &PLUGIN
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_basic() {
        let tokenizer = Tokenizer::new(Config { lowercase: false });
        let tokens = tokenizer.tokenize("Hello World");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].2, "Hello");
        assert_eq!(tokens[1].2, "World");
    }

    #[test]
    fn test_tokenizer_lowercase() {
        let tokenizer = Tokenizer::new(Config { lowercase: true });
        let tokens = tokenizer.tokenize("Hello World");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].2, "hello");
        assert_eq!(tokens[1].2, "world");
    }

    #[test]
    fn test_tokenizer_multiple_spaces() {
        let tokenizer = Tokenizer::new(Config { lowercase: false });
        let tokens = tokenizer.tokenize("  Hello   World  ");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, 2); // offset
        assert_eq!(tokens[0].1, 7);
        assert_eq!(tokens[1].0, 10);
        assert_eq!(tokens[1].1, 15);
    }
}
