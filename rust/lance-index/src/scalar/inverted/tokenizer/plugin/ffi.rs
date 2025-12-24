// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! FFI definitions for tokenizer plugin interface.
//!
//! These types mirror the C header file `include/lance_tokenizer_plugin.h`.

use std::ffi::{c_char, c_void};

pub const PLUGIN_API_VERSION: u32 = 1;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct CToken {
    /// Start byte offset in the original text (UTF-8)
    pub offset_from: u32,

    /// End byte offset in the original text (UTF-8)
    pub offset_to: u32,

    /// Position of this token in the sequence (0-indexed)
    pub position: u32,

    /// Pointer to the token text (null-terminated UTF-8)
    pub text: *const c_char,

    /// Length of the token text in bytes (not including null terminator)
    pub text_length: u32,

    /// Position length (usually 1, but can be > 1 for synonyms)
    pub position_length: u32,
}

impl Default for CToken {
    fn default() -> Self {
        Self {
            offset_from: 0,
            offset_to: 0,
            position: 0,
            text: std::ptr::null(),
            text_length: 0,
            position_length: 1,
        }
    }
}

pub type LanceTokenizerFactory = c_void;
pub type LanceTokenizer = c_void;
pub type LanceTokenStream = c_void;

#[repr(C)]
pub struct CTokenizerPlugin {
    pub api_version: unsafe extern "C" fn() -> u32,
    pub create_factory: unsafe extern "C" fn(
        config_json: *const c_char,
        config_len: u32,
    ) -> *mut LanceTokenizerFactory,
    pub destroy_factory: unsafe extern "C" fn(factory: *mut LanceTokenizerFactory),
    pub create_tokenizer:
        unsafe extern "C" fn(factory: *mut LanceTokenizerFactory) -> *mut LanceTokenizer,
    pub destroy_tokenizer: unsafe extern "C" fn(tokenizer: *mut LanceTokenizer),
    pub create_stream: unsafe extern "C" fn(
        tokenizer: *mut LanceTokenizer,
        text: *const c_char,
        text_length: u32,
    ) -> *mut LanceTokenStream,
    pub destroy_stream: unsafe extern "C" fn(stream: *mut LanceTokenStream),

    /// Get the next token from the stream.
    /// Returns 1 if a token was produced, 0 if no more tokens, negative on error.
    pub next_token: unsafe extern "C" fn(stream: *mut LanceTokenStream, token: *mut CToken) -> i32,

    /// Get the last error message.
    pub get_error: unsafe extern "C" fn(factory: *mut LanceTokenizerFactory) -> *const c_char,

    pub name: unsafe extern "C" fn() -> *const c_char,
    pub version: unsafe extern "C" fn() -> *const c_char,
}

pub type GetPluginFn = unsafe extern "C" fn() -> *const CTokenizerPlugin;
pub const ENTRY_POINT_SYMBOL: &[u8] = b"lance_tokenizer_get_plugin";

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn test_ctoken_layout() {
        // Verify CToken has expected size and alignment for C interop
        // On 64-bit systems: 3*u32 + padding + pointer + 2*u32 = 32 bytes
        // On 32-bit systems: 5*u32 + pointer = 24 bytes
        let expected_size = if mem::size_of::<*const c_char>() == 8 {
            32 // 64-bit: includes padding before pointer
        } else {
            24 // 32-bit: no padding needed
        };
        assert_eq!(mem::size_of::<CToken>(), expected_size);

        // Verify alignment matches pointer alignment (for C interop)
        assert_eq!(mem::align_of::<CToken>(), mem::align_of::<*const c_char>());
    }

    #[test]
    fn test_ctoken_default() {
        let token = CToken::default();
        assert_eq!(token.offset_from, 0);
        assert_eq!(token.offset_to, 0);
        assert_eq!(token.position, 0);
        assert!(token.text.is_null());
        assert_eq!(token.text_length, 0);
        assert_eq!(token.position_length, 1);
    }
}
