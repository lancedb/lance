// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! FFI definitions for tokenizer plugin interface.
//!
//! These types mirror the C header file `include/lance_tokenizer_plugin.h`.

use std::borrow::Cow;
use std::ffi::{c_char, c_void};

pub const PLUGIN_API_VERSION: u32 = 1;

/// A reference to a UTF-8 string that provides a zero-copy way to pass strings between Rust and C.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CStringRef {
    pub data: *const c_char,
    pub length: u32,
}

impl CStringRef {
    /// Create a CStringRef from a Rust string slice.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        Self {
            data: s.as_ptr() as *const c_char,
            length: s.len() as u32,
        }
    }

    /// Convert to a Rust string, safely handling invalid UTF-8.
    ///
    /// This method uses lossy conversion, replacing invalid UTF-8 sequences
    /// with the Unicode replacement character (U+FFFD). Use this when reading
    /// data from untrusted sources (e.g., plugin error messages).
    ///
    /// # Safety
    /// The caller must ensure the data pointer is valid and points to allocated memory.
    pub unsafe fn to_string_lossy(&self) -> Cow<'_, str> {
        if self.data.is_null() || self.length == 0 {
            Cow::Borrowed("")
        } else {
            let slice = std::slice::from_raw_parts(self.data as *const u8, self.length as usize);
            String::from_utf8_lossy(slice)
        }
    }
}

impl Default for CStringRef {
    fn default() -> Self {
        Self {
            data: std::ptr::null(),
            length: 0,
        }
    }
}

/// Error information returned by plugin functions.
/// The message is valid until the next call on the same object or until destruction.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CError {
    pub message: CStringRef,
}

impl CError {
    /// Check if this error contains a message.
    pub fn has_message(&self) -> bool {
        !self.message.data.is_null() && self.message.length > 0
    }

    /// Get the error message as a string, safely handling invalid UTF-8.
    ///
    /// Since error messages come from plugins (untrusted sources),
    /// invalid UTF-8 sequences are replaced with the replacement character.
    ///
    /// # Safety
    /// The caller must ensure the message data pointer is valid.
    pub unsafe fn message_str(&self) -> Cow<'_, str> {
        self.message.to_string_lossy()
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct CToken {
    /// Start byte offset in the original text (UTF-8)
    pub offset_from: u32,

    /// End byte offset in the original text (UTF-8)
    pub offset_to: u32,

    /// Position of this token in the sequence (0-indexed)
    pub position: u32,

    /// Position length (usually 1, but can be > 1 for synonyms)
    pub position_length: u32,

    /// Token text (UTF-8)
    pub text: CStringRef,
}

impl Default for CToken {
    fn default() -> Self {
        Self {
            offset_from: 0,
            offset_to: 0,
            position: 0,
            position_length: 1,
            text: CStringRef::default(),
        }
    }
}

pub type LanceTokenizerFactory = c_void;
pub type LanceTokenizer = c_void;
pub type LanceTokenStream = c_void;

#[repr(C)]
pub struct CTokenizerPlugin {
    pub api_version: unsafe extern "C" fn() -> u32,
    pub create_factory:
        unsafe extern "C" fn(config: CStringRef, error: *mut CError) -> *mut LanceTokenizerFactory,
    pub destroy_factory: unsafe extern "C" fn(factory: *mut LanceTokenizerFactory),
    pub create_tokenizer: unsafe extern "C" fn(
        factory: *mut LanceTokenizerFactory,
        error: *mut CError,
    ) -> *mut LanceTokenizer,
    pub destroy_tokenizer: unsafe extern "C" fn(tokenizer: *mut LanceTokenizer),
    pub create_stream: unsafe extern "C" fn(
        tokenizer: *mut LanceTokenizer,
        text: CStringRef,
        error: *mut CError,
    ) -> *mut LanceTokenStream,
    pub destroy_stream: unsafe extern "C" fn(stream: *mut LanceTokenStream),

    /// Get the next token from the stream.
    /// Returns 1 if a token was produced, 0 if no more tokens, negative on error.
    pub next_token: unsafe extern "C" fn(
        stream: *mut LanceTokenStream,
        token: *mut CToken,
        error: *mut CError,
    ) -> i32,

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
    fn test_cstring_ref_from_str() {
        let s = "hello";
        let sr = CStringRef::from_str(s);
        assert_eq!(sr.length, 5);
        assert!(!sr.data.is_null());

        unsafe {
            assert_eq!(sr.to_string_lossy(), "hello");
        }
    }

    #[test]
    fn test_cstring_ref_empty() {
        let sr = CStringRef::default();
        assert!(sr.data.is_null());
        assert_eq!(sr.length, 0);

        unsafe {
            assert_eq!(sr.to_string_lossy(), "");
        }
    }

    #[test]
    fn test_ctoken_layout() {
        // Verify CToken has expected size and alignment for C interop
        // CToken: 4*u32 + CStringRef (pointer + u32)
        let expected_size = if mem::size_of::<*const c_char>() == 8 {
            32 // 64-bit: 4*4 + 8 + 4 + padding = 32
        } else {
            24 // 32-bit: 4*4 + 4 + 4 = 24
        };
        assert_eq!(mem::size_of::<CToken>(), expected_size);

        // Verify alignment matches pointer alignment (for C interop)
        assert_eq!(mem::align_of::<CToken>(), mem::align_of::<*const c_char>());
    }

    #[test]
    fn test_cstringref_to_string_lossy_with_invalid_utf8() {
        let invalid_utf8: [u8; 6] = [0x68, 0x65, 0x6c, 0x6c, 0x6F, 0xFF]; // "hello" + invalid byte
        let sr = CStringRef {
            data: invalid_utf8.as_ptr() as *const c_char,
            length: 6,
        };

        unsafe {
            let result = sr.to_string_lossy();
            // Invalid byte should be replaced with U+FFFD
            assert_eq!(result, "hello\u{FFFD}");
        }
    }
}
