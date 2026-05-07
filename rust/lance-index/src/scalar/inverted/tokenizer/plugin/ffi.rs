// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! FFI definitions for tokenizer plugin interface.
//!
//! These types mirror the C header file `include/lance_tokenizer_plugin.h`,
//! which is the authoritative source for the ABI contract.

use std::borrow::Cow;
use std::ffi::{c_char, c_void};

use lance_core::{Error, Result};

pub const PLUGIN_API_VERSION: u32 = 1;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CStringRef {
    pub data: *const c_char,
    pub length: u32,
}

impl CStringRef {
    /// The plugin C ABI encodes string lengths as `u32`. Casting `usize`
    /// would silently wrap and corrupt FTS results by emitting a partial
    /// token stream, so reject inputs that don't fit instead.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self> {
        Self::from_str_with_limit(s, u32::MAX)
    }

    /// `from_str` with an injectable limit so unit tests can exercise the
    /// rejection path without a 4 GiB allocation.
    pub(crate) fn from_str_with_limit(s: &str, limit: u32) -> Result<Self> {
        if s.len() > limit as usize {
            return Err(Error::invalid_input(format!(
                "input is {} bytes; the plugin C ABI uses a u32 length, so \
                 values larger than {} bytes cannot be passed without \
                 silently truncating the token stream",
                s.len(),
                limit,
            )));
        }
        Ok(Self {
            data: s.as_ptr() as *const c_char,
            length: s.len() as u32,
        })
    }

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

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CError {
    pub message: CStringRef,
}

impl CError {
    pub fn has_message(&self) -> bool {
        !self.message.data.is_null() && self.message.length > 0
    }

    /// # Safety
    /// The caller must ensure the message data pointer is valid.
    pub unsafe fn message_str(&self) -> Cow<'_, str> {
        self.message.to_string_lossy()
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct CToken {
    pub offset_from: u32,
    pub offset_to: u32,
    pub position: u32,
    pub position_length: u32,
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

/// Plugin vtable. Function pointers are nullable (`Option<extern fn>`) to
/// match the C ABI of a struct that may carry `NULL` for un-implemented
/// callbacks; the loader rejects any such NULL before the rest of the
/// codebase touches it. `Option<extern "C" fn>` is laid out as a single
/// pointer thanks to the niche optimization, so it is ABI-compatible with
/// a bare function pointer.
#[repr(C)]
pub struct CTokenizerPlugin {
    pub api_version: Option<unsafe extern "C" fn() -> u32>,
    pub create_factory: Option<
        unsafe extern "C" fn(config: CStringRef, error: *mut CError) -> *mut LanceTokenizerFactory,
    >,
    pub destroy_factory: Option<unsafe extern "C" fn(factory: *mut LanceTokenizerFactory)>,
    pub create_tokenizer: Option<
        unsafe extern "C" fn(
            factory: *mut LanceTokenizerFactory,
            error: *mut CError,
        ) -> *mut LanceTokenizer,
    >,
    pub destroy_tokenizer: Option<unsafe extern "C" fn(tokenizer: *mut LanceTokenizer)>,
    pub create_stream: Option<
        unsafe extern "C" fn(
            tokenizer: *mut LanceTokenizer,
            text: CStringRef,
            error: *mut CError,
        ) -> *mut LanceTokenStream,
    >,
    pub destroy_stream: Option<unsafe extern "C" fn(stream: *mut LanceTokenStream)>,
    pub next_token: Option<
        unsafe extern "C" fn(
            stream: *mut LanceTokenStream,
            token: *mut CToken,
            error: *mut CError,
        ) -> i32,
    >,
    pub name: Option<unsafe extern "C" fn() -> *const c_char>,
    pub version: Option<unsafe extern "C" fn() -> *const c_char>,
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
        let sr = CStringRef::from_str(s).expect("ascii input must fit");
        assert_eq!(sr.length, 5);
        assert!(!sr.data.is_null());

        unsafe {
            assert_eq!(sr.to_string_lossy(), "hello");
        }
    }

    /// `from_str_with_limit` must reject inputs that would silently wrap
    /// when cast to the ABI's `u32` length.
    #[test]
    fn test_cstring_ref_rejects_input_over_u32_len_limit() {
        let s = "abcdef"; // 6 bytes
        let err = CStringRef::from_str_with_limit(s, 5)
            .expect_err("string longer than the limit must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("u32 length"), "got: {}", msg);
        assert!(msg.contains("silently truncating"), "got: {}", msg);
    }

    /// Pin the inclusive boundary so a future "make the wrap impossible"
    /// rewrite cannot quietly tighten the contract.
    #[test]
    fn test_cstring_ref_accepts_input_at_limit() {
        let s = "abcde"; // 5 bytes, exactly equals the limit
        let sr = CStringRef::from_str_with_limit(s, 5).expect("input at the limit must succeed");
        assert_eq!(sr.length, 5);
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
            // Invalid byte should be replaced with U+FFFD
            assert_eq!(sr.to_string_lossy(), "hello\u{FFFD}");
        }
    }
}
