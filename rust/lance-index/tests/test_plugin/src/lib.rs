// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Test tokenizer plugin for integration tests.
//!
//! This is a simple whitespace tokenizer that can optionally lowercase tokens.

use std::ffi::{c_char, c_void, CString};
use std::ptr;
use std::sync::Mutex;

const PLUGIN_API_VERSION: u32 = 1;

#[repr(C)]
pub struct LanceToken {
    pub offset_from: u32,
    pub offset_to: u32,
    pub position: u32,
    pub text: *const c_char,
    pub text_len: u32,
    pub position_length: u32,
}

#[repr(C)]
pub struct LanceTokenizerPlugin {
    pub api_version: unsafe extern "C" fn() -> u32,
    pub create_factory: unsafe extern "C" fn(*const c_char, u32) -> *mut c_void,
    pub destroy_factory: unsafe extern "C" fn(*mut c_void),
    pub create_tokenizer: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    pub destroy_tokenizer: unsafe extern "C" fn(*mut c_void),
    pub create_stream: unsafe extern "C" fn(*mut c_void, *const c_char, u32) -> *mut c_void,
    pub destroy_stream: unsafe extern "C" fn(*mut c_void),
    pub next_token: unsafe extern "C" fn(*mut c_void, *mut LanceToken) -> i32,
    pub get_error: unsafe extern "C" fn(*mut c_void) -> *const c_char,
    pub name: unsafe extern "C" fn() -> *const c_char,
    pub version: unsafe extern "C" fn() -> *const c_char,
}

#[derive(Default, Clone)]
struct Config {
    lowercase: bool,
}

struct Factory {
    config: Config,
    last_error: Mutex<Option<CString>>,
}

struct Tokenizer {
    config: Config,
}

struct TokenStream {
    tokens: Vec<(usize, usize, String)>,
    index: usize,
    current_token_text: CString,
}

impl Factory {
    fn new(config_json: &str) -> Self {
        let lowercase = config_json.contains("\"lowercase\":true")
            || config_json.contains("\"lowercase\": true");
        Self {
            config: Config { lowercase },
            last_error: Mutex::new(None),
        }
    }

    fn get_error(&self) -> *const c_char {
        let guard = self.last_error.lock().unwrap();
        match &*guard {
            Some(s) => s.as_ptr(),
            None => ptr::null(),
        }
    }
}

impl Tokenizer {
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
    fn next(&mut self, token: &mut LanceToken) -> bool {
        if self.index >= self.tokens.len() {
            return false;
        }

        let (start, end, ref text) = self.tokens[self.index];
        self.current_token_text = CString::new(text.as_str()).unwrap_or_default();

        token.offset_from = start as u32;
        token.offset_to = end as u32;
        token.position = self.index as u32;
        token.text = self.current_token_text.as_ptr();
        token.text_len = text.len() as u32;
        token.position_length = 1;

        self.index += 1;
        true
    }
}

unsafe extern "C" fn api_version() -> u32 {
    PLUGIN_API_VERSION
}

unsafe extern "C" fn create_factory(config_json: *const c_char, config_len: u32) -> *mut c_void {
    let config_str = if config_json.is_null() || config_len == 0 {
        ""
    } else {
        let slice = std::slice::from_raw_parts(config_json as *const u8, config_len as usize);
        std::str::from_utf8(slice).unwrap_or("{}")
    };
    Box::into_raw(Box::new(Factory::new(config_str))) as *mut c_void
}

unsafe extern "C" fn destroy_factory(factory: *mut c_void) {
    if !factory.is_null() {
        drop(Box::from_raw(factory as *mut Factory));
    }
}

unsafe extern "C" fn create_tokenizer(factory: *mut c_void) -> *mut c_void {
    if factory.is_null() {
        return ptr::null_mut();
    }
    let factory = &*(factory as *const Factory);
    let tokenizer = Tokenizer {
        config: factory.config.clone(),
    };
    Box::into_raw(Box::new(tokenizer)) as *mut c_void
}

unsafe extern "C" fn destroy_tokenizer(tokenizer: *mut c_void) {
    if !tokenizer.is_null() {
        drop(Box::from_raw(tokenizer as *mut Tokenizer));
    }
}

unsafe extern "C" fn create_stream(
    tokenizer: *mut c_void,
    text: *const c_char,
    text_len: u32,
) -> *mut c_void {
    if tokenizer.is_null() || text.is_null() {
        return ptr::null_mut();
    }
    let tokenizer = &*(tokenizer as *const Tokenizer);
    let text_slice = std::slice::from_raw_parts(text as *const u8, text_len as usize);
    let text_str = String::from_utf8_lossy(text_slice).into_owned();
    let tokens = tokenizer.tokenize(&text_str);
    let stream = TokenStream {
        tokens,
        index: 0,
        current_token_text: CString::default(),
    };
    Box::into_raw(Box::new(stream)) as *mut c_void
}

unsafe extern "C" fn destroy_stream(stream: *mut c_void) {
    if !stream.is_null() {
        drop(Box::from_raw(stream as *mut TokenStream));
    }
}

unsafe extern "C" fn next_token(stream: *mut c_void, token: *mut LanceToken) -> i32 {
    if stream.is_null() || token.is_null() {
        return -1;
    }
    let stream = &mut *(stream as *mut TokenStream);
    if stream.next(&mut *token) {
        1
    } else {
        0
    }
}

unsafe extern "C" fn get_error(factory: *mut c_void) -> *const c_char {
    if factory.is_null() {
        return ptr::null();
    }
    let factory = &*(factory as *const Factory);
    factory.get_error()
}

unsafe extern "C" fn name() -> *const c_char {
    static NAME: &[u8] = b"test_whitespace_tokenizer\0";
    NAME.as_ptr() as *const c_char
}

unsafe extern "C" fn version() -> *const c_char {
    static VERSION: &[u8] = b"0.1.0\0";
    VERSION.as_ptr() as *const c_char
}

static PLUGIN: LanceTokenizerPlugin = LanceTokenizerPlugin {
    api_version,
    create_factory,
    destroy_factory,
    create_tokenizer,
    destroy_tokenizer,
    create_stream,
    destroy_stream,
    next_token,
    get_error,
    name,
    version,
};

#[no_mangle]
pub extern "C" fn lance_tokenizer_get_plugin() -> *const LanceTokenizerPlugin {
    &PLUGIN
}
