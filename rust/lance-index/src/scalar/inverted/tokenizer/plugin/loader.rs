// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ffi::CStr;
use std::path::Path;
use std::sync::Arc;

use lance_core::{Error, Result};
use libloading::Library;

use super::ffi::{
    CError, CStringRef, CToken, CTokenizerPlugin, ENTRY_POINT_SYMBOL, GetPluginFn,
    LanceTokenStream, LanceTokenizer, LanceTokenizerFactory, PLUGIN_API_VERSION,
};

pub struct TokenizerPluginLibrary {
    _library: Library,

    plugin: *const CTokenizerPlugin,
}

// SAFETY: TokenizerPluginLibrary can be shared across threads because:
// 1. `_library` (libloading::Library) is Send + Sync
// 2. `plugin` points to an immutable vtable of function pointers that
//    remains valid as long as `_library` is alive
// 3. The plugin functions themselves are stateless; mutable state is
//    confined to Factory/Tokenizer/Stream instances which are NOT
//    Send/Sync and must be used on a single thread
unsafe impl Send for TokenizerPluginLibrary {}
unsafe impl Sync for TokenizerPluginLibrary {}

impl TokenizerPluginLibrary {
    pub fn load(path: impl AsRef<Path>) -> Result<Arc<Self>> {
        let path = path.as_ref();

        let library = unsafe { Library::new(path) }.map_err(|e| {
            Error::invalid_input(format!(
                "failed to load tokenizer plugin from {:?}: {}",
                path, e
            ))
        })?;

        let get_plugin: libloading::Symbol<GetPluginFn> =
            unsafe { library.get(ENTRY_POINT_SYMBOL) }.map_err(|e| {
                Error::invalid_input(format!(
                    "tokenizer plugin {:?} missing entry point '{}': {}",
                    path,
                    String::from_utf8_lossy(ENTRY_POINT_SYMBOL),
                    e
                ))
            })?;

        let plugin = unsafe { get_plugin() };
        if plugin.is_null() {
            return Err(Error::invalid_input(format!(
                "tokenizer plugin {:?} returned null plugin interface",
                path
            )));
        }

        // Check API version
        let api_version = unsafe { ((*plugin).api_version)() };
        if api_version != PLUGIN_API_VERSION {
            return Err(Error::invalid_input(format!(
                "tokenizer plugin {:?} has incompatible API version {} (expected {})",
                path, api_version, PLUGIN_API_VERSION
            )));
        }

        Ok(Arc::new(Self {
            _library: library,
            plugin,
        }))
    }

    pub fn name(&self) -> &str {
        unsafe {
            let name_ptr = ((*self.plugin).name)();
            if name_ptr.is_null() {
                "unknown"
            } else {
                CStr::from_ptr(name_ptr).to_str().unwrap_or("unknown")
            }
        }
    }

    pub fn version(&self) -> &str {
        unsafe {
            let version_ptr = ((*self.plugin).version)();
            if version_ptr.is_null() {
                "unknown"
            } else {
                CStr::from_ptr(version_ptr).to_str().unwrap_or("unknown")
            }
        }
    }

    pub fn create_factory(&self, config: &str) -> Result<PluginFactory<'_>> {
        let mut error = CError::default();
        let factory =
            unsafe { ((*self.plugin).create_factory)(CStringRef::from_str(config), &mut error) };

        if factory.is_null() {
            let error_msg = if error.has_message() {
                unsafe { error.message_str().to_string() }
            } else {
                "unknown error".to_string()
            };
            return Err(Error::invalid_input(format!(
                "failed to create tokenizer factory: {}",
                error_msg
            )));
        }

        Ok(PluginFactory {
            library: self,
            factory,
        })
    }

    unsafe fn destroy_factory(&self, factory: *mut LanceTokenizerFactory) {
        if !factory.is_null() {
            ((*self.plugin).destroy_factory)(factory);
        }
    }

    unsafe fn create_tokenizer(
        &self,
        factory: *mut LanceTokenizerFactory,
    ) -> Result<*mut LanceTokenizer> {
        let mut error = CError::default();
        let tokenizer = ((*self.plugin).create_tokenizer)(factory, &mut error);
        if tokenizer.is_null() {
            let error_msg = if error.has_message() {
                error.message_str().to_string()
            } else {
                "unknown error".to_string()
            };
            return Err(Error::invalid_input(format!(
                "failed to create tokenizer: {}",
                error_msg
            )));
        }
        Ok(tokenizer)
    }

    unsafe fn destroy_tokenizer(&self, tokenizer: *mut LanceTokenizer) {
        if !tokenizer.is_null() {
            ((*self.plugin).destroy_tokenizer)(tokenizer);
        }
    }

    unsafe fn create_stream(
        &self,
        tokenizer: *mut LanceTokenizer,
        text: &str,
    ) -> Result<*mut LanceTokenStream> {
        let mut error = CError::default();
        let stream =
            ((*self.plugin).create_stream)(tokenizer, CStringRef::from_str(text), &mut error);
        if stream.is_null() {
            let error_msg = if error.has_message() {
                error.message_str().to_string()
            } else {
                "unknown error".to_string()
            };
            return Err(Error::invalid_input(format!(
                "failed to create token stream: {}",
                error_msg
            )));
        }
        Ok(stream)
    }

    unsafe fn destroy_stream(&self, stream: *mut LanceTokenStream) {
        if !stream.is_null() {
            ((*self.plugin).destroy_stream)(stream);
        }
    }

    unsafe fn next_token(
        &self,
        stream: *mut LanceTokenStream,
        token: &mut CToken,
        error: &mut CError,
    ) -> i32 {
        ((*self.plugin).next_token)(stream, token, error)
    }
}

// Note: PluginFactory, PluginTokenizerInstance, and PluginTokenStream
//       are not Send/Sync because they hold raw pointers to plugin state.
//       Each thread should create its own instances.

/// PluginFactory holds shared resources (like dictionaries) and can create
/// multiple tokenizer instances.
pub struct PluginFactory<'a> {
    library: &'a TokenizerPluginLibrary,
    factory: *mut LanceTokenizerFactory,
}

impl<'a> PluginFactory<'a> {
    pub fn create_tokenizer(&self) -> Result<PluginTokenizerInstance<'a>> {
        let tokenizer = unsafe { self.library.create_tokenizer(self.factory)? };
        Ok(PluginTokenizerInstance {
            library: self.library,
            tokenizer,
        })
    }
}

impl Drop for PluginFactory<'_> {
    fn drop(&mut self) {
        unsafe {
            self.library.destroy_factory(self.factory);
        }
    }
}

/// A tokenizer instance created from a plugin factory.
pub struct PluginTokenizerInstance<'a> {
    library: &'a TokenizerPluginLibrary,
    tokenizer: *mut LanceTokenizer,
}

impl<'a> PluginTokenizerInstance<'a> {
    pub fn create_stream(&self, text: &str) -> Result<PluginTokenStream<'a>> {
        let stream = unsafe { self.library.create_stream(self.tokenizer, text)? };
        Ok(PluginTokenStream {
            library: self.library,
            stream,
        })
    }
}

impl Drop for PluginTokenizerInstance<'_> {
    fn drop(&mut self) {
        unsafe {
            self.library.destroy_tokenizer(self.tokenizer);
        }
    }
}

/// A token stream from a plugin tokenizer.
pub struct PluginTokenStream<'a> {
    library: &'a TokenizerPluginLibrary,
    stream: *mut LanceTokenStream,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NextTokenResult {
    Token,
    EndOfStream,
    Error(i32, String),
}

impl PluginTokenStream<'_> {
    pub fn next_token(&mut self, token: &mut CToken) -> NextTokenResult {
        let mut error = CError::default();
        let result = unsafe { self.library.next_token(self.stream, token, &mut error) };
        if result > 0 {
            NextTokenResult::Token
        } else if result == 0 {
            NextTokenResult::EndOfStream
        } else {
            let error_msg = if error.has_message() {
                unsafe { error.message_str().to_string() }
            } else {
                format!("tokenizer error code: {}", result)
            };
            NextTokenResult::Error(result, error_msg)
        }
    }
}

impl Drop for PluginTokenStream<'_> {
    fn drop(&mut self) {
        unsafe {
            self.library.destroy_stream(self.stream);
        }
    }
}

/// An owned plugin factory that holds an Arc to the library.
/// This allows the factory to be cached and reused across multiple tokenizations.
pub struct OwnedPluginFactory {
    library: Arc<TokenizerPluginLibrary>,
    factory: *mut LanceTokenizerFactory,
}

// SAFETY: OwnedPluginFactory can be sent/shared across threads because:
// 1. `library` (Arc<TokenizerPluginLibrary>) is Send + Sync
// 2. `factory` is a raw pointer to plugin state that is only accessed through
//    the library's thread-safe function pointers
// 3. The factory is used with &mut self, ensuring single-threaded access
unsafe impl Send for OwnedPluginFactory {}
unsafe impl Sync for OwnedPluginFactory {}

impl OwnedPluginFactory {
    /// Create a new owned factory from a library and config.
    pub fn new(library: Arc<TokenizerPluginLibrary>, config: &str) -> Result<Self> {
        let mut error = CError::default();
        let factory =
            unsafe { ((*library.plugin).create_factory)(CStringRef::from_str(config), &mut error) };

        if factory.is_null() {
            let error_msg = if error.has_message() {
                unsafe { error.message_str().to_string() }
            } else {
                "unknown error".to_string()
            };
            return Err(Error::invalid_input(format!(
                "failed to create tokenizer factory: {}",
                error_msg
            )));
        }

        Ok(Self { library, factory })
    }

    /// Create a tokenizer instance from this factory.
    pub fn create_tokenizer(&self) -> Result<OwnedPluginTokenizerInstance> {
        let mut error = CError::default();
        let tokenizer =
            unsafe { ((*self.library.plugin).create_tokenizer)(self.factory, &mut error) };
        if tokenizer.is_null() {
            let error_msg = if error.has_message() {
                unsafe { error.message_str().to_string() }
            } else {
                "unknown error".to_string()
            };
            return Err(Error::invalid_input(format!(
                "failed to create tokenizer: {}",
                error_msg
            )));
        }
        Ok(OwnedPluginTokenizerInstance {
            library: Arc::clone(&self.library),
            tokenizer,
        })
    }
}

impl Drop for OwnedPluginFactory {
    fn drop(&mut self) {
        unsafe {
            self.library.destroy_factory(self.factory);
        }
    }
}

/// An owned tokenizer instance created from an owned factory.
pub struct OwnedPluginTokenizerInstance {
    library: Arc<TokenizerPluginLibrary>,
    tokenizer: *mut LanceTokenizer,
}

impl OwnedPluginTokenizerInstance {
    /// Create a token stream for the given text.
    pub fn create_stream(&self, text: &str) -> Result<OwnedPluginTokenStream> {
        let mut error = CError::default();
        let stream = unsafe {
            ((*self.library.plugin).create_stream)(
                self.tokenizer,
                CStringRef::from_str(text),
                &mut error,
            )
        };
        if stream.is_null() {
            let error_msg = if error.has_message() {
                unsafe { error.message_str().to_string() }
            } else {
                "unknown error".to_string()
            };
            return Err(Error::invalid_input(format!(
                "failed to create token stream: {}",
                error_msg
            )));
        }
        Ok(OwnedPluginTokenStream {
            library: Arc::clone(&self.library),
            stream,
        })
    }
}

impl Drop for OwnedPluginTokenizerInstance {
    fn drop(&mut self) {
        unsafe {
            self.library.destroy_tokenizer(self.tokenizer);
        }
    }
}

/// An owned token stream from an owned tokenizer instance.
pub struct OwnedPluginTokenStream {
    library: Arc<TokenizerPluginLibrary>,
    stream: *mut LanceTokenStream,
}

impl OwnedPluginTokenStream {
    pub fn next_token(&mut self, token: &mut CToken) -> NextTokenResult {
        let mut error = CError::default();
        let result = unsafe { ((*self.library.plugin).next_token)(self.stream, token, &mut error) };
        if result > 0 {
            NextTokenResult::Token
        } else if result == 0 {
            NextTokenResult::EndOfStream
        } else {
            let error_msg = if error.has_message() {
                unsafe { error.message_str().to_string() }
            } else {
                format!("tokenizer error code: {}", result)
            };
            NextTokenResult::Error(result, error_msg)
        }
    }
}

impl Drop for OwnedPluginTokenStream {
    fn drop(&mut self) {
        unsafe {
            self.library.destroy_stream(self.stream);
        }
    }
}
