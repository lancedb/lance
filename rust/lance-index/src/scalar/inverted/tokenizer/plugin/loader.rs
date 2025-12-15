// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Dynamic library loader for tokenizer plugins.

use std::ffi::{c_char, CStr, CString};
use std::path::Path;
use std::sync::Arc;

use lance_core::{Error, Result};
use libloading::Library;
use snafu::location;

use super::ffi::{
    CToken, CTokenizerPlugin, GetPluginFn, LanceTokenStream, LanceTokenizer, LanceTokenizerFactory,
    ENTRY_POINT_SYMBOL, PLUGIN_API_VERSION,
};

/// A loaded tokenizer plugin library.
///
/// This struct manages the lifetime of a dynamically loaded plugin library.
/// The library is kept alive as long as this struct exists.
pub struct TokenizerPluginLibrary {
    /// The loaded library (kept alive to prevent unloading)
    _library: Library,

    /// Pointer to the plugin interface
    plugin: *const CTokenizerPlugin,
}

// SAFETY: The plugin interface is thread-safe as long as we don't share
// mutable state across threads. Each tokenizer instance is independent.
unsafe impl Send for TokenizerPluginLibrary {}
unsafe impl Sync for TokenizerPluginLibrary {}

impl TokenizerPluginLibrary {
    /// Load a tokenizer plugin from the given path.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the dynamic library (.so, .dylib, or .dll)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The library cannot be loaded
    /// - The entry point symbol is not found
    /// - The plugin API version is incompatible
    pub fn load(path: impl AsRef<Path>) -> Result<Arc<Self>> {
        let path = path.as_ref();

        // SAFETY: Loading a dynamic library is inherently unsafe.
        // We trust that the library at the given path is a valid tokenizer plugin.
        let library = unsafe { Library::new(path) }.map_err(|e| Error::InvalidInput {
            source: format!("failed to load tokenizer plugin from {:?}: {}", path, e).into(),
            location: location!(),
        })?;

        // Get the entry point function
        // SAFETY: We trust the symbol exists and has the correct signature
        let get_plugin: libloading::Symbol<GetPluginFn> =
            unsafe { library.get(ENTRY_POINT_SYMBOL) }.map_err(|e| Error::InvalidInput {
                source: format!(
                    "tokenizer plugin {:?} missing entry point '{}': {}",
                    path,
                    String::from_utf8_lossy(ENTRY_POINT_SYMBOL),
                    e
                )
                .into(),
                location: location!(),
            })?;

        // Get the plugin interface
        // SAFETY: We trust the function returns a valid plugin pointer
        let plugin = unsafe { get_plugin() };
        if plugin.is_null() {
            return Err(Error::InvalidInput {
                source: format!("tokenizer plugin {:?} returned null plugin interface", path)
                    .into(),
                location: location!(),
            });
        }

        // Check API version
        let api_version = unsafe { ((*plugin).api_version)() };
        if api_version != PLUGIN_API_VERSION {
            return Err(Error::InvalidInput {
                source: format!(
                    "tokenizer plugin {:?} has incompatible API version {} (expected {})",
                    path, api_version, PLUGIN_API_VERSION
                )
                .into(),
                location: location!(),
            });
        }

        Ok(Arc::new(Self {
            _library: library,
            plugin,
        }))
    }

    /// Get the plugin name.
    pub fn name(&self) -> &str {
        // SAFETY: plugin is valid and name() returns a static string
        unsafe {
            let name_ptr = ((*self.plugin).name)();
            if name_ptr.is_null() {
                "unknown"
            } else {
                CStr::from_ptr(name_ptr).to_str().unwrap_or("unknown")
            }
        }
    }

    /// Get the plugin version.
    pub fn version(&self) -> &str {
        // SAFETY: plugin is valid and version() returns a static string
        unsafe {
            let version_ptr = ((*self.plugin).version)();
            if version_ptr.is_null() {
                "unknown"
            } else {
                CStr::from_ptr(version_ptr).to_str().unwrap_or("unknown")
            }
        }
    }

    /// Get the last error message from the factory.
    fn get_error(&self, factory: *mut LanceTokenizerFactory) -> Option<String> {
        // SAFETY: plugin is valid
        unsafe {
            let error_ptr = ((*self.plugin).get_error)(factory);
            if error_ptr.is_null() {
                None
            } else {
                Some(CStr::from_ptr(error_ptr).to_string_lossy().into_owned())
            }
        }
    }

    /// Create a tokenizer factory with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config_json` - JSON configuration string
    ///
    /// # Returns
    ///
    /// A factory handle that can be used to create tokenizer instances.
    pub fn create_factory(&self, config_json: &str) -> Result<PluginFactory<'_>> {
        let config_cstring = CString::new(config_json).map_err(|e| Error::InvalidInput {
            source: format!("invalid config JSON (contains null byte): {}", e).into(),
            location: location!(),
        })?;

        // SAFETY: plugin is valid and config_cstring is properly null-terminated
        let factory = unsafe {
            ((*self.plugin).create_factory)(config_cstring.as_ptr(), config_json.len() as u32)
        };

        if factory.is_null() {
            let error = self.get_error(std::ptr::null_mut());
            return Err(Error::InvalidInput {
                source: format!(
                    "failed to create tokenizer factory: {}",
                    error.unwrap_or_else(|| "unknown error".to_string())
                )
                .into(),
                location: location!(),
            });
        }

        Ok(PluginFactory {
            library: self,
            factory,
        })
    }

    /// Destroy a factory.
    ///
    /// # Safety
    ///
    /// The factory must have been created by this library.
    unsafe fn destroy_factory(&self, factory: *mut LanceTokenizerFactory) {
        if !factory.is_null() {
            ((*self.plugin).destroy_factory)(factory);
        }
    }

    /// Create a tokenizer from a factory.
    ///
    /// # Safety
    ///
    /// The factory must have been created by this library.
    unsafe fn create_tokenizer(
        &self,
        factory: *mut LanceTokenizerFactory,
    ) -> Result<*mut LanceTokenizer> {
        let tokenizer = ((*self.plugin).create_tokenizer)(factory);
        if tokenizer.is_null() {
            let error = self.get_error(factory);
            return Err(Error::InvalidInput {
                source: format!(
                    "failed to create tokenizer: {}",
                    error.unwrap_or_else(|| "unknown error".to_string())
                )
                .into(),
                location: location!(),
            });
        }
        Ok(tokenizer)
    }

    /// Destroy a tokenizer.
    ///
    /// # Safety
    ///
    /// The tokenizer must have been created by this library.
    unsafe fn destroy_tokenizer(&self, tokenizer: *mut LanceTokenizer) {
        if !tokenizer.is_null() {
            ((*self.plugin).destroy_tokenizer)(tokenizer);
        }
    }

    /// Create a token stream.
    ///
    /// # Safety
    ///
    /// The tokenizer must have been created by this library.
    unsafe fn create_stream(
        &self,
        tokenizer: *mut LanceTokenizer,
        text: &str,
    ) -> Result<*mut LanceTokenStream> {
        let stream = ((*self.plugin).create_stream)(
            tokenizer,
            text.as_ptr() as *const c_char,
            text.len() as u32,
        );
        if stream.is_null() {
            return Err(Error::InvalidInput {
                source: "failed to create token stream".to_string().into(),
                location: location!(),
            });
        }
        Ok(stream)
    }

    /// Destroy a token stream.
    ///
    /// # Safety
    ///
    /// The stream must have been created by this library.
    unsafe fn destroy_stream(&self, stream: *mut LanceTokenStream) {
        if !stream.is_null() {
            ((*self.plugin).destroy_stream)(stream);
        }
    }

    /// Get the next token from a stream.
    ///
    /// # Safety
    ///
    /// The stream must have been created by this library.
    unsafe fn next_token(&self, stream: *mut LanceTokenStream, token: &mut CToken) -> i32 {
        ((*self.plugin).next_token)(stream, token)
    }
}

/// A tokenizer factory created from a plugin.
///
/// The factory holds shared resources (like dictionaries) and can create
/// multiple tokenizer instances.
pub struct PluginFactory<'a> {
    library: &'a TokenizerPluginLibrary,
    factory: *mut LanceTokenizerFactory,
}

impl<'a> PluginFactory<'a> {
    /// Create a tokenizer instance from this factory.
    pub fn create_tokenizer(&self) -> Result<PluginTokenizerInstance<'a>> {
        // SAFETY: factory was created by this library
        let tokenizer = unsafe { self.library.create_tokenizer(self.factory)? };
        Ok(PluginTokenizerInstance {
            library: self.library,
            tokenizer,
        })
    }
}

impl Drop for PluginFactory<'_> {
    fn drop(&mut self) {
        // SAFETY: factory was created by this library
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
    /// Create a token stream for the given text.
    pub fn create_stream(&self, text: &str) -> Result<PluginTokenStream<'a>> {
        // SAFETY: tokenizer was created by this library
        let stream = unsafe { self.library.create_stream(self.tokenizer, text)? };
        Ok(PluginTokenStream {
            library: self.library,
            stream,
        })
    }
}

impl Drop for PluginTokenizerInstance<'_> {
    fn drop(&mut self) {
        // SAFETY: tokenizer was created by this library
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

impl PluginTokenStream<'_> {
    /// Get the next token from the stream.
    ///
    /// Returns `Some(token)` if a token is available, `None` if the stream is exhausted.
    pub fn next_token(&mut self, token: &mut CToken) -> Option<()> {
        // SAFETY: stream was created by this library
        let result = unsafe { self.library.next_token(self.stream, token) };
        if result > 0 {
            Some(())
        } else {
            None
        }
    }
}

impl Drop for PluginTokenStream<'_> {
    fn drop(&mut self) {
        // SAFETY: stream was created by this library
        unsafe {
            self.library.destroy_stream(self.stream);
        }
    }
}

// Note: PluginFactory, PluginTokenizerInstance, and PluginTokenStream
// are not Send/Sync because they hold raw pointers to plugin state.
// This is intentional - each thread should create its own instances.
