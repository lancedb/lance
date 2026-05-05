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

/// Reject plugin vtables that are missing any required callback. Returning
/// `Err` here keeps the failure on the user-input boundary instead of letting
/// a NULL function pointer crash the host the first time it is dereferenced.
fn validate_vtable(p: &CTokenizerPlugin, path: &Path) -> Result<()> {
    let required: [(&str, bool); 10] = [
        ("api_version", p.api_version.is_some()),
        ("create_factory", p.create_factory.is_some()),
        ("destroy_factory", p.destroy_factory.is_some()),
        ("create_tokenizer", p.create_tokenizer.is_some()),
        ("destroy_tokenizer", p.destroy_tokenizer.is_some()),
        ("create_stream", p.create_stream.is_some()),
        ("destroy_stream", p.destroy_stream.is_some()),
        ("next_token", p.next_token.is_some()),
        ("name", p.name.is_some()),
        ("version", p.version.is_some()),
    ];
    for (callback_name, present) in required {
        if !present {
            return Err(Error::invalid_input(format!(
                "tokenizer plugin {:?} missing required callback `{}` \
                 (NULL function pointer in vtable)",
                path, callback_name
            )));
        }
    }
    Ok(())
}

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

        // Validate that the plugin's vtable has every required callback.
        // A broken or older-ABI plugin can leave fields as NULL; calling
        // a NULL function pointer would crash the host process. Reject
        // such plugins here as user input errors instead.
        let p = unsafe { &*plugin };
        validate_vtable(p, path)?;

        // Safe: validated above.
        let api_version = unsafe { (p.api_version.unwrap())() };
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
            let name_ptr = ((*self.plugin).name.unwrap())();
            if name_ptr.is_null() {
                "unknown"
            } else {
                CStr::from_ptr(name_ptr).to_str().unwrap_or("unknown")
            }
        }
    }

    pub fn version(&self) -> &str {
        unsafe {
            let version_ptr = ((*self.plugin).version.unwrap())();
            if version_ptr.is_null() {
                "unknown"
            } else {
                CStr::from_ptr(version_ptr).to_str().unwrap_or("unknown")
            }
        }
    }

    pub fn create_factory(&self, config: &str) -> Result<PluginFactory<'_>> {
        let mut error = CError::default();
        let factory = unsafe {
            ((*self.plugin).create_factory.unwrap())(CStringRef::from_str(config), &mut error)
        };

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
            ((*self.plugin).destroy_factory.unwrap())(factory);
        }
    }

    unsafe fn create_tokenizer(
        &self,
        factory: *mut LanceTokenizerFactory,
    ) -> Result<*mut LanceTokenizer> {
        let mut error = CError::default();
        let tokenizer = ((*self.plugin).create_tokenizer.unwrap())(factory, &mut error);
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
            ((*self.plugin).destroy_tokenizer.unwrap())(tokenizer);
        }
    }

    unsafe fn create_stream(
        &self,
        tokenizer: *mut LanceTokenizer,
        text: &str,
    ) -> Result<*mut LanceTokenStream> {
        let mut error = CError::default();
        let stream = ((*self.plugin).create_stream.unwrap())(
            tokenizer,
            CStringRef::from_str(text),
            &mut error,
        );
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
            ((*self.plugin).destroy_stream.unwrap())(stream);
        }
    }

    unsafe fn next_token(
        &self,
        stream: *mut LanceTokenStream,
        token: &mut CToken,
        error: &mut CError,
    ) -> i32 {
        ((*self.plugin).next_token.unwrap())(stream, token, error)
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
        let factory = unsafe {
            ((*library.plugin).create_factory.unwrap())(CStringRef::from_str(config), &mut error)
        };

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
            unsafe { ((*self.library.plugin).create_tokenizer.unwrap())(self.factory, &mut error) };
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
            ((*self.library.plugin).create_stream.unwrap())(
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
        let result =
            unsafe { ((*self.library.plugin).next_token.unwrap())(self.stream, token, &mut error) };
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::c_char;

    // Stub callbacks used to populate a "valid" vtable for tests.
    unsafe extern "C" fn stub_api_version() -> u32 {
        PLUGIN_API_VERSION
    }
    unsafe extern "C" fn stub_create_factory(
        _config: CStringRef,
        _error: *mut CError,
    ) -> *mut LanceTokenizerFactory {
        std::ptr::null_mut()
    }
    unsafe extern "C" fn stub_destroy_factory(_factory: *mut LanceTokenizerFactory) {}
    unsafe extern "C" fn stub_create_tokenizer(
        _factory: *mut LanceTokenizerFactory,
        _error: *mut CError,
    ) -> *mut LanceTokenizer {
        std::ptr::null_mut()
    }
    unsafe extern "C" fn stub_destroy_tokenizer(_tokenizer: *mut LanceTokenizer) {}
    unsafe extern "C" fn stub_create_stream(
        _tokenizer: *mut LanceTokenizer,
        _text: CStringRef,
        _error: *mut CError,
    ) -> *mut LanceTokenStream {
        std::ptr::null_mut()
    }
    unsafe extern "C" fn stub_destroy_stream(_stream: *mut LanceTokenStream) {}
    unsafe extern "C" fn stub_next_token(
        _stream: *mut LanceTokenStream,
        _token: *mut CToken,
        _error: *mut CError,
    ) -> i32 {
        0
    }
    unsafe extern "C" fn stub_name() -> *const c_char {
        std::ptr::null()
    }
    unsafe extern "C" fn stub_version() -> *const c_char {
        std::ptr::null()
    }

    fn full_vtable() -> CTokenizerPlugin {
        CTokenizerPlugin {
            api_version: Some(stub_api_version),
            create_factory: Some(stub_create_factory),
            destroy_factory: Some(stub_destroy_factory),
            create_tokenizer: Some(stub_create_tokenizer),
            destroy_tokenizer: Some(stub_destroy_tokenizer),
            create_stream: Some(stub_create_stream),
            destroy_stream: Some(stub_destroy_stream),
            next_token: Some(stub_next_token),
            name: Some(stub_name),
            version: Some(stub_version),
        }
    }

    #[test]
    fn test_validate_vtable_accepts_full_table() {
        let vtable = full_vtable();
        validate_vtable(&vtable, Path::new("/test/plugin.so")).expect("full vtable should pass");
    }

    /// Every required callback must be flagged when missing. A broken or
    /// older-ABI plugin can leave individual fields as NULL; calling them
    /// would crash the host process.
    #[test]
    fn test_validate_vtable_rejects_each_missing_callback() {
        type Clear = fn(&mut CTokenizerPlugin);
        let path = Path::new("/test/plugin.so");
        let cases: &[(&str, Clear)] = &[
            ("api_version", |p| p.api_version = None),
            ("create_factory", |p| p.create_factory = None),
            ("destroy_factory", |p| p.destroy_factory = None),
            ("create_tokenizer", |p| p.create_tokenizer = None),
            ("destroy_tokenizer", |p| p.destroy_tokenizer = None),
            ("create_stream", |p| p.create_stream = None),
            ("destroy_stream", |p| p.destroy_stream = None),
            ("next_token", |p| p.next_token = None),
            ("name", |p| p.name = None),
            ("version", |p| p.version = None),
        ];
        for (name, clear) in cases {
            let mut vtable = full_vtable();
            clear(&mut vtable);
            let err = validate_vtable(&vtable, path)
                .expect_err(&format!("missing `{}` should be rejected", name));
            assert!(
                err.to_string().contains(name),
                "error message should mention the missing callback `{}`, got: {}",
                name,
                err
            );
        }
    }
}
