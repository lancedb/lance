// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ffi::CStr;
use std::marker::PhantomData;
use std::path::Path;
use std::sync::{Arc, Mutex};

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

/// Verify that a freshly returned plugin pointer is compatible with the
/// `CTokenizerPlugin` layout this crate was built against.
///
/// The check order matters: `api_version` MUST be read and compared before
/// touching any other field. The plugin may have been compiled against a
/// shorter / older vtable layout, so reading the full struct up front
/// could be an out-of-bounds read against the plugin's actual memory.
/// `api_version` is, by contract, the first field of every vtable
/// revision, so reading just that one field is always safe.
fn verify_plugin_compat(plugin: *const CTokenizerPlugin, path: &Path) -> Result<()> {
    if plugin.is_null() {
        return Err(Error::invalid_input(format!(
            "tokenizer plugin {:?} returned null plugin interface",
            path
        )));
    }

    // Read only the first field. `Option<unsafe extern "C" fn() -> u32>`
    // has the same single-pointer layout as the bare function pointer,
    // so this dereferences exactly the bytes of `api_version` and no
    // more.
    let api_version_fn = unsafe {
        let api_version_ptr = plugin as *const Option<unsafe extern "C" fn() -> u32>;
        *api_version_ptr
    };
    let api_version_fn = api_version_fn.ok_or_else(|| {
        Error::invalid_input(format!(
            "tokenizer plugin {:?} has NULL api_version callback",
            path
        ))
    })?;
    let api_version = unsafe { api_version_fn() };
    if api_version != PLUGIN_API_VERSION {
        return Err(Error::invalid_input(format!(
            "tokenizer plugin {:?} has incompatible API version {} (expected {})",
            path, api_version, PLUGIN_API_VERSION
        )));
    }

    // API version matches the layout we were compiled against — it is
    // now safe to read the rest of the vtable.
    let p = unsafe { &*plugin };
    validate_vtable(p, path)
}

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
            // The underlying libloading error already names the precise cause
            // (file not found, missing dependent .so, permission denied, etc.).
            // Pass it through verbatim and add a hint pointing the user at the
            // path they configured.
            Error::invalid_input(format!(
                "failed to load tokenizer plugin from {:?}: {}. \
                 Verify the path is correct and the file is readable.",
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
        verify_plugin_compat(plugin, path)?;

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
//
// Lifetime chain: a stream borrows from its tokenizer instance and from the
// input text; a tokenizer instance borrows from the factory. The plugin C ABI
// allows children to alias parent state (e.g. a tokenizer holds pointers into
// the factory's dictionary, or a stream zero-copies the input text), so we
// encode that as a borrow chain in the type system. Without this, safe Rust
// would let a caller `drop(factory)` while a tokenizer/stream is still alive,
// or pass `&format!(...)` as the input text and outlive the temporary string,
// either of which is a use-after-free in the plugin.

/// PluginFactory holds shared resources (like dictionaries) and can create
/// multiple tokenizer instances.
///
/// The borrow chain `factory -> instance -> stream` is enforced at compile
/// time so a caller cannot drop a parent while a child is still alive.
///
/// Dropping the factory while a tokenizer derived from it is still alive
/// must be rejected by the borrow checker:
///
/// ```compile_fail
/// # use lance_index::scalar::inverted::tokenizer::plugin::loader::PluginFactory;
/// fn use_after_factory_drop<'a>(factory: PluginFactory<'a>) {
///     let inst = factory.create_tokenizer().unwrap();
///     drop(factory);
///     let _ = inst.create_stream("hi");
/// }
/// ```
///
/// Likewise, returning a stream that borrows a temporary input string
/// (e.g. `&format!(...)`) must be rejected:
///
/// ```compile_fail
/// # use lance_index::scalar::inverted::tokenizer::plugin::loader::{
/// #     PluginFactory, PluginTokenStream,
/// # };
/// fn temp_input_outlives_stream<'a>(
///     factory: &'a PluginFactory<'a>,
/// ) -> PluginTokenStream<'a> {
///     let inst = factory.create_tokenizer().unwrap();
///     inst.create_stream(&format!("temp {}", 1)).unwrap()
/// }
/// ```
pub struct PluginFactory<'a> {
    library: &'a TokenizerPluginLibrary,
    factory: *mut LanceTokenizerFactory,
}

impl<'a> PluginFactory<'a> {
    /// The returned instance borrows from `self`, so the borrow checker
    /// rejects dropping the factory while any instance is still alive.
    pub fn create_tokenizer<'b>(&'b self) -> Result<PluginTokenizerInstance<'b>>
    where
        'a: 'b,
    {
        let tokenizer = unsafe { self.library.create_tokenizer(self.factory)? };
        Ok(PluginTokenizerInstance {
            library: self.library,
            tokenizer,
            _factory: PhantomData,
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

/// A tokenizer instance created from a plugin factory. Borrows from the
/// factory so the factory cannot be dropped while the instance is alive.
pub struct PluginTokenizerInstance<'a> {
    library: &'a TokenizerPluginLibrary,
    tokenizer: *mut LanceTokenizer,
    /// Tie the instance's lifetime to the factory it was created from.
    _factory: PhantomData<&'a PluginFactory<'a>>,
}

impl<'a> PluginTokenizerInstance<'a> {
    /// The returned stream borrows from both `self` (the tokenizer instance)
    /// and `text`, so the borrow checker rejects dropping either while the
    /// stream is still alive — including the common foot-gun of passing
    /// `&format!(...)` as the input.
    pub fn create_stream<'b>(&'b self, text: &'b str) -> Result<PluginTokenStream<'b>>
    where
        'a: 'b,
    {
        let stream = unsafe { self.library.create_stream(self.tokenizer, text)? };
        Ok(PluginTokenStream {
            library: self.library,
            stream,
            _instance: PhantomData,
            _text: PhantomData,
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

/// A token stream from a plugin tokenizer. Borrows from the parent
/// tokenizer instance and the input text — the plugin may zero-copy either,
/// so dropping either while the stream is alive would be a use-after-free.
pub struct PluginTokenStream<'a> {
    library: &'a TokenizerPluginLibrary,
    stream: *mut LanceTokenStream,
    _instance: PhantomData<&'a PluginTokenizerInstance<'a>>,
    _text: PhantomData<&'a str>,
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
    /// Raw factory handle from the C plugin, guarded by a Mutex.
    ///
    /// The plugin C ABI does not require `create_tokenizer` to be safe under
    /// concurrent calls against the same factory handle — a stateful plugin
    /// (e.g. one that lazy-initializes a dictionary on first use) could
    /// data-race or crash. Since `OwnedPluginFactory` is `pub` and `Sync`,
    /// callers may legitimately share an `Arc<OwnedPluginFactory>` across
    /// threads, so we serialize access here rather than rely on every caller
    /// to wrap it externally.
    factory: Mutex<*mut LanceTokenizerFactory>,
}

// SAFETY: OwnedPluginFactory can be sent/shared across threads because:
// 1. `library` (Arc<TokenizerPluginLibrary>) is Send + Sync.
// 2. `factory` is a raw pointer to plugin state, but every call into the
//    plugin that uses it goes through the `Mutex` above, so there is at
//    most one thread inside the plugin's `create_tokenizer` for a given
//    factory at any time. The plugin's other vtable callbacks
//    (create_factory / destroy_factory) are only invoked from `new` /
//    `Drop` where `&self` exclusivity is guaranteed by the borrow checker.
// 3. `Mutex<*mut T>` is not auto-`Send`/`Sync` because of the raw pointer,
//    so we declare both impls manually.
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

        Ok(Self {
            library,
            factory: Mutex::new(factory),
        })
    }

    /// Create a tokenizer instance from this factory.
    ///
    /// The returned instance keeps an `Arc` to the factory so the C-side
    /// factory state cannot be destroyed while the tokenizer (or any stream
    /// it produced) is still alive. Without this back-reference a caller
    /// could write `let inst = factory.create_tokenizer()?; drop(factory);`
    /// and then dereference the freed factory state through `inst`.
    pub fn create_tokenizer(self: &Arc<Self>) -> Result<Arc<OwnedPluginTokenizerInstance>> {
        let factory_ptr = self.factory.lock().expect("plugin factory mutex poisoned");
        let mut error = CError::default();
        let tokenizer =
            unsafe { ((*self.library.plugin).create_tokenizer.unwrap())(*factory_ptr, &mut error) };
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
        Ok(Arc::new(OwnedPluginTokenizerInstance {
            library: Arc::clone(&self.library),
            tokenizer: Mutex::new(tokenizer),
            _factory: Arc::clone(self),
        }))
    }
}

impl Drop for OwnedPluginFactory {
    fn drop(&mut self) {
        // Exclusive access in Drop, so `get_mut` is sufficient and avoids the
        // poisoning check entirely.
        let factory_ptr = *self
            .factory
            .get_mut()
            .expect("plugin factory mutex poisoned");
        unsafe {
            self.library.destroy_factory(factory_ptr);
        }
    }
}

/// An owned tokenizer instance created from an owned factory.
///
/// Keeps an `Arc<OwnedPluginFactory>` back-reference so the factory cannot
/// be dropped (which would invoke the plugin's `destroy_factory`) while any
/// instance derived from it is still alive.
pub struct OwnedPluginTokenizerInstance {
    library: Arc<TokenizerPluginLibrary>,
    /// Raw tokenizer handle, guarded by a Mutex for the same reason
    /// `OwnedPluginFactory.factory` is — a stateful plugin's
    /// `create_stream` is not required by the C ABI to be safe under
    /// concurrent calls against the same tokenizer handle, and instances
    /// are shared via `Arc` so multiple threads may legitimately call
    /// `create_stream` on the same instance.
    tokenizer: Mutex<*mut LanceTokenizer>,
    _factory: Arc<OwnedPluginFactory>,
}

// SAFETY: same rationale as OwnedPluginFactory — the only access to the raw
// tokenizer pointer goes through the Mutex above. `Mutex<*mut T>` is not
// auto-`Send`/`Sync` because of the raw pointer, so we declare both
// manually.
unsafe impl Send for OwnedPluginTokenizerInstance {}
unsafe impl Sync for OwnedPluginTokenizerInstance {}

impl OwnedPluginTokenizerInstance {
    /// Create a token stream for the given text.
    ///
    /// The returned stream owns the input `text` (as `String`) and keeps an
    /// `Arc` to the parent instance. This is necessary because the plugin
    /// ABI lets streams alias either the tokenizer state or the input bytes
    /// (for zero-copy / lazy implementations). Without this composition the
    /// safe API would allow `let stream = inst.create_stream(&format!(...))`
    /// — the temporary `String` would drop while the C-side stream still
    /// holds a pointer into it.
    pub fn create_stream(
        self: &Arc<Self>,
        text: impl Into<String>,
    ) -> Result<OwnedPluginTokenStream> {
        let text = text.into();
        let tokenizer_ptr = self
            .tokenizer
            .lock()
            .expect("plugin tokenizer mutex poisoned");
        let mut error = CError::default();
        let stream = unsafe {
            ((*self.library.plugin).create_stream.unwrap())(
                *tokenizer_ptr,
                CStringRef::from_str(&text),
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
            // Field declaration order matters for Drop: `stream` is dropped
            // first (custom Drop calls destroy_stream while text/instance
            // are still alive), then text/instance/library are released.
            stream,
            library: Arc::clone(&self.library),
            _instance: Arc::clone(self),
            _text: text,
        })
    }
}

impl Drop for OwnedPluginTokenizerInstance {
    fn drop(&mut self) {
        let tokenizer_ptr = *self
            .tokenizer
            .get_mut()
            .expect("plugin tokenizer mutex poisoned");
        unsafe {
            self.library.destroy_tokenizer(tokenizer_ptr);
        }
    }
}

/// An owned token stream from an owned tokenizer instance.
///
/// Owns the input `String` and keeps an `Arc` to the parent instance so
/// neither can be freed while the C-side stream is still alive. The plugin
/// is allowed to zero-copy from either, so dropping them earlier would be
/// a use-after-free.
pub struct OwnedPluginTokenStream {
    stream: *mut LanceTokenStream,
    library: Arc<TokenizerPluginLibrary>,
    _instance: Arc<OwnedPluginTokenizerInstance>,
    _text: String,
}

// SAFETY: `next_token` takes `&mut self`, so the borrow checker already
// guarantees exclusive access — no per-stream Mutex is needed. The struct
// merely holds a raw pointer alongside Arc-counted owners, so the auto
// Send/Sync impls would fall through if not for the raw pointer.
unsafe impl Send for OwnedPluginTokenStream {}
unsafe impl Sync for OwnedPluginTokenStream {}

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

    /// API version mismatch must be detected before any other vtable
    /// field is read. Reading the rest of the vtable first risks an
    /// out-of-bounds access against a plugin built with a smaller /
    /// older `CTokenizerPlugin` layout. The test sets a vtable whose
    /// `api_version` returns the wrong number AND has a missing
    /// callback; the version error must surface first.
    #[test]
    fn test_verify_plugin_compat_checks_api_version_before_vtable() {
        unsafe extern "C" fn wrong_api_version() -> u32 {
            PLUGIN_API_VERSION + 999
        }
        let mut vtable = full_vtable();
        vtable.api_version = Some(wrong_api_version);
        // Also clear an unrelated callback. If `validate_vtable` ran
        // before the version check, the "missing callback" message
        // would surface instead of the version mismatch.
        vtable.next_token = None;

        let path = Path::new("/test/plugin.so");
        let err = verify_plugin_compat(&vtable as *const _, path)
            .expect_err("incompatible API version must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("incompatible API version"),
            "version mismatch must be reported before vtable validation, got: {}",
            msg
        );
        assert!(
            !msg.contains("next_token"),
            "vtable validation must not run before the version check, got: {}",
            msg
        );
    }

    #[test]
    fn test_verify_plugin_compat_rejects_null_api_version() {
        let mut vtable = full_vtable();
        vtable.api_version = None;
        let err = verify_plugin_compat(&vtable as *const _, Path::new("/test/plugin.so"))
            .expect_err("NULL api_version must be rejected");
        assert!(
            err.to_string().contains("NULL api_version callback"),
            "got: {}",
            err
        );
    }

    #[test]
    fn test_verify_plugin_compat_rejects_null_plugin_pointer() {
        let err = verify_plugin_compat(std::ptr::null(), Path::new("/test/plugin.so"))
            .expect_err("null plugin pointer must be rejected");
        assert!(
            err.to_string().contains("null plugin interface"),
            "got: {}",
            err
        );
    }

    #[test]
    fn test_verify_plugin_compat_accepts_valid_vtable() {
        let vtable = full_vtable();
        verify_plugin_compat(&vtable as *const _, Path::new("/test/plugin.so"))
            .expect("matching API version with full vtable should pass");
    }
}
