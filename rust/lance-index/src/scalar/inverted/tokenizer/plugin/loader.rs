// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ffi::CStr;
use std::marker::PhantomData;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

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

// SAFETY: `_library` is Send+Sync; `plugin` points to an immutable vtable
// that lives as long as `_library`. Per-call mutable state lives in
// Factory/Tokenizer/Stream instances, not here.
unsafe impl Send for TokenizerPluginLibrary {}
unsafe impl Sync for TokenizerPluginLibrary {}

/// Build an `Error` from an out-parameter `CError` returned by a plugin
/// callback that signalled failure (typically by returning a NULL handle).
unsafe fn plugin_error(error: &CError, context: &str) -> Error {
    let msg = if error.has_message() {
        error.message_str().to_string()
    } else {
        "unknown error".to_string()
    };
    Error::invalid_input(format!("{}: {}", context, msg))
}

/// Recover the inner pointer from a `Mutex` in `Drop`, even if the lock is
/// poisoned. Lets `destroy_*` callbacks still run instead of double-panicking
/// during unwind (which would escalate to `abort`).
fn lock_or_recover<T: Copy>(mutex: &mut Mutex<T>) -> T {
    *mutex.get_mut().unwrap_or_else(PoisonError::into_inner)
}

/// `api_version` MUST be read before any other vtable field. The plugin may
/// have been compiled against a shorter / older `CTokenizerPlugin` layout, so
/// reading the full struct up front could be an out-of-bounds read against
/// the plugin's actual memory. `api_version` is contractually the first field
/// of every revision, so reading just that one field is always safe.
fn verify_plugin_compat(plugin: *const CTokenizerPlugin, path: &Path) -> Result<()> {
    if plugin.is_null() {
        return Err(Error::invalid_input(format!(
            "tokenizer plugin {:?} returned null plugin interface",
            path
        )));
    }

    // `Option<unsafe extern "C" fn() -> u32>` has the same single-pointer
    // layout as the bare function pointer, so this dereferences only the
    // `api_version` slot.
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

    let p = unsafe { &*plugin };
    validate_vtable(p, path)
}

const REQUIRED_VTABLE_CALLBACKS: usize = 10;

// `CTokenizerPlugin` is `#[repr(C)]` and every field is one pointer-sized
// `Option<extern "C" fn>`. Adding a field without updating
// `REQUIRED_VTABLE_CALLBACKS` and the `required` table fires this assertion
// at compile time, so a NULL slot can't slip past validation.
const _: () = {
    assert!(
        std::mem::size_of::<CTokenizerPlugin>()
            == REQUIRED_VTABLE_CALLBACKS * std::mem::size_of::<usize>(),
        "CTokenizerPlugin layout changed; update REQUIRED_VTABLE_CALLBACKS and \
         the `required` table in validate_vtable to cover every callback"
    );
};

fn validate_vtable(p: &CTokenizerPlugin, path: &Path) -> Result<()> {
    let required: [(&str, bool); REQUIRED_VTABLE_CALLBACKS] = [
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

    // The `(*self.plugin).<callback>.unwrap()` pattern below (and at every
    // other vtable dispatch site in this file) is sound because `load` runs
    // every plugin pointer through `verify_plugin_compat` → `validate_vtable`,
    // which rejects the load unless all 10 required callbacks are `Some`.

    pub fn name(&self) -> &str {
        unsafe { read_c_str_or((*self.plugin).name.unwrap()()) }
    }

    pub fn version(&self) -> &str {
        unsafe { read_c_str_or((*self.plugin).version.unwrap()()) }
    }

    pub fn create_factory(&self, config: &str) -> Result<PluginFactory<'_>> {
        let config_ref = CStringRef::from_str(config)?;
        let mut error = CError::default();
        let factory = unsafe { ((*self.plugin).create_factory.unwrap())(config_ref, &mut error) };

        if factory.is_null() {
            return Err(unsafe { plugin_error(&error, "failed to create tokenizer factory") });
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
            return Err(plugin_error(&error, "failed to create tokenizer"));
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
        let text_ref = CStringRef::from_str(text)?;
        let mut error = CError::default();
        let stream = ((*self.plugin).create_stream.unwrap())(tokenizer, text_ref, &mut error);
        if stream.is_null() {
            return Err(plugin_error(&error, "failed to create token stream"));
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

unsafe fn read_c_str_or(ptr: *const std::ffi::c_char) -> &'static str {
    if ptr.is_null() {
        "unknown"
    } else {
        CStr::from_ptr(ptr).to_str().unwrap_or("unknown")
    }
}

/// PluginFactory holds shared resources (e.g. dictionaries) and creates
/// tokenizer instances. The borrow chain `factory -> instance -> stream` is
/// encoded in the type system so a caller cannot drop a parent while a child
/// is still alive — without it the safe API would let `&format!(...)` outlive
/// a stream that zero-copies it, or let `drop(factory)` run while a tokenizer
/// borrowing the C-side state is still around. Both would be use-after-frees
/// in the plugin.
///
/// ```compile_fail
/// # use lance_index::scalar::inverted::tokenizer::plugin::loader::PluginFactory;
/// fn use_after_factory_drop<'a>(factory: PluginFactory<'a>) {
///     let mut inst = factory.create_tokenizer().unwrap();
///     drop(factory);
///     let _ = inst.create_stream("hi");
/// }
/// ```
pub struct PluginFactory<'a> {
    library: &'a TokenizerPluginLibrary,
    factory: *mut LanceTokenizerFactory,
}

impl<'a> PluginFactory<'a> {
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

pub struct PluginTokenizerInstance<'a> {
    library: &'a TokenizerPluginLibrary,
    tokenizer: *mut LanceTokenizer,
    _factory: PhantomData<&'a PluginFactory<'a>>,
}

impl<'a> PluginTokenizerInstance<'a> {
    /// `&mut self` (rather than `&self`) enforces the C ABI rule that the
    /// previous stream must be destroyed before another is created from the
    /// same tokenizer: while the returned stream lives it holds the unique
    /// borrow. Plugins are allowed to share scratch state between a tokenizer
    /// and its single live stream, so two overlapping streams would be a data
    /// race or use-after-free in the plugin.
    pub fn create_stream<'b>(&'b mut self, text: &'b str) -> Result<PluginTokenStream<'b>>
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

pub struct PluginTokenStream<'a> {
    library: &'a TokenizerPluginLibrary,
    stream: *mut LanceTokenStream,
    _instance: PhantomData<&'a mut PluginTokenizerInstance<'a>>,
    _text: PhantomData<&'a str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NextTokenResult {
    Token,
    EndOfStream,
    Error(i32, String),
}

impl NextTokenResult {
    /// SAFETY: `error` must point to a valid `CError` whose `message`
    /// payload (if `has_message()` is true) lives until this call returns.
    unsafe fn from_raw(result: i32, error: &CError) -> Self {
        if result > 0 {
            Self::Token
        } else if result == 0 {
            Self::EndOfStream
        } else {
            let error_msg = if error.has_message() {
                error.message_str().to_string()
            } else {
                format!("tokenizer error code: {}", result)
            };
            Self::Error(result, error_msg)
        }
    }
}

impl PluginTokenStream<'_> {
    pub fn next_token(&mut self, token: &mut CToken) -> NextTokenResult {
        let mut error = CError::default();
        let result = unsafe { self.library.next_token(self.stream, token, &mut error) };
        unsafe { NextTokenResult::from_raw(result, &error) }
    }
}

impl Drop for PluginTokenStream<'_> {
    fn drop(&mut self) {
        unsafe {
            self.library.destroy_stream(self.stream);
        }
    }
}

/// An owned plugin factory that holds an `Arc` to the library, suitable for
/// sharing across threads via `Arc<OwnedPluginFactory>`. The C ABI does not
/// require `create_*` callbacks to be safe under concurrent calls against the
/// same handle (e.g. a stateful plugin lazy-initializing a dictionary), so
/// access goes through an internal `Mutex`. A plugin callback that
/// synchronously waits on another thread which is itself dispatching against
/// the same handle will deadlock — see `include/lance_tokenizer_plugin.h` for
/// the full reentrancy contract.
pub struct OwnedPluginFactory {
    library: Arc<TokenizerPluginLibrary>,
    factory: Mutex<*mut LanceTokenizerFactory>,
}

// SAFETY: All access to the raw `factory` pointer is serialized by the
// `Mutex`. `Mutex<*mut T>` is not auto-Send/Sync because of the raw pointer,
// so we declare both manually.
unsafe impl Send for OwnedPluginFactory {}
unsafe impl Sync for OwnedPluginFactory {}

impl OwnedPluginFactory {
    pub fn new(library: Arc<TokenizerPluginLibrary>, config: &str) -> Result<Self> {
        let config_ref = CStringRef::from_str(config)?;
        let mut error = CError::default();
        let factory =
            unsafe { ((*library.plugin).create_factory.unwrap())(config_ref, &mut error) };

        if factory.is_null() {
            return Err(unsafe { plugin_error(&error, "failed to create tokenizer factory") });
        }

        Ok(Self {
            library,
            factory: Mutex::new(factory),
        })
    }

    /// The returned instance keeps an `Arc` to this factory so the C-side
    /// factory state cannot be destroyed while the tokenizer (or any stream
    /// it produced) is still alive.
    pub fn create_tokenizer(self: &Arc<Self>) -> Result<Arc<OwnedPluginTokenizerInstance>> {
        // Surface poisoning as `Err` (not panic) so the `create_stream` caller
        // contract is consistent across both fallible operations on the chain.
        let factory_ptr = self
            .factory
            .lock()
            .map_err(|e| Error::invalid_input(format!("plugin factory mutex poisoned: {}", e)))?;
        let mut error = CError::default();
        let tokenizer =
            unsafe { ((*self.library.plugin).create_tokenizer.unwrap())(*factory_ptr, &mut error) };
        if tokenizer.is_null() {
            return Err(unsafe { plugin_error(&error, "failed to create tokenizer") });
        }
        Ok(Arc::new(OwnedPluginTokenizerInstance {
            library: Arc::clone(&self.library),
            tokenizer: Mutex::new(tokenizer),
            active_stream: AtomicBool::new(false),
            _factory: Arc::clone(self),
        }))
    }
}

impl Drop for OwnedPluginFactory {
    fn drop(&mut self) {
        let factory_ptr = lock_or_recover(&mut self.factory);
        unsafe {
            self.library.destroy_factory(factory_ptr);
        }
    }
}

pub struct OwnedPluginTokenizerInstance {
    library: Arc<TokenizerPluginLibrary>,
    tokenizer: Mutex<*mut LanceTokenizer>,
    /// Guards the C ABI rule "the stream must be destroyed before creating
    /// another from the same tokenizer". We can't span the lifetime of the
    /// returned stream with a `MutexGuard` (a guard cannot be stored in the
    /// stream alongside the `Arc<Self>` it borrows from), so we use an
    /// atomic flag that stream `Drop` clears.
    active_stream: AtomicBool,
    _factory: Arc<OwnedPluginFactory>,
}

// SAFETY: same rationale as OwnedPluginFactory — Mutex serializes raw
// pointer access; the auto impls would fall through but for the raw pointer.
unsafe impl Send for OwnedPluginTokenizerInstance {}
unsafe impl Sync for OwnedPluginTokenizerInstance {}

impl OwnedPluginTokenizerInstance {
    /// The returned stream owns the input `text` (as `String`) and holds an
    /// `Arc` to this instance — necessary because the plugin ABI lets streams
    /// alias either tokenizer state or the input bytes (zero-copy / lazy
    /// implementations). Without this, passing `&format!(...)` would drop the
    /// temporary while the C-side stream still holds a pointer into it.
    pub fn create_stream(
        self: &Arc<Self>,
        text: impl Into<String>,
    ) -> Result<OwnedPluginTokenStream> {
        // Validate the text length before reserving the active_stream slot:
        // a length-overflow failure leaves the slot in the same state a
        // successful build would have left it (free).
        let text = text.into();
        let text_ref = CStringRef::from_str(&text)?;

        // C ABI forbids overlapping streams on the same tokenizer; reserve
        // the per-instance slot before calling into the plugin.
        self.active_stream
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .map_err(|_| {
                Error::invalid_input(
                    "tokenizer instance already has a live token stream; \
                     the plugin C ABI requires the previous stream to be \
                     dropped before creating another from the same tokenizer",
                )
            })?;

        let result = self.create_stream_locked(text, text_ref);
        if result.is_err() {
            self.active_stream.store(false, Ordering::Release);
        }
        result
    }

    fn create_stream_locked(
        self: &Arc<Self>,
        text: String,
        text_ref: CStringRef,
    ) -> Result<OwnedPluginTokenStream> {
        let tokenizer_ptr: MutexGuard<*mut LanceTokenizer> = self
            .tokenizer
            .lock()
            .map_err(|e| Error::invalid_input(format!("plugin tokenizer mutex poisoned: {}", e)))?;
        let mut error = CError::default();
        let stream = unsafe {
            ((*self.library.plugin).create_stream.unwrap())(*tokenizer_ptr, text_ref, &mut error)
        };
        if stream.is_null() {
            return Err(unsafe { plugin_error(&error, "failed to create token stream") });
        }
        Ok(OwnedPluginTokenStream {
            // Drop order: `stream` first (calls destroy_stream while text /
            // instance / library are still alive), then text/instance/library.
            stream,
            library: Arc::clone(&self.library),
            _instance: Arc::clone(self),
            _text: text,
        })
    }
}

impl Drop for OwnedPluginTokenizerInstance {
    fn drop(&mut self) {
        let tokenizer_ptr = lock_or_recover(&mut self.tokenizer);
        unsafe {
            self.library.destroy_tokenizer(tokenizer_ptr);
        }
    }
}

pub struct OwnedPluginTokenStream {
    stream: *mut LanceTokenStream,
    library: Arc<TokenizerPluginLibrary>,
    _instance: Arc<OwnedPluginTokenizerInstance>,
    _text: String,
}

// SAFETY: `next_token` takes `&mut self`, so the borrow checker guarantees
// exclusive access; no per-stream Mutex is needed. `Sync` is intentionally
// *not* asserted: only `&mut self` access is exposed today, so concurrent
// shared use is not a real shape. Adding a `&self` method later requires
// fresh review.
unsafe impl Send for OwnedPluginTokenStream {}

impl OwnedPluginTokenStream {
    pub fn next_token(&mut self, token: &mut CToken) -> NextTokenResult {
        let mut error = CError::default();
        let result =
            unsafe { ((*self.library.plugin).next_token.unwrap())(self.stream, token, &mut error) };
        unsafe { NextTokenResult::from_raw(result, &error) }
    }
}

impl Drop for OwnedPluginTokenStream {
    fn drop(&mut self) {
        unsafe {
            self.library.destroy_stream(self.stream);
        }
        // `Release` pairs with `Acquire` in `create_stream`'s CAS.
        self._instance.active_stream.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::c_char;

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

    /// Every required callback must be flagged when missing — calling a NULL
    /// slot would crash the host.
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
                "error must mention the missing callback `{}`, got: {}",
                name,
                err
            );
        }
    }

    /// API version mismatch must surface before any other vtable field is
    /// read — reading the rest first risks an OOB access against an
    /// older-layout plugin.
    #[test]
    fn test_verify_plugin_compat_checks_api_version_before_vtable() {
        unsafe extern "C" fn wrong_api_version() -> u32 {
            PLUGIN_API_VERSION + 999
        }
        let mut vtable = full_vtable();
        vtable.api_version = Some(wrong_api_version);
        // Also clear an unrelated callback; the version error must still surface first.
        vtable.next_token = None;

        let path = Path::new("/test/plugin.so");
        let err = verify_plugin_compat(&vtable as *const _, path)
            .expect_err("incompatible API version must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("incompatible API version"), "got: {}", msg);
        assert!(!msg.contains("next_token"), "got: {}", msg);
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
