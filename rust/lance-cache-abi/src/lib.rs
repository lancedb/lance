// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! C ABI for dynamically loaded Lance cache backends.
//!
//! The ABI is byte-oriented by design. Lance owns typed cache entries and
//! codecs; dynamic backends only receive stable key parts and serialized bytes.
//!
//! ## ABI contract
//!
//! All ABI functions return one of the `DYNAMIC_CACHE_STATUS_*` constants.
//! `DYNAMIC_CACHE_STATUS_OK` means the operation completed synchronously before
//! returning. `DYNAMIC_CACHE_STATUS_HIT` and `DYNAMIC_CACHE_STATUS_MISS` are
//! only valid return values for [`DynamicCacheGetFn`].
//! `DYNAMIC_CACHE_STATUS_ERROR` means the backend did not complete the
//! operation. Hosts must ignore all out parameters after an error unless the
//! function-specific contract says otherwise.
//!
//! All pointer parameters are non-null unless explicitly documented otherwise.
//! `DynamicCacheByteSlice` values are borrowed for the duration of the ABI call
//! only. A byte slice with `len > 0` must have a non-null `ptr` that is valid
//! for `len` readable bytes. A byte slice with `len == 0` has no readable
//! bytes; consumers must ignore `ptr` and must not dereference it. Rust helpers
//! in this crate use a null `ptr` for empty slices. Dynamic backends must not
//! retain borrowed pointers after returning.
//!
//! `DynamicCacheOwnedBytes` values are owned by the dynamic backend allocator.
//! When `len > 0`, `ptr` must be non-null, readable for `len` bytes, and
//! allocated with at least `capacity` bytes where `capacity >= len`. The host
//! must not read from `ptr` when `len == 0`. The host copies bytes returned by
//! [`DynamicCacheGetFn`] on
//! [`DYNAMIC_CACHE_STATUS_HIT`] and then calls [`DynamicCacheFreeBytesFn`]
//! exactly once. The host must not call [`DynamicCacheFreeBytesFn`] for
//! [`DYNAMIC_CACHE_STATUS_MISS`] or [`DYNAMIC_CACHE_STATUS_ERROR`] results.
//!
//! Out parameters must be treated as uninitialized storage by dynamic backends.
//! Backends must fully initialize every out parameter required by a successful
//! return before returning that status. Hosts must ignore out parameters after
//! any status that does not require them to be written.
//!
//! ### Threading and lifetime
//!
//! Hosts may call [`DynamicCacheGetFn`], [`DynamicCacheInsertFn`],
//! [`DynamicCacheInvalidatePrefixFn`], [`DynamicCacheClearFn`], and
//! [`DynamicCacheMeasureFn`] concurrently on the same backend instance from
//! multiple threads. Dynamic backends must make all shared backend state
//! internally thread-safe; the host does not serialize calls for the plugin.
//!
//! Hosts guarantee [`DynamicCacheDestroyFn`] is called at most once for a
//! backend instance, and only after all in-flight operations on that instance
//! have completed and no future operations will start. Backends may release all
//! state associated with the backend pointer during destroy.
//!
//! Hosts may call [`DynamicCacheFreeBytesFn`] from any thread after a successful
//! [`DynamicCacheGetFn`] hit. The backend's owned-byte deallocation path must
//! therefore be thread-safe and must not rely on thread-local state from the
//! original `get` call.

use std::ffi::c_void;

/// ABI version understood by this crate.
pub const DYNAMIC_CACHE_ABI_VERSION: u32 = 1;

/// Exported symbol every dynamic cache backend must provide.
pub const DYNAMIC_CACHE_BACKEND_INIT_SYMBOL: &[u8] = b"lance_cache_backend_init\0";

/// Dynamic cache operation status.
///
/// ABI function signatures use raw `u32` return values so hosts can handle
/// unknown future status codes without constructing an invalid Rust enum. This
/// enum is a Rust-side helper for known v1 status values.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DynamicCacheStatus {
    /// Operation completed successfully.
    Ok = 0,
    /// Lookup found a byte payload.
    Hit = 1,
    /// Lookup did not find a byte payload.
    Miss = 2,
    /// Operation failed inside the dynamic backend.
    Error = 3,
}

impl DynamicCacheStatus {
    /// Return the raw ABI status code.
    pub const fn as_u32(self) -> u32 {
        self as u32
    }
}

impl TryFrom<u32> for DynamicCacheStatus {
    type Error = UnknownDynamicCacheStatus;

    fn try_from(value: u32) -> Result<Self, UnknownDynamicCacheStatus> {
        match value {
            DYNAMIC_CACHE_STATUS_OK => Ok(Self::Ok),
            DYNAMIC_CACHE_STATUS_HIT => Ok(Self::Hit),
            DYNAMIC_CACHE_STATUS_MISS => Ok(Self::Miss),
            DYNAMIC_CACHE_STATUS_ERROR => Ok(Self::Error),
            _ => Err(UnknownDynamicCacheStatus(value)),
        }
    }
}

/// Raw status code that is not known to this ABI crate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnknownDynamicCacheStatus(pub u32);

/// Operation completed successfully.
pub const DYNAMIC_CACHE_STATUS_OK: u32 = DynamicCacheStatus::Ok as u32;
/// Lookup found a byte payload.
pub const DYNAMIC_CACHE_STATUS_HIT: u32 = DynamicCacheStatus::Hit as u32;
/// Lookup did not find a byte payload.
pub const DYNAMIC_CACHE_STATUS_MISS: u32 = DynamicCacheStatus::Miss as u32;
/// Operation failed inside the dynamic backend.
pub const DYNAMIC_CACHE_STATUS_ERROR: u32 = DynamicCacheStatus::Error as u32;

/// Number of pointer-sized slots reserved for future optional functions.
pub const DYNAMIC_CACHE_BACKEND_VTABLE_RESERVED_SLOTS: usize = 8;

/// Borrowed bytes passed through the dynamic cache ABI.
///
/// A non-empty slice must have a non-null `ptr` that is valid for `len`
/// readable bytes. Empty slices have no readable bytes; consumers must ignore
/// `ptr` and must not dereference it.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DynamicCacheByteSlice {
    /// Pointer to the first byte. Ignored when `len == 0`.
    pub ptr: *const u8,
    /// Number of bytes available at `ptr`.
    pub len: usize,
}

impl DynamicCacheByteSlice {
    /// Create an ABI slice from Rust bytes.
    pub fn from_slice(bytes: &[u8]) -> Self {
        if bytes.is_empty() {
            return Self {
                ptr: std::ptr::null(),
                len: 0,
            };
        }
        Self {
            ptr: bytes.as_ptr(),
            len: bytes.len(),
        }
    }

    /// Borrow the slice described by this ABI value.
    ///
    /// # Safety
    ///
    /// When `len > 0`, the caller must ensure `ptr` is valid for `len` bytes
    /// and outlives the returned slice.
    pub unsafe fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// Bytes allocated by the dynamic backend and released through `free_bytes`.
///
/// A non-empty buffer must have a non-null `ptr`, `capacity >= len`, and an
/// allocation that can be released by the backend's [`DynamicCacheFreeBytesFn`].
/// Empty buffers have no readable bytes; hosts must not read from `ptr`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DynamicCacheOwnedBytes {
    /// Pointer to the first byte. Ignored when `len == 0`.
    pub ptr: *mut u8,
    /// Number of initialized bytes.
    pub len: usize,
    /// Allocation capacity needed by the backend that owns the allocator.
    pub capacity: usize,
}

impl DynamicCacheOwnedBytes {
    /// Empty owned byte buffer.
    pub fn empty() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            len: 0,
            capacity: 0,
        }
    }

    /// Convert a Rust `Vec<u8>` into ABI-owned bytes.
    ///
    /// The same dynamic library that calls this must later reconstruct the
    /// vector in its `free_bytes` function. The Lance host copies these bytes
    /// and then calls `free_bytes`; it must not free them directly.
    pub fn from_vec(mut bytes: Vec<u8>) -> Self {
        if bytes.is_empty() {
            return Self::empty();
        }
        let owned = Self {
            ptr: bytes.as_mut_ptr(),
            len: bytes.len(),
            capacity: bytes.capacity(),
        };
        std::mem::forget(bytes);
        owned
    }

    /// Reconstruct the `Vec<u8>` originally passed to [`Self::from_vec`].
    ///
    /// # Safety
    ///
    /// This must only be called by the dynamic library that allocated the
    /// bytes, using a value originally returned by [`Self::from_vec`].
    pub unsafe fn into_vec(self) -> Vec<u8> {
        if self.capacity == 0 {
            return Vec::new();
        }
        unsafe { Vec::from_raw_parts(self.ptr, self.len, self.capacity) }
    }
}

impl Default for DynamicCacheOwnedBytes {
    fn default() -> Self {
        Self::empty()
    }
}

/// Cache key view passed through the dynamic cache ABI.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DynamicCacheKey {
    /// Dataset/index prefix.
    pub prefix: DynamicCacheByteSlice,
    /// User-visible cache key within the prefix.
    pub key: DynamicCacheByteSlice,
    /// Stable cache value type name.
    pub type_name: DynamicCacheByteSlice,
}

impl DynamicCacheKey {
    /// Borrow the key prefix bytes.
    ///
    /// # Safety
    ///
    /// The caller must ensure this key was provided by Lance for the duration
    /// of the current ABI call.
    pub unsafe fn prefix(&self) -> &[u8] {
        unsafe { self.prefix.as_slice() }
    }

    /// Borrow the key bytes.
    ///
    /// # Safety
    ///
    /// The caller must ensure this key was provided by Lance for the duration
    /// of the current ABI call.
    pub unsafe fn key(&self) -> &[u8] {
        unsafe { self.key.as_slice() }
    }

    /// Borrow the value type name bytes.
    ///
    /// # Safety
    ///
    /// The caller must ensure this key was provided by Lance for the duration
    /// of the current ABI call.
    pub unsafe fn type_name(&self) -> &[u8] {
        unsafe { self.type_name.as_slice() }
    }
}

/// Lookup result filled by a dynamic backend on a hit.
///
/// Backends must fully initialize this value before returning
/// [`DYNAMIC_CACHE_STATUS_HIT`] from [`DynamicCacheGetFn`]. Hosts ignore this
/// value for [`DYNAMIC_CACHE_STATUS_MISS`] and [`DYNAMIC_CACHE_STATUS_ERROR`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DynamicCacheGetResult {
    /// Serialized payload bytes. Valid only when `get` returns
    /// [`DYNAMIC_CACHE_STATUS_HIT`].
    pub bytes: DynamicCacheOwnedBytes,
    /// Host-side cache weight originally associated with this entry.
    pub size_bytes: usize,
}

/// Create an opaque backend instance.
///
/// `out_backend` must be non-null. On [`DYNAMIC_CACHE_STATUS_OK`], the backend
/// must write a non-null pointer that remains valid until passed to
/// [`DynamicCacheDestroyFn`]. On [`DYNAMIC_CACHE_STATUS_ERROR`], the backend
/// must write null to `out_backend`.
pub type DynamicCacheCreateFn = unsafe extern "C" fn(
    config: DynamicCacheByteSlice,
    memory_capacity_bytes: usize,
    out_backend: *mut *mut c_void,
) -> u32;

/// Destroy a backend instance previously created by [`DynamicCacheCreateFn`].
///
/// `backend` must be a non-null pointer returned by a successful
/// [`DynamicCacheCreateFn`] call. This function consumes the backend instance.
/// The host calls it only after all in-flight operations on the backend have
/// completed.
pub type DynamicCacheDestroyFn = unsafe extern "C" fn(backend: *mut c_void);

/// Look up a serialized byte payload.
///
/// `backend`, `key`, and `out_result` must be non-null. On
/// [`DYNAMIC_CACHE_STATUS_HIT`], the backend must fully initialize `out_result`
/// with a valid owned byte buffer and its cache weight. On
/// [`DYNAMIC_CACHE_STATUS_MISS`] or [`DYNAMIC_CACHE_STATUS_ERROR`], the backend
/// must not return owned bytes; hosts ignore `out_result` and must not call
/// [`DynamicCacheFreeBytesFn`].
pub type DynamicCacheGetFn = unsafe extern "C" fn(
    backend: *mut c_void,
    key: *const DynamicCacheKey,
    out_result: *mut DynamicCacheGetResult,
) -> u32;

/// Insert a serialized byte payload.
///
/// `backend` and `key` must be non-null. `value` is borrowed for this call only
/// and must be copied by the backend if it is retained. [`DYNAMIC_CACHE_STATUS_OK`]
/// means the value is available to later lookups before the function returns.
pub type DynamicCacheInsertFn = unsafe extern "C" fn(
    backend: *mut c_void,
    key: *const DynamicCacheKey,
    value: DynamicCacheByteSlice,
    size_bytes: usize,
) -> u32;

/// Invalidate all entries whose prefix starts with `prefix`.
///
/// `backend` must be non-null. [`DYNAMIC_CACHE_STATUS_OK`] means all matching
/// entries have been invalidated before the function returns. Asynchronous or
/// best-effort invalidation must return [`DYNAMIC_CACHE_STATUS_ERROR`] unless a
/// future ABI revision adds an explicit async capability.
pub type DynamicCacheInvalidatePrefixFn =
    unsafe extern "C" fn(backend: *mut c_void, prefix: DynamicCacheByteSlice) -> u32;

/// Clear all entries.
///
/// `backend` must be non-null. [`DYNAMIC_CACHE_STATUS_OK`] means all entries
/// have been removed before the function returns.
pub type DynamicCacheClearFn = unsafe extern "C" fn(backend: *mut c_void) -> u32;

/// Return approximate entry count and total weighted bytes.
///
/// `backend`, `out_num_entries`, and `out_size_bytes` must be non-null. On
/// [`DYNAMIC_CACHE_STATUS_OK`], the backend must write both out parameters. On
/// [`DYNAMIC_CACHE_STATUS_ERROR`], hosts ignore both out parameters.
pub type DynamicCacheMeasureFn = unsafe extern "C" fn(
    backend: *mut c_void,
    out_num_entries: *mut usize,
    out_size_bytes: *mut usize,
) -> u32;

/// Free bytes returned by [`DynamicCacheGetFn`].
///
/// `bytes` must be a value returned by this backend from a successful
/// [`DynamicCacheGetFn`] hit and must be freed exactly once. Empty buffers are
/// valid and should be treated as no-ops. Hosts may call this function from a
/// different thread than the original [`DynamicCacheGetFn`] call.
pub type DynamicCacheFreeBytesFn = unsafe extern "C" fn(bytes: DynamicCacheOwnedBytes);

/// Reserved optional function slot.
///
/// Future ABI revisions may replace reserved slots with typed optional
/// function pointers. Backends must initialize all reserved slots to `None`.
pub type DynamicCacheReservedFn = unsafe extern "C" fn();

/// Function table exported by a dynamic cache backend.
///
/// All functions in this table, except [`DynamicCacheDestroyFn`], may be called
/// concurrently for the same backend instance.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DynamicCacheBackendVTable {
    /// Size in bytes of the vtable filled by the backend.
    ///
    /// Backends must set this to [`DYNAMIC_CACHE_BACKEND_VTABLE_SIZE`]. Hosts
    /// must validate this before reading fields that may not exist in older
    /// vtables.
    pub struct_size: usize,
    /// ABI version used to fill this table.
    pub abi_version: u32,
    /// Create an opaque backend instance.
    pub create: DynamicCacheCreateFn,
    /// Destroy an opaque backend instance.
    pub destroy: DynamicCacheDestroyFn,
    /// Look up serialized bytes.
    pub get: DynamicCacheGetFn,
    /// Insert serialized bytes.
    pub insert: DynamicCacheInsertFn,
    /// Invalidate by prefix.
    pub invalidate_prefix: DynamicCacheInvalidatePrefixFn,
    /// Clear all entries.
    pub clear: DynamicCacheClearFn,
    /// Measure approximate backend size.
    pub measure: DynamicCacheMeasureFn,
    /// Release bytes returned by `get`.
    pub free_bytes: DynamicCacheFreeBytesFn,
    /// Reserved optional function slots for future ABI revisions.
    ///
    /// Backends must initialize every slot to `None`. Future ABI revisions may
    /// replace slots from the start of this array with typed optional function
    /// pointers while keeping the vtable size stable.
    pub reserved: [Option<DynamicCacheReservedFn>; DYNAMIC_CACHE_BACKEND_VTABLE_RESERVED_SLOTS],
}

impl std::fmt::Debug for DynamicCacheBackendVTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicCacheBackendVTable")
            .field("struct_size", &self.struct_size)
            .field("abi_version", &self.abi_version)
            .finish_non_exhaustive()
    }
}

impl DynamicCacheBackendVTable {
    /// Create a vtable with current layout metadata and empty reserved slots.
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        abi_version: u32,
        create: DynamicCacheCreateFn,
        destroy: DynamicCacheDestroyFn,
        get: DynamicCacheGetFn,
        insert: DynamicCacheInsertFn,
        invalidate_prefix: DynamicCacheInvalidatePrefixFn,
        clear: DynamicCacheClearFn,
        measure: DynamicCacheMeasureFn,
        free_bytes: DynamicCacheFreeBytesFn,
    ) -> Self {
        Self {
            struct_size: DYNAMIC_CACHE_BACKEND_VTABLE_SIZE,
            abi_version,
            create,
            destroy,
            get,
            insert,
            invalidate_prefix,
            clear,
            measure,
            free_bytes,
            reserved: [None; DYNAMIC_CACHE_BACKEND_VTABLE_RESERVED_SLOTS],
        }
    }

    /// Whether this vtable is large enough to contain all required v1 fields.
    pub const fn has_required_fields(&self) -> bool {
        self.struct_size >= DYNAMIC_CACHE_BACKEND_VTABLE_REQUIRED_SIZE
    }
}

/// Current size in bytes of [`DynamicCacheBackendVTable`].
pub const DYNAMIC_CACHE_BACKEND_VTABLE_SIZE: usize =
    std::mem::size_of::<DynamicCacheBackendVTable>();

/// Minimum size in bytes needed to read all required v1 vtable fields.
pub const DYNAMIC_CACHE_BACKEND_VTABLE_REQUIRED_SIZE: usize =
    std::mem::offset_of!(DynamicCacheBackendVTable, reserved);

/// Initialization function exported as [`DYNAMIC_CACHE_BACKEND_INIT_SYMBOL`].
///
/// The host passes the ABI version it understands and the size in bytes of the
/// vtable buffer at `out_vtable`. Backends must not write more than
/// `host_vtable_size` bytes and must set [`DynamicCacheBackendVTable::struct_size`]
/// to the size they actually filled. Initialization is single-threaded for the
/// specific backend instance being created; concurrent operation calls may only
/// start after initialization returns [`DYNAMIC_CACHE_STATUS_OK`].
pub type DynamicCacheBackendInitFn = unsafe extern "C" fn(
    host_abi_version: u32,
    host_vtable_size: usize,
    out_vtable: *mut DynamicCacheBackendVTable,
) -> u32;

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn create_backend(
        _config: DynamicCacheByteSlice,
        _memory_capacity_bytes: usize,
        _out_backend: *mut *mut c_void,
    ) -> u32 {
        DYNAMIC_CACHE_STATUS_OK
    }

    unsafe extern "C" fn destroy_backend(_backend: *mut c_void) {}

    unsafe extern "C" fn get_entry(
        _backend: *mut c_void,
        _key: *const DynamicCacheKey,
        _out_result: *mut DynamicCacheGetResult,
    ) -> u32 {
        DYNAMIC_CACHE_STATUS_MISS
    }

    unsafe extern "C" fn insert_entry(
        _backend: *mut c_void,
        _key: *const DynamicCacheKey,
        _value: DynamicCacheByteSlice,
        _size_bytes: usize,
    ) -> u32 {
        DYNAMIC_CACHE_STATUS_OK
    }

    unsafe extern "C" fn invalidate_prefix(
        _backend: *mut c_void,
        _prefix: DynamicCacheByteSlice,
    ) -> u32 {
        DYNAMIC_CACHE_STATUS_OK
    }

    unsafe extern "C" fn clear_entries(_backend: *mut c_void) -> u32 {
        DYNAMIC_CACHE_STATUS_OK
    }

    unsafe extern "C" fn measure_entries(
        _backend: *mut c_void,
        _out_num_entries: *mut usize,
        _out_size_bytes: *mut usize,
    ) -> u32 {
        DYNAMIC_CACHE_STATUS_OK
    }

    unsafe extern "C" fn free_bytes(_bytes: DynamicCacheOwnedBytes) {}

    fn test_vtable() -> DynamicCacheBackendVTable {
        DynamicCacheBackendVTable::new(
            DYNAMIC_CACHE_ABI_VERSION,
            create_backend,
            destroy_backend,
            get_entry,
            insert_entry,
            invalidate_prefix,
            clear_entries,
            measure_entries,
            free_bytes,
        )
    }

    #[test]
    fn owned_bytes_round_trip_in_allocating_library() {
        let bytes = vec![1, 2, 3];
        let owned = DynamicCacheOwnedBytes::from_vec(bytes);
        // SAFETY: This test reconstructs the vector in the same crate that
        // allocated it, which mirrors a plugin's `free_bytes` implementation.
        let bytes = unsafe { owned.into_vec() };
        assert_eq!(bytes, vec![1, 2, 3]);
    }

    #[test]
    fn empty_byte_slice_uses_null_pointer() {
        let slice = DynamicCacheByteSlice::from_slice(&[]);
        assert!(slice.ptr.is_null());
        assert_eq!(slice.len, 0);

        // SAFETY: Empty slices do not require a readable pointer.
        assert_eq!(unsafe { slice.as_slice() }, &[]);
    }

    #[test]
    fn empty_owned_bytes_use_null_pointer_even_with_vec_capacity() {
        let bytes = Vec::with_capacity(8);
        let owned = DynamicCacheOwnedBytes::from_vec(bytes);
        assert!(owned.ptr.is_null());
        assert_eq!(owned.len, 0);
        assert_eq!(owned.capacity, 0);

        // SAFETY: Empty owned bytes contain no allocation to reconstruct.
        let bytes = unsafe { owned.into_vec() };
        assert!(bytes.is_empty());
        assert_eq!(bytes.capacity(), 0);
    }

    #[test]
    fn status_codes_match_rust_status_enum() {
        assert_eq!(DynamicCacheStatus::Ok.as_u32(), DYNAMIC_CACHE_STATUS_OK);
        assert_eq!(DynamicCacheStatus::Hit.as_u32(), DYNAMIC_CACHE_STATUS_HIT);
        assert_eq!(DynamicCacheStatus::Miss.as_u32(), DYNAMIC_CACHE_STATUS_MISS);
        assert_eq!(
            DynamicCacheStatus::Error.as_u32(),
            DYNAMIC_CACHE_STATUS_ERROR
        );

        assert_eq!(
            DynamicCacheStatus::try_from(DYNAMIC_CACHE_STATUS_OK),
            Ok(DynamicCacheStatus::Ok)
        );
        assert_eq!(
            DynamicCacheStatus::try_from(DYNAMIC_CACHE_STATUS_HIT),
            Ok(DynamicCacheStatus::Hit)
        );
        assert_eq!(
            DynamicCacheStatus::try_from(DYNAMIC_CACHE_STATUS_MISS),
            Ok(DynamicCacheStatus::Miss)
        );
        assert_eq!(
            DynamicCacheStatus::try_from(DYNAMIC_CACHE_STATUS_ERROR),
            Ok(DynamicCacheStatus::Error)
        );
        assert_eq!(
            DynamicCacheStatus::try_from(99),
            Err(UnknownDynamicCacheStatus(99))
        );
    }

    #[test]
    fn vtable_new_fills_current_layout_metadata() {
        let vtable = test_vtable();
        assert_eq!(vtable.struct_size, DYNAMIC_CACHE_BACKEND_VTABLE_SIZE);
        assert_eq!(vtable.abi_version, DYNAMIC_CACHE_ABI_VERSION);
        assert!(vtable.has_required_fields());
        assert!(vtable.reserved.iter().all(Option::is_none));
    }

    #[test]
    fn vtable_required_size_excludes_reserved_slots() {
        assert_eq!(
            DYNAMIC_CACHE_BACKEND_VTABLE_REQUIRED_SIZE,
            std::mem::offset_of!(DynamicCacheBackendVTable, reserved)
        );
        const {
            assert!(DYNAMIC_CACHE_BACKEND_VTABLE_REQUIRED_SIZE < DYNAMIC_CACHE_BACKEND_VTABLE_SIZE);
        }
    }

    #[test]
    fn vtable_rejects_short_required_layout() {
        let mut vtable = test_vtable();
        vtable.struct_size = DYNAMIC_CACHE_BACKEND_VTABLE_REQUIRED_SIZE - 1;
        assert!(!vtable.has_required_fields());
    }
}
