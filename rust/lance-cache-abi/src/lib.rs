// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! C ABI for dynamically loaded Lance cache backends.
//!
//! The ABI is byte-oriented by design. Lance owns typed cache entries and
//! codecs; dynamic backends only receive stable key parts and serialized bytes.

use std::ffi::c_void;

/// ABI version understood by this crate.
pub const DYNAMIC_CACHE_ABI_VERSION: u32 = 1;

/// Exported symbol every dynamic cache backend must provide.
pub const DYNAMIC_CACHE_BACKEND_INIT_SYMBOL: &[u8] = b"lance_cache_backend_init\0";

/// Operation completed successfully.
pub const DYNAMIC_CACHE_STATUS_OK: u32 = 0;
/// Lookup found a byte payload.
pub const DYNAMIC_CACHE_STATUS_HIT: u32 = 1;
/// Lookup did not find a byte payload.
pub const DYNAMIC_CACHE_STATUS_MISS: u32 = 2;
/// Operation failed inside the dynamic backend.
pub const DYNAMIC_CACHE_STATUS_ERROR: u32 = 3;

/// Borrowed bytes passed through the dynamic cache ABI.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DynamicCacheByteSlice {
    /// Pointer to the first byte, or null when `len == 0`.
    pub ptr: *const u8,
    /// Number of bytes available at `ptr`.
    pub len: usize,
}

impl DynamicCacheByteSlice {
    /// Create an ABI slice from Rust bytes.
    pub fn from_slice(bytes: &[u8]) -> Self {
        Self {
            ptr: bytes.as_ptr(),
            len: bytes.len(),
        }
    }

    /// Borrow the slice described by this ABI value.
    ///
    /// # Safety
    ///
    /// The caller must ensure `ptr` is valid for `len` bytes and outlives the
    /// returned slice.
    pub unsafe fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// Bytes allocated by the dynamic backend and released through `free_bytes`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DynamicCacheOwnedBytes {
    /// Pointer to the first byte, or null when `len == 0`.
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
        if bytes.capacity() == 0 {
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
    /// bytes.
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
pub type DynamicCacheCreateFn = unsafe extern "C" fn(
    config: DynamicCacheByteSlice,
    memory_capacity_bytes: usize,
    out_backend: *mut *mut c_void,
) -> u32;

/// Destroy a backend instance previously created by [`DynamicCacheCreateFn`].
pub type DynamicCacheDestroyFn = unsafe extern "C" fn(backend: *mut c_void);

/// Look up a serialized byte payload.
pub type DynamicCacheGetFn = unsafe extern "C" fn(
    backend: *mut c_void,
    key: *const DynamicCacheKey,
    out_result: *mut DynamicCacheGetResult,
) -> u32;

/// Insert a serialized byte payload.
pub type DynamicCacheInsertFn = unsafe extern "C" fn(
    backend: *mut c_void,
    key: *const DynamicCacheKey,
    value: DynamicCacheByteSlice,
    size_bytes: usize,
) -> u32;

/// Invalidate all entries whose prefix starts with `prefix`.
pub type DynamicCacheInvalidatePrefixFn =
    unsafe extern "C" fn(backend: *mut c_void, prefix: DynamicCacheByteSlice) -> u32;

/// Clear all entries.
pub type DynamicCacheClearFn = unsafe extern "C" fn(backend: *mut c_void) -> u32;

/// Return approximate entry count and total weighted bytes.
pub type DynamicCacheMeasureFn = unsafe extern "C" fn(
    backend: *mut c_void,
    out_num_entries: *mut usize,
    out_size_bytes: *mut usize,
) -> u32;

/// Free bytes returned by [`DynamicCacheGetFn`].
pub type DynamicCacheFreeBytesFn = unsafe extern "C" fn(bytes: DynamicCacheOwnedBytes);

/// Function table exported by a dynamic cache backend.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DynamicCacheBackendVTable {
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
}

impl std::fmt::Debug for DynamicCacheBackendVTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicCacheBackendVTable")
            .field("abi_version", &self.abi_version)
            .finish_non_exhaustive()
    }
}

/// Initialization function exported as [`DYNAMIC_CACHE_BACKEND_INIT_SYMBOL`].
pub type DynamicCacheBackendInitFn =
    unsafe extern "C" fn(abi_version: u32, out_vtable: *mut DynamicCacheBackendVTable) -> u32;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_bytes_round_trip_in_allocating_library() {
        let bytes = vec![1, 2, 3];
        let owned = DynamicCacheOwnedBytes::from_vec(bytes);
        // SAFETY: This test reconstructs the vector in the same crate that
        // allocated it, which mirrors a plugin's `free_bytes` implementation.
        let bytes = unsafe { owned.into_vec() };
        assert_eq!(bytes, vec![1, 2, 3]);
    }
}
