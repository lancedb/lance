// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Thread-local error handling for FFI.
//!
//! After any C function returns an error indicator (NULL pointer or negative int),
//! the caller retrieves the error code and message from thread-local storage.

use std::cell::RefCell;
use std::ffi::{c_char, CString};
use std::ptr;

/// Error codes returned by `lance_last_error_code()`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LanceErrorCode {
    Ok = 0,
    InvalidArgument = 1,
    IoError = 2,
    NotFound = 3,
    DatasetAlreadyExists = 4,
    IndexError = 5,
    Internal = 6,
    NotSupported = 7,
    CommitConflict = 8,
}

struct LastError {
    code: LanceErrorCode,
    message: CString,
}

thread_local! {
    static LAST_ERROR: RefCell<Option<LastError>> = const { RefCell::new(None) };
}

pub fn clear_last_error() {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = None;
    });
}

pub fn set_last_error(code: LanceErrorCode, message: impl AsRef<str>) {
    let message = match CString::new(message.as_ref()) {
        Ok(v) => v,
        Err(_) => CString::new(message.as_ref().replace('\0', "\\0"))
            .unwrap_or_else(|_| CString::new("invalid error message").unwrap()),
    };
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = Some(LastError { code, message });
    });
}

/// Map a `lance_core::Error` to an `LanceErrorCode`.
pub fn error_code_from_lance(err: &lance_core::Error) -> LanceErrorCode {
    use lance_core::Error;
    match err {
        Error::InvalidInput { .. } => LanceErrorCode::InvalidArgument,
        Error::DatasetAlreadyExists { .. } => LanceErrorCode::DatasetAlreadyExists,
        Error::CommitConflict { .. } => LanceErrorCode::CommitConflict,
        Error::DatasetNotFound { .. } | Error::NotFound { .. } => LanceErrorCode::NotFound,
        Error::IO { .. } => LanceErrorCode::IoError,
        Error::Index { .. } => LanceErrorCode::IndexError,
        Error::NotSupported { .. } => LanceErrorCode::NotSupported,
        _ => LanceErrorCode::Internal,
    }
}

/// Set the thread-local error from a `lance_core::Error`.
pub fn set_lance_error(err: &lance_core::Error) {
    set_last_error(error_code_from_lance(err), err.to_string());
}

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------

/// Return the error code from the last failed operation on this thread.
/// Returns `LanceErrorCode::Ok` if no error is pending.
#[unsafe(no_mangle)]
pub extern "C" fn lance_last_error_code() -> LanceErrorCode {
    LAST_ERROR.with(|e| {
        e.borrow()
            .as_ref()
            .map(|v| v.code)
            .unwrap_or(LanceErrorCode::Ok)
    })
}

/// Return the error message from the last failed operation on this thread.
/// The caller must free the returned string with `lance_free_string()`.
/// Returns NULL if no error is pending.
#[unsafe(no_mangle)]
pub extern "C" fn lance_last_error_message() -> *const c_char {
    LAST_ERROR.with(|e| match e.borrow_mut().take() {
        Some(err) => err.message.into_raw() as *const c_char,
        None => ptr::null(),
    })
}

/// Free a string returned by `lance_last_error_message()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lance_free_string(s: *const c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s as *mut c_char);
        }
    }
}

// ---------------------------------------------------------------------------
// Helper macro for FFI functions
// ---------------------------------------------------------------------------

/// Wrap an FFI function body: on error, set thread-local error and return $err_val.
/// On success, clear the error and return the value.
macro_rules! ffi_try {
    ($body:expr, null) => {
        match $body {
            Ok(val) => {
                $crate::error::clear_last_error();
                val
            }
            Err(err) => {
                $crate::error::set_lance_error(&err);
                std::ptr::null_mut()
            }
        }
    };
    ($body:expr, neg) => {
        match $body {
            Ok(val) => {
                $crate::error::clear_last_error();
                val
            }
            Err(err) => {
                $crate::error::set_lance_error(&err);
                -1
            }
        }
    };
}

pub(crate) use ffi_try;
