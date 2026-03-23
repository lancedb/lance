// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! C string parsing utilities.

use std::collections::HashMap;
use std::ffi::{c_char, CStr};

use lance_core::{Error, Result};
use snafu::location;

/// Parse a nullable C string pointer into an `Option<&str>`.
pub unsafe fn parse_c_string<'a>(ptr: *const c_char) -> Result<Option<&'a str>> {
    if ptr.is_null() {
        return Ok(None);
    }
    let cstr = unsafe { CStr::from_ptr(ptr) };
    let s = cstr.to_str().map_err(|e| Error::InvalidInput {
        source: Box::new(e),
        location: location!(),
    })?;
    Ok(Some(s))
}

/// Parse a NULL-terminated array of C strings into `Option<Vec<String>>`.
/// If `ptr` is NULL, returns `None` (meaning "all columns" / "no filter").
/// The array must end with a NULL pointer sentinel.
pub unsafe fn parse_c_string_array(ptr: *const *const c_char) -> Result<Option<Vec<String>>> {
    if ptr.is_null() {
        return Ok(None);
    }
    let mut result = Vec::new();
    let mut i = 0;
    loop {
        let entry = unsafe { *ptr.add(i) };
        if entry.is_null() {
            break;
        }
        let s = unsafe { parse_c_string(entry)? };
        match s {
            Some(s) => result.push(s.to_string()),
            None => break,
        }
        i += 1;
    }
    Ok(Some(result))
}

/// Parse NULL-terminated key-value pairs into a `HashMap`.
/// Format: `["key1", "val1", "key2", "val2", NULL]`
/// If `ptr` is NULL, returns an empty map.
pub unsafe fn parse_storage_options(ptr: *const *const c_char) -> Result<HashMap<String, String>> {
    let mut map = HashMap::new();
    if ptr.is_null() {
        return Ok(map);
    }
    let mut i = 0;
    loop {
        let key_ptr = unsafe { *ptr.add(i) };
        if key_ptr.is_null() {
            break;
        }
        let val_ptr = unsafe { *ptr.add(i + 1) };
        if val_ptr.is_null() {
            return Err(Error::InvalidInput {
                source: "storage options must be key-value pairs; odd number of entries".into(),
                location: location!(),
            });
        }
        let key = unsafe { parse_c_string(key_ptr)? }.unwrap().to_string();
        let val = unsafe { parse_c_string(val_ptr)? }.unwrap().to_string();
        map.insert(key, val);
        i += 2;
    }
    Ok(map)
}
