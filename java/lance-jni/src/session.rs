// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use jni::objects::JObject;
use jni::sys::jlong;
use jni::JNIEnv;
use lance::dataset::{DEFAULT_INDEX_CACHE_SIZE, DEFAULT_METADATA_CACHE_SIZE};
use lance::session::Session as LanceSession;
use lance_io::object_store::ObjectStoreRegistry;

use crate::error::{Error, Result};
use crate::ok_or_throw_with_return;

/// Creates a new Session and returns a handle to it.
///
/// The handle is a raw pointer to a Box<Arc<LanceSession>>, which allows
/// the session to be shared between multiple datasets.
#[no_mangle]
pub extern "system" fn Java_org_lance_Session_createNative(
    mut env: JNIEnv,
    _obj: JObject,
    index_cache_size_bytes: jlong,
    metadata_cache_size_bytes: jlong,
) -> jlong {
    ok_or_throw_with_return!(
        env,
        create_session(index_cache_size_bytes, metadata_cache_size_bytes),
        0
    )
}

fn create_session(
    index_cache_size_bytes: jlong,
    metadata_cache_size_bytes: jlong,
) -> Result<jlong> {
    let index_cache_size = if index_cache_size_bytes >= 0 {
        index_cache_size_bytes as usize
    } else {
        DEFAULT_INDEX_CACHE_SIZE
    };

    let metadata_cache_size = if metadata_cache_size_bytes >= 0 {
        metadata_cache_size_bytes as usize
    } else {
        DEFAULT_METADATA_CACHE_SIZE
    };

    let session = LanceSession::new(
        index_cache_size,
        metadata_cache_size,
        Arc::new(ObjectStoreRegistry::default()),
    );

    // Wrap in Arc and Box, then convert to raw pointer
    let boxed: Box<Arc<LanceSession>> = Box::new(Arc::new(session));
    let handle = Box::into_raw(boxed) as jlong;
    Ok(handle)
}

/// Returns the current size of the session in bytes.
#[no_mangle]
pub extern "system" fn Java_org_lance_Session_sizeBytesNative(
    mut env: JNIEnv,
    obj: JObject,
) -> jlong {
    ok_or_throw_with_return!(env, size_bytes_native(&mut env, obj), 0)
}

fn size_bytes_native(env: &mut JNIEnv, obj: JObject) -> Result<jlong> {
    let handle = get_session_handle(env, &obj)?;
    if handle == 0 {
        return Err(Error::input_error("Session is closed".to_string()));
    }

    // Safety: We trust that the handle is valid and was created by createNative
    let session_arc = unsafe { &*(handle as *const Arc<LanceSession>) };
    Ok(session_arc.size_bytes() as jlong)
}

/// Releases the native session handle.
#[no_mangle]
pub extern "system" fn Java_org_lance_Session_releaseNative(
    _env: JNIEnv,
    _obj: JObject,
    handle: jlong,
) {
    if handle != 0 {
        // Safety: We trust that the handle is valid and was created by createNative
        let _ = unsafe { Box::from_raw(handle as *mut Arc<LanceSession>) };
        // The Box is dropped here, which decrements the Arc reference count
    }
}

/// Helper function to get the session handle from a Session object
fn get_session_handle(env: &mut JNIEnv, obj: &JObject) -> Result<jlong> {
    let handle = env.get_field(obj, "nativeSessionHandle", "J")?;
    Ok(handle.j()?)
}

/// Creates an Arc<LanceSession> from a raw handle.
/// This is used when passing a session to dataset operations.
///
/// # Safety
/// The handle must be a valid pointer created by `create_session`.
pub fn session_from_handle(handle: jlong) -> Option<Arc<LanceSession>> {
    if handle == 0 {
        return None;
    }

    // Safety: We trust that the handle is valid and was created by createNative
    let session_arc = unsafe { &*(handle as *const Arc<LanceSession>) };
    Some(session_arc.clone())
}

/// Creates a raw handle from an Arc<LanceSession>.
/// This is used when returning a session handle from a dataset.
///
/// Note: This creates a new Box, so the caller is responsible for
/// managing its lifetime or converting it back to a Java Session object.
pub fn handle_from_session(session: Arc<LanceSession>) -> jlong {
    let boxed: Box<Arc<LanceSession>> = Box::new(session);
    Box::into_raw(boxed) as jlong
}

/// Compares two session handles to see if they point to the same underlying session.
/// This is needed because each call to handle_from_session creates a new Box,
/// resulting in different pointer addresses even for the same session.
#[no_mangle]
pub extern "system" fn Java_org_lance_Session_isSameAsNative(
    _env: JNIEnv,
    _obj: JObject,
    handle1: jlong,
    handle2: jlong,
) -> jni::sys::jboolean {
    if handle1 == 0 || handle2 == 0 {
        return 0; // false
    }

    // Safety: We trust that the handles are valid and were created by createNative
    let session1 = unsafe { &*(handle1 as *const Arc<LanceSession>) };
    let session2 = unsafe { &*(handle2 as *const Arc<LanceSession>) };

    if Arc::ptr_eq(session1, session2) {
        1 // true
    } else {
        0 // false
    }
}
