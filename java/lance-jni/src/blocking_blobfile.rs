// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::error::Result;
use crate::traits::IntoJava;
use crate::RT;
use jni::objects::JObject;
use jni::sys::{jbyteArray, jlong};
use jni::JNIEnv;
use lance::dataset::blob::BlobFile;
use std::sync::Arc;

pub const NATIVE_BLOB_FILE: &str = "nativeBlobFileHandle";

/// Wrapper for the Rust BlobFile to handle JNI interactions
#[derive(Clone)]
pub struct BlockingBlobFile {
    pub(crate) inner: Arc<BlobFile>,
}

impl BlockingBlobFile {
    pub fn new(inner: Arc<BlobFile>) -> Self {
        Self { inner }
    }

    pub fn read(&self) -> Result<Vec<u8>> {
        RT.block_on(async {
            let bytes = self.inner.read().await?;
            Ok(bytes.to_vec())
        })
    }

    pub fn read_async(&self) -> Result<Vec<u8>> {
        // For async version, we still use blocking here but in a real implementation
        // this would be handled differently with proper async JNI support
        self.read()
    }

    pub fn read_up_to(&self, length: usize) -> Result<Vec<u8>> {
        RT.block_on(async {
            let bytes = self.inner.read_up_to(length).await?;
            Ok(bytes.to_vec())
        })
    }

    pub fn read_up_to_async(&self, length: usize) -> Result<Vec<u8>> {
        self.read_up_to(length)
    }

    pub fn seek(&self, position: u64) -> Result<()> {
        RT.block_on(async { Ok(self.inner.seek(position).await?) })
    }

    pub fn seek_async(&self, position: u64) -> Result<()> {
        self.seek(position)
    }

    pub fn tell(&self) -> Result<u64> {
        RT.block_on(async { Ok(self.inner.tell().await?) })
    }

    pub fn tell_async(&self) -> Result<u64> {
        self.tell()
    }

    pub fn size(&self) -> u64 {
        self.inner.size()
    }

    pub fn close(&self) -> Result<()> {
        RT.block_on(async { self.inner.close().await })
    }
}

impl IntoJava for BlockingBlobFile {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        attach_native_blob_file(env, self)
    }
}

fn attach_native_blob_file<'local>(
    env: &mut JNIEnv<'local>,
    blob_file: BlockingBlobFile,
) -> Result<JObject<'local>> {
    let j_blob_file = create_java_blob_file_object(env)?;
    unsafe { env.set_rust_field(&j_blob_file, NATIVE_BLOB_FILE, blob_file) }?;
    Ok(j_blob_file)
}

fn create_java_blob_file_object<'a>(env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
    let object = env.new_object("com/lancedb/lance/blob/BlobFile", "(J)V", &[jlong::from(0).into()])?;
    Ok(object)
}

// JNI method implementations for BlobFile

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_blob_BlobFile_nativeRead(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
) -> jbyteArray {
    ok_or_throw_with_return!(
        env,
        inner_blob_file_read(&mut env, handle),
        std::ptr::null_mut()
    )
}

fn inner_blob_file_read(env: &mut JNIEnv, handle: jlong) -> Result<jbyteArray> {
    let blob_file = unsafe {
        std::ptr::read(handle as *const BlockingBlobFile)
    };
    let data = blob_file.read()?;
    let byte_array = env.byte_array_from_slice(&data)?;
    Ok(**byte_array)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_blob_BlobFile_nativeReadAsync(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
) -> jbyteArray {
    ok_or_throw_with_return!(
        env,
        inner_blob_file_read_async(&mut env, handle),
        std::ptr::null_mut()
    )
}

fn inner_blob_file_read_async(env: &mut JNIEnv, handle: jlong) -> Result<jbyteArray> {
    let blob_file = unsafe {
        std::ptr::read(handle as *const BlockingBlobFile)
    };
    let data = blob_file.read_async()?;
    let byte_array = env.byte_array_from_slice(&data)?;
    Ok(**byte_array)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_blob_BlobFile_nativeReadUpTo(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    length: jni::sys::jint,
) -> jbyteArray {
    ok_or_throw_with_return!(
        env,
        inner_blob_file_read_up_to(&mut env, handle, length),
        std::ptr::null_mut()
    )
}

fn inner_blob_file_read_up_to(env: &mut JNIEnv, handle: jlong, length: jni::sys::jint) -> Result<jbyteArray> {
    let blob_file = unsafe {
        std::ptr::read(handle as *const BlockingBlobFile)
    };
    let data = blob_file.read_up_to(length as usize)?;
    let byte_array = env.byte_array_from_slice(&data)?;
    Ok(**byte_array)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_blob_BlobFile_nativeReadUpToAsync(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    length: jni::sys::jint,
) -> jbyteArray {
    ok_or_throw_with_return!(
        env,
        inner_blob_file_read_up_to_async(&mut env, handle, length),
        std::ptr::null_mut()
    )
}

fn inner_blob_file_read_up_to_async(env: &mut JNIEnv, handle: jlong, length: jni::sys::jint) -> Result<jbyteArray> {
    let blob_file = unsafe {
        std::ptr::read(handle as *const BlockingBlobFile)
    };
    let data = blob_file.read_up_to_async(length as usize)?;
    let byte_array = env.byte_array_from_slice(&data)?;
    Ok(**byte_array)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_blob_BlobFile_nativeSeek(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    position: jlong,
) {
    ok_or_throw_without_return!(env, inner_blob_file_seek(&mut env, handle, position))
}

fn inner_blob_file_seek(env: &mut JNIEnv, handle: jlong, position: jlong) -> Result<()> {
    let blob_file = unsafe {
        std::ptr::read(handle as *const BlockingBlobFile)
    };
    blob_file.seek(position as u64)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_blob_BlobFile_nativeSeekAsync(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    position: jlong,
) {
    ok_or_throw_without_return!(env, inner_blob_file_seek_async(&mut env, handle, position))
}

fn inner_blob_file_seek_async(env: &mut JNIEnv, handle: jlong, position: jlong) -> Result<()> {
    let blob_file = unsafe {
        std::ptr::read(handle as *const BlockingBlobFile)
    };
    blob_file.seek_async(position as u64)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_blob_BlobFile_nativeTell(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
) -> jlong {
    ok_or_throw_with_return!(env, inner_blob_file_tell(&mut env, handle), -1) as jlong
}

fn inner_blob_file_tell(env: &mut JNIEnv, handle: jlong) -> Result<u64> {
    let blob_file = unsafe {
        std::ptr::read(handle as *const BlockingBlobFile)
    };
    blob_file.tell()
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_blob_BlobFile_nativeTellAsync(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
) -> jlong {
    ok_or_throw_with_return!(env, inner_blob_file_tell_async(&mut env, handle), -1) as jlong
}

fn inner_blob_file_tell_async(env: &mut JNIEnv, handle: jlong) -> Result<u64> {
    let blob_file = unsafe {
        std::ptr::read(handle as *const BlockingBlobFile)
    };
    blob_file.tell_async()
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_blob_BlobFile_nativeSize(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
) -> jlong {
    let blob_file = unsafe {
        std::ptr::read(handle as *const BlockingBlobFile)
    };
    blob_file.size() as jlong
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_blob_BlobFile_nativeClose(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
) {
    ok_or_throw_without_return!(env, inner_blob_file_close(&mut env, handle))
}

fn inner_blob_file_close(env: &mut JNIEnv, handle: jlong) -> Result<()> {
    let blob_file = unsafe {
        std::ptr::read(handle as *const BlockingBlobFile)
    };
    blob_file.close()
}