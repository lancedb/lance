// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::blocking_dataset::{BlockingDataset, NATIVE_DATASET};
use crate::error::Result;
use crate::traits::{FromJString, IntoJava};
use crate::{JNIEnvExt, RT};
use jni::objects::{JByteArray, JObject, JString, JValueGen};
use jni::sys::{jbyteArray, jint, jlong};
use jni::JNIEnv;
use lance::dataset::BlobFile;
use std::mem::transmute;
use std::sync::Arc;

const BLOB_FILE_CLASS: &str = "com/lancedb/lance/BlobFile";
const BLOB_FILE_CTOR_SIG: &str = "()V";
const NATIVE_BLOB: &str = "nativeBlobHandle";

pub struct BlockingBlobFile {
    pub(crate) inner: BlobFile,
}

impl IntoJava for BlockingBlobFile {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let j_blob = env.new_object(BLOB_FILE_CLASS, BLOB_FILE_CTOR_SIG, &[])?;
        unsafe { env.set_rust_field(&j_blob, NATIVE_BLOB, self) }?;
        Ok(j_blob)
    }
}

impl From<BlobFile> for BlockingBlobFile {
    fn from(b: BlobFile) -> Self {
        Self { inner: b }
    }
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeTakeBlobs<'local>(
    mut env: JNIEnv<'local>,
    jdataset: JObject,
    row_ids_obj: JObject, // List<Long>
    column: JString,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_take_blobs(&mut env, jdataset, row_ids_obj, column)
    )
}

fn inner_take_blobs<'local>(
    env: &mut JNIEnv<'local>,
    jdataset: JObject,
    row_ids_obj: JObject, // List<Long>
    column: JString,
) -> Result<JObject<'local>> {
    let row_ids_i64 = env.get_longs(&row_ids_obj)?;
    let row_ids_u64: Vec<u64> = row_ids_i64.into_iter().map(|x| x as u64).collect();
    let col_name: String = column.extract(env)?;
    let blobs = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }?;
        RT.block_on(Arc::new(dataset.inner.clone()).take_blobs(&row_ids_u64, col_name))?
    };
    let j_blobs = blobs
        .into_iter()
        .map(BlockingBlobFile::from)
        .collect::<Vec<BlockingBlobFile>>();
    transform_vec(env, j_blobs)
}

fn transform_vec<'local>(
    env: &mut JNIEnv<'local>,
    vec: Vec<BlockingBlobFile>,
) -> Result<JObject<'local>> {
    let array_list_class = env.find_class("java/util/ArrayList")?;
    let array_list = env.new_object(array_list_class, "()V", &[])?;
    for blob_file in vec {
        let blob_file_obj = blob_file.into_java(env)?;
        env.call_method(
            &array_list,
            "add",
            "(Ljava/lang/Object;)Z",
            &[JValueGen::Object(&blob_file_obj)],
        )?;
    }
    Ok(array_list)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeTakeBlobsByIndices<'local>(
    mut env: JNIEnv<'local>,
    jdataset: JObject,
    row_indices_obj: JObject, // List<Long>
    column: JString,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_take_blobs_by_indices(&mut env, jdataset, row_indices_obj, column)
    )
}

fn inner_take_blobs_by_indices<'local>(
    env: &mut JNIEnv<'local>,
    jdataset: JObject,
    row_indices_obj: JObject, // List<Long>
    column: JString,
) -> Result<JObject<'local>> {
    let row_indices_i64 = env.get_longs(&row_indices_obj)?;
    let row_indices_u64: Vec<u64> = row_indices_i64.into_iter().map(|x| x as u64).collect();
    let col_name: String = column.extract(env)?;
    let blobs = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }?;
        RT.block_on(
            Arc::new(dataset.inner.clone()).take_blobs_by_indices(&row_indices_u64, col_name),
        )?
    };
    let j_blobs = blobs
        .into_iter()
        .map(BlockingBlobFile::from)
        .collect::<Vec<BlockingBlobFile>>();
    transform_vec(env, j_blobs)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_BlobFile_nativeRead(
    mut env: JNIEnv,
    jblob: JObject,
) -> jbyteArray {
    ok_or_throw_with_return!(
        env,
        inner_blob_read(&mut env, jblob).map(|arr| arr.into_raw()),
        JByteArray::default().into_raw()
    )
}

fn inner_blob_read<'local>(env: &mut JNIEnv<'local>, jblob: JObject) -> Result<JByteArray<'local>> {
    let bytes = {
        let blob = unsafe { env.get_rust_field::<_, _, BlockingBlobFile>(jblob, NATIVE_BLOB) }?;
        RT.block_on(blob.inner.read())?
    };
    let arr = env.new_byte_array(bytes.len() as jint)?;
    let u8_slice: &[u8] = bytes.as_ref();
    let i8_slice: &[i8] = unsafe { transmute(u8_slice) };

    env.set_byte_array_region(&arr, 0, i8_slice)?;
    Ok(arr)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_BlobFile_nativeReadUpTo<'local>(
    mut env: JNIEnv<'local>,
    jblob: JObject,
    len: jint,
) -> jbyteArray {
    ok_or_throw_with_return!(
        env,
        inner_blob_read_up_to(&mut env, jblob, len).map(|arr| arr.into_raw()),
        JByteArray::default().into_raw()
    )
}

fn inner_blob_read_up_to<'local>(
    env: &mut JNIEnv<'local>,
    jblob: JObject,
    len: jint,
) -> Result<JByteArray<'local>> {
    let bytes = {
        let blob = unsafe { env.get_rust_field::<_, _, BlockingBlobFile>(jblob, NATIVE_BLOB) }?;
        RT.block_on(blob.inner.read_up_to(len as usize))?
    };
    let arr = env.new_byte_array(bytes.len() as jint)?;
    let u8_slice: &[u8] = bytes.as_ref();
    let i8_slice: &[i8] = unsafe { transmute(u8_slice) };

    env.set_byte_array_region(&arr, 0, i8_slice)?;
    Ok(arr)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_BlobFile_nativeSeek(
    mut env: JNIEnv,
    jblob: JObject,
    new_cursor: jlong,
) {
    ok_or_throw_without_return!(env, inner_blob_seek(&mut env, jblob, new_cursor));
}

fn inner_blob_seek(env: &mut JNIEnv, jblob: JObject, new_cursor: jlong) -> Result<()> {
    let blob = unsafe { env.get_rust_field::<_, _, BlockingBlobFile>(jblob, NATIVE_BLOB) }?;
    RT.block_on(blob.inner.seek(new_cursor as u64))?;
    Ok(())
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_BlobFile_nativeTell(
    mut env: JNIEnv,
    jblob: JObject,
) -> jlong {
    ok_or_throw_with_return!(env, inner_blob_tell(&mut env, jblob), -1) as jlong
}

fn inner_blob_tell(env: &mut JNIEnv, jblob: JObject) -> Result<u64> {
    let blob = unsafe { env.get_rust_field::<_, _, BlockingBlobFile>(jblob, NATIVE_BLOB) }?;
    Ok(RT.block_on(blob.inner.tell())?)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_BlobFile_nativeSize(
    mut env: JNIEnv,
    jblob: JObject,
) -> jlong {
    ok_or_throw_with_return!(env, inner_blob_size(&mut env, jblob), -1) as jlong
}

fn inner_blob_size(env: &mut JNIEnv, jblob: JObject) -> Result<u64> {
    let blob = unsafe { env.get_rust_field::<_, _, BlockingBlobFile>(jblob, NATIVE_BLOB) }?;
    Ok(blob.inner.size())
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_BlobFile_nativeClose(
    mut env: JNIEnv,
    jblob: JObject,
) {
    ok_or_throw_without_return!(env, inner_blob_close(&mut env, jblob));
}

fn inner_blob_close(env: &mut JNIEnv, jblob: JObject) -> Result<()> {
    let blob = unsafe { env.take_rust_field::<_, _, BlockingBlobFile>(jblob, NATIVE_BLOB)? };
    RT.block_on(blob.inner.close())?;
    Ok(())
}
