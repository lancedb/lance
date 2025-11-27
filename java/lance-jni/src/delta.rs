// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::blocking_dataset::{BlockingDataset, NATIVE_DATASET};
use crate::error::Result;
use crate::ffi::JNIEnvExt;
use crate::transaction::convert_to_java_transaction;
use crate::RT;
use arrow::ffi_stream::FFI_ArrowArrayStream;
use jni::objects::{JObject, JValue};
use jni::sys::jlong;
use jni::JNIEnv;
use lance::dataset::delta::DatasetDelta as RustDatasetDelta;
use lance::dataset::scanner::DatasetRecordBatchStream;
use lance::dataset::transaction::Transaction;
use lance_io::ffi::to_ffi_arrow_array_stream;

pub const NATIVE_DELTA: &str = "nativeDeltaHandle";

// Removed Clone derive because RustDatasetDelta does not implement Clone
pub struct BlockingDatasetDelta {
    pub(crate) inner: RustDatasetDelta,
}

fn attach_native_delta<'local>(
    env: &mut JNIEnv<'local>,
    delta: BlockingDatasetDelta,
    java_dataset: &JObject<'local>,
) -> Result<JObject<'local>> {
    let j_delta = env.new_object("org/lance/delta/DatasetDelta", "()V", &[])?;

    // Set native Rust field
    unsafe { env.set_rust_field(&j_delta, NATIVE_DELTA, delta) }?;

    // Attach base Dataset Java object for Transaction conversion
    env.set_field(
        &j_delta,
        "dataset",
        "Lorg/lance/Dataset;",
        JValue::Object(java_dataset),
    )?;
    Ok(j_delta)
}

#[no_mangle]
pub extern "system" fn Java_org_lance_delta_DatasetDeltaBuilder_nativeBuild<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject<'local>,
    java_dataset: JObject<'local>,
    compared_against_obj: JObject<'local>, // Optional<Long>
    begin_version_obj: JObject<'local>,    // Optional<Long>
    end_version_obj: JObject<'local>,      // Optional<Long>
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_native_build(
            &mut env,
            java_dataset,
            compared_against_obj,
            begin_version_obj,
            end_version_obj
        )
    )
}

#[no_mangle]
pub extern "system" fn Java_org_lance_Dataset_nativeBuildDelta<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject<'local>,
    compared_against_obj: JObject<'local>, // Optional<Long>
    begin_version_obj: JObject<'local>,    // Optional<Long>
    end_version_obj: JObject<'local>,      // Optional<Long>
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_native_build(
            &mut env,
            java_dataset,
            compared_against_obj,
            begin_version_obj,
            end_version_obj
        )
    )
}

fn inner_native_build<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject<'local>,
    compared_against_obj: JObject<'local>,
    begin_version_obj: JObject<'local>,
    end_version_obj: JObject<'local>,
) -> Result<JObject<'local>> {
    // Extract Option<u64> from Optional<Long> first to avoid overlapping borrows of env
    let compared_against = env.get_u64_opt(&compared_against_obj)?;
    let begin_version = env.get_u64_opt(&begin_version_obj)?;
    let end_version = env.get_u64_opt(&end_version_obj)?;

    // Get the Rust Dataset guard and build the delta inside a scope to drop the guard early
    let delta = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(&java_dataset, NATIVE_DATASET)? };

        // Build Rust DatasetDelta using the builder semantics
        let mut builder = dataset_guard.inner.delta();
        if let Some(compared) = compared_against {
            builder = builder.compared_against_version(compared);
        } else {
            if let Some(begin) = begin_version {
                builder = builder.with_begin_version(begin);
            }
            if let Some(end) = end_version {
                builder = builder.with_end_version(end);
            }
        }
        builder.build()?
    };

    let blocking_delta = BlockingDatasetDelta { inner: delta };
    attach_native_delta(env, blocking_delta, &java_dataset)
}

#[no_mangle]
pub extern "system" fn Java_org_lance_delta_DatasetDelta_listTransactions<'local>(
    mut env: JNIEnv<'local>,
    j_delta: JObject<'local>,
) -> JObject<'local> {
    ok_or_throw!(env, inner_list_transactions(&mut env, j_delta))
}

fn inner_list_transactions<'local>(
    env: &mut JNIEnv<'local>,
    j_delta: JObject<'local>,
) -> Result<JObject<'local>> {
    // Borrow env to get the guard and fetch transactions inside a small scope so the guard is dropped
    let txs: Vec<Transaction> = {
        let delta_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDatasetDelta>(&j_delta, NATIVE_DELTA) }?;
        RT.block_on(delta_guard.inner.list_transactions())?
    };

    // Retrieve the Java Dataset attached to the DatasetDelta for conversion context
    let java_dataset = env
        .get_field(&j_delta, "dataset", "Lorg/lance/Dataset;")?
        .l()?;

    // Convert to Java List<Transaction>
    let array_list = env.new_object("java/util/ArrayList", "()V", &[])?;
    for tx in txs.into_iter() {
        let jtx = convert_to_java_transaction(env, tx, &java_dataset)?;
        env.call_method(
            &array_list,
            "add",
            "(Ljava/lang/Object;)Z",
            &[JValue::Object(&jtx)],
        )?;
    }
    Ok(array_list)
}

#[no_mangle]
pub extern "system" fn Java_org_lance_delta_DatasetDelta_getInsertedRows<'local>(
    mut env: JNIEnv<'local>,
    j_delta: JObject,
    stream_addr: jlong,
) {
    ok_or_throw_without_return!(env, inner_get_inserted_rows(&mut env, j_delta, stream_addr))
}

fn inner_get_inserted_rows(env: &mut JNIEnv, j_delta: JObject, stream_addr: jlong) -> Result<()> {
    let delta_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDatasetDelta>(&j_delta, NATIVE_DELTA) }?;

    let stream: DatasetRecordBatchStream = RT.block_on(delta_guard.inner.get_inserted_rows())?;
    let ffi_stream = to_ffi_arrow_array_stream(stream, RT.handle().clone())?;

    unsafe { std::ptr::write_unaligned(stream_addr as *mut FFI_ArrowArrayStream, ffi_stream) }
    Ok(())
}

#[no_mangle]
pub extern "system" fn Java_org_lance_delta_DatasetDelta_getUpdatedRows<'local>(
    mut env: JNIEnv<'local>,
    j_delta: JObject,
    stream_addr: jlong,
) {
    ok_or_throw_without_return!(env, inner_get_updated_rows(&mut env, j_delta, stream_addr))
}

fn inner_get_updated_rows(env: &mut JNIEnv, j_delta: JObject, stream_addr: jlong) -> Result<()> {
    let delta_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDatasetDelta>(&j_delta, NATIVE_DELTA) }?;

    let stream: DatasetRecordBatchStream = RT.block_on(delta_guard.inner.get_updated_rows())?;
    let ffi_stream = to_ffi_arrow_array_stream(stream, RT.handle().clone())?;

    unsafe { std::ptr::write_unaligned(stream_addr as *mut FFI_ArrowArrayStream, ffi_stream) }
    Ok(())
}

#[no_mangle]
pub extern "system" fn Java_org_lance_delta_DatasetDelta_releaseNativeDelta(
    mut env: JNIEnv,
    j_delta: JObject,
) {
    ok_or_throw_without_return!(env, inner_release_native_delta(&mut env, j_delta));
}

fn inner_release_native_delta(env: &mut JNIEnv, j_delta: JObject) -> Result<()> {
    let _: BlockingDatasetDelta = unsafe { env.take_rust_field(j_delta, NATIVE_DELTA) }?;
    Ok(())
}
