// Copyright 2024 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use jni::objects::{JClass, JObject, JObjectArray, JString};
use jni::sys::{jint, jlong, jobject};
use jni::JNIEnv;
use std::time::Duration;

use lance::dataset::{Dataset as LanceDataset, MergeInsertBuilder as LanceMergeInsertBuilder};
use lance::dataset::write::merge_insert::{WhenMatched, WhenNotMatched, WhenNotMatchedBySource};
use arrow::ffi::FFI_ArrowArrayStream;
use arrow::array::RecordBatchReader;
use arrow::record_batch::RecordBatch;

#[macro_export]
macro_rules! ok_or_throw {
    ($env:expr, $result:expr) => {
        match $result {
            Ok(value) => value,
            Err(err) => {
                err.throw(&mut $env);
                return JObject::null();
            }
        }
    };
}

macro_rules! ok_or_throw_without_return {
    ($env:expr, $result:expr) => {
        match $result {
            Ok(value) => value,
            Err(err) => {
                err.throw(&mut $env);
                return;
            }
        }
    };
}

#[macro_export]
macro_rules! ok_or_throw_with_return {
    ($env:expr, $result:expr, $ret:expr) => {
        match $result {
            Ok(value) => value,
            Err(err) => {
                err.throw(&mut $env);
                return $ret;
            }
        }
    };
}

mod blocking_dataset;
mod blocking_scanner;
pub mod error;
pub mod ffi;
mod file_reader;
mod file_writer;
mod fragment;
pub mod traits;
pub mod utils;

pub use error::Error;
pub use error::Result;
pub use ffi::JNIEnvExt;

use env_logger::{Builder, Env};
use std::sync::Arc;

use std::sync::LazyLock;

pub static RT: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to create tokio runtime")
});

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_JniLoader_initLanceLogger() {
    let env = Env::new()
        .filter_or("LANCE_LOG", "warn")
        .write_style("LANCE_LOG_STYLE");
    let mut log_builder = Builder::from_env(env);
    let logger = Arc::new(log_builder.build());
    let max_level = logger.filter();
    log::set_boxed_logger(Box::new(logger.clone())).unwrap();
    log::set_max_level(max_level);
    // todo: add tracing
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_MergeInsertBuilder_createNativeBuilder(
    mut env: JNIEnv,
    _class: JClass,
    dataset_handle: jlong,
    on_columns: JObjectArray,
) -> jlong {
    let dataset = unsafe { &*(dataset_handle as *const LanceDataset) };

    let len = env.get_array_length(&on_columns).unwrap() as usize;
    let mut columns = Vec::with_capacity(len);

    for i in 0..len {
        let jstr = env.get_object_array_element(&on_columns, i as i32).unwrap();
        let jstr = JString::from(jstr);
        let rust_str: String = env.get_string(&jstr).unwrap().into();
        columns.push(rust_str);
    }

    match LanceMergeInsertBuilder::try_new(Arc::new(dataset.clone()), columns) {
        Ok(mut builder) => {
            builder
                .when_matched(WhenMatched::DoNothing)
                .when_not_matched(WhenNotMatched::DoNothing);

            let builder_ptr = Box::into_raw(Box::new(builder));
            builder_ptr as jlong
        }
        Err(e) => {
            env.throw_new("java/lang/RuntimeException", &e.to_string()).unwrap();
            0
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_MergeInsertBuilder_whenMatchedUpdateAllNative(
    mut env: JNIEnv,
    _class: JClass,
    builder_handle: jlong,
    condition: JString,
) {
    let builder = unsafe { &mut *(builder_handle as *mut LanceMergeInsertBuilder) };

    let when_matched = if condition.is_null() {
        WhenMatched::UpdateAll
    } else {
        let condition_str: String = env.get_string(&condition).unwrap().into();
        match WhenMatched::update_if(&builder.dataset, &condition_str) {
            Ok(matched) => matched,
            Err(e) => {
                env.throw_new("java/lang/RuntimeException", &e.to_string()).unwrap();
                return;
            }
        }
    };

    builder.when_matched(when_matched);
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_MergeInsertBuilder_whenNotMatchedInsertAllNative(
    _env: JNIEnv,
    _class: JClass,
    builder_handle: jlong,
) {
    let builder = unsafe { &mut *(builder_handle as *mut LanceMergeInsertBuilder) };
    builder.when_not_matched(WhenNotMatched::InsertAll);
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_MergeInsertBuilder_whenNotMatchedBySourceDeleteNative(
    mut env: JNIEnv,
    _class: JClass,
    builder_handle: jlong,
    expr: JString,
) {
    let builder = unsafe { &mut *(builder_handle as *mut LanceMergeInsertBuilder) };

    let when_not_matched_by_source = if expr.is_null() {
        WhenNotMatchedBySource::Delete
    } else {
        let expr_str: String = env.get_string(&expr).unwrap().into();
        match WhenNotMatchedBySource::delete_if(&builder.dataset, &expr_str) {
            Ok(not_matched) => not_matched,
            Err(e) => {
                env.throw_new("java/lang/RuntimeException", &e.to_string()).unwrap();
                return;
            }
        }
    };

    builder.when_not_matched_by_source(when_not_matched_by_source);
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_MergeInsertBuilder_conflictRetriesNative(
    _env: JNIEnv,
    _class: JClass,
    builder_handle: jlong,
    max_retries: jint,
) {
    let builder = unsafe { &mut *(builder_handle as *mut LanceMergeInsertBuilder) };
    builder.conflict_retries(max_retries as u32);
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_MergeInsertBuilder_retryTimeoutNative(
    _env: JNIEnv,
    _class: JClass,
    builder_handle: jlong,
    timeout_millis: jlong,
) {
    let builder = unsafe { &mut *(builder_handle as *mut LanceMergeInsertBuilder) };
    builder.retry_timeout(Duration::from_millis(timeout_millis as u64));
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_MergeInsertBuilder_executeNative(
    mut env: JNIEnv,
    _class: JClass,
    builder_handle: jlong,
    stream_address: jlong,
) -> jobject {
    let builder = unsafe { Box::from_raw(builder_handle as *mut LanceMergeInsertBuilder) };

    // 从内存地址创建 ArrowArrayStream
    let stream_reader = unsafe {
        arrow::ffi::ArrowArrayStreamReader::from_raw(stream_address as *mut FFI_ArrowArrayStream)
    };

    // 执行 merge insert
    let runtime = tokio::runtime::Runtime::new().unwrap();
    match runtime.block_on(async {
        let job = builder.try_build()?;
        job.execute(Box::new(stream_reader)).await
    }) {
        Ok((_, stats)) => {
            // 创建 MergeInsertResult Java 对象
            let result_class = env.find_class("com/lancedb/lance/MergeInsertResult").unwrap();
            let result = env.new_object(
                result_class,
                "(JJJ)V",
                &[
                    (stats.num_inserted_rows as jlong).into(),
                    (stats.num_updated_rows as jlong).into(),
                    (stats.num_deleted_rows as jlong).into(),
                ],
            ).unwrap();
            result.into_raw()
        }
        Err(e) => {
            env.throw_new("java/lang/RuntimeException", &e.to_string()).unwrap();
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_MergeInsertBuilder_closeNative(
    _env: JNIEnv,
    _class: JClass,
    builder_handle: jlong,
) {
    if builder_handle != 0 {
        unsafe {
            let _ = Box::from_raw(builder_handle as *mut LanceMergeInsertBuilder);
        }
    }
}