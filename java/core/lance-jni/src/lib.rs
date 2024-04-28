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

use std::iter::empty;
use std::sync::Arc;

use arrow::array::{RecordBatchIterator, RecordBatchReader};
use arrow::ffi::FFI_ArrowSchema;
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow_schema::Schema;
use blocking_dataset::NATIVE_DATASET;
use ffi::JNIEnvExt;
use jni::objects::{JObject, JString};
use jni::sys::{jint, jlong};
use jni::JNIEnv;
use lance::dataset::transaction::Operation;
use lance::table::format::Fragment;
use lance_io::ffi::to_ffi_arrow_array_stream;
use lazy_static::lazy_static;
use snafu::{location, Location};
use traits::IntoJava;

use crate::utils::extract_write_params;

#[macro_export]
macro_rules! ok_or_throw {
    ($env:expr, $result:expr) => {
        match $result {
            Ok(value) => value,
            Err(err) => {
                Error::from(err).throw(&mut $env);
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
                Error::from(err).throw(&mut $env);
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
                Error::from(err).throw(&mut $env);
                return $ret;
            }
        }
    };
}

mod blocking_dataset;
pub mod error;
mod ffi;
mod fragment;
mod traits;
mod utils;

use self::traits::FromJString;
use crate::blocking_dataset::BlockingDataset;
pub use error::{Error, Result};

lazy_static! {
    static ref RT: tokio::runtime::Runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to create tokio runtime");
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_createWithFfiSchema<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    arrow_schema_addr: jlong,
    path: JString,
    max_rows_per_file: JObject,  // Optional<Integer>
    max_rows_per_group: JObject, // Optional<Integer>
    max_bytes_per_file: JObject, // Optional<Long>
    mode: JObject,               // Optional<String>
) -> JObject<'local> {
    let c_schema_ptr = arrow_schema_addr as *mut FFI_ArrowSchema;
    let c_schema = unsafe { FFI_ArrowSchema::from_raw(c_schema_ptr) };
    let schema = ok_or_throw!(
        env,
        Schema::try_from(&c_schema).map_err(|e| Error::Arrow {
            message: e.to_string(),
            location: location!(),
        })
    );

    let reader = RecordBatchIterator::new(empty(), Arc::new(schema));
    ok_or_throw!(
        env,
        create_dataset(
            &mut env,
            path,
            max_rows_per_file,
            max_rows_per_group,
            max_bytes_per_file,
            mode,
            reader
        )
    )
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_writeWithFfiStream<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    arrow_array_stream_addr: jlong,
    path: JString,
    max_rows_per_file: JObject,  // Optional<Integer>
    max_rows_per_group: JObject, // Optional<Integer>
    max_bytes_per_file: JObject, // Optional<Long>
    mode: JObject,               // Optional<String>
) -> JObject<'local> {
    let stream_ptr = arrow_array_stream_addr as *mut FFI_ArrowArrayStream;
    let reader = ok_or_throw!(
        env,
        unsafe { ArrowArrayStreamReader::from_raw(stream_ptr) }.map_err(|e| Error::Arrow {
            message: e.to_string(),
            location: location!(),
        })
    );

    ok_or_throw!(
        env,
        create_dataset(
            &mut env,
            path,
            max_rows_per_file,
            max_rows_per_group,
            max_bytes_per_file,
            mode,
            reader
        )
    )
}

fn create_dataset<'local>(
    env: &mut JNIEnv<'local>,
    path: JString,
    max_rows_per_file: JObject,
    max_rows_per_group: JObject,
    max_bytes_per_file: JObject,
    mode: JObject,
    reader: impl RecordBatchReader + Send + 'static,
) -> Result<JObject<'local>> {
    let path_str = path.extract(env)?;

    let write_params = extract_write_params(
        env,
        &max_rows_per_file,
        &max_rows_per_group,
        &max_bytes_per_file,
        &mode,
    )?;

    let dataset = BlockingDataset::write(reader, &path_str, Some(write_params))?;
    Ok(dataset.into_java(env))
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_openNative<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    path: JString,
) -> JObject<'local> {
    let path_str: String = ok_or_throw!(env, path.extract(&mut env));

    let dataset = ok_or_throw!(env, BlockingDataset::open(&path_str));
    dataset.into_java(&mut env)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_commitAppend<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    path: JString,
    read_version_obj: JObject, // Optional<Long>
    fragments_obj: JObject,    // List<String>, String is json serialized Fragment
) -> JObject<'local> {
    let json_fragments = ok_or_throw!(env, env.get_strings(&fragments_obj));
    let mut fragments: Vec<Fragment> = Vec::new();
    for json_fragment in json_fragments {
        let fragment = ok_or_throw!(
            env,
            Fragment::from_json(&json_fragment).map_err(|err| Error::IO {
                message: err.to_string(),
                location: location!()
            })
        );
        fragments.push(fragment);
    }
    let op = Operation::Append { fragments };
    let path_str: String = ok_or_throw!(env, path.extract(&mut env));

    let read_version = ok_or_throw!(env, env.get_u64_opt(&read_version_obj));
    let dataset = ok_or_throw!(env, BlockingDataset::commit(&path_str, op, read_version));
    dataset.into_java(&mut env)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_version(
    mut env: JNIEnv,
    java_dataset: JObject,
) -> jlong {
    let dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, "nativeDatasetHandle") };
    match dataset_guard {
        Ok(dataset) => dataset.inner.version().version as jlong,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_latestVersion(
    mut env: JNIEnv,
    java_dataset: JObject,
) -> jlong {
    let dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, "nativeDatasetHandle") };
    match dataset_guard {
        Ok(dataset) => dataset
            .latest_version()
            .expect("Failed to get the latest version.") as jlong,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_countRows(
    mut env: JNIEnv,
    java_dataset: JObject,
) -> jint {
    let dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, "nativeDatasetHandle") };
    match dataset_guard {
        Ok(dataset) => dataset
            .count_rows(None)
            .expect("Faild to get the row count from dataset's metadata.")
            as jint,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_ipc_DatasetScanner_openStream(
    mut env: JNIEnv,
    _reader: JObject,
    jdataset: JObject,
    fragment_id_obj: JObject,      // Optional<Integer>
    columns: JObject,              // Optional<String[]>
    substrait_filter_obj: JObject, // Optional<ByteBuffer>
    filter_obj: JObject,           // Optional<String>
    batch_size: jlong,
    stream_addr: jlong,
) {
    let dataset = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }
                .expect("Dataset handle not set");
        dataset.clone()
    };
    let mut scanner = dataset.inner.scan();
    let fragment_id_opt = ok_or_throw_without_return!(env, env.get_int_opt(&fragment_id_obj));
    if let Some(fragment_id) = fragment_id_opt {
        let Some(fragment) = dataset.inner.get_fragment(fragment_id as usize) else {
            env.throw_new(
                "java/lang/RuntimeException",
                format!("fragment id {fragment_id} not found"),
            )
            .expect("failed to throw java exception");
            return;
        };
        scanner.with_fragments(vec![fragment.metadata().clone()]);
    }
    let columns = ok_or_throw_without_return!(env, env.get_strings_opt(&columns));
    if let Some(cols) = columns {
        ok_or_throw_without_return!(env, scanner.project(&cols));
    };
    let substrait = ok_or_throw_without_return!(env, env.get_bytes_opt(&substrait_filter_obj));
    if let Some(substrait) = substrait {
        ok_or_throw_without_return!(
            env,
            RT.block_on(async { scanner.filter_substrait(substrait).await })
        );
    }
    let filter = ok_or_throw_without_return!(env, env.get_string_opt(&filter_obj));
    if let Some(filter) = filter {
        ok_or_throw_without_return!(env, scanner.filter(filter.as_str()));
    }
    scanner.batch_size(batch_size as usize);

    let stream =
        ok_or_throw_without_return!(env, RT.block_on(async { scanner.try_into_stream().await }));
    let ffi_stream = to_ffi_arrow_array_stream(stream, RT.handle().clone()).unwrap();
    unsafe { std::ptr::write_unaligned(stream_addr as *mut FFI_ArrowArrayStream, ffi_stream) }
}
