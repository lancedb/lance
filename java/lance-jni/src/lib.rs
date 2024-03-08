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

use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use jni::objects::{JMap, JObject, JString};
use jni::sys::{jint, jlong};
use jni::JNIEnv;
use lance::dataset::{WriteMode, WriteParams};
use lazy_static::lazy_static;
use snafu::{location, Location};
use traits::IntoJava;

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

mod blocking_dataset;
pub mod error;
mod ffi;
mod fragment;
mod traits;

use self::traits::{FromJString, JMapExt};
use crate::blocking_dataset::BlockingDataset;
pub use error::{Error, Result};

lazy_static! {
    static ref RT: tokio::runtime::Runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to create tokio runtime");
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_writeWithFfiStream<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    arrow_array_stream_addr: jlong,
    path: JString,
    params: JObject,
) -> JObject<'local> {
    let path_str: String = ok_or_throw!(env, path.extract(&mut env));

    let write_params = ok_or_throw!(env, extract_write_params(&mut env, &params));

    let stream_ptr = arrow_array_stream_addr as *mut FFI_ArrowArrayStream;
    let reader = ok_or_throw!(
        env,
        unsafe { ArrowArrayStreamReader::from_raw(stream_ptr) }.map_err(|e| Error::Arrow {
            message: e.to_string(),
            location: location!(),
        })
    );

    let dataset = ok_or_throw!(
        env,
        BlockingDataset::write(reader, &path_str, Some(write_params))
    );
    dataset.into_java(&mut env)
}

pub fn extract_write_params(env: &mut JNIEnv, params: &JObject) -> Result<WriteParams> {
    let params_map = JMap::from_env(env, params)?;

    let mut write_params = WriteParams::default();

    if let Some(max_rows) = params_map.get_i32(env, "max_row_per_file")? {
        write_params.max_rows_per_group = max_rows as usize;
    }
    if let Some(max_bytes) = params_map.get_i64(env, "max_bytes_per_file")? {
        write_params.max_bytes_per_file = max_bytes as usize;
    }
    if let Some(mode) = params_map.get_string(env, "mode")? {
        write_params.mode = WriteMode::try_from(mode.as_str())?;
    }
    Ok(write_params)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_open<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    path: JString,
) -> JObject<'local> {
    let path_str: String = ok_or_throw!(env, path.extract(&mut env));

    let dataset = ok_or_throw!(env, BlockingDataset::open(&path_str));
    dataset.into_java(&mut env)
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
            .count_rows()
            .expect("Faild to get the row count from dataset's metadata.")
            as jint,
        Err(_) => -1,
    }
}
