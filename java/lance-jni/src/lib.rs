// Copyright 2023 Lance Developers.
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

mod blocking_dataset;
mod jni_helpers;

use crate::blocking_dataset::BlockingDataset;
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use jni::objects::{JMap, JObject, JString};
use jni::sys::{jint, jlong};
use jni::JNIEnv;
use lance::dataset::{WriteMode, WriteParams};
use lazy_static::lazy_static;

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
    let path_str = match jni_helpers::extract_path_str(&mut env, &path) {
        Ok(value) => value,
        Err(err_str) => {
            env.throw_new("java/lang/IllegalArgumentException", err_str)
                .expect("Error throwing exception");
            return JObject::null();
        }
    };

    let write_params = match extract_write_params(&mut env, &params) {
        Ok(value) => value,
        Err(err_str) => {
            env.throw_new("java/lang/IllegalArgumentException", err_str)
                .expect("Error throwing exception");
            return JObject::null();
        }
    };

    let stream_ptr = arrow_array_stream_addr as *mut FFI_ArrowArrayStream;
    let stream = unsafe { ArrowArrayStreamReader::from_raw(stream_ptr) };
    match stream {
        Ok(reader) => match BlockingDataset::write(reader, &path_str, Some(write_params)) {
            Ok(dataset) => attach_native_dataset(&mut env, dataset),
            Err(err) => {
                jni_helpers::throw_java_exception(
                    &mut env,
                    format!("Failed to write from arrow array stream: {}", err).as_str(),
                );
                JObject::null()
            }
        },
        Err(err) => {
            jni_helpers::throw_java_exception(
                &mut env,
                &format!("Failed to extract arrow array stream: {}", err),
            );
            JObject::null()
        }
    }
}

pub fn extract_write_params(env: &mut JNIEnv, params: &JObject) -> Result<WriteParams, String> {
    let params_map = JMap::from_env(env, params)
        .map_err(|err| format!("Failed to extract JMap write parameters: {}", err))?;

    let mut write_params = WriteParams::default();

    jni_helpers::extract_and_process_jni_map_value::<i32, _>(
        env,
        &params_map,
        "max_rows_per_file",
        |v| {
            write_params.max_rows_per_file = v as usize;
            Ok(())
        },
    )?;

    jni_helpers::extract_and_process_jni_map_value::<i32, _>(
        env,
        &params_map,
        "max_rows_per_group",
        |v| {
            write_params.max_rows_per_group = v as usize;
            Ok(())
        },
    )?;

    jni_helpers::extract_and_process_jni_map_value::<i64, _>(
        env,
        &params_map,
        "max_bytes_per_file",
        |v| {
            write_params.max_bytes_per_file = v as usize;
            Ok(())
        },
    )?;

    jni_helpers::extract_and_process_jni_map_value::<String, _>(
        env,
        &params_map,
        "mode",
        |mode_str| match mode_str.to_uppercase().as_str() {
            "CREATE" => {
                write_params.mode = WriteMode::Create;
                Ok(())
            }
            "APPEND" => {
                write_params.mode = WriteMode::Append;
                Ok(())
            }
            "OVERWRITE" => {
                write_params.mode = WriteMode::Overwrite;
                Ok(())
            }
            _ => Err(format!("Unsupported write mode provided: {}", mode_str)),
        },
    )?;

    Ok(write_params)
}

fn attach_native_dataset<'local>(
    env: &mut JNIEnv<'local>,
    dataset: BlockingDataset,
) -> JObject<'local> {
    let j_dataset = create_java_dataset_object(env);
    // This block sets a native Rust object (dataset) as a field in the Java object (j_dataset).
    // Caution: This creates a potential for memory leaks. The Rust object (dataset) is not
    // automatically garbage-collected by Java, and its memory will not be freed unless
    // explicitly handled.
    //
    // To prevent memory leaks, ensure the following:
    // 1. The Java object (`j_dataset`) should implement the `java.io.Closeable` interface.
    // 2. Users of this Java object should be instructed to always use it within a try-with-resources
    //    statement (or manually call the `close()` method) to ensure that `self.close()` is invoked.
        match unsafe { env.set_rust_field(&j_dataset, "nativeDatasetHandle", dataset) } {
            Ok(_) => j_dataset,
            Err(err) => {
                env.throw_new(
                    "java/lang/RuntimeException",
                    format!("Failed to set native handle: {}", err),
                )
                .expect("Error throwing exception");
                JObject::null()
            }
        }
    }
}

fn create_java_dataset_object<'a>(env: &mut JNIEnv<'a>) -> JObject<'a> {
    env.new_object("com/lancedb/lance/Dataset", "()V", &[])
        .expect("Failed to create Java Dataset instance")
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_open<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    path: JString,
) -> JObject<'local> {
    let Ok(path_str) = jni_helpers::extract_path_str(&mut env, &path) else {
        return JObject::null();
    };

    match BlockingDataset::open(&path_str) {
        Ok(dataset) => attach_native_dataset(&mut env, dataset),
        Err(err) => {
            env.throw_new(
                "java/lang/RuntimeException",
                format!("Failed to open dataset: {}", err),
            )
            .expect("Error throwing exception");
            JObject::null()
        }
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
            .count_rows()
            .expect("Faild to get the row count from dataset's metadata.")
            as jint,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_releaseNativeDataset(
    mut env: JNIEnv,
    obj: JObject,
) {
    let dataset: BlockingDataset = unsafe {
        env.take_rust_field(obj, "nativeDatasetHandle")
            .expect("Failed to take native dataset handle")
    };
    dataset.close()
}
