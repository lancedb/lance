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

use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use crate::blocking_dataset::BlockingDataset;
use jni::objects::{JObject, JString};
use jni::JNIEnv;
use jni::sys::{jint, jlong};
use snafu::{location, Location};

use lance::dataset::{WriteMode, WriteParams};
use lance::Error;

mod blocking_dataset;

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_writeWithFFIStream<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    arrow_array_stream_addr: jlong,
    path: JString,
    max_rows_per_file: jint,
    max_rows_per_group: jint,
    max_bytes_per_file: jlong,
    mode: JString,
) -> JObject<'local> {
    let path_str = match extract_path_str(&mut env, &path) {
        Ok(value) => value,
        Err(_) => return JObject::null(),
    };

    let write_params = match extract_write_params(&mut env,
        &max_rows_per_file, &max_rows_per_group, &max_bytes_per_file, &mode) {
        Ok(value) => value,
        Err(_) => return JObject::null(),
    };

    unsafe {
        match ArrowArrayStreamReader::from_raw(arrow_array_stream_addr as *mut FFI_ArrowArrayStream) {
            Ok(reader) => {
                match BlockingDataset::write(reader, &path_str, Some(write_params)) {
                    Ok(dataset) => attach_native_dataset(&mut env, dataset),
                    Err(err) => {
                        throw_java_exception(&mut env, format!("Failed to write from arrow array stream: {}", err).as_str());
                        JObject::null()
                    }
                }
            },
            Err(err) => {
                throw_java_exception(&mut env, &format!("Failed to extract arrow array stream: {}", err));
                JObject::null()
            }
        }
    }
}

fn attach_native_dataset<'local>(env: &mut JNIEnv<'local>, dataset: BlockingDataset) -> JObject<'local> {
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
    unsafe {
        match env.set_rust_field(&j_dataset, "nativeDatasetHandle", dataset) {
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

fn throw_java_exception(env: &mut JNIEnv, err_msg: &str) {
    env.throw_new(
        "java/lang/RuntimeException",
        err_msg
    ).expect("Error throwing exception");
}

fn create_java_dataset_object<'a>(env: &mut JNIEnv<'a>) -> JObject<'a> {
    env
        .new_object("com/lancedb/lance/Dataset", "()V", &[])
        .expect("Failed to create Java Dataset instance")
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_open<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    path: JString,
) -> JObject<'local> {
    let path_str = match extract_path_str(&mut env, &path) {
        Ok(value) => value,
        Err(_) => return JObject::null(),
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
    let dataset_guard = unsafe {
        env.get_rust_field::<_, _, BlockingDataset>(java_dataset, "nativeDatasetHandle")
    };
    match dataset_guard {
        Ok(dataset) =>
            dataset.count_rows()
                .expect("Faild to get the row count from dataset's metadata.") as jint,
        Err(_) => {
            -1
        }
    }
}

fn extract_path_str(env: &mut JNIEnv, path: &JString) -> Result<String, Error> {
    match env.get_string(path) {
        Ok(path) => Ok(path.into()),
        Err(err) => {
            env.throw_new(
                "java/lang/IllegalArgumentException",
                format!("Invalid path string: {}", err),
            )
                .expect("Error throwing exception");
            Err(Error::InvalidTableLocation { message: format!("Invalid path string: {}", err)})
        }
    }
}

fn extract_write_params<'a>(
    env: &mut JNIEnv<'a>,
    max_rows_per_file: &jint,
    max_rows_per_group: &jint,
    max_bytes_per_file: &jlong,
    mode: &JString,
) -> Result<WriteParams, Error> {
    let mode_str: String = match env.get_string(mode) {
        Ok(mode) => Ok(mode.into()),
        Err(err) => {
            env.throw_new(
                "java/lang/IllegalArgumentException",
                format!("Invalid mode string: {}", err),
            )
                .expect("Error throwing exception");
            Err(Error::InvalidInput { source: "Invalid mode string".into(), location: location!() })
        }
    }?;

    let mode = match mode_str.as_str() {
        "CREATE" => WriteMode::Create,
        "APPEND" => WriteMode::Append,
        "OVERWRITE" => WriteMode::Overwrite,
        _ => {
            env.throw_new(
                "java/lang/IllegalArgumentException",
                format!("Unsupported write mode provided: {}", mode_str)
            )
                .expect("Error throwing exception");
            return Err(Error::InvalidInput { source: "Invalid mode string".into(), location: location!() });
        },
    };

    Ok(WriteParams {
        max_rows_per_file: *max_rows_per_file as usize,
        max_rows_per_group: *max_rows_per_group as usize,
        max_bytes_per_file: *max_bytes_per_file as usize,
        mode,
        ..Default::default()
    })
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_releaseNativeDataset(
    mut env: JNIEnv,
    obj: JObject,
) {
    unsafe {
        let dataset: BlockingDataset = env
            .take_rust_field(obj, "nativeDatasetHandle")
            .expect("Failed to take native dataset handle");
        dataset.close()
    }
}