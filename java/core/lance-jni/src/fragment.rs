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

use arrow::array::{RecordBatch, StructArray};
use arrow::ffi::{from_ffi_and_data_type, FFI_ArrowArray, FFI_ArrowSchema};
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow_schema::{DataType, Schema};
use jni::{
    objects::{JObject, JString},
    sys::{jint, jlong},
    JNIEnv,
};
use snafu::{location, Location};

use lance::dataset::{fragment::FileFragment, scanner::Scanner};
use lance_io::ffi::to_ffi_arrow_array_stream;

use crate::traits::SingleRecordBatchReader;
use crate::{
    blocking_dataset::{BlockingDataset, NATIVE_DATASET},
    error::{Error, Result},
    ffi::JNIEnvExt,
    traits::FromJString,
    utils::extract_write_params,
    RT,
};

fn fragment_count_rows(dataset: &BlockingDataset, fragment_id: jlong) -> Result<jint> {
    let Some(fragment) = dataset.inner.get_fragment(fragment_id as usize) else {
        return Err(Error::InvalidArgument {
            message: format!("Fragment not found: {}", fragment_id),
        });
    };
    Ok(RT.block_on(fragment.count_rows())? as jint)
}

struct FragmentScanner {
    fragment: FileFragment,
}

impl FragmentScanner {
    async fn try_open(dataset: &BlockingDataset, fragment_id: usize) -> Result<Self> {
        let fragment = dataset
            .inner
            .get_fragment(fragment_id)
            .ok_or_else(|| Error::IO {
                message: format!("Fragment not found: {}", fragment_id),
                location: location!(),
            })?;
        Ok(Self { fragment })
    }

    /// Returns the schema of the scanner, with optional columns.
    fn schema(&self, columns: Option<Vec<String>>) -> Result<Schema> {
        let schema = self.fragment.schema();
        let schema = if let Some(columns) = columns {
            schema
                .project(&columns)
                .map_err(|e| Error::InvalidArgument {
                    message: format!("Failed to select columns: {}", e),
                })?
        } else {
            schema.clone()
        };
        Ok((&schema).into())
    }
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Fragment_createWithFfiArray(
    mut env: JNIEnv,
    _obj: JObject,
    dataset_uri: JString,
    arrow_array_addr: jlong,
    arrow_schema_addr: jlong,
    fragment_id: JObject,        // Optional<Integer>
    max_rows_per_file: JObject,  // Optional<Integer>
    max_rows_per_group: JObject, // Optional<Integer>
    max_bytes_per_file: JObject, // Optional<Long>
    mode: JObject,               // Optional<String>
) -> jint {
    let c_array_ptr = arrow_array_addr as *mut FFI_ArrowArray;
    let c_schema_ptr = arrow_schema_addr as *mut FFI_ArrowSchema;

    let c_array = unsafe { FFI_ArrowArray::from_raw(c_array_ptr) };
    let c_schema = unsafe { FFI_ArrowSchema::from_raw(c_schema_ptr) };
    let data_type = ok_or_throw_with_return!(
        env,
        DataType::try_from(&c_schema).map_err(|e| {
            Error::Arrow {
                message: e.to_string(),
                location: location!(),
            }
        }),
        -1
    );

    let array_data = ok_or_throw_with_return!(
        env,
        unsafe {
            from_ffi_and_data_type(c_array, data_type).map_err(|e| Error::Arrow {
                message: e.to_string(),
                location: location!(),
            })
        },
        -1
    );

    let record_batch = RecordBatch::from(StructArray::from(array_data));
    let batch_schema = record_batch.schema().clone();
    let reader = SingleRecordBatchReader::new(Some(record_batch), batch_schema);

    let path_str: String = ok_or_throw_with_return!(env, dataset_uri.extract(&mut env), -1);
    let fragment_id_opts = ok_or_throw_with_return!(env, env.get_int_opt(&fragment_id), -1);

    let write_params = ok_or_throw_with_return!(
        env,
        extract_write_params(
            &mut env,
            &max_rows_per_file,
            &max_rows_per_group,
            &max_bytes_per_file,
            &mode
        ),
        -1
    );

    match RT.block_on(FileFragment::create(
        &path_str,
        fragment_id_opts.unwrap_or(0) as usize,
        reader,
        Some(write_params),
    )) {
        Ok(fragment) => fragment.id as jint,
        Err(e) => {
            Error::IO {
                message: e.to_string(),
                location: location!(),
            }
            .throw(&mut env);
            -1
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Fragment_createWithFfiStream(
    mut env: JNIEnv,
    _obj: JObject,
    dataset_uri: JString,
    arrow_array_stream_addr: jlong,
    fragment_id: JObject,        // Optional<Integer>
    max_rows_per_file: JObject,  // Optional<Integer>
    max_rows_per_group: JObject, // Optional<Integer>
    max_bytes_per_file: JObject, // Optional<Long>
    mode: JObject,               // Optional<String>
) -> jint {
    let path_str: String = ok_or_throw_with_return!(env, dataset_uri.extract(&mut env), -1);
    let stream_ptr = arrow_array_stream_addr as *mut FFI_ArrowArrayStream;
    let reader = ok_or_throw_with_return!(
        env,
        unsafe { ArrowArrayStreamReader::from_raw(stream_ptr) }.map_err(|e| Error::Arrow {
            message: e.to_string(),
            location: location!(),
        }),
        -1
    );

    let fragment_id_opts = ok_or_throw_with_return!(env, env.get_int_opt(&fragment_id), -1);

    let write_params = ok_or_throw_with_return!(
        env,
        extract_write_params(
            &mut env,
            &max_rows_per_file,
            &max_rows_per_group,
            &max_bytes_per_file,
            &mode
        ),
        -1
    );

    match RT.block_on(FileFragment::create(
        &path_str,
        fragment_id_opts.unwrap_or(0) as usize,
        reader,
        Some(write_params),
    )) {
        Ok(fragment) => fragment.id as jint,
        Err(e) => {
            Error::IO {
                message: e.to_string(),
                location: location!(),
            }
            .throw(&mut env);
            -1
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Fragment_countRowsNative(
    mut env: JNIEnv,
    _jfragment: JObject,
    jdataset: JObject,
    fragment_id: jlong,
) -> jint {
    let res = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }
                .expect("Dataset handle not set");
        fragment_count_rows(&dataset, fragment_id)
    };
    match res {
        Ok(r) => r,
        Err(e) => {
            e.throw(&mut env);
            -1
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_ipc_FragmentScanner_getSchema(
    mut env: JNIEnv,
    _scanner: JObject,
    jdataset: JObject,
    fragment_id: jint,
    columns: JObject, // Optional<String[]>
) -> jlong {
    let columns = match env.get_strings_opt(&columns) {
        Ok(c) => c,
        Err(e) => {
            env.throw(e.to_string()).expect("Failed to throw exception");
            return -1;
        }
    };

    let res = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }
                .expect("Dataset handle not set");
        dataset.clone()
    };
    let scanner = ok_or_throw_with_return!(
        env,
        RT.block_on(async { FragmentScanner::try_open(&res, fragment_id as usize).await }),
        0
    );

    let schema = ok_or_throw_with_return!(env, scanner.schema(columns), 0);
    let ffi_schema =
        Box::new(FFI_ArrowSchema::try_from(&schema).expect("Failed to convert schema"));
    Box::into_raw(ffi_schema) as jlong
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_ipc_FragmentScanner_openStream(
    mut env: JNIEnv,
    _reader: JObject,
    jdataset: JObject,
    fragment_id: jint,
    columns: JObject, // Optional<String[]>
    batch_size: jlong,
    stream_addr: jlong,
) {
    let columns = match env.get_strings_opt(&columns) {
        Ok(c) => c,
        Err(e) => {
            env.throw(e.to_string()).expect("Failed to throw exception");
            return;
        }
    };
    let dataset = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }
                .expect("Dataset handle not set");
        dataset.clone()
    };
    let Some(fragment) = dataset.inner.get_fragment(fragment_id as usize) else {
        env.throw("Fragment not found")
            .expect("Throw exception failed");
        return;
    };
    let mut scanner: Scanner = fragment.scan();
    if let Some(cols) = columns {
        if let Err(e) = scanner.project(&cols) {
            env.throw(format!("Setting scanner projection: {}", e))
                .expect("Throw exception failed");
            return;
        }
    };
    scanner.batch_size(batch_size as usize);

    let stream = match RT.block_on(async { scanner.try_into_stream().await }) {
        Ok(s) => s,
        Err(e) => {
            env.throw(e.to_string()).expect("Throw exception failed");
            return;
        }
    };
    let ffi_stream = to_ffi_arrow_array_stream(stream, RT.handle().clone()).unwrap();
    unsafe { std::ptr::write_unaligned(stream_addr as *mut FFI_ArrowArrayStream, ffi_stream) }
}
