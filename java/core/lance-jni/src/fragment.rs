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

use arrow::array::{RecordBatch, RecordBatchIterator, RecordBatchReader, StructArray};
use arrow::ffi::{from_ffi_and_data_type, FFI_ArrowArray, FFI_ArrowSchema};
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow_schema::{DataType, Schema};
use jni::{
    objects::{JObject, JString},
    sys::{jint, jlong},
    JNIEnv,
};
use snafu::{location, Location};
use std::iter::once;

use lance::dataset::fragment::FileFragment;

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
            location: location!(),
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
                    location: location!(),
                })?
        } else {
            schema.clone()
        };
        Ok((&schema).into())
    }
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Fragment_createWithFfiArray<'a>(
    mut env: JNIEnv<'a>,
    _obj: JObject,
    dataset_uri: JString,
    arrow_array_addr: jlong,
    arrow_schema_addr: jlong,
    fragment_id: JObject,        // Optional<Integer>
    max_rows_per_file: JObject,  // Optional<Integer>
    max_rows_per_group: JObject, // Optional<Integer>
    max_bytes_per_file: JObject, // Optional<Long>
    mode: JObject,               // Optional<String>
) -> JString<'a> {
    let c_array_ptr = arrow_array_addr as *mut FFI_ArrowArray;
    let c_schema_ptr = arrow_schema_addr as *mut FFI_ArrowSchema;

    let c_array = unsafe { FFI_ArrowArray::from_raw(c_array_ptr) };
    let c_schema = unsafe { FFI_ArrowSchema::from_raw(c_schema_ptr) };
    let data_type =
        ok_or_throw_with_return!(env, DataType::try_from(&c_schema), JString::default());

    let array_data = ok_or_throw_with_return!(
        env,
        unsafe { from_ffi_and_data_type(c_array, data_type) },
        JString::default()
    );

    let record_batch = RecordBatch::from(StructArray::from(array_data));
    let batch_schema = record_batch.schema().clone();
    let reader = RecordBatchIterator::new(once(Ok(record_batch)), batch_schema);

    ok_or_throw_with_return!(
        env,
        create_fragment(
            &mut env,
            dataset_uri,
            fragment_id,
            max_rows_per_file,
            max_rows_per_group,
            max_bytes_per_file,
            mode,
            reader
        ),
        JString::default()
    )
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Fragment_createWithFfiStream<'a>(
    mut env: JNIEnv<'a>,
    _obj: JObject,
    dataset_uri: JString,
    arrow_array_stream_addr: jlong,
    fragment_id: JObject,        // Optional<Integer>
    max_rows_per_file: JObject,  // Optional<Integer>
    max_rows_per_group: JObject, // Optional<Integer>
    max_bytes_per_file: JObject, // Optional<Long>
    mode: JObject,               // Optional<String>
) -> JString<'a> {
    let stream_ptr = arrow_array_stream_addr as *mut FFI_ArrowArrayStream;
    let reader = ok_or_throw_with_return!(
        env,
        unsafe { ArrowArrayStreamReader::from_raw(stream_ptr) }.map_err(|e| Error::Arrow {
            message: e.to_string(),
            location: location!(),
        }),
        JString::default()
    );

    ok_or_throw_with_return!(
        env,
        create_fragment(
            &mut env,
            dataset_uri,
            fragment_id,
            max_rows_per_file,
            max_rows_per_group,
            max_bytes_per_file,
            mode,
            reader
        ),
        JString::default()
    )
}

#[allow(clippy::too_many_arguments)]
fn create_fragment<'a>(
    env: &mut JNIEnv<'a>,
    dataset_uri: JString,
    fragment_id: JObject,        // Optional<Integer>
    max_rows_per_file: JObject,  // Optional<Integer>
    max_rows_per_group: JObject, // Optional<Integer>
    max_bytes_per_file: JObject, // Optional<Long>
    mode: JObject,               // Optional<String>
    reader: impl RecordBatchReader + Send + 'static,
) -> Result<JString<'a>> {
    let path_str = dataset_uri.extract(env)?;

    let fragment_id_opts = env.get_int_opt(&fragment_id)?;

    let write_params = extract_write_params(
        env,
        &max_rows_per_file,
        &max_rows_per_group,
        &max_bytes_per_file,
        &mode,
    )?;
    let fragment = RT.block_on(FileFragment::create(
        &path_str,
        fragment_id_opts.unwrap_or(0) as usize,
        reader,
        Some(write_params),
    ))?;
    let json_string = serde_json::to_string(&fragment)?;
    Ok(env.new_string(json_string)?)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_DatasetFragment_countRowsNative(
    mut env: JNIEnv,
    _jfragment: JObject,
    jdataset: JObject,
    fragment_id: jlong,
) -> jint {
    ok_or_throw_with_return!(
        env,
        {
            let dataset =
                unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }
                    .expect("Dataset handle not set");
            fragment_count_rows(&dataset, fragment_id)
        },
        -1
    )
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_ipc_FragmentScanner_importFfiSchema(
    mut env: JNIEnv,
    _scanner: JObject,
    jdataset: JObject,
    arrow_schema_addr: jlong,
    fragment_id: jint,
    columns: JObject, // Optional<String[]>
) {
    let columns = ok_or_throw_without_return!(env, env.get_strings_opt(&columns));

    let res = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }
                .expect("Dataset handle not set");
        dataset.clone()
    };
    let scanner = ok_or_throw_without_return!(
        env,
        RT.block_on(async { FragmentScanner::try_open(&res, fragment_id as usize).await })
    );

    let schema = ok_or_throw_without_return!(env, scanner.schema(columns));
    let c_schema = ok_or_throw_without_return!(env, FFI_ArrowSchema::try_from(&schema));
    let out_c_schema = arrow_schema_addr as *mut FFI_ArrowSchema;
    unsafe {
        std::ptr::copy(std::ptr::addr_of!(c_schema), out_c_schema, 1);
        std::mem::forget(c_schema);
    };
}
