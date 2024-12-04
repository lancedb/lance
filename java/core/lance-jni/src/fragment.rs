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

use arrow::array::{RecordBatch, RecordBatchIterator, StructArray};
use arrow::ffi::{from_ffi_and_data_type, FFI_ArrowArray, FFI_ArrowSchema};
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow_schema::DataType;
use jni::{
    objects::{JObject, JString},
    sys::{jint, jlong},
    JNIEnv,
};
use std::iter::once;

use lance::dataset::fragment::FileFragment;
use lance_datafusion::utils::StreamingWriteSource;

use crate::error::{Error, Result};
use crate::{
    blocking_dataset::{BlockingDataset, NATIVE_DATASET},
    traits::FromJString,
    utils::extract_write_params,
    RT,
};

//////////////////
// Read Methods //
//////////////////
#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_DatasetFragment_countRowsNative(
    mut env: JNIEnv,
    _jfragment: JObject,
    jdataset: JObject,
    fragment_id: jlong,
) -> jint {
    ok_or_throw_with_return!(
        env,
        inner_count_rows_native(&mut env, jdataset, fragment_id),
        -1
    ) as jint
}

fn inner_count_rows_native(
    env: &mut JNIEnv,
    jdataset: JObject,
    fragment_id: jlong,
) -> Result<usize> {
    let dataset = unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }?;
    let Some(fragment) = dataset.inner.get_fragment(fragment_id as usize) else {
        return Err(Error::input_error(format!(
            "Fragment not found: {fragment_id}"
        )));
    };
    let res = RT.block_on(fragment.count_rows())?;
    Ok(res)
}

///////////////////
// Write Methods //
///////////////////
#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Fragment_createWithFfiArray<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    dataset_uri: JString,
    arrow_array_addr: jlong,
    arrow_schema_addr: jlong,
    max_rows_per_file: JObject,   // Optional<Integer>
    max_rows_per_group: JObject,  // Optional<Integer>
    max_bytes_per_file: JObject,  // Optional<Long>
    mode: JObject,                // Optional<String>
    storage_options_obj: JObject, // Map<String, String>
) -> JString<'local> {
    ok_or_throw_with_return!(
        env,
        inner_create_with_ffi_array(
            &mut env,
            dataset_uri,
            arrow_array_addr,
            arrow_schema_addr,
            max_rows_per_file,
            max_rows_per_group,
            max_bytes_per_file,
            mode,
            storage_options_obj
        ),
        JString::default()
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_create_with_ffi_array<'local>(
    env: &mut JNIEnv<'local>,
    dataset_uri: JString,
    arrow_array_addr: jlong,
    arrow_schema_addr: jlong,
    max_rows_per_file: JObject,   // Optional<Integer>
    max_rows_per_group: JObject,  // Optional<Integer>
    max_bytes_per_file: JObject,  // Optional<Long>
    mode: JObject,                // Optional<String>
    storage_options_obj: JObject, // Map<String, String>
) -> Result<JString<'local>> {
    let c_array_ptr = arrow_array_addr as *mut FFI_ArrowArray;
    let c_schema_ptr = arrow_schema_addr as *mut FFI_ArrowSchema;

    let c_array = unsafe { FFI_ArrowArray::from_raw(c_array_ptr) };
    let c_schema = unsafe { FFI_ArrowSchema::from_raw(c_schema_ptr) };
    let data_type = DataType::try_from(&c_schema)?;

    let array_data = unsafe { from_ffi_and_data_type(c_array, data_type) }?;

    let record_batch = RecordBatch::from(StructArray::from(array_data));
    let batch_schema = record_batch.schema().clone();
    let reader = RecordBatchIterator::new(once(Ok(record_batch)), batch_schema);

    create_fragment(
        env,
        dataset_uri,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        mode,
        storage_options_obj,
        reader,
    )
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Fragment_createWithFfiStream<'a>(
    mut env: JNIEnv<'a>,
    _obj: JObject,
    dataset_uri: JString,
    arrow_array_stream_addr: jlong,
    max_rows_per_file: JObject,   // Optional<Integer>
    max_rows_per_group: JObject,  // Optional<Integer>
    max_bytes_per_file: JObject,  // Optional<Long>
    mode: JObject,                // Optional<String>
    storage_options_obj: JObject, // Map<String, String>
) -> JString<'a> {
    ok_or_throw_with_return!(
        env,
        inner_create_with_ffi_stream(
            &mut env,
            dataset_uri,
            arrow_array_stream_addr,
            max_rows_per_file,
            max_rows_per_group,
            max_bytes_per_file,
            mode,
            storage_options_obj
        ),
        JString::default()
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_create_with_ffi_stream<'local>(
    env: &mut JNIEnv<'local>,
    dataset_uri: JString,
    arrow_array_stream_addr: jlong,
    max_rows_per_file: JObject,   // Optional<Integer>
    max_rows_per_group: JObject,  // Optional<Integer>
    max_bytes_per_file: JObject,  // Optional<Long>
    mode: JObject,                // Optional<String>
    storage_options_obj: JObject, // Map<String, String>
) -> Result<JString<'local>> {
    let stream_ptr = arrow_array_stream_addr as *mut FFI_ArrowArrayStream;
    let reader = unsafe { ArrowArrayStreamReader::from_raw(stream_ptr) }?;

    create_fragment(
        env,
        dataset_uri,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        mode,
        storage_options_obj,
        reader,
    )
}

#[allow(clippy::too_many_arguments)]
fn create_fragment<'a>(
    env: &mut JNIEnv<'a>,
    dataset_uri: JString,
    max_rows_per_file: JObject,   // Optional<Integer>
    max_rows_per_group: JObject,  // Optional<Integer>
    max_bytes_per_file: JObject,  // Optional<Long>
    mode: JObject,                // Optional<String>
    storage_options_obj: JObject, // Map<String, String>
    source: impl StreamingWriteSource,
) -> Result<JString<'a>> {
    let path_str = dataset_uri.extract(env)?;

    let write_params = extract_write_params(
        env,
        &max_rows_per_file,
        &max_rows_per_group,
        &max_bytes_per_file,
        &mode,
        &storage_options_obj,
    )?;
    let fragment = RT.block_on(FileFragment::create_fragments(
        &path_str,
        source,
        Some(write_params),
    ))?;
    let json_string = serde_json::to_string(&fragment)?;
    let res = env.new_string(json_string)?;
    Ok(res)
}
