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
use crate::ffi::JNIEnvExt;
use crate::error::{JavaResult, JavaErrorExt};
use crate::{traits::IntoJava, RT};
use crate::utils::extract_write_params;
use arrow::array::RecordBatchReader;
use jni::sys::jlong;
use jni::{objects::JObject, JNIEnv};
use lance::dataset::transaction::Operation;
use lance::dataset::{Dataset, WriteParams};
use lance::table::format::Fragment;
use jni::sys::jint;
use jni::objects::JString;
use arrow::ffi::FFI_ArrowSchema;
use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatchIterator;
use std::sync::Arc;
use arrow::ffi_stream::FFI_ArrowArrayStream;
use arrow::ffi_stream::ArrowArrayStreamReader;
use crate::traits::FromJString;

pub const NATIVE_DATASET: &str = "nativeDatasetHandle";

#[derive(Clone)]
pub struct BlockingDataset {
    pub(crate) inner: Dataset,
}

impl BlockingDataset {
    pub fn write(
        reader: impl RecordBatchReader + Send + 'static,
        uri: &str,
        params: Option<WriteParams>,
    ) -> JavaResult<Self> {
        let inner = RT.block_on(Dataset::write(reader, uri, params)).infer_error()?;
        Ok(Self { inner })
    }

    pub fn open(uri: &str) -> JavaResult<Self> {
        let inner = RT.block_on(Dataset::open(uri)).infer_error()?;
        Ok(Self { inner })
    }

    pub fn commit(uri: &str, operation: Operation, read_version: Option<u64>) -> JavaResult<Self> {
        let inner = RT.block_on(Dataset::commit(uri, operation, read_version, None, None)).infer_error()?;
        Ok(Self { inner })
    }

    pub fn latest_version(&self) -> JavaResult<u64> {
        Ok(RT.block_on(self.inner.latest_version_id()).infer_error()?)
    }

    pub fn count_rows(&self, filter: Option<String>) -> JavaResult<usize> {
        Ok(RT.block_on(self.inner.count_rows(filter)).infer_error()?)
    }

    pub fn close(&self) {}
}

///////////////////
// Write Methods //
///////////////////
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
    ok_or_throw!(env, inner_create_with_ffi_schema(&mut env, arrow_schema_addr, path, max_rows_per_file, max_rows_per_group, max_bytes_per_file, mode))
}

fn inner_create_with_ffi_schema<'local> (
    env: &mut JNIEnv<'local>,
    arrow_schema_addr: jlong,
    path: JString,
    max_rows_per_file: JObject,  // Optional<Integer>
    max_rows_per_group: JObject, // Optional<Integer>
    max_bytes_per_file: JObject, // Optional<Long>
    mode: JObject,               // Optional<String>
) -> JavaResult<JObject<'local>> {
    let c_schema_ptr = arrow_schema_addr as *mut FFI_ArrowSchema;
    let c_schema = unsafe { FFI_ArrowSchema::from_raw(c_schema_ptr) };
    let schema = Schema::try_from(&c_schema).infer_error()?;

    let reader = RecordBatchIterator::new(empty(), Arc::new(schema));
    create_dataset(
            env,
            path,
            max_rows_per_file,
            max_rows_per_group,
            max_bytes_per_file,
            mode,
            reader
        )
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_createWithFfiStream<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    arrow_array_stream_addr: jlong,
    path: JString,
    max_rows_per_file: JObject,  // Optional<Integer>
    max_rows_per_group: JObject, // Optional<Integer>
    max_bytes_per_file: JObject, // Optional<Long>
    mode: JObject,               // Optional<String>
) -> JObject<'local> {
    ok_or_throw!(env, inner_create_with_ffi_stream(&mut env, arrow_array_stream_addr, path, max_rows_per_file, max_rows_per_group, max_bytes_per_file, mode))
}

fn inner_create_with_ffi_stream<'local>(
    env: &mut JNIEnv<'local>,
    arrow_array_stream_addr: jlong,
    path: JString,
    max_rows_per_file: JObject,  // Optional<Integer>
    max_rows_per_group: JObject, // Optional<Integer>
    max_bytes_per_file: JObject, // Optional<Long>
    mode: JObject,               // Optional<String>
) -> JavaResult<JObject<'local>> {
    let stream_ptr = arrow_array_stream_addr as *mut FFI_ArrowArrayStream;
    let reader = unsafe { ArrowArrayStreamReader::from_raw(stream_ptr) }.infer_error()?;
    create_dataset(
        env,
        path,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        mode,
        reader
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
) -> JavaResult<JObject<'local>> {
    let path_str = path.extract(env)?;

    let write_params = extract_write_params(
        env,
        &max_rows_per_file,
        &max_rows_per_group,
        &max_bytes_per_file,
        &mode,
    )?;

    let dataset = BlockingDataset::write(reader, &path_str, Some(write_params))?;
    dataset.into_java(env)
}

impl IntoJava for BlockingDataset {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> JavaResult<JObject<'a>> {
        attach_native_dataset(env, self)
    }
}

fn attach_native_dataset<'local>(
    env: &mut JNIEnv<'local>,
    dataset: BlockingDataset,
) -> JavaResult<JObject<'local>> {
    let j_dataset = create_java_dataset_object(env)?;
    // This block sets a native Rust object (dataset) as a field in the Java object (j_dataset).
    // Caution: This creates a potential for memory leaks. The Rust object (dataset) is not
    // automatically garbage-collected by Java, and its memory will not be freed unless
    // explicitly handled.
    //
    // To prevent memory leaks, ensure the following:
    // 1. The Java object (`j_dataset`) should implement the `java.io.Closeable` interface.
    // 2. Users of this Java object should be instructed to always use it within a try-with-resources
    //    statement (or manually call the `close()` method) to ensure that `self.close()` is invoked.
   unsafe { env.set_rust_field(&j_dataset, NATIVE_DATASET, dataset) }.infer_error()?;
   Ok(j_dataset)
}

fn create_java_dataset_object<'a>(env: &mut JNIEnv<'a>) -> JavaResult<JObject<'a>> {
    Ok(env.new_object("com/lancedb/lance/Dataset", "()V", &[]).infer_error()?)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_commitAppend<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    path: JString,
    read_version_obj: JObject, // Optional<Long>
    fragments_obj: JObject,    // List<String>, String is json serialized Fragment
) -> JObject<'local> {
    ok_or_throw!(env, inner_commit_append(&mut env, path, read_version_obj, fragments_obj))
}

pub fn inner_commit_append<'local>(
    env: &mut JNIEnv<'local>,
    path: JString,
    read_version_obj: JObject, // Optional<Long>
    fragments_obj: JObject,    // List<String>, String is json serialized Fragment)
) -> JavaResult<JObject<'local>> {
    let json_fragments = env.get_strings(&fragments_obj)?;
    let mut fragments: Vec<Fragment> = Vec::new();
    for json_fragment in json_fragments {
        let fragment = Fragment::from_json(&json_fragment).infer_error()?;
        fragments.push(fragment);
    }
    let op = Operation::Append { fragments };
    let path_str = path.extract(env)?;
    let read_version = env.get_u64_opt(&read_version_obj)?;
    let dataset = BlockingDataset::commit(&path_str, op, read_version)?;
    dataset.into_java(env)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_releaseNativeDataset(
    mut env: JNIEnv,
    obj: JObject,
) {
    ok_or_throw_without_return!(env, inner_release_native_dataset(&mut env, obj))
}

fn inner_release_native_dataset(
    env: &mut JNIEnv,
    obj: JObject,
) -> JavaResult<()> {
    let dataset: BlockingDataset = unsafe {
        env.take_rust_field(obj, NATIVE_DATASET).infer_error()?
    };
    dataset.close();
    Ok(())
}

//////////////////
// Read Methods //
//////////////////
#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_openNative<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    path: JString,
) -> JObject<'local> {
    ok_or_throw!(env, inner_open_native(&mut env, path))
}

fn inner_open_native<'local>(
    env: &mut JNIEnv<'local>,
    path: JString,
) -> JavaResult<JObject<'local>> {
    let path_str: String = path.extract(env)?;
    let dataset = BlockingDataset::open(&path_str)?;
    dataset.into_java(env)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_getJsonFragments<'a>(
    mut env: JNIEnv<'a>,
    jdataset: JObject,
) -> JObject<'a> {
    ok_or_throw!(env, inner_get_json_fragments(&mut env, jdataset))
}

fn inner_get_json_fragments<'local>(
    env: &mut JNIEnv<'local>,
    jdataset: JObject,
) -> JavaResult<JObject<'local>> {
    let fragments = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }.infer_error()?;
        dataset.inner.get_fragments()
    };

    let array_list_class = env.find_class("java/util/ArrayList").infer_error()?;

    let array_list = env.new_object(array_list_class, "()V", &[]).infer_error()?;

    for fragment in fragments {
        let json_string = serde_json::to_string(fragment.metadata()).infer_error()?;
        let jstring = env.new_string(json_string).infer_error()?;
        env.call_method(
            &array_list,
            "add",
            "(Ljava/lang/Object;)Z",
            &[(&jstring).into()],
        ).infer_error()?;
    }
    Ok(array_list)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_importFfiSchema(
    mut env: JNIEnv,
    jdataset: JObject,
    arrow_schema_addr: jlong,
) {
    ok_or_throw_without_return!(env, inner_import_ffi_schema(&mut env, jdataset, arrow_schema_addr))
}

fn inner_import_ffi_schema(
    env: &mut JNIEnv,
    jdataset: JObject,
    arrow_schema_addr: jlong,
) -> JavaResult<()>{
    let dataset = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }.infer_error()?;
        dataset.clone()
    };
    let schema = Schema::from(dataset.inner.schema());

    let c_schema = FFI_ArrowSchema::try_from(&schema).infer_error()?;
    let out_c_schema = unsafe { &mut *(arrow_schema_addr as *mut FFI_ArrowSchema) };
    let _old = std::mem::replace(out_c_schema, c_schema);
    Ok(())
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_version(
    mut env: JNIEnv,
    java_dataset: JObject,
) -> jlong {
    ok_or_throw_with_return!(env, inner_version(&mut env, java_dataset), -1) as jlong
}

fn inner_version(env: &mut JNIEnv, java_dataset: JObject) -> JavaResult<u64> {
    let dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }.infer_error()?;
    Ok(dataset_guard.inner.version().version)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_latestVersion(
    mut env: JNIEnv,
    java_dataset: JObject,
) -> jlong {
    ok_or_throw_with_return!(env, inner_latest_version(&mut env, java_dataset), -1) as jlong
}

fn inner_latest_version(env: &mut JNIEnv, java_dataset: JObject) -> JavaResult<u64> {
    let dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }.infer_error()?;
    dataset_guard.latest_version()
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_countRows(
    mut env: JNIEnv,
    java_dataset: JObject,
) -> jint {
    ok_or_throw_with_return!(env, inner_count_rows(&mut env, java_dataset), -1) as jint
}

fn inner_count_rows(env: &mut JNIEnv, java_dataset: JObject) -> JavaResult<usize> {
    let dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }.infer_error()?;
    dataset_guard.count_rows(None)
}