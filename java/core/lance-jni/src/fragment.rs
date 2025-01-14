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
use jni::objects::{JIntArray, JValueGen};
use jni::{
    objects::{JObject, JString},
    sys::{jint, jlong},
    JNIEnv,
};
use lance::table::format::{DataFile, DeletionFile, DeletionFileType, Fragment, RowIdMeta};
use std::iter::once;

use lance::dataset::fragment::FileFragment;
use lance_datafusion::utils::StreamingWriteSource;

use crate::error::{Error, Result};
use crate::traits::{export_vec, import_vec, FromJObjectWithEnv, IntoJava, JLance};
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
pub extern "system" fn Java_com_lancedb_lance_Fragment_countRowsNative(
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
    let res = RT.block_on(fragment.count_rows(None))?;
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
) -> JObject<'local> {
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
        JObject::default()
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
) -> Result<JObject<'local>> {
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
) -> JObject<'a> {
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
        JObject::null()
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
) -> Result<JObject<'local>> {
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
) -> Result<JObject<'a>> {
    let path_str = dataset_uri.extract(env)?;

    let write_params = extract_write_params(
        env,
        &max_rows_per_file,
        &max_rows_per_group,
        &max_bytes_per_file,
        &mode,
        &storage_options_obj,
    )?;
    let fragments = RT.block_on(FileFragment::create_fragments(
        &path_str,
        source,
        Some(write_params),
    ))?;
    export_vec(env, &fragments)
}

const DATA_FILE_CLASS: &str = "com/lancedb/lance/fragment/DataFile";
const DATA_FILE_CONSTRUCTOR_SIG: &str = "(Ljava/lang/String;[I[III)V";
const DELETE_FILE_CLASS: &str = "com/lancedb/lance/fragment/DeletionFile";
const DELETE_FILE_CONSTRUCTOR_SIG: &str =
    "(JJLjava/lang/Long;Lcom/lancedb/lance/fragment/DeletionFileType;)V";
const DELETE_FILE_TYPE_CLASS: &str = "com/lancedb/lance/fragment/DeletionFileType";
const FRAGMENT_METADATA_CLASS: &str = "com/lancedb/lance/FragmentMetadata";
const FRAGMENT_METADATA_CONSTRUCTOR_SIG: &str ="(ILjava/util/List;Ljava/lang/Long;Lcom/lancedb/lance/fragment/DeletionFile;Lcom/lancedb/lance/fragment/RowIdMeta;)V";
const ROW_ID_META_CLASS: &str = "com/lancedb/lance/fragment/RowIdMeta";
const ROW_ID_META_CONSTRUCTOR_SIG: &str = "(Ljava/lang/String;)V";

impl IntoJava for &DataFile {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let path = env.new_string(self.path.clone())?.into();
        let fields = JLance(self.fields.clone()).into_java(env)?;
        let column_indices = JLance(self.column_indices.clone()).into_java(env)?;
        Ok(env.new_object(
            DATA_FILE_CLASS,
            DATA_FILE_CONSTRUCTOR_SIG,
            &[
                JValueGen::Object(&path),
                JValueGen::Object(&fields),
                JValueGen::Object(&column_indices),
                JValueGen::Int(self.file_major_version as i32),
                JValueGen::Int(self.file_minor_version as i32),
            ],
        )?)
    }
}

impl IntoJava for &DeletionFileType {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let name = match self {
            lance::table::format::DeletionFileType::Array => "ARRAY",
            lance::table::format::DeletionFileType::Bitmap => "BITMAP",
        };
        env.get_static_field(
            DELETE_FILE_TYPE_CLASS,
            name,
            format!("L{};", DELETE_FILE_TYPE_CLASS),
        )?
        .l()
        .map_err(|e| {
            Error::runtime_error(format!("failed to get {}: {}", DELETE_FILE_TYPE_CLASS, e))
        })
    }
}

impl IntoJava for &DeletionFile {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let num_deleted_rows = match self.num_deleted_rows {
            Some(f) => JLance(f).into_java(env)?,
            None => JObject::null(),
        };
        let file_type = self.file_type.into_java(env)?;
        Ok(env.new_object(
            DELETE_FILE_CLASS,
            DELETE_FILE_CONSTRUCTOR_SIG,
            &[
                JValueGen::Long(self.id as i64),
                JValueGen::Long(self.read_version as i64),
                JValueGen::Object(&num_deleted_rows),
                JValueGen::Object(&file_type),
            ],
        )?)
    }
}

impl IntoJava for &RowIdMeta {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let json_str = serde_json::to_string(self)?;
        let json = env.new_string(json_str)?.into();
        Ok(env.new_object(
            ROW_ID_META_CLASS,
            ROW_ID_META_CONSTRUCTOR_SIG,
            &[JValueGen::Object(&json)],
        )?)
    }
}

impl IntoJava for &Fragment {
    fn into_java<'local>(self, env: &mut JNIEnv<'local>) -> Result<JObject<'local>> {
        let files = self.files.clone();
        let files = export_vec::<DataFile>(env, &files)?;
        let deletion_file = match &self.deletion_file {
            Some(f) => f.into_java(env)?,
            None => JObject::null(),
        };
        let physical_rows = &JLance(self.physical_rows).into_java(env)?;
        let row_id_meta = match &self.row_id_meta {
            Some(m) => m.into_java(env)?,
            None => JObject::null(),
        };

        env.new_object(
            FRAGMENT_METADATA_CLASS,
            FRAGMENT_METADATA_CONSTRUCTOR_SIG,
            &[
                JValueGen::Int(self.id as i32),
                JValueGen::Object(&files),
                JValueGen::Object(physical_rows),
                JValueGen::Object(&deletion_file),
                JValueGen::Object(&row_id_meta),
            ],
        )
        .map_err(|e| {
            Error::runtime_error(format!("failed to get {}: {}", FRAGMENT_METADATA_CLASS, e))
        })
    }
}

impl FromJObjectWithEnv<RowIdMeta> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<RowIdMeta> {
        let metadata = env
            .call_method(self, "getMetadata", "()Ljava/lang/String;", &[])?
            .l()?;
        let s: String = env.get_string(&JString::from(metadata))?.into();
        let meta: RowIdMeta = serde_json::from_str(&s)?;
        Ok(meta)
    }
}

impl FromJObjectWithEnv<Fragment> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<Fragment> {
        let id = env.call_method(self, "getId", "()I", &[])?.i()? as u64;
        let file_objs = env
            .call_method(self, "getFiles", "()Ljava/util/List;", &[])?
            .l()?;
        let physical_rows = env.call_method(self, "getPhysicalRows", "()J", &[])?.j()? as usize;
        let file_objs = import_vec(env, &file_objs)?;
        let mut files = Vec::with_capacity(file_objs.len());
        for f in file_objs {
            files.push(f.extract_object(env)?);
        }
        let deletion_file = env
            .call_method(
                self,
                "getDeletionFile",
                format!("()L{};", DELETE_FILE_CLASS),
                &[],
            )?
            .l()?;
        let deletion_file = if deletion_file.is_null() {
            None
        } else {
            Some(deletion_file.extract_object(env)?)
        };

        let row_id_meta = env
            .call_method(
                self,
                "getRowIdMeta",
                format!("()L{};", ROW_ID_META_CLASS),
                &[],
            )?
            .l()?;
        let row_id_meta = if row_id_meta.is_null() {
            None
        } else {
            Some(row_id_meta.extract_object(env)?)
        };
        Ok(Fragment {
            id,
            files,
            deletion_file,
            physical_rows: Some(physical_rows),
            row_id_meta,
        })
    }
}

impl FromJObjectWithEnv<DeletionFile> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<DeletionFile> {
        let id = env.call_method(self, "getId", "()J", &[])?.j()? as u64;
        let read_version = env.call_method(self, "getReadVersion", "()J", &[])?.j()? as u64;
        let num_deleted_rows: Option<i64> = env
            .call_method(self, "getNumDeletedRows", "()Ljava/lang/Long;", &[])?
            .l()?
            .extract_object(env)?;
        let num_deleted_rows = num_deleted_rows.map(|r| r as usize);
        let file_type: DeletionFileType = env
            .call_method(
                self,
                "getFileType",
                format!("()L{};", DELETE_FILE_TYPE_CLASS),
                &[],
            )?
            .l()?
            .extract_object(env)?;
        Ok(DeletionFile {
            read_version,
            id,
            num_deleted_rows,
            file_type,
        })
    }
}

impl FromJObjectWithEnv<DeletionFileType> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<DeletionFileType> {
        let s = env
            .call_method(self, "toString", "()Ljava.lang.String;", &[])?
            .l()?;
        let s: String = env.get_string(&JString::from(s))?.into();
        let t = if s == "ARRAY" {
            DeletionFileType::Array
        } else {
            DeletionFileType::Bitmap
        };
        Ok(t)
    }
}

impl FromJObjectWithEnv<DataFile> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<DataFile> {
        let path = env
            .call_method(self, "getPath", "()Ljava/lang/String;", &[])?
            .l()?;
        let path: String = env.get_string(&JString::from(path))?.into();
        let fields = env.call_method(self, "getFields", "()[I", &[])?.l()?;
        let fields = JIntArray::from(fields).extract_object(env)?;
        let column_indices = env
            .call_method(self, "getColumnIndices", "()[I", &[])?
            .l()?;
        let column_indices = JIntArray::from(column_indices).extract_object(env)?;
        let file_major_version = env
            .call_method(self, "getFileMajorVersion", "()I", &[])?
            .i()? as u32;
        let file_minor_version = env
            .call_method(self, "getFileMinorVersion", "()I", &[])?
            .i()? as u32;
        Ok(DataFile {
            path,
            fields,
            column_indices,
            file_major_version,
            file_minor_version,
        })
    }
}
