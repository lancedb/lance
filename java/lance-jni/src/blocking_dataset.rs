// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::error::{Error, Result};
use crate::ffi::JNIEnvExt;
use crate::storage_options::JavaStorageOptionsProvider;
use crate::traits::{export_vec, import_vec, FromJObjectWithEnv, FromJString};
use crate::utils::{
    build_compaction_options, extract_storage_options, extract_write_params,
    get_scalar_index_params, get_vector_index_params, to_rust_map,
};
use crate::{traits::IntoJava, RT};
use arrow::array::RecordBatchReader;
use arrow::datatypes::Schema;
use arrow::ffi::FFI_ArrowSchema;
use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::ffi_stream::FFI_ArrowArrayStream;
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatchIterator;
use arrow_schema::DataType;
use arrow_schema::Schema as ArrowSchema;
use chrono::{DateTime, Utc};
use jni::objects::{JMap, JString, JValue};
use jni::sys::{jboolean, jint};
use jni::sys::{jbyteArray, jlong};
use jni::{objects::JObject, JNIEnv};
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::cleanup::{CleanupPolicy, RemovalStats};
use lance::dataset::index::LanceIndexStoreExt;
use lance::dataset::optimize::{compact_files, CompactionOptions as RustCompactionOptions};
use lance::dataset::refs::{Ref, TagContents};
use lance::dataset::statistics::{DataStatistics, DatasetStatisticsExt};
use lance::dataset::transaction::{Operation, Transaction};
use lance::dataset::{
    ColumnAlteration, CommitBuilder, Dataset, NewColumnTransform, ProjectionRequest, ReadParams,
    Version, WriteParams,
};
use lance::io::{ObjectStore, ObjectStoreParams};
use lance::table::format::Fragment;
use lance::table::format::IndexMetadata;
use lance_core::datatypes::Schema as LanceSchema;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::DatasetIndexExt;
use lance_index::{IndexParams, IndexType};
use lance_io::object_store::ObjectStoreRegistry;
use lance_io::object_store::StorageOptionsProvider;
use std::collections::HashMap;
use std::future::IntoFuture;
use std::iter::empty;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, UNIX_EPOCH};

pub const NATIVE_DATASET: &str = "nativeDatasetHandle";

#[derive(Clone)]
pub struct BlockingDataset {
    pub(crate) inner: Dataset,
}

impl BlockingDataset {
    pub fn drop(uri: &str, storage_options: HashMap<String, String>) -> Result<()> {
        RT.block_on(async move {
            let registry = Arc::new(ObjectStoreRegistry::default());
            let object_store_params = ObjectStoreParams {
                storage_options: Some(storage_options.clone()),
                ..Default::default()
            };
            let (object_store, path) =
                ObjectStore::from_uri_and_params(registry, uri, &object_store_params)
                    .await
                    .map_err(|e| Error::io_error(e.to_string()))?;
            object_store
                .remove_dir_all(path)
                .await
                .map_err(|e| Error::io_error(e.to_string()))
        })
    }
    pub fn write(
        reader: impl RecordBatchReader + Send + 'static,
        uri: &str,
        params: Option<WriteParams>,
    ) -> Result<Self> {
        let inner = RT.block_on(Dataset::write(reader, uri, params))?;
        Ok(Self { inner })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn open(
        uri: &str,
        version: Option<i32>,
        block_size: Option<i32>,
        index_cache_size_bytes: i64,
        metadata_cache_size_bytes: i64,
        storage_options: HashMap<String, String>,
        serialized_manifest: Option<&[u8]>,
        storage_options_provider: Option<Arc<dyn StorageOptionsProvider>>,
        s3_credentials_refresh_offset_seconds: Option<u64>,
    ) -> Result<Self> {
        let mut store_params = ObjectStoreParams {
            block_size: block_size.map(|size| size as usize),
            storage_options: Some(storage_options.clone()),
            ..Default::default()
        };
        if let Some(offset_seconds) = s3_credentials_refresh_offset_seconds {
            store_params.s3_credentials_refresh_offset =
                std::time::Duration::from_secs(offset_seconds);
        }
        if let Some(provider) = storage_options_provider.clone() {
            store_params.storage_options_provider = Some(provider);
        }
        let params = ReadParams {
            index_cache_size_bytes: index_cache_size_bytes as usize,
            metadata_cache_size_bytes: metadata_cache_size_bytes as usize,
            store_options: Some(store_params),
            ..Default::default()
        };

        let mut builder = DatasetBuilder::from_uri(uri).with_read_params(params);

        if let Some(ver) = version {
            builder = builder.with_version(ver as u64);
        }
        builder = builder.with_storage_options(storage_options);
        if let Some(provider) = storage_options_provider {
            builder = builder.with_storage_options_provider(provider)
        }
        if let Some(offset_seconds) = s3_credentials_refresh_offset_seconds {
            builder = builder
                .with_s3_credentials_refresh_offset(std::time::Duration::from_secs(offset_seconds));
        }

        if let Some(serialized_manifest) = serialized_manifest {
            builder = builder.with_serialized_manifest(serialized_manifest)?;
        }

        let inner = RT.block_on(builder.load())?;
        Ok(Self { inner })
    }

    pub fn commit(
        uri: &str,
        operation: Operation,
        read_version: Option<u64>,
        storage_options: HashMap<String, String>,
    ) -> Result<Self> {
        let inner = RT.block_on(Dataset::commit(
            uri,
            operation,
            read_version,
            Some(ObjectStoreParams {
                storage_options: Some(storage_options),
                ..Default::default()
            }),
            None,
            Default::default(),
            false, // TODO: support enable_v2_manifest_paths
        ))?;
        Ok(Self { inner })
    }

    pub fn latest_version(&self) -> Result<u64> {
        let version = RT.block_on(self.inner.latest_version_id())?;
        Ok(version)
    }

    pub fn list_versions(&self) -> Result<Vec<Version>> {
        let versions = RT.block_on(self.inner.versions())?;
        Ok(versions)
    }

    pub fn version(&self) -> Result<Version> {
        Ok(self.inner.version())
    }

    pub fn checkout_version(&mut self, version: u64) -> Result<Self> {
        let inner = RT.block_on(self.inner.checkout_version(version))?;
        Ok(Self { inner })
    }

    pub fn checkout_tag(&mut self, tag: &str) -> Result<Self> {
        let inner = RT.block_on(self.inner.checkout_version(tag))?;
        Ok(Self { inner })
    }

    pub fn checkout_latest(&mut self) -> Result<()> {
        RT.block_on(self.inner.checkout_latest())?;
        Ok(())
    }

    pub fn restore(&mut self) -> Result<()> {
        RT.block_on(self.inner.restore())?;
        Ok(())
    }

    pub fn list_tags(&self) -> Result<HashMap<String, TagContents>> {
        let tags = RT.block_on(self.inner.tags().list())?;
        Ok(tags)
    }

    pub fn list_branches(&self) -> Result<HashMap<String, lance::dataset::refs::BranchContents>> {
        let branches = RT.block_on(self.inner.list_branches())?;
        Ok(branches)
    }

    pub fn create_branch(
        &mut self,
        branch: &str,
        version: u64,
        source_branch: Option<&str>,
    ) -> Result<Self> {
        let reference = match source_branch {
            Some(b) => Ref::from((b, version)),
            None => Ref::from(version),
        };
        let inner = RT.block_on(self.inner.create_branch(branch, reference, None))?;
        Ok(Self { inner })
    }

    pub fn delete_branch(&mut self, branch: &str) -> Result<()> {
        RT.block_on(self.inner.delete_branch(branch))?;
        Ok(())
    }

    pub fn checkout_reference(
        &mut self,
        branch: Option<String>,
        version: Option<u64>,
        tag: Option<String>,
    ) -> Result<Self> {
        let reference = if let Some(tag_name) = tag {
            Ref::from(tag_name.as_str())
        } else {
            Ref::Version(branch, version)
        };
        let inner = RT.block_on(self.inner.checkout_version(reference))?;
        Ok(Self { inner })
    }

    pub fn create_tag(
        &mut self,
        tag: &str,
        version_number: u64,
        branch: Option<&str>,
    ) -> Result<()> {
        RT.block_on(
            self.inner
                .tags()
                .create_on_branch(tag, version_number, branch),
        )?;
        Ok(())
    }

    pub fn delete_tag(&mut self, tag: &str) -> Result<()> {
        RT.block_on(self.inner.tags().delete(tag))?;
        Ok(())
    }

    pub fn update_tag(&mut self, tag: &str, version: u64, branch: Option<&str>) -> Result<()> {
        RT.block_on(self.inner.tags().update_on_branch(tag, version, branch))?;
        Ok(())
    }

    pub fn get_version(&self, tag: &str) -> Result<u64> {
        let version = RT.block_on(self.inner.tags().get_version(tag))?;
        Ok(version)
    }

    pub fn count_rows(&self, filter: Option<String>) -> Result<usize> {
        let rows = RT.block_on(self.inner.count_rows(filter))?;
        Ok(rows)
    }

    pub fn calculate_data_stats(&self) -> Result<DataStatistics> {
        let stats = RT.block_on(Arc::new(self.clone().inner).calculate_data_stats())?;
        Ok(stats)
    }

    pub fn list_indexes(&self) -> Result<Arc<Vec<IndexMetadata>>> {
        let indexes = RT.block_on(self.inner.load_indices())?;
        Ok(indexes)
    }

    pub fn commit_transaction(
        &mut self,
        transaction: Transaction,
        write_params: HashMap<String, String>,
    ) -> Result<Self> {
        let new_dataset = RT.block_on(
            CommitBuilder::new(Arc::new(self.clone().inner))
                .with_store_params(ObjectStoreParams {
                    storage_options: Some(write_params),
                    ..Default::default()
                })
                .execute(transaction),
        )?;
        Ok(BlockingDataset { inner: new_dataset })
    }

    pub fn read_transaction(&self) -> Result<Option<Transaction>> {
        let transaction = RT.block_on(self.inner.read_transaction())?;
        Ok(transaction)
    }

    pub fn get_table_metadata(&self) -> Result<HashMap<String, String>> {
        Ok(self.inner.metadata().clone())
    }

    pub fn compact(&mut self, options: RustCompactionOptions) -> Result<()> {
        RT.block_on(compact_files(&mut self.inner, options, None))?;
        Ok(())
    }

    pub fn cleanup_with_policy(&mut self, policy: CleanupPolicy) -> Result<RemovalStats> {
        Ok(RT.block_on(self.inner.cleanup_with_policy(policy))?)
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
    max_rows_per_file: JObject,     // Optional<Integer>
    max_rows_per_group: JObject,    // Optional<Integer>
    max_bytes_per_file: JObject,    // Optional<Long>
    mode: JObject,                  // Optional<String>
    enable_stable_row_ids: JObject, // Optional<Boolean>
    data_storage_version: JObject,  // Optional<String>
    storage_options_obj: JObject,   // Map<String, String>
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_create_with_ffi_schema(
            &mut env,
            arrow_schema_addr,
            path,
            max_rows_per_file,
            max_rows_per_group,
            max_bytes_per_file,
            mode,
            enable_stable_row_ids,
            data_storage_version,
            storage_options_obj
        )
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_create_with_ffi_schema<'local>(
    env: &mut JNIEnv<'local>,
    arrow_schema_addr: jlong,
    path: JString,
    max_rows_per_file: JObject,     // Optional<Integer>
    max_rows_per_group: JObject,    // Optional<Integer>
    max_bytes_per_file: JObject,    // Optional<Long>
    mode: JObject,                  // Optional<String>
    enable_stable_row_ids: JObject, // Optional<Boolean>
    data_storage_version: JObject,  // Optional<String>
    storage_options_obj: JObject,   // Map<String, String>
) -> Result<JObject<'local>> {
    let c_schema_ptr = arrow_schema_addr as *mut FFI_ArrowSchema;
    let c_schema = unsafe { FFI_ArrowSchema::from_raw(c_schema_ptr) };
    let schema = Schema::try_from(&c_schema)?;

    let reader = RecordBatchIterator::new(empty(), Arc::new(schema));
    create_dataset(
        env,
        path,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        mode,
        enable_stable_row_ids,
        data_storage_version,
        storage_options_obj,
        reader,
    )
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_drop<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    path: JString<'local>,
    storage_options_obj: JObject<'local>,
) -> JObject<'local> {
    let path_str = ok_or_throw!(env, path.extract(&mut env));
    let storage_options =
        ok_or_throw!(env, extract_storage_options(&mut env, &storage_options_obj));
    ok_or_throw!(env, BlockingDataset::drop(&path_str, storage_options));
    JObject::null()
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_createWithFfiStream<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    arrow_array_stream_addr: jlong,
    path: JString,
    max_rows_per_file: JObject,     // Optional<Integer>
    max_rows_per_group: JObject,    // Optional<Integer>
    max_bytes_per_file: JObject,    // Optional<Long>
    mode: JObject,                  // Optional<String>
    enable_stable_row_ids: JObject, // Optional<Boolean>
    data_storage_version: JObject,  // Optional<String>
    storage_options_obj: JObject,   // Map<String, String>
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_create_with_ffi_stream(
            &mut env,
            arrow_array_stream_addr,
            path,
            max_rows_per_file,
            max_rows_per_group,
            max_bytes_per_file,
            mode,
            enable_stable_row_ids,
            data_storage_version,
            storage_options_obj
        )
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_create_with_ffi_stream<'local>(
    env: &mut JNIEnv<'local>,
    arrow_array_stream_addr: jlong,
    path: JString,
    max_rows_per_file: JObject,     // Optional<Integer>
    max_rows_per_group: JObject,    // Optional<Integer>
    max_bytes_per_file: JObject,    // Optional<Long>
    mode: JObject,                  // Optional<String>
    enable_stable_row_ids: JObject, // Optional<Boolean>
    data_storage_version: JObject,  // Optional<String>
    storage_options_obj: JObject,   // Map<String, String>
) -> Result<JObject<'local>> {
    let stream_ptr = arrow_array_stream_addr as *mut FFI_ArrowArrayStream;
    let reader = unsafe { ArrowArrayStreamReader::from_raw(stream_ptr) }?;
    create_dataset(
        env,
        path,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        mode,
        enable_stable_row_ids,
        data_storage_version,
        storage_options_obj,
        reader,
    )
}

#[allow(clippy::too_many_arguments)]
fn create_dataset<'local>(
    env: &mut JNIEnv<'local>,
    path: JString,
    max_rows_per_file: JObject,
    max_rows_per_group: JObject,
    max_bytes_per_file: JObject,
    mode: JObject,
    enable_stable_row_ids: JObject,
    data_storage_version: JObject,
    storage_options_obj: JObject,
    reader: impl RecordBatchReader + Send + 'static,
) -> Result<JObject<'local>> {
    let path_str = path.extract(env)?;

    let write_params = extract_write_params(
        env,
        &max_rows_per_file,
        &max_rows_per_group,
        &max_bytes_per_file,
        &mode,
        &enable_stable_row_ids,
        &data_storage_version,
        &storage_options_obj,
    )?;

    let dataset = BlockingDataset::write(reader, &path_str, Some(write_params))?;
    dataset.into_java(env)
}

impl IntoJava for BlockingDataset {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        attach_native_dataset(env, self)
    }
}

impl IntoJava for Version {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let timestamp_str = self.timestamp.to_rfc3339();
        let jtimestamp = env.new_string(timestamp_str)?;
        let zdt = env
            .call_static_method(
                "java/time/ZonedDateTime",
                "parse",
                "(Ljava/lang/CharSequence;)Ljava/time/ZonedDateTime;",
                &[JValue::Object(&jtimestamp)],
            )?
            .l()?;

        let jmap = env.new_object("java/util/TreeMap", "()V", &[])?;
        let map = JMap::from_env(env, &jmap)?;

        for (k, v) in self.metadata {
            let jkey = env.new_string(k)?;
            let jval = env.new_string(v)?;
            map.put(env, &jkey, &jval).expect("ERROR: calling jmap.put");
        }

        let java_version = env.new_object(
            "com/lancedb/lance/Version",
            "(JLjava/time/ZonedDateTime;Ljava/util/TreeMap;)V",
            &[
                JValue::Long(self.version as i64),
                JValue::Object(&zdt),
                JValue::Object(&jmap),
            ],
        )?;
        Ok(java_version)
    }
}

fn attach_native_dataset<'local>(
    env: &mut JNIEnv<'local>,
    dataset: BlockingDataset,
) -> Result<JObject<'local>> {
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
    unsafe { env.set_rust_field(&j_dataset, NATIVE_DATASET, dataset) }?;
    Ok(j_dataset)
}

fn create_java_dataset_object<'a>(env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
    let object = env.new_object("com/lancedb/lance/Dataset", "()V", &[])?;
    Ok(object)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_commitAppend<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    path: JString,
    read_version_obj: JObject,    // Optional<Long>
    fragments_obj: JObject,       // List<FragmentMetadata>
    storage_options_obj: JObject, // Map<String, String>
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_commit_append(
            &mut env,
            path,
            read_version_obj,
            fragments_obj,
            storage_options_obj
        )
    )
}

pub fn inner_commit_append<'local>(
    env: &mut JNIEnv<'local>,
    path: JString,
    read_version_obj: JObject,    // Optional<Long>
    fragment_objs: JObject,       // List<FragmentMetadata>
    storage_options_obj: JObject, // Map<String, String>
) -> Result<JObject<'local>> {
    let fragment_objs = import_vec(env, &fragment_objs)?;
    let mut fragments = Vec::with_capacity(fragment_objs.len());
    for f in fragment_objs {
        fragments.push(f.extract_object(env)?);
    }
    let op = Operation::Append { fragments };
    let path_str = path.extract(env)?;
    let read_version = env.get_u64_opt(&read_version_obj)?;
    let storage_options = extract_storage_options(env, &storage_options_obj)?;
    let dataset = BlockingDataset::commit(&path_str, op, read_version, storage_options)?;
    dataset.into_java(env)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_commitOverwrite<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    path: JString,
    arrow_schema_addr: jlong,
    read_version_obj: JObject,    // Optional<Long>
    fragments_obj: JObject,       // List<FragmentMetadata>
    storage_options_obj: JObject, // Map<String, String>
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_commit_overwrite(
            &mut env,
            path,
            arrow_schema_addr,
            read_version_obj,
            fragments_obj,
            storage_options_obj
        )
    )
}

pub fn inner_commit_overwrite<'local>(
    env: &mut JNIEnv<'local>,
    path: JString,
    arrow_schema_addr: jlong,
    read_version_obj: JObject,    // Optional<Long>
    fragments_obj: JObject,       // List<FragmentMetadata>
    storage_options_obj: JObject, // Map<String, String>
) -> Result<JObject<'local>> {
    let fragment_objs = import_vec(env, &fragments_obj)?;
    let mut fragments = Vec::with_capacity(fragment_objs.len());
    for f in fragment_objs {
        fragments.push(f.extract_object(env)?);
    }
    let c_schema_ptr = arrow_schema_addr as *mut FFI_ArrowSchema;
    let c_schema = unsafe { FFI_ArrowSchema::from_raw(c_schema_ptr) };
    let arrow_schema = Schema::try_from(&c_schema)?;
    let schema = LanceSchema::try_from(&arrow_schema)?;

    let op = Operation::Overwrite {
        fragments,
        schema,
        config_upsert_values: None,
        initial_bases: None,
    };
    let path_str = path.extract(env)?;
    let read_version = env.get_u64_opt(&read_version_obj)?;
    let jmap = JMap::from_env(env, &storage_options_obj)?;
    let storage_options = to_rust_map(env, &jmap)?;
    let dataset = BlockingDataset::commit(&path_str, op, read_version, storage_options)?;
    dataset.into_java(env)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_releaseNativeDataset(
    mut env: JNIEnv,
    obj: JObject,
) {
    ok_or_throw_without_return!(env, inner_release_native_dataset(&mut env, obj))
}

fn inner_release_native_dataset(env: &mut JNIEnv, obj: JObject) -> Result<()> {
    let dataset: BlockingDataset = unsafe { env.take_rust_field(obj, NATIVE_DATASET)? };
    dataset.close();
    Ok(())
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeCreateIndex(
    mut env: JNIEnv,
    java_dataset: JObject,
    columns_jobj: JObject, // List<String>
    index_type_code_jobj: jint,
    name_jobj: JObject,       // Optional<String>
    params_jobj: JObject,     // IndexParams
    replace_jobj: jboolean,   // replace
    train_jobj: jboolean,     // train
    fragments_jobj: JObject,  // List<Integer>
    index_uuid_jobj: JObject, // String
) {
    ok_or_throw_without_return!(
        env,
        inner_create_index(
            &mut env,
            java_dataset,
            columns_jobj,
            index_type_code_jobj,
            name_jobj,
            params_jobj,
            replace_jobj,
            train_jobj,
            fragments_jobj,
            index_uuid_jobj
        )
    );
}

#[allow(clippy::too_many_arguments)]
fn inner_create_index(
    env: &mut JNIEnv,
    java_dataset: JObject,
    columns_jobj: JObject, // List<String>
    index_type_code_jobj: jint,
    name_jobj: JObject,       // Optional<String>
    params_jobj: JObject,     // IndexParams
    replace_jobj: jboolean,   // replace
    train_jobj: jboolean,     // train
    fragments_jobj: JObject,  // Optional<List<String>>
    index_uuid_jobj: JObject, // Optional<String>
) -> Result<()> {
    let columns = env.get_strings(&columns_jobj)?;
    let index_type = IndexType::try_from(index_type_code_jobj)?;
    let name = env.get_string_opt(&name_jobj)?;
    let columns_slice: Vec<&str> = columns.iter().map(AsRef::as_ref).collect();
    let replace = replace_jobj != 0;
    let train = train_jobj != 0;
    let fragment_ids = env
        .get_ints_opt(&fragments_jobj)?
        .map(|vec| vec.into_iter().map(|i| i as u32).collect());
    let index_uuid = env.get_string_opt(&index_uuid_jobj)?;

    // Handle scalar vs vector indices differently and get params before borrowing dataset
    let params_result: Result<Box<dyn IndexParams>> = match index_type {
        IndexType::Scalar
        | IndexType::BTree
        | IndexType::Bitmap
        | IndexType::LabelList
        | IndexType::Inverted
        | IndexType::NGram
        | IndexType::ZoneMap
        | IndexType::BloomFilter => {
            // For scalar indices, create a scalar IndexParams
            let (index_type_str, params_opt) = get_scalar_index_params(env, params_jobj)?;
            let scalar_params = lance_index::scalar::ScalarIndexParams {
                index_type: index_type_str,
                params: params_opt,
            };
            Ok(Box::new(scalar_params))
        }
        IndexType::FragmentReuse | IndexType::MemWal => {
            // System indices - not user-creatable
            Err(Error::input_error(format!(
                "Cannot create system index type: {:?}. System indices are managed internally.",
                index_type
            )))
        }
        IndexType::Vector
        | IndexType::IvfFlat
        | IndexType::IvfSq
        | IndexType::IvfPq
        | IndexType::IvfRq
        | IndexType::IvfHnswSq
        | IndexType::IvfHnswPq
        | IndexType::IvfHnswFlat => {
            // For vector indices, use the existing parameter handling
            get_vector_index_params(env, params_jobj)
        }
    };

    let params = params_result?;
    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;

    let mut index_builder = dataset_guard
        .inner
        .create_index_builder(&columns_slice, index_type, params.as_ref())
        .replace(replace)
        .train(train);

    if let Some(name) = name {
        index_builder = index_builder.name(name);
    }

    let has_fragment_ids = fragment_ids.is_some();

    if let Some(fragment_ids) = fragment_ids {
        index_builder = index_builder.fragments(fragment_ids);
    }

    if let Some(index_uuid) = index_uuid {
        index_builder = index_builder.index_uuid(index_uuid);
    }

    if has_fragment_ids {
        RT.block_on(index_builder.execute_uncommitted())?;
    } else {
        RT.block_on(index_builder.into_future())?
    }

    Ok(())
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_innerMergeIndexMetadata<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    index_uuid: JString,
    index_type_code_jobj: jint,
    batch_readhead_jobj: JObject, // Optional<Integer>
) {
    ok_or_throw_without_return!(
        env,
        inner_merge_index_metadata(
            &mut env,
            java_dataset,
            index_uuid,
            index_type_code_jobj,
            batch_readhead_jobj
        )
    );
}

fn inner_merge_index_metadata(
    env: &mut JNIEnv,
    java_dataset: JObject,
    index_uuid: JString,
    index_type_code_jobj: jint,
    batch_readhead_jobj: JObject, // Optional<Integer>
) -> Result<()> {
    let index_uuid = index_uuid.extract(env)?;
    let index_type = IndexType::try_from(index_type_code_jobj)?;
    let batch_readhead = env
        .get_int_opt(&batch_readhead_jobj)?
        .map(|val| val as usize);

    let dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;

    RT.block_on(async {
        let index_store = LanceIndexStore::from_dataset_for_new(&dataset_guard.inner, &index_uuid)?;
        let object_store = dataset_guard.inner.object_store();
        let index_dir = dataset_guard.inner.indices_dir().child(index_uuid);

        match index_type {
            IndexType::Inverted => lance_index::scalar::inverted::builder::merge_index_files(
                object_store,
                &index_dir,
                Arc::new(index_store),
            )
            .await
            .map_err(|e| {
                Error::runtime_error(format!(
                    "Cannot create index of type: {:?}. Caused by: {:?}",
                    index_type,
                    e.to_string()
                ))
            }),
            IndexType::BTree => lance_index::scalar::btree::merge_index_files(
                object_store,
                &index_dir,
                Arc::new(index_store),
                batch_readhead,
            )
            .await
            .map_err(|e| {
                Error::runtime_error(format!(
                    "Cannot create index of type: {:?}. Caused by: {:?}",
                    index_type,
                    e.to_string()
                ))
            }),
            _ => Err(Error::input_error(format!(
                "Cannot merge index type: {:?}. Only supports BTREE and INVERTED now.",
                index_type
            ))),
        }
    })
}

//////////////////
// Read Methods //
//////////////////
#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_openNative<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    path: JString,
    version_obj: JObject,    // Optional<Integer>
    block_size_obj: JObject, // Optional<Integer>
    index_cache_size_bytes: jlong,
    metadata_cache_size_bytes: jlong,
    storage_options_obj: JObject,          // Map<String, String>
    serialized_manifest: JObject,          // Optional<ByteBuffer>
    storage_options_provider_obj: JObject, // Optional<StorageOptionsProvider>
    s3_credentials_refresh_offset_seconds_obj: JObject, // Optional<Long>
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_open_native(
            &mut env,
            path,
            version_obj,
            block_size_obj,
            index_cache_size_bytes,
            metadata_cache_size_bytes,
            storage_options_obj,
            serialized_manifest,
            storage_options_provider_obj,
            s3_credentials_refresh_offset_seconds_obj
        )
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_open_native<'local>(
    env: &mut JNIEnv<'local>,
    path: JString,
    version_obj: JObject,    // Optional<Integer>
    block_size_obj: JObject, // Optional<Integer>
    index_cache_size_bytes: jlong,
    metadata_cache_size_bytes: jlong,
    storage_options_obj: JObject,          // Map<String, String>
    serialized_manifest: JObject,          // Optional<ByteBuffer>
    storage_options_provider_obj: JObject, // Optional<StorageOptionsProvider>
    s3_credentials_refresh_offset_seconds_obj: JObject, // Optional<Long>
) -> Result<JObject<'local>> {
    let path_str: String = path.extract(env)?;
    let version = env.get_int_opt(&version_obj)?;
    let block_size = env.get_int_opt(&block_size_obj)?;
    let jmap = JMap::from_env(env, &storage_options_obj)?;
    let storage_options = to_rust_map(env, &jmap)?;

    // Extract storage options provider first (before get_bytes_opt which borrows env)
    let storage_options_provider = if !storage_options_provider_obj.is_null() {
        // Check if it's an Optional.empty()
        let is_present = env
            .call_method(&storage_options_provider_obj, "isPresent", "()Z", &[])?
            .z()?;
        if is_present {
            // Get the value from Optional
            let provider_obj = env
                .call_method(
                    &storage_options_provider_obj,
                    "get",
                    "()Ljava/lang/Object;",
                    &[],
                )?
                .l()?;
            Some(JavaStorageOptionsProvider::new(env, provider_obj)?)
        } else {
            None
        }
    } else {
        None
    };

    let storage_options_provider_arc =
        storage_options_provider.map(|v| Arc::new(v) as Arc<dyn StorageOptionsProvider>);

    // Extract s3_credentials_refresh_offset_seconds
    let s3_credentials_refresh_offset_seconds =
        if !s3_credentials_refresh_offset_seconds_obj.is_null() {
            let is_present = env
                .call_method(
                    &s3_credentials_refresh_offset_seconds_obj,
                    "isPresent",
                    "()Z",
                    &[],
                )?
                .z()?;
            if is_present {
                let value = env
                    .call_method(
                        &s3_credentials_refresh_offset_seconds_obj,
                        "get",
                        "()Ljava/lang/Object;",
                        &[],
                    )?
                    .l()?;
                let long_value = env.call_method(&value, "longValue", "()J", &[])?.j()?;
                Some(long_value as u64)
            } else {
                None
            }
        } else {
            None
        };

    let serialized_manifest = env.get_bytes_opt(&serialized_manifest)?;
    let dataset = BlockingDataset::open(
        &path_str,
        version,
        block_size,
        index_cache_size_bytes,
        metadata_cache_size_bytes,
        storage_options,
        serialized_manifest,
        storage_options_provider_arc,
        s3_credentials_refresh_offset_seconds,
    )?;
    dataset.into_java(env)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_getFragmentsNative<'a>(
    mut env: JNIEnv<'a>,
    jdataset: JObject,
) -> JObject<'a> {
    ok_or_throw!(env, inner_get_fragments(&mut env, jdataset))
}

fn inner_get_fragments<'local>(
    env: &mut JNIEnv<'local>,
    jdataset: JObject,
) -> Result<JObject<'local>> {
    let fragments = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }?;
        dataset.inner.get_fragments()
    };
    let fragments = fragments
        .iter()
        .map(|f| f.metadata().clone())
        .collect::<Vec<Fragment>>();
    export_vec(env, &fragments)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_getFragmentNative<'a>(
    mut env: JNIEnv<'a>,
    jdataset: JObject,
    fragment_id: jint,
) -> JObject<'a> {
    ok_or_throw!(env, inner_get_fragment(&mut env, jdataset, fragment_id))
}

fn inner_get_fragment<'local>(
    env: &mut JNIEnv<'local>,
    jdataset: JObject,
    fragment_id: jint,
) -> Result<JObject<'local>> {
    let fragment = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }?;
        dataset.inner.get_fragment(fragment_id as usize)
    };
    let obj = match fragment {
        Some(f) => f.metadata().into_java(env)?,
        None => JObject::default(),
    };
    Ok(obj)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeGetLanceSchema<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_get_lance_schema(&mut env, java_dataset))
}

fn inner_get_lance_schema<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let schema = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset.inner.schema().clone()
    };
    schema.into_java(env)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_importFfiSchema(
    mut env: JNIEnv,
    jdataset: JObject,
    arrow_schema_addr: jlong,
) {
    ok_or_throw_without_return!(
        env,
        inner_import_ffi_schema(&mut env, jdataset, arrow_schema_addr)
    )
}

fn inner_import_ffi_schema(
    env: &mut JNIEnv,
    jdataset: JObject,
    arrow_schema_addr: jlong,
) -> Result<()> {
    let schema = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }?;
        Schema::from(dataset.inner.schema())
    };

    let ffi_schema = FFI_ArrowSchema::try_from(&schema)?;
    unsafe { std::ptr::write_unaligned(arrow_schema_addr as *mut FFI_ArrowSchema, ffi_schema) }
    Ok(())
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeUri<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JString<'local> {
    ok_or_throw_with_return!(
        env,
        inner_uri(&mut env, java_dataset).map_err(|err| Error::input_error(err.to_string())),
        JString::from(JObject::null())
    )
}

fn inner_uri<'local>(env: &mut JNIEnv<'local>, java_dataset: JObject) -> Result<JString<'local>> {
    let uri = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.inner.uri().to_string()
    };

    let jstring_uri = env.new_string(uri)?;
    Ok(jstring_uri)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeListVersions<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_list_versions(&mut env, java_dataset))
}

fn inner_list_versions<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let versions = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.list_versions()?
    };
    let array_list = env.new_object("java/util/ArrayList", "()V", &[])?;

    versions
        .into_iter()
        .map(|inner_ver| inner_ver.into_java(env))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .try_for_each(|java_ver| -> Result<()> {
            env.call_method(
                &array_list,
                "add",
                "(Ljava/lang/Object;)Z",
                &[JValue::Object(&java_ver)],
            )?;
            Ok(())
        })?;
    Ok(array_list)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeGetVersion<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_get_version(&mut env, java_dataset))
}

fn inner_get_version<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let version = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.version()?
    };
    version.into_java(env)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeGetLatestVersionId(
    mut env: JNIEnv,
    java_dataset: JObject,
) -> jlong {
    ok_or_throw_with_return!(env, inner_latest_version_id(&mut env, java_dataset), -1) as jlong
}

fn inner_latest_version_id(env: &mut JNIEnv, java_dataset: JObject) -> Result<u64> {
    let dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    dataset_guard.latest_version()
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeCheckoutLatest(
    mut env: JNIEnv,
    java_dataset: JObject,
) {
    ok_or_throw_without_return!(env, inner_checkout_latest(&mut env, java_dataset));
}

fn inner_checkout_latest(env: &mut JNIEnv, java_dataset: JObject) -> Result<()> {
    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    dataset_guard.checkout_latest()
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeCheckoutVersion<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    version: jlong,
) -> JObject<'local> {
    ok_or_throw!(env, inner_checkout_version(&mut env, java_dataset, version))
}

fn inner_checkout_version<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    version: jlong,
) -> Result<JObject<'local>> {
    let new_dataset = {
        let mut dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.checkout_version(version as u64)?
    };

    new_dataset.into_java(env)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeCheckoutTag<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    jtag: JString,
) -> JObject<'local> {
    ok_or_throw!(env, inner_checkout_tag(&mut env, java_dataset, jtag))
}

fn inner_checkout_tag<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    jtag_name: JString,
) -> Result<JObject<'local>> {
    let tag_name = jtag_name.extract(env)?;
    let new_dataset = {
        let mut dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.checkout_tag(tag_name.as_str())?
    };

    new_dataset.into_java(env)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeRestore(
    mut env: JNIEnv,
    java_dataset: JObject,
) {
    ok_or_throw_without_return!(env, inner_restore(&mut env, java_dataset))
}

fn inner_restore(env: &mut JNIEnv, java_dataset: JObject) -> Result<()> {
    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    dataset_guard.restore()
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeShallowClone<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    target_path: JString,
    reference: JObject,
    storage_options: JObject,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_shallow_clone(
            &mut env,
            java_dataset,
            target_path,
            reference,
            storage_options
        )
    )
}

fn inner_shallow_clone<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    target_path: JString,
    reference: JObject,
    storage_options: JObject,
) -> Result<JObject<'local>> {
    let target_path_str = target_path.extract(env)?;
    let storage_options = env.get_optional(&storage_options, |env, map_obj| {
        let jmap = JMap::from_env(env, map_obj)?;
        to_rust_map(env, &jmap)
    })?;

    let reference = {
        let version_number = env.get_optional_u64_from_method(&reference, "getVersionNumber")?;
        let tag_name = env.get_optional_string_from_method(&reference, "getTagName")?;
        let branch_name = env.get_optional_string_from_method(&reference, "getBranchName")?;
        match (version_number, branch_name, tag_name) {
            (Some(version_number), branch_name, None) => {
                Ref::Version(branch_name, Some(version_number))
            }
            (None, None, Some(tag_name)) => Ref::Tag(tag_name),
            _ => {
                return Err(Error::input_error(
                    "One of (optional branch, version_number) and tag must be specified"
                        .to_string(),
                ))
            }
        }
    };

    let new_ds = {
        let mut dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        RT.block_on(
            dataset_guard.inner.shallow_clone(
                &target_path_str,
                reference,
                storage_options
                    .map(|options| {
                        Some(ObjectStoreParams {
                            storage_options: Some(options),
                            ..Default::default()
                        })
                    })
                    .unwrap_or(None),
            ),
        )?
    };

    BlockingDataset { inner: new_ds }.into_java(env)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeCountRows(
    mut env: JNIEnv,
    java_dataset: JObject,
    filter_jobj: JObject, // Optional<String>
) -> jlong {
    ok_or_throw_with_return!(
        env,
        inner_count_rows(&mut env, java_dataset, filter_jobj),
        -1
    ) as jlong
}

fn inner_count_rows(
    env: &mut JNIEnv,
    java_dataset: JObject,
    filter_jobj: JObject,
) -> Result<usize> {
    let filter = env.get_string_opt(&filter_jobj)?;
    let dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    dataset_guard.count_rows(filter)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeGetDataStatistics<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_get_data_statistics(&mut env, java_dataset))
}

fn inner_get_data_statistics<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let stats = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.calculate_data_stats()?
    };
    let data_stats = env.new_object("com/lancedb/lance/ipc/DataStatistics", "()V", &[])?;

    for field in stats.fields {
        let id = field.id as jint;
        let byte_size = field.bytes_on_disk as jlong;
        let filed_jobj = env.new_object(
            "com/lancedb/lance/ipc/FieldStatistics",
            "(IJ)V",
            &[JValue::Int(id), JValue::Long(byte_size)],
        )?;
        env.call_method(
            &data_stats,
            "addFiledStatistics",
            "(Lcom/lancedb/lance/ipc/FieldStatistics;)V",
            &[JValue::Object(&filed_jobj)],
        )?;
    }
    Ok(data_stats)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeListIndexes<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_list_indexes(&mut env, java_dataset))
}

fn inner_list_indexes<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let index_names = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        let indexes = dataset_guard.list_indexes()?;
        indexes
            .iter()
            .map(|index| index.name.clone())
            .collect::<Vec<String>>()
    };

    let array_list = env.new_object("java/util/ArrayList", "()V", &[])?;

    for name in index_names {
        let java_string = env.new_string(&name)?;
        env.call_method(
            &array_list,
            "add",
            "(Ljava/lang/Object;)Z",
            &[JValue::Object(&java_string)],
        )?;
    }

    Ok(array_list)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeGetConfig<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_get_config(&mut env, java_dataset))
}

fn inner_get_config<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let config = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.inner.config().clone()
    };

    let java_hashmap = env
        .new_object("java/util/HashMap", "()V", &[])
        .expect("Failed to create Java HashMap");

    for (k, v) in config {
        let java_key = env
            .new_string(&k)
            .expect("Failed to create Java String (key)");
        let java_value = env
            .new_string(&v)
            .expect("Failed to create Java String (value)");

        env.call_method(
            &java_hashmap,
            "put",
            "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;",
            &[JValue::Object(&java_key), JValue::Object(&java_value)],
        )
        .expect("Failed to call HashMap.put()");
    }

    Ok(java_hashmap)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeTake(
    mut env: JNIEnv,
    java_dataset: JObject,
    indices_obj: JObject, // List<Long>
    columns_obj: JObject, // List<String>
) -> jbyteArray {
    match inner_take(&mut env, java_dataset, indices_obj, columns_obj) {
        Ok(byte_array) => byte_array,
        Err(e) => {
            let _ = env.throw_new("java/lang/RuntimeException", format!("{:?}", e));
            std::ptr::null_mut()
        }
    }
}

fn inner_take(
    env: &mut JNIEnv,
    java_dataset: JObject,
    indices_obj: JObject, // List<Long>
    columns_obj: JObject, // List<String>
) -> Result<jbyteArray> {
    let indices: Vec<i64> = env.get_longs(&indices_obj)?;
    let indices_u64: Vec<u64> = indices.iter().map(|&x| x as u64).collect();
    let indices_slice: &[u64] = &indices_u64;
    let columns: Vec<String> = env.get_strings(&columns_obj)?;

    let result = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        let dataset = &dataset_guard.inner;

        let projection = ProjectionRequest::from_columns(columns, dataset.schema());

        match RT.block_on(dataset.take(indices_slice, projection)) {
            Ok(res) => res,
            Err(e) => {
                return Err(e.into());
            }
        }
    };

    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &result.schema())?;
        writer.write(&result)?;
        writer.finish()?;
    }

    let byte_array = env.byte_array_from_slice(&buffer)?;
    Ok(**byte_array)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeDelete(
    mut env: JNIEnv,
    java_dataset: JObject,
    predicate: JString,
) {
    ok_or_throw_without_return!(env, inner_delete(&mut env, java_dataset, predicate))
}

fn inner_delete(env: &mut JNIEnv, java_dataset: JObject, predicate: JString) -> Result<()> {
    let predicate_str = predicate.extract(env)?;
    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    RT.block_on(dataset_guard.inner.delete(&predicate_str))?;
    Ok(())
}

//////////////////////////////
// Schema evolution Methods //
//////////////////////////////
#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeDropColumns(
    mut env: JNIEnv,
    java_dataset: JObject,
    columns_obj: JObject, // List<String>
) {
    ok_or_throw_without_return!(env, inner_drop_columns(&mut env, java_dataset, columns_obj))
}

fn inner_drop_columns(
    env: &mut JNIEnv,
    java_dataset: JObject,
    columns_obj: JObject, // List<String>
) -> Result<()> {
    let columns: Vec<String> = env.get_strings(&columns_obj)?;
    let columns_slice: Vec<&str> = columns.iter().map(AsRef::as_ref).collect();
    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    RT.block_on(dataset_guard.inner.drop_columns(&columns_slice))?;
    Ok(())
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeAlterColumns(
    mut env: JNIEnv,
    java_dataset: JObject,
    column_alterations_obj: JObject, // List<ColumnAlteration>
) {
    ok_or_throw_without_return!(
        env,
        inner_alter_columns(&mut env, java_dataset, column_alterations_obj)
    )
}

fn create_column_alteration(
    env: &mut JNIEnv,
    column_alteration_jobj: JObject, // ColumnAlteration
) -> Result<ColumnAlteration> {
    let path_obj = env
        .get_field(&column_alteration_jobj, "path", "Ljava/lang/String;")?
        .l()?;
    let path_jstring: JString = path_obj.into();
    let path: String = env.get_string(&path_jstring)?.into();

    let rename_obj = env
        .get_field(&column_alteration_jobj, "rename", "Ljava/util/Optional;")?
        .l()?;
    let rename = if env.call_method(&rename_obj, "isPresent", "()Z", &[])?.z()? {
        let jstring: JObject = env
            .call_method(rename_obj, "get", "()Ljava/lang/Object;", &[])?
            .l()?;
        let jstring: JString = jstring.into();
        let rename_str: String = env.get_string(&jstring)?.into(); // Intermediate variable
        Some(rename_str)
    } else {
        None
    };

    let nullable_obj = env
        .get_field(&column_alteration_jobj, "nullable", "Ljava/util/Optional;")?
        .l()?;
    let nullable = if env
        .call_method(&nullable_obj, "isPresent", "()Z", &[])?
        .z()?
    {
        let nullable_value = env
            .call_method(nullable_obj, "get", "()Ljava/lang/Object;", &[])?
            .l()?;
        Some(
            env.call_method(nullable_value, "booleanValue", "()Z", &[])?
                .z()?,
        )
    } else {
        None
    };

    let data_type_obj = env
        .get_field(&column_alteration_jobj, "dataType", "Ljava/util/Optional;")?
        .l()?;
    let data_type = if env
        .call_method(&data_type_obj, "isPresent", "()Z", &[])?
        .z()?
    {
        let j_data_type: JObject = env
            .call_method(data_type_obj, "get", "()Ljava/lang/Object;", &[])?
            .l()?;
        let jstring: JString = env
            .call_method(j_data_type, "toString", "()Ljava/lang/String;", &[])?
            .l()?
            .into();
        let data_type_str: String = env.get_string(&jstring)?.into(); // Intermediate variable
        DataType::from_str(&data_type_str)
            .map_err(|e| Error::input_error(e.to_string()))
            .ok()
    } else {
        None
    };

    Ok(ColumnAlteration {
        path,
        rename,
        nullable,
        data_type,
    })
}

fn inner_alter_columns(
    env: &mut JNIEnv,
    java_dataset: JObject,
    column_alterations_obj: JObject, // List<ColumnAlteration>
) -> Result<()> {
    let list = env.get_list(&column_alterations_obj)?;
    let mut iter = list.iter(env)?;
    let mut column_alterations = Vec::new();

    while let Some(elem) = iter.next(env)? {
        let alteration = create_column_alteration(env, elem)?;
        column_alterations.push(alteration);
    }

    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;

    RT.block_on(dataset_guard.inner.alter_columns(&column_alterations))?;
    Ok(())
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeAddColumnsBySqlExpressions(
    mut env: JNIEnv,
    java_dataset: JObject,
    sql_expressions: JObject, // SqlExpressions
    batch_size: JObject,      // Optional<Long>
) {
    ok_or_throw_without_return!(
        env,
        inner_add_columns_by_sql_expressions(&mut env, java_dataset, sql_expressions, batch_size)
    )
}

fn inner_add_columns_by_sql_expressions(
    env: &mut JNIEnv,
    java_dataset: JObject,
    sql_expressions: JObject, // SqlExpressions
    batch_size: JObject,      // Optional<Long>
) -> Result<()> {
    let sql_expressions_obj = env
        .get_field(sql_expressions, "sqlExpressions", "Ljava/util/List;")?
        .l()?;

    let sql_expressions_obj_list = env.get_list(&sql_expressions_obj)?;
    let mut expressions: Vec<(String, String)> = Vec::new();

    let mut iterator = sql_expressions_obj_list.iter(env)?;

    while let Some(item) = iterator.next(env)? {
        let name = env
            .call_method(&item, "getName", "()Ljava/lang/String;", &[])?
            .l()?;
        let value = env
            .call_method(&item, "getExpression", "()Ljava/lang/String;", &[])?
            .l()?;
        let key_str: String = env.get_string(&JString::from(name))?.into();
        let value_str: String = env.get_string(&JString::from(value))?.into();
        expressions.push((key_str, value_str));
    }

    let rust_transform = NewColumnTransform::SqlExpressions(expressions);

    let batch_size = if env.call_method(&batch_size, "isPresent", "()Z", &[])?.z()? {
        let batch_size_value = env.get_long_opt(&batch_size)?;
        match batch_size_value {
            Some(value) => Some(
                value
                    .try_into()
                    .map_err(|_| Error::input_error("Batch size conversion error".to_string()))?,
            ),
            None => None,
        }
    } else {
        None
    };

    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;

    RT.block_on(
        dataset_guard
            .inner
            .add_columns(rust_transform, None, batch_size),
    )?;
    Ok(())
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeAddColumnsByReader(
    mut env: JNIEnv,
    java_dataset: JObject,
    arrow_array_stream_addr: jlong,
    batch_size: JObject, // Optional<Long>
) {
    ok_or_throw_without_return!(
        env,
        inner_add_columns_by_reader(&mut env, java_dataset, arrow_array_stream_addr, batch_size)
    )
}

fn inner_add_columns_by_reader(
    env: &mut JNIEnv,
    java_dataset: JObject,
    arrow_array_stream_addr: jlong,
    batch_size: JObject, // Optional<Long>
) -> Result<()> {
    let stream_ptr = arrow_array_stream_addr as *mut FFI_ArrowArrayStream;

    let reader = unsafe { ArrowArrayStreamReader::from_raw(stream_ptr) }?;

    let transform = NewColumnTransform::Reader(Box::new(reader));

    let batch_size = if env.call_method(&batch_size, "isPresent", "()Z", &[])?.z()? {
        let batch_size_value = env.get_long_opt(&batch_size)?;
        match batch_size_value {
            Some(value) => Some(
                value
                    .try_into()
                    .map_err(|_| Error::input_error("Batch size conversion error".to_string()))?,
            ),
            None => None,
        }
    } else {
        None
    };

    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;

    RT.block_on(dataset_guard.inner.add_columns(transform, None, batch_size))?;

    Ok(())
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeAddColumnsBySchema(
    mut env: JNIEnv,
    java_dataset: JObject,
    schema_ptr: jlong, // Schema pointer
) {
    ok_or_throw_without_return!(
        env,
        inner_add_columns_by_schema(&mut env, java_dataset, schema_ptr)
    )
}

fn inner_add_columns_by_schema(
    env: &mut JNIEnv,
    java_dataset: JObject,
    schema_ptr: jlong,
) -> Result<()> {
    let c_schema = unsafe { FFI_ArrowSchema::from_raw(schema_ptr as *mut _) };

    let schema = ArrowSchema::try_from(&c_schema)
        .map_err(|_| Error::input_error("ArrowSchema conversion error".to_string()))?;

    let transform = NewColumnTransform::AllNulls(Arc::new(schema));
    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;

    RT.block_on(dataset_guard.inner.add_columns(transform, None, None))?;

    Ok(())
}

//////////////////////////////
// Tag operation Methods    //
//////////////////////////////
#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeListTags<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_list_tags(&mut env, java_dataset))
}

fn inner_list_tags<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let tag_map = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.list_tags()?
    };
    let array_list = env.new_object("java/util/ArrayList", "()V", &[])?;

    for (tag_name, tag_contents) in tag_map {
        let java_tag = env.new_object(
            "com/lancedb/lance/Tag",
            "(Ljava/lang/String;JI)V",
            &[
                JValue::Object(&env.new_string(tag_name)?.into()),
                JValue::Long(tag_contents.version as i64),
                JValue::Int(tag_contents.manifest_size as i32),
            ],
        )?;
        env.call_method(
            &array_list,
            "add",
            "(Ljava/lang/Object;)Z",
            &[JValue::Object(&java_tag)],
        )?;
    }
    Ok(array_list)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeCreateTag(
    mut env: JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
    jtag_version: jlong,
) {
    ok_or_throw_without_return!(
        env,
        inner_create_tag(&mut env, java_dataset, jtag_name, jtag_version)
    )
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeCreateTagOnBranch(
    mut env: JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
    jtag_version: jlong,
    jbranch: JString,
) {
    ok_or_throw_without_return!(
        env,
        inner_create_tag_on_branch(&mut env, java_dataset, jtag_name, jtag_version, jbranch)
    )
}

fn inner_create_tag(
    env: &mut JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
    jtag_version: jlong,
) -> Result<()> {
    let tag = jtag_name.extract(env)?;
    let mut dataset_guard =
        { unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }? };
    dataset_guard.create_tag(tag.as_str(), jtag_version as u64, None)?;
    Ok(())
}

fn inner_create_tag_on_branch(
    env: &mut JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
    jtag_version: jlong,
    jbranch: JString,
) -> Result<()> {
    let tag = jtag_name.extract(env)?;
    let branch = jbranch.extract(env)?;
    let mut dataset_guard =
        { unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }? };
    dataset_guard.create_tag(tag.as_str(), jtag_version as u64, Some(branch.as_str()))?;
    Ok(())
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeDeleteTag(
    mut env: JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
) {
    ok_or_throw_without_return!(env, inner_delete_tag(&mut env, java_dataset, jtag_name))
}

fn inner_delete_tag(env: &mut JNIEnv, java_dataset: JObject, jtag_name: JString) -> Result<()> {
    let tag = { jtag_name.extract(env)? };
    let mut dataset_guard =
        { unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }? };
    dataset_guard.delete_tag(tag.as_str())
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeUpdateTag(
    mut env: JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
    jtag_version: jlong,
) {
    ok_or_throw_without_return!(
        env,
        inner_update_tag(&mut env, java_dataset, jtag_name, jtag_version)
    )
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeUpdateTagOnBranch(
    mut env: JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
    jtag_version: jlong,
    jbranch: JString,
) {
    ok_or_throw_without_return!(
        env,
        inner_update_tag_on_branch(&mut env, java_dataset, jtag_name, jtag_version, jbranch)
    )
}

fn inner_update_tag_on_branch(
    env: &mut JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
    jtag_version: jlong,
    jbranch: JString,
) -> Result<()> {
    let tag = jtag_name.extract(env)?;
    let branch = jbranch.extract(env)?;
    let mut dataset_guard =
        { unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }? };
    dataset_guard.update_tag(tag.as_str(), jtag_version as u64, Some(branch.as_str()))?;
    Ok(())
}

fn inner_update_tag(
    env: &mut JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
    jtag_version: jlong,
) -> Result<()> {
    let tag = jtag_name.extract(env)?;
    let mut dataset_guard =
        { unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }? };
    dataset_guard.update_tag(tag.as_str(), jtag_version as u64, None)?;
    Ok(())
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeGetVersionByTag(
    mut env: JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
) -> jlong {
    ok_or_throw_with_return!(
        env,
        inner_get_version_by_tag(&mut env, java_dataset, jtag_name),
        -1
    ) as jlong
}

fn inner_get_version_by_tag(
    env: &mut JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
) -> Result<u64> {
    let tag = { jtag_name.extract(env)? };
    let dataset_guard =
        { unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }? };
    dataset_guard.get_version(tag.as_str())
}

//////////////////////////////
// Branch operation Methods  //
//////////////////////////////
#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeListBranches<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_list_branches(&mut env, java_dataset))
}

fn inner_list_branches<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let branches = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.list_branches()?
    };
    let array_list = env.new_object("java/util/ArrayList", "()V", &[])?;

    for (name, contents) in branches {
        let jname = env.new_string(name)?;
        let jparent = if let Some(p) = contents.parent_branch {
            env.new_string(p)?.into()
        } else {
            JObject::null()
        };
        let jbranch = env.new_object(
            "com/lancedb/lance/Branch",
            "(Ljava/lang/String;Ljava/lang/String;JJI)V",
            &[
                JValue::Object(&jname),
                JValue::Object(&jparent),
                JValue::Long(contents.parent_version as i64),
                JValue::Long(contents.create_at as i64),
                JValue::Int(contents.manifest_size as i32),
            ],
        )?;
        env.call_method(
            &array_list,
            "add",
            "(Ljava/lang/Object;)Z",
            &[JValue::Object(&jbranch)],
        )?;
    }
    Ok(array_list)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeCreateBranch<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    jbranch: JString,
    jversion: jlong,
    source_branch_obj: JObject, // Optional<String>
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_create_branch(&mut env, java_dataset, jbranch, jversion, source_branch_obj)
    )
}

fn inner_create_branch<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    jbranch: JString,
    jversion: jlong,
    source_branch_obj: JObject, // Optional<String>
) -> Result<JObject<'local>> {
    let branch_name: String = jbranch.extract(env)?;
    let version = jversion as u64;
    let source_branch = env.get_string_opt(&source_branch_obj)?;
    let new_dataset = {
        let mut dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.create_branch(&branch_name, version, source_branch.as_deref())?
    };
    new_dataset.into_java(env)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeCreateBranchOnTag<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    jbranch: JString,
    jtag_name: JString,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_create_branch_on_tag(&mut env, java_dataset, jbranch, jtag_name)
    )
}

fn inner_create_branch_on_tag<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    jbranch: JString,
    jtag_name: JString,
) -> Result<JObject<'local>> {
    let branch_name: String = jbranch.extract(env)?;
    let tag_name: String = jtag_name.extract(env)?;
    let reference = Ref::from(tag_name.as_str());

    let new_blocking_dataset = {
        let mut dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        let inner = RT.block_on(dataset_guard.inner.create_branch(
            branch_name.as_str(),
            reference,
            None,
        ))?;
        BlockingDataset { inner }
    };
    new_blocking_dataset.into_java(env)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeDeleteBranch(
    mut env: JNIEnv,
    java_dataset: JObject,
    jbranch: JString,
) {
    ok_or_throw_without_return!(env, inner_delete_branch(&mut env, java_dataset, jbranch))
}

fn inner_delete_branch(env: &mut JNIEnv, java_dataset: JObject, jbranch: JString) -> Result<()> {
    let branch_name: String = jbranch.extract(env)?;
    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    dataset_guard.delete_branch(branch_name.as_str())
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeCheckout<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    reference_obj: JObject, // Reference
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_checkout_ref(&mut env, java_dataset, reference_obj)
    )
}

fn inner_checkout_ref<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    reference_obj: JObject, // Reference
) -> Result<JObject<'local>> {
    // Extract Optional fields from Reference
    let branch_opt_obj = env
        .call_method(
            &reference_obj,
            "getBranchName",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let version_opt_obj = env
        .call_method(
            &reference_obj,
            "getVersionNumber",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let tag_opt_obj = env
        .call_method(&reference_obj, "getTagName", "()Ljava/util/Optional;", &[])?
        .l()?;

    let branch_opt = env.get_string_opt(&branch_opt_obj)?;
    let version_opt = env.get_u64_opt(&version_opt_obj)?;
    let tag_opt = env.get_string_opt(&tag_opt_obj)?;

    let new_dataset = {
        let mut dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.checkout_reference(branch_opt, version_opt, tag_opt)?
    };
    new_dataset.into_java(env)
}

// Unified metadata API JNI methods

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeGetTableMetadata<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_get_table_metadata(&mut env, java_dataset))
}

fn inner_get_table_metadata<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let table_metadata = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.get_table_metadata()?
    };

    let java_hashmap = env
        .new_object("java/util/HashMap", "()V", &[])
        .expect("Failed to create Java HashMap");

    for (k, v) in table_metadata {
        let java_key = env
            .new_string(&k)
            .expect("Failed to create Java String (key)");
        let java_value = env
            .new_string(&v)
            .expect("Failed to create Java String (value)");

        env.call_method(
            &java_hashmap,
            "put",
            "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;",
            &[JValue::Object(&java_key), JValue::Object(&java_value)],
        )
        .expect("Failed to call HashMap.put()");
    }

    Ok(java_hashmap)
}

//////////////////////////////
// Compaction Methods       //
//////////////////////////////
#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeCompact(
    mut env: JNIEnv,
    java_dataset: JObject,
    compaction_options: JObject, // CompactionOptions
) {
    ok_or_throw_without_return!(
        env,
        inner_compact(&mut env, java_dataset, compaction_options)
    )
}

fn inner_compact(
    env: &mut JNIEnv,
    java_dataset: JObject,
    compaction_options: JObject, // CompactionOptions
) -> Result<()> {
    let rust_options = convert_java_compaction_options_to_rust(env, compaction_options)?;
    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    dataset_guard.compact(rust_options)?;
    Ok(())
}

fn convert_java_compaction_options_to_rust(
    env: &mut JNIEnv,
    java_options: JObject,
) -> Result<RustCompactionOptions> {
    let target_rows_per_fragment = env
        .call_method(
            &java_options,
            "getTargetRowsPerFragment",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let max_rows_per_group = env
        .call_method(
            &java_options,
            "getMaxRowsPerGroup",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let max_bytes_per_file = env
        .call_method(
            &java_options,
            "getMaxBytesPerFile",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let materialize_deletions = env
        .call_method(
            &java_options,
            "getMaterializeDeletions",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let materialize_deletions_threshold = env
        .call_method(
            &java_options,
            "getMaterializeDeletionsThreshold",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let num_threads = env
        .call_method(
            &java_options,
            "getNumThreads",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let batch_size = env
        .call_method(&java_options, "getBatchSize", "()Ljava/util/Optional;", &[])?
        .l()?;
    let defer_index_remap = env
        .call_method(
            &java_options,
            "getDeferIndexRemap",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;

    build_compaction_options(
        env,
        &target_rows_per_fragment,
        &max_rows_per_group,
        &max_bytes_per_file,
        &materialize_deletions,
        &materialize_deletions_threshold,
        &num_threads,
        &batch_size,
        &defer_index_remap,
    )
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeCleanupWithPolicy<'local>(
    mut env: JNIEnv<'local>,
    jdataset: JObject,
    jpolicy: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_cleanup_with_policy(&mut env, jdataset, jpolicy))
}

fn inner_cleanup_with_policy<'local>(
    env: &mut JNIEnv<'local>,
    jdataset: JObject,
    jpolicy: JObject,
) -> Result<JObject<'local>> {
    let before_ts_millis =
        env.get_optional_u64_from_method(&jpolicy, "getBeforeTimestampMillis")?;
    let before_timestamp = before_ts_millis.map(|millis| {
        let st = UNIX_EPOCH + Duration::from_millis(millis);
        DateTime::<Utc>::from(st)
    });

    let before_version = env.get_optional_u64_from_method(&jpolicy, "getBeforeVersion")?;

    let delete_unverified = env
        .get_optional_from_method(&jpolicy, "getDeleteUnverified", |env, obj| {
            Ok(env.call_method(obj, "booleanValue", "()Z", &[])?.z()?)
        })?
        .unwrap_or(false);

    let error_if_tagged_old_versions = env
        .get_optional_from_method(&jpolicy, "getErrorIfTaggedOldVersions", |env, obj| {
            Ok(env.call_method(obj, "booleanValue", "()Z", &[])?.z()?)
        })?
        .unwrap_or(true);

    let policy = CleanupPolicy {
        before_timestamp,
        before_version,
        delete_unverified,
        error_if_tagged_old_versions,
    };

    let stats = {
        let mut dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }?;
        dataset.cleanup_with_policy(policy)
    }?;

    let jstats = env.new_object(
        "com/lancedb/lance/cleanup/RemovalStats",
        "(JJ)V",
        &[
            JValue::Long(stats.bytes_removed as i64),
            JValue::Long(stats.old_versions as i64),
        ],
    )?;

    Ok(jstats)
}
