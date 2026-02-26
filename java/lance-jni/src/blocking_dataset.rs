// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::error::{Error, Result};
use crate::ffi::JNIEnvExt;
use crate::namespace::{
    create_java_lance_namespace, BlockingDirectoryNamespace, BlockingRestNamespace,
};
use crate::session::{handle_from_session, session_from_handle};
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
use lance::dataset::optimize::{compact_files, CompactionOptions as RustCompactionOptions};
use lance::dataset::refs::{Ref, TagContents};
use lance::dataset::statistics::{DataStatistics, DatasetStatisticsExt};
use lance::dataset::transaction::{Operation, Transaction};
use lance::dataset::{
    ColumnAlteration, CommitBuilder, Dataset, NewColumnTransform, ProjectionRequest, ReadParams,
    Version, WriteParams,
};
use lance::io::commit::namespace_manifest::LanceNamespaceExternalManifestStore;
use lance::io::{ObjectStore, ObjectStoreParams};
use lance::session::Session as LanceSession;
use lance::table::format::IndexMetadata;
use lance::table::format::{BasePath, Fragment};
use lance_core::datatypes::Schema as LanceSchema;
use lance_file::version::LanceFileVersion;
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::btree::BTreeParameters;
use lance_index::DatasetIndexExt;
use lance_index::IndexCriteria as RustIndexCriteria;
use lance_index::{IndexParams, IndexType};
use lance_io::object_store::ObjectStoreRegistry;
use lance_io::object_store::StorageOptionsProvider;
use lance_namespace::LanceNamespace;
use lance_table::io::commit::external_manifest::ExternalManifestCommitHandler;
use lance_table::io::commit::CommitHandler;
use std::collections::HashMap;
use std::future::IntoFuture;
use std::iter::empty;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, UNIX_EPOCH};

pub const NATIVE_DATASET: &str = "nativeDatasetHandle";

impl FromJObjectWithEnv<BasePath> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<BasePath> {
        let id = env.get_u32_from_method(self, "getId")?;
        let name = env.get_optional_string_from_method(self, "getName")?;
        let path = env.get_string_from_method(self, "getPath")?;
        let is_dataset_root = env.get_boolean_from_method(self, "isDatasetRoot")?;
        Ok(BasePath {
            id,
            name,
            path,
            is_dataset_root,
        })
    }
}

#[derive(Clone)]
pub struct BlockingDataset {
    pub(crate) inner: Dataset,
}

impl BlockingDataset {
    /// Get the initial storage options used to open this dataset.
    ///
    /// Returns the options that were provided when the dataset was opened,
    /// without any refresh from the provider. Returns None if no storage options
    /// were provided.
    pub fn initial_storage_options(&self) -> Option<HashMap<String, String>> {
        self.inner.initial_storage_options().cloned()
    }

    /// Get the latest storage options, potentially refreshed from the provider.
    ///
    /// If a storage options provider was configured and credentials are expiring,
    /// this will refresh them.
    pub fn latest_storage_options(&self) -> Result<Option<HashMap<String, String>>> {
        RT.block_on(async { self.inner.latest_storage_options().await })
            .map(|opt| opt.map(|opts| opts.0))
            .map_err(|e| Error::io_error(e.to_string()))
    }

    pub fn drop(uri: &str, storage_options: HashMap<String, String>) -> Result<()> {
        RT.block_on(async move {
            let registry = Arc::new(ObjectStoreRegistry::default());
            let object_store_params = ObjectStoreParams {
                storage_options_accessor: Some(Arc::new(
                    lance::io::StorageOptionsAccessor::with_static_options(storage_options),
                )),
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
        version: Option<u64>,
        block_size: Option<i32>,
        index_cache_size_bytes: i64,
        metadata_cache_size_bytes: i64,
        storage_options: HashMap<String, String>,
        serialized_manifest: Option<&[u8]>,
        storage_options_provider: Option<Arc<dyn StorageOptionsProvider>>,
        session: Option<Arc<LanceSession>>,
        namespace: Option<Arc<dyn LanceNamespace>>,
        table_id: Option<Vec<String>>,
    ) -> Result<Self> {
        // Create storage options accessor from storage_options and provider
        let accessor = match (storage_options.is_empty(), storage_options_provider) {
            (false, Some(provider)) => Some(Arc::new(
                lance::io::StorageOptionsAccessor::with_initial_and_provider(
                    storage_options,
                    provider,
                ),
            )),
            (false, None) => Some(Arc::new(
                lance::io::StorageOptionsAccessor::with_static_options(storage_options),
            )),
            (true, Some(provider)) => Some(Arc::new(
                lance::io::StorageOptionsAccessor::with_provider(provider),
            )),
            (true, None) => None,
        };

        let store_params = ObjectStoreParams {
            block_size: block_size.map(|size| size as usize),
            storage_options_accessor: accessor,
            ..Default::default()
        };
        let params = ReadParams {
            index_cache_size_bytes: index_cache_size_bytes as usize,
            metadata_cache_size_bytes: metadata_cache_size_bytes as usize,
            store_options: Some(store_params),
            session,
            ..Default::default()
        };

        let mut builder = DatasetBuilder::from_uri(uri).with_read_params(params);

        if let Some(ver) = version {
            builder = builder.with_version(ver);
        }

        if let Some(serialized_manifest) = serialized_manifest {
            builder = builder.with_serialized_manifest(serialized_manifest)?;
        }

        // Set up namespace commit handler if namespace and table_id are provided
        if let (Some(ns), Some(tid)) = (namespace, table_id) {
            let external_store = LanceNamespaceExternalManifestStore::new(ns, tid);
            let commit_handler: Arc<dyn CommitHandler> = Arc::new(ExternalManifestCommitHandler {
                external_manifest_store: Arc::new(external_store),
            });
            builder = builder.with_commit_handler(commit_handler);
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
        let accessor = if storage_options.is_empty() {
            None
        } else {
            Some(Arc::new(
                lance::io::StorageOptionsAccessor::with_static_options(storage_options),
            ))
        };
        let inner = RT.block_on(Dataset::commit(
            uri,
            operation,
            read_version,
            Some(ObjectStoreParams {
                storage_options_accessor: accessor,
                ..Default::default()
            }),
            None,
            Default::default(),
            false,
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
        let branches = RT.block_on(self.inner.branches().list())?;
        Ok(branches)
    }

    pub fn delete_branch(&mut self, branch: &str) -> Result<()> {
        RT.block_on(self.inner.branches().delete(branch, true))?;
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

    pub fn create_tag(&mut self, tag: &str, reference: Ref) -> Result<()> {
        RT.block_on(self.inner.tags().create(tag, reference))?;
        Ok(())
    }

    pub fn delete_tag(&mut self, tag: &str) -> Result<()> {
        RT.block_on(self.inner.tags().delete(tag))?;
        Ok(())
    }

    pub fn update_tag(&mut self, tag: &str, reference: Ref) -> Result<()> {
        RT.block_on(self.inner.tags().update(tag, reference))?;
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

    #[allow(clippy::too_many_arguments)]
    pub fn commit_transaction(
        &mut self,
        transaction: Transaction,
        store_params: ObjectStoreParams,
        detached: bool,
        enable_v2_manifest_paths: bool,
        use_stable_row_ids: Option<bool>,
        storage_format: Option<LanceFileVersion>,
        max_retries: u32,
        skip_auto_cleanup: bool,
    ) -> Result<Self> {
        let mut builder = CommitBuilder::new(Arc::new(self.clone().inner))
            .with_store_params(store_params)
            .with_detached(detached)
            .enable_v2_manifest_paths(enable_v2_manifest_paths);
        if let Some(use_stable) = use_stable_row_ids {
            builder = builder.use_stable_row_ids(use_stable);
        }
        if let Some(format) = storage_format {
            builder = builder.with_storage_format(format);
        }
        if max_retries > 0 {
            builder = builder.with_max_retries(max_retries);
        }
        if skip_auto_cleanup {
            builder = builder.with_skip_auto_cleanup(true);
        }
        let new_dataset = RT.block_on(builder.execute(transaction))?;
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
pub extern "system" fn Java_org_lance_Dataset_createWithFfiSchema<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    arrow_schema_addr: jlong,
    path: JString,
    max_rows_per_file: JObject,        // Optional<Integer>
    max_rows_per_group: JObject,       // Optional<Integer>
    max_bytes_per_file: JObject,       // Optional<Long>
    mode: JObject,                     // Optional<String>
    enable_stable_row_ids: JObject,    // Optional<Boolean>
    data_storage_version: JObject,     // Optional<String>
    enable_v2_manifest_paths: JObject, // Optional<Boolean>
    storage_options_obj: JObject,      // Map<String, String>
    initial_bases: JObject,
    target_bases: JObject,
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
            enable_v2_manifest_paths,
            storage_options_obj,
            initial_bases,
            target_bases,
        )
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_create_with_ffi_schema<'local>(
    env: &mut JNIEnv<'local>,
    arrow_schema_addr: jlong,
    path: JString,
    max_rows_per_file: JObject,        // Optional<Integer>
    max_rows_per_group: JObject,       // Optional<Integer>
    max_bytes_per_file: JObject,       // Optional<Long>
    mode: JObject,                     // Optional<String>
    enable_stable_row_ids: JObject,    // Optional<Boolean>
    data_storage_version: JObject,     // Optional<String>
    enable_v2_manifest_paths: JObject, // Optional<Boolean>
    storage_options_obj: JObject,      // Map<String, String>
    initial_bases: JObject,
    target_bases: JObject,
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
        enable_v2_manifest_paths,
        storage_options_obj,
        JObject::null(), // No provider for schema-only creation
        initial_bases,
        target_bases,
        reader,
        None, // No namespace for schema-only creation
    )
}

#[no_mangle]
pub extern "system" fn Java_org_lance_Dataset_drop<'local>(
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
pub extern "system" fn Java_org_lance_Dataset_nativeMigrateManifestPathsV2(
    mut env: JNIEnv,
    java_dataset: JObject,
) {
    ok_or_throw_without_return!(
        env,
        inner_native_migrate_manifest_paths_v2(&mut env, java_dataset)
    )
}

fn inner_native_migrate_manifest_paths_v2(env: &mut JNIEnv, java_dataset: JObject) -> Result<()> {
    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    RT.block_on(dataset_guard.inner.migrate_manifest_paths_v2())?;
    Ok(())
}

#[no_mangle]
pub extern "system" fn Java_org_lance_Dataset_createWithFfiStream<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    arrow_array_stream_addr: jlong,
    path: JString,
    max_rows_per_file: JObject,        // Optional<Integer>
    max_rows_per_group: JObject,       // Optional<Integer>
    max_bytes_per_file: JObject,       // Optional<Long>
    mode: JObject,                     // Optional<String>
    enable_stable_row_ids: JObject,    // Optional<Boolean>
    data_storage_version: JObject,     // Optional<String>
    enable_v2_manifest_paths: JObject, // Optional<Boolean>
    storage_options_obj: JObject,      // Map<String, String>
    initial_bases: JObject,
    target_bases: JObject,
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
            enable_v2_manifest_paths,
            storage_options_obj,
            JObject::null(),
            initial_bases,
            target_bases,
            JObject::null(), // No namespace
            JObject::null(), // No table_id
        )
    )
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_org_lance_Dataset_createWithFfiStreamAndProvider<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    arrow_array_stream_addr: jlong,
    path: JString,
    max_rows_per_file: JObject,            // Optional<Integer>
    max_rows_per_group: JObject,           // Optional<Integer>
    max_bytes_per_file: JObject,           // Optional<Long>
    mode: JObject,                         // Optional<String>
    enable_stable_row_ids: JObject,        // Optional<Boolean>
    data_storage_version: JObject,         // Optional<String>
    enable_v2_manifest_paths: JObject,     // Optional<Boolean>
    storage_options_obj: JObject,          // Map<String, String>
    storage_options_provider_obj: JObject, // Optional<StorageOptionsProvider>
    initial_bases: JObject,                // Optional<List<BasePath>>
    target_bases: JObject,                 // Optional<List<String>>
    namespace_obj: JObject,                // LanceNamespace (can be null)
    table_id_obj: JObject,                 // List<String> (can be null)
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
            enable_v2_manifest_paths,
            storage_options_obj,
            storage_options_provider_obj,
            initial_bases,
            target_bases,
            namespace_obj,
            table_id_obj,
        )
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_create_with_ffi_stream<'local>(
    env: &mut JNIEnv<'local>,
    arrow_array_stream_addr: jlong,
    path: JString,
    max_rows_per_file: JObject,            // Optional<Integer>
    max_rows_per_group: JObject,           // Optional<Integer>
    max_bytes_per_file: JObject,           // Optional<Long>
    mode: JObject,                         // Optional<String>
    enable_stable_row_ids: JObject,        // Optional<Boolean>
    data_storage_version: JObject,         // Optional<String>
    enable_v2_manifest_paths: JObject,     // Optional<Boolean>
    storage_options_obj: JObject,          // Map<String, String>
    storage_options_provider_obj: JObject, // Optional<StorageOptionsProvider>
    initial_bases: JObject,                // Optional<List<BasePath>>
    target_bases: JObject,                 // Optional<List<String>>
    namespace_obj: JObject,                // LanceNamespace (can be null)
    table_id_obj: JObject,                 // List<String> (can be null)
) -> Result<JObject<'local>> {
    let stream_ptr = arrow_array_stream_addr as *mut FFI_ArrowArrayStream;
    let reader = unsafe { ArrowArrayStreamReader::from_raw(stream_ptr) }?;

    // Create the namespace wrapper for commit handling (if provided)
    let namespace_info = extract_namespace_info(env, &namespace_obj, &table_id_obj)?;

    create_dataset(
        env,
        path,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        mode,
        enable_stable_row_ids,
        data_storage_version,
        enable_v2_manifest_paths,
        storage_options_obj,
        storage_options_provider_obj,
        initial_bases,
        target_bases,
        reader,
        namespace_info,
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
    enable_v2_manifest_paths: JObject,
    storage_options_obj: JObject,
    storage_options_provider_obj: JObject, // Optional<StorageOptionsProvider>
    initial_bases: JObject,
    target_bases: JObject,
    reader: impl RecordBatchReader + Send + 'static,
    namespace_info: Option<(Arc<dyn LanceNamespace>, Vec<String>)>,
) -> Result<JObject<'local>> {
    let path_str = path.extract(env)?;

    let mut write_params = extract_write_params(
        env,
        &max_rows_per_file,
        &max_rows_per_group,
        &max_bytes_per_file,
        &mode,
        &enable_stable_row_ids,
        &data_storage_version,
        Some(&enable_v2_manifest_paths),
        &storage_options_obj,
        &storage_options_provider_obj,
        &initial_bases,
        &target_bases,
    )?;

    // Set up namespace commit handler if provided
    if let Some((namespace, table_id)) = namespace_info {
        let external_store = LanceNamespaceExternalManifestStore::new(namespace, table_id);
        let commit_handler: Arc<dyn CommitHandler> = Arc::new(ExternalManifestCommitHandler {
            external_manifest_store: Arc::new(external_store),
        });
        write_params.commit_handler = Some(commit_handler);
    }

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
            "org/lance/Version",
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
    let object = env.new_object("org/lance/Dataset", "()V", &[])?;
    Ok(object)
}

#[no_mangle]
pub extern "system" fn Java_org_lance_Dataset_commitAppend<'local>(
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
pub extern "system" fn Java_org_lance_Dataset_commitOverwrite<'local>(
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
pub extern "system" fn Java_org_lance_Dataset_releaseNativeDataset(mut env: JNIEnv, obj: JObject) {
    ok_or_throw_without_return!(env, inner_release_native_dataset(&mut env, obj))
}

fn inner_release_native_dataset(env: &mut JNIEnv, obj: JObject) -> Result<()> {
    let dataset: BlockingDataset = unsafe { env.take_rust_field(obj, NATIVE_DATASET)? };
    dataset.close();
    Ok(())
}

#[no_mangle]
pub extern "system" fn Java_org_lance_Dataset_nativeCreateIndex<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject<'local>,
    columns_jobj: JObject<'local>, // List<String>
    index_type_code_jobj: jint,
    name_jobj: JObject<'local>,              // Optional<String>
    params_jobj: JObject<'local>,            // IndexParams
    replace_jobj: jboolean,                  // replace
    train_jobj: jboolean,                    // train
    fragments_jobj: JObject<'local>,         // List<Integer>
    index_uuid_jobj: JObject<'local>,        // String
    arrow_stream_addr_jobj: JObject<'local>, // Optional<Long>
) -> JObject<'local> {
    ok_or_throw!(
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
            index_uuid_jobj,
            arrow_stream_addr_jobj,
        )
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_create_index<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject<'local>,
    columns_jobj: JObject<'local>, // List<String>
    index_type_code_jobj: jint,
    name_jobj: JObject<'local>,              // Optional<String>
    params_jobj: JObject<'local>,            // IndexParams
    replace_jobj: jboolean,                  // replace
    train_jobj: jboolean,                    // train
    fragments_jobj: JObject<'local>,         // Optional<List<String>>
    index_uuid_jobj: JObject<'local>,        // Optional<String>
    arrow_stream_addr_jobj: JObject<'local>, // Optional<Long>
) -> Result<JObject<'local>> {
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
    let arrow_stream_addr_opt = env.get_long_opt(&arrow_stream_addr_jobj)?;
    let batch_reader = if let Some(arrow_stream_addr) = arrow_stream_addr_opt {
        let stream_ptr = arrow_stream_addr as *mut FFI_ArrowArrayStream;
        let reader = unsafe { ArrowArrayStreamReader::from_raw(stream_ptr) }?;
        Some(reader)
    } else {
        None
    };

    // we should skip committing index when building distributed indices.
    let mut skip_commit = fragment_ids.is_some();

    // Handle scalar vs vector indices differently and get params before borrowing dataset
    let params_result: Result<Box<dyn IndexParams>> = match index_type {
        IndexType::Scalar
        | IndexType::BTree
        | IndexType::Bitmap
        | IndexType::LabelList
        | IndexType::Inverted
        | IndexType::NGram
        | IndexType::ZoneMap
        | IndexType::BloomFilter
        | IndexType::RTree => {
            // For scalar indices, create a scalar IndexParams
            let (index_type_str, params_opt) = get_scalar_index_params(env, params_jobj)?;
            let scalar_params = lance_index::scalar::ScalarIndexParams {
                index_type: index_type_str,
                params: params_opt.clone(),
            };
            skip_commit = skip_commit || should_skip_commit(index_type, &params_opt)?;
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

    // Execute index creation in a block to ensure dataset_guard is dropped
    // before we call into_java (which needs to borrow env again)
    let index_metadata = {
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

        if let Some(fragment_ids) = fragment_ids {
            index_builder = index_builder.fragments(fragment_ids);
        }

        if let Some(index_uuid) = index_uuid {
            index_builder = index_builder.index_uuid(index_uuid);
        }

        if let Some(reader) = batch_reader {
            index_builder = index_builder.preprocessed_data(Box::new(reader));
        }

        if skip_commit {
            RT.block_on(index_builder.execute_uncommitted())?
        } else {
            RT.block_on(index_builder.into_future())?
        }
    };

    (&index_metadata).into_java(env)
}

fn should_skip_commit(index_type: IndexType, params_opt: &Option<String>) -> Result<bool> {
    match index_type {
        IndexType::BTree => {
            // Should defer the commit if we are building range-based BTree index
            if let Some(params) = params_opt {
                let btree_parameters = serde_json::from_str::<BTreeParameters>(params)?;
                return Ok(btree_parameters.range_id.is_some());
            }
            Ok(false)
        }
        _ => Ok(false),
    }
}

#[no_mangle]
pub extern "system" fn Java_org_lance_Dataset_innerMergeIndexMetadata<'local>(
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
        dataset_guard
            .inner
            .merge_index_metadata(&index_uuid, index_type, batch_readhead)
            .await
    })?;
    Ok(())
}

#[no_mangle]
pub extern "system" fn Java_org_lance_Dataset_nativeOptimizeIndices(
    mut env: JNIEnv,
    java_dataset: JObject,
    options_obj: JObject, // OptimizeOptions
) {
    ok_or_throw_without_return!(
        env,
        inner_optimize_indices(&mut env, java_dataset, options_obj)
    );
}

fn inner_optimize_indices(
    env: &mut JNIEnv,
    java_dataset: JObject,
    java_options: JObject, // OptimizeOptions
) -> Result<()> {
    let mut options = OptimizeOptions::default();

    if !java_options.is_null() {
        options.num_indices_to_merge =
            env.get_optional_usize_from_method(&java_options, "getNumIndicesToMerge")?;

        // getIndexNames(): Optional<List<String>>
        let index_names_obj = env
            .call_method(
                &java_options,
                "getIndexNames",
                "()Ljava/util/Optional;",
                &[],
            )?
            .l()?;
        let index_names = env.get_strings_opt(&index_names_obj)?;
        options.index_names = index_names;

        // isRetrain(): boolean
        let retrain = env
            .call_method(&java_options, "isRetrain", "()Z", &[])?
            .z()?;
        options.retrain = retrain;
    }

    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    RT.block_on(dataset_guard.inner.optimize_indices(&options))?;
    Ok(())
}

//////////////////
// Read Methods //
//////////////////
#[no_mangle]
pub extern "system" fn Java_org_lance_Dataset_openNative<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    path: JString,
    version_obj: JObject,    // Optional<Long>
    block_size_obj: JObject, // Optional<Integer>
    index_cache_size_bytes: jlong,
    metadata_cache_size_bytes: jlong,
    storage_options_obj: JObject,          // Map<String, String>
    serialized_manifest: JObject,          // Optional<ByteBuffer>
    storage_options_provider_obj: JObject, // Optional<StorageOptionsProvider>
    session_handle: jlong,                 // Session handle, 0 means no session
    namespace_obj: JObject,                // LanceNamespace object, null if no namespace
    table_id_obj: JObject,                 // List<String>, null if no namespace
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
            session_handle,
            namespace_obj,
            table_id_obj,
        )
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_open_native<'local>(
    env: &mut JNIEnv<'local>,
    path: JString,
    version_obj: JObject,    // Optional<Long>
    block_size_obj: JObject, // Optional<Integer>
    index_cache_size_bytes: jlong,
    metadata_cache_size_bytes: jlong,
    storage_options_obj: JObject,          // Map<String, String>
    serialized_manifest: JObject,          // Optional<ByteBuffer>
    storage_options_provider_obj: JObject, // Optional<StorageOptionsProvider>
    session_handle: jlong,                 // Session handle, 0 means no session
    namespace_obj: JObject,                // LanceNamespace object, null if no namespace
    table_id_obj: JObject,                 // List<String>, null if no namespace
) -> Result<JObject<'local>> {
    let path_str: String = path.extract(env)?;
    let version = env.get_u64_opt(&version_obj)?;
    let block_size = env.get_int_opt(&block_size_obj)?;
    let jmap = JMap::from_env(env, &storage_options_obj)?;
    let storage_options = to_rust_map(env, &jmap)?;

    // Extract storage options provider first (before get_bytes_opt which borrows env)
    let storage_options_provider = env
        .get_optional(&storage_options_provider_obj, |env, provider_obj| {
            JavaStorageOptionsProvider::new(env, provider_obj)
        })?;

    let storage_options_provider_arc =
        storage_options_provider.map(|v| Arc::new(v) as Arc<dyn StorageOptionsProvider>);

    // Extract namespace and table_id if provided (before get_bytes_opt which holds borrow)
    let namespace_info = extract_namespace_info(env, &namespace_obj, &table_id_obj)?;
    let (namespace, table_id) = match namespace_info {
        Some((ns, tid)) => (Some(ns), Some(tid)),
        None => (None, None),
    };

    let serialized_manifest = env.get_bytes_opt(&serialized_manifest)?;

    // Convert session handle to Arc<LanceSession> if provided
    let session = session_from_handle(session_handle);

    let dataset = BlockingDataset::open(
        &path_str,
        version,
        block_size,
        index_cache_size_bytes,
        metadata_cache_size_bytes,
        storage_options,
        serialized_manifest,
        storage_options_provider_arc,
        session,
        namespace,
        table_id,
    )?;
    dataset.into_java(env)
}

/// Check if the Java object is an instance of DirectoryNamespace.
fn is_directory_namespace(env: &mut JNIEnv, namespace_obj: &JObject) -> Result<bool> {
    let class = env
        .find_class("org/lance/namespace/DirectoryNamespace")
        .map_err(|e| {
            Error::runtime_error(format!("Failed to find DirectoryNamespace class: {}", e))
        })?;
    env.is_instance_of(namespace_obj, class)
        .map_err(|e| Error::runtime_error(format!("Failed to check instanceof: {}", e)))
}

/// Check if the Java object is an instance of RestNamespace.
fn is_rest_namespace(env: &mut JNIEnv, namespace_obj: &JObject) -> Result<bool> {
    let class = env
        .find_class("org/lance/namespace/RestNamespace")
        .map_err(|e| Error::runtime_error(format!("Failed to find RestNamespace class: {}", e)))?;
    env.is_instance_of(namespace_obj, class)
        .map_err(|e| Error::runtime_error(format!("Failed to check instanceof: {}", e)))
}

/// Get the native handle from a Java LanceNamespace object.
fn get_native_namespace_handle(env: &mut JNIEnv, namespace_obj: &JObject) -> Result<jlong> {
    env.call_method(namespace_obj, "getNativeHandle", "()J", &[])
        .map_err(|e| Error::runtime_error(format!("Failed to call getNativeHandle: {}", e)))?
        .j()
        .map_err(|e| Error::runtime_error(format!("getNativeHandle did not return a long: {}", e)))
}

/// Extract namespace and table_id from Java objects into Rust types.
///
/// Returns `None` if `namespace_obj` is null, otherwise returns the namespace
/// and table_id pair.
#[allow(clippy::type_complexity)]
pub(crate) fn extract_namespace_info(
    env: &mut JNIEnv,
    namespace_obj: &JObject,
    table_id_obj: &JObject,
) -> Result<Option<(Arc<dyn LanceNamespace>, Vec<String>)>> {
    if namespace_obj.is_null() {
        return Ok(None);
    }

    let namespace: Arc<dyn LanceNamespace> = if is_directory_namespace(env, namespace_obj)? {
        let native_handle = get_native_namespace_handle(env, namespace_obj)?;
        let ns = unsafe { &*(native_handle as *const BlockingDirectoryNamespace) };
        ns.inner.clone()
    } else if is_rest_namespace(env, namespace_obj)? {
        let native_handle = get_native_namespace_handle(env, namespace_obj)?;
        let ns = unsafe { &*(native_handle as *const BlockingRestNamespace) };
        ns.inner.clone()
    } else {
        create_java_lance_namespace(env, namespace_obj)?
    };

    let table_id = env.get_strings(table_id_obj)?;
    Ok(Some((namespace, table_id)))
}

#[no_mangle]
pub extern "system" fn Java_org_lance_Dataset_getFragmentsNative<'a>(
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
pub extern "system" fn Java_org_lance_Dataset_getFragmentNative<'a>(
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
pub extern "system" fn Java_org_lance_Dataset_nativeGetLanceSchema<'local>(
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
pub extern "system" fn Java_org_lance_Dataset_importFfiSchema(
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
pub extern "system" fn Java_org_lance_Dataset_nativeUri<'local>(
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
pub extern "system" fn Java_org_lance_Dataset_nativeListVersions<'local>(
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
pub extern "system" fn Java_org_lance_Dataset_nativeGetVersion<'local>(
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
pub extern "system" fn Java_org_lance_Dataset_nativeGetLatestVersionId(
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
pub extern "system" fn Java_org_lance_Dataset_nativeGetInitialStorageOptions<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_get_initial_storage_options(&mut env, java_dataset)
    )
}

fn inner_get_initial_storage_options<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let storage_options = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.initial_storage_options()
    };
    match storage_options {
        Some(opts) => opts.into_java(env),
        None => Ok(JObject::null()),
    }
}

#[no_mangle]
pub extern "system" fn Java_org_lance_Dataset_nativeGetLatestStorageOptions<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_get_latest_storage_options(&mut env, java_dataset)
    )
}

fn inner_get_latest_storage_options<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let storage_options = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.latest_storage_options()?
    };
    match storage_options {
        Some(opts) => opts.into_java(env),
        None => Ok(JObject::null()),
    }
}

#[no_mangle]
pub extern "system" fn Java_org_lance_Dataset_nativeCheckoutLatest(
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
pub extern "system" fn Java_org_lance_Dataset_nativeCheckoutVersion<'local>(
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
pub extern "system" fn Java_org_lance_Dataset_nativeCheckoutTag<'local>(
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
pub extern "system" fn Java_org_lance_Dataset_nativeRestore(
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
pub extern "system" fn Java_org_lance_Dataset_nativeShallowClone<'local>(
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
    jref: JObject,
    storage_options: JObject,
) -> Result<JObject<'local>> {
    let target_path_str = target_path.extract(env)?;
    let reference = transform_jref_to_ref(jref, env)?;
    let storage_opts = transform_jstorage_options(storage_options, env)?;
    let new_ds = {
        let mut dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        RT.block_on(dataset_guard.inner.shallow_clone(
            target_path_str.as_str(),
            reference,
            storage_opts,
        ))?
    };

    BlockingDataset { inner: new_ds }.into_java(env)
}

#[no_mangle]
pub extern "system" fn Java_org_lance_Dataset_nativeCountRows(
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
pub extern "system" fn Java_org_lance_Dataset_nativeGetDataStatistics<'local>(
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
    let data_stats = env.new_object("org/lance/ipc/DataStatistics", "()V", &[])?;

    for field in stats.fields {
        let id = field.id as jint;
        let byte_size = field.bytes_on_disk as jlong;
        let filed_jobj = env.new_object(
            "org/lance/ipc/FieldStatistics",
            "(IJ)V",
            &[JValue::Int(id), JValue::Long(byte_size)],
        )?;
        env.call_method(
            &data_stats,
            "addFieldStatistics",
            "(Lorg/lance/ipc/FieldStatistics;)V",
            &[JValue::Object(&filed_jobj)],
        )?;
    }
    Ok(data_stats)
}

#[no_mangle]
pub extern "system" fn Java_org_lance_Dataset_nativeListIndexes<'local>(
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
pub extern "system" fn Java_org_lance_Dataset_nativeGetConfig<'local>(
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
pub extern "system" fn Java_org_lance_Dataset_nativeTake(
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
pub extern "system" fn Java_org_lance_Dataset_nativeDelete(
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

#[no_mangle]
pub extern "system" fn Java_org_lance_Dataset_nativeTruncateTable(
    mut env: JNIEnv,
    java_dataset: JObject,
) {
    ok_or_throw_without_return!(env, inner_truncate_table(&mut env, java_dataset))
}

fn inner_truncate_table(env: &mut JNIEnv, java_dataset: JObject) -> Result<()> {
    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    RT.block_on(dataset_guard.inner.truncate_table())?;
    Ok(())
}

//////////////////////////////
// Schema evolution Methods //
//////////////////////////////
#[no_mangle]
pub extern "system" fn Java_org_lance_Dataset_nativeDropColumns(
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
pub extern "system" fn Java_org_lance_Dataset_nativeAlterColumns(
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
pub extern "system" fn Java_org_lance_Dataset_nativeAddColumnsBySqlExpressions(
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

    let batch_size = match env.get_long_opt(&batch_size)? {
        Some(value) => Some(
            value
                .try_into()
                .map_err(|_| Error::input_error("Batch size conversion error".to_string()))?,
        ),
        None => None,
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
pub extern "system" fn Java_org_lance_Dataset_nativeAddColumnsByReader(
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

    let batch_size = match env.get_long_opt(&batch_size)? {
        Some(value) => Some(
            value
                .try_into()
                .map_err(|_| Error::input_error("Batch size conversion error".to_string()))?,
        ),
        None => None,
    };

    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;

    RT.block_on(dataset_guard.inner.add_columns(transform, None, batch_size))?;

    Ok(())
}

#[no_mangle]
pub extern "system" fn Java_org_lance_Dataset_nativeAddColumnsBySchema(
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
pub extern "system" fn Java_org_lance_Dataset_nativeListTags<'local>(
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
        let branch_name: JObject = if let Some(branch_name) = tag_contents.branch.as_ref() {
            env.new_string(branch_name)?.into()
        } else {
            JObject::null()
        };
        let java_tag = env.new_object(
            "org/lance/Tag",
            "(Ljava/lang/String;Ljava/lang/String;JI)V",
            &[
                JValue::Object(&env.new_string(tag_name)?.into()),
                JValue::Object(&branch_name),
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
pub extern "system" fn Java_org_lance_Dataset_nativeCreateTag(
    mut env: JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
    jref: JObject,
) {
    ok_or_throw_without_return!(
        env,
        inner_create_tag(&mut env, java_dataset, jtag_name, jref)
    )
}

fn inner_create_tag(
    env: &mut JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
    jref: JObject,
) -> Result<()> {
    let tag = jtag_name.extract(env)?;
    let reference = transform_jref_to_ref(jref, env)?;
    let mut dataset_guard =
        { unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }? };
    dataset_guard.create_tag(tag.as_str(), reference)?;
    Ok(())
}

#[no_mangle]
pub extern "system" fn Java_org_lance_Dataset_nativeDeleteTag(
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
pub extern "system" fn Java_org_lance_Dataset_nativeUpdateTag(
    mut env: JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
    jref: JObject,
) {
    ok_or_throw_without_return!(
        env,
        inner_update_tag(&mut env, java_dataset, jtag_name, jref)
    )
}

fn inner_update_tag(
    env: &mut JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
    jref: JObject,
) -> Result<()> {
    let tag = jtag_name.extract(env)?;
    let reference = transform_jref_to_ref(jref, env)?;
    let mut dataset_guard =
        { unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }? };
    dataset_guard.update_tag(tag.as_str(), reference)
}

#[no_mangle]
pub extern "system" fn Java_org_lance_Dataset_nativeGetVersionByTag(
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
pub extern "system" fn Java_org_lance_Dataset_nativeListBranches<'local>(
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
            "org/lance/Branch",
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
pub extern "system" fn Java_org_lance_Dataset_nativeCreateBranch<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    jbranch: JString,
    jref: JObject,
    jstorage_options: JObject, // Optional<String>
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_create_branch(&mut env, java_dataset, jbranch, jref, jstorage_options)
    )
}

fn inner_create_branch<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    jbranch: JString,
    jref: JObject,
    jstorage_options: JObject, // Optional<String>
) -> Result<JObject<'local>> {
    let branch_name: String = jbranch.extract(env)?;
    let reference = transform_jref_to_ref(jref, env)?;
    let storage_opts = transform_jstorage_options(jstorage_options, env)?;

    let new_blocking_dataset = {
        let mut dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        let inner = RT.block_on(dataset_guard.inner.create_branch(
            branch_name.as_str(),
            reference,
            storage_opts,
        ))?;
        BlockingDataset { inner }
    };
    new_blocking_dataset.into_java(env)
}

fn transform_jref_to_ref(jref: JObject, env: &mut JNIEnv) -> Result<Ref> {
    let source_tag_name = env.get_optional_string_from_method(&jref, "getTagName")?;
    let source_version_number = env.get_optional_u64_from_method(&jref, "getVersionNumber")?;
    let source_branch = env.get_optional_string_from_method(&jref, "getBranchName")?;
    if let Some(tag_name) = source_tag_name {
        Ok(Ref::Tag(tag_name))
    } else {
        Ok(Ref::Version(source_branch, source_version_number))
    }
}

fn transform_jstorage_options(
    jstorage_options: JObject,
    env: &mut JNIEnv,
) -> Result<Option<ObjectStoreParams>> {
    let storage_options = env.get_optional(&jstorage_options, |env, map_obj| {
        let jmap = JMap::from_env(env, &map_obj)?;
        to_rust_map(env, &jmap)
    })?;
    Ok(storage_options
        .map(|options| {
            Some(ObjectStoreParams {
                storage_options_accessor: Some(Arc::new(
                    lance::io::StorageOptionsAccessor::with_static_options(options),
                )),
                ..Default::default()
            })
        })
        .unwrap_or(None))
}

#[no_mangle]
pub extern "system" fn Java_org_lance_Dataset_nativeDeleteBranch(
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
pub extern "system" fn Java_org_lance_Dataset_nativeCheckout<'local>(
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
pub extern "system" fn Java_org_lance_Dataset_nativeGetTableMetadata<'local>(
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
pub extern "system" fn Java_org_lance_Dataset_nativeCompact(
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
pub extern "system" fn Java_org_lance_Dataset_nativeCleanupWithPolicy<'local>(
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

    let clean_referenced_branches = env
        .get_optional_from_method(&jpolicy, "getCleanReferencedBranches", |env, obj| {
            Ok(env.call_method(obj, "booleanValue", "()Z", &[])?.z()?)
        })?
        .unwrap_or(false);

    let policy = CleanupPolicy {
        before_timestamp,
        before_version,
        delete_unverified,
        error_if_tagged_old_versions,
        clean_referenced_branches,
    };

    let stats = {
        let mut dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }?;
        dataset.cleanup_with_policy(policy)
    }?;

    let jstats = env.new_object(
        "org/lance/cleanup/RemovalStats",
        "(JJ)V",
        &[
            JValue::Long(stats.bytes_removed as i64),
            JValue::Long(stats.old_versions as i64),
        ],
    )?;

    Ok(jstats)
}

//////////////////////////////
// Index operation Methods   //
//////////////////////////////

#[no_mangle]
pub extern "system" fn Java_org_lance_Dataset_nativeGetIndexes<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_get_indexes(&mut env, java_dataset))
}

fn inner_get_indexes<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let indexes = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.list_indexes()?
    };

    let array_list = env.new_object("java/util/ArrayList", "()V", &[])?;

    for index_meta in indexes.iter() {
        let java_index = index_meta.into_java(env)?;
        env.call_method(
            &array_list,
            "add",
            "(Ljava/lang/Object;)Z",
            &[JValue::Object(&java_index)],
        )?;
    }

    Ok(array_list)
}

#[no_mangle]
pub extern "system" fn Java_org_lance_Dataset_nativeGetIndexStatistics<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    jindex_name: JString,
) -> JString<'local> {
    ok_or_throw_with_return!(
        env,
        inner_get_index_statistics(&mut env, java_dataset, jindex_name),
        JString::from(JObject::null())
    )
}

fn inner_get_index_statistics<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    jindex_name: JString,
) -> Result<JString<'local>> {
    let index_name: String = jindex_name.extract(env)?;
    let stats_json = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        RT.block_on(dataset_guard.inner.index_statistics(&index_name))?
    };
    let jstats = env.new_string(stats_json)?;
    Ok(jstats)
}

#[no_mangle]
pub extern "system" fn Java_org_lance_Dataset_nativeDescribeIndices<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    criteria_obj: JObject,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_describe_indices(&mut env, java_dataset, criteria_obj)
    )
}

fn inner_describe_indices<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    java_index_criteria: JObject,
) -> Result<JObject<'local>> {
    let mut for_column = None;
    let mut has_name = None;
    let index_criteria = env.get_optional(&java_index_criteria, |env, obj| {
        for_column = env.get_optional_string_from_method(&obj, "getForColumn")?;
        has_name = env.get_optional_string_from_method(&obj, "getHasName")?;
        let must_support_fts = env.get_boolean_from_method(&obj, "mustSupportFts")?;
        let must_support_exact_equality =
            env.get_boolean_from_method(&obj, "mustSupportExactEquality")?;
        Ok(RustIndexCriteria {
            for_column: for_column.as_deref(),
            has_name: has_name.as_deref(),
            must_support_fts,
            must_support_exact_equality,
        })
    })?;

    let descriptions = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        RT.block_on(dataset_guard.inner.describe_indices(index_criteria))?
    };

    export_vec(env, &descriptions)
}

#[no_mangle]
pub extern "system" fn Java_org_lance_Dataset_nativeCountIndexedRows(
    mut env: JNIEnv,
    java_dataset: JObject,
    jindex_name: JString,
    jfilter: JString,
    jfragment_ids: JObject, // Optional<List<Integer>>
) -> jlong {
    ok_or_throw_with_return!(
        env,
        inner_count_indexed_rows(&mut env, java_dataset, jindex_name, jfilter, jfragment_ids),
        -1
    )
}

fn inner_count_indexed_rows(
    env: &mut JNIEnv,
    java_dataset: JObject,
    _jindex_name: JString,
    jfilter: JString,
    jfragment_ids: JObject, // Optional<List<Integer>>
) -> Result<i64> {
    let filter: String = jfilter.extract(env)?;

    // Extract optional fragment IDs
    let fragment_ids: Option<Vec<u32>> = if env
        .call_method(&jfragment_ids, "isPresent", "()Z", &[])?
        .z()?
    {
        let list_obj = env
            .call_method(&jfragment_ids, "get", "()Ljava/lang/Object;", &[])?
            .l()?;
        let list = env.get_list(&list_obj)?;
        let mut ids = Vec::new();
        let mut iter = list.iter(env)?;
        while let Some(elem) = iter.next(env)? {
            let int_val = env.call_method(&elem, "intValue", "()I", &[])?.i()?;
            ids.push(int_val as u32);
        }
        Some(ids)
    } else {
        None
    };

    let count = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;

        // Use a scanner with fragment filtering to count rows
        // This ensures we only count rows in the specified fragments
        let inner = dataset_guard.inner.clone();

        RT.block_on(async {
            let mut scanner = inner.scan();

            // Apply filter
            if !filter.is_empty() {
                scanner.filter(&filter)?;
            }

            // Empty projection and enable row_id for count_rows to work
            // count_rows() requires metadata-only projection
            scanner.project::<String>(&[])?;
            scanner.with_row_id();

            // Apply fragment filter if specified
            if let Some(frag_ids) = fragment_ids {
                // Convert FileFragment to Fragment by extracting metadata
                let filtered_fragments: Vec<_> = inner
                    .get_fragments()
                    .into_iter()
                    .filter(|f| frag_ids.contains(&(f.id() as u32)))
                    .map(|f| f.metadata().clone())
                    .collect();
                scanner.with_fragments(filtered_fragments);
            }

            // Use the scanner's count_rows method
            let count = scanner.count_rows().await?;

            Ok::<i64, lance::Error>(count as i64)
        })?
    };

    Ok(count)
}

//////////////////////////////
// Session Methods          //
//////////////////////////////

/// Returns the session handle from a dataset.
/// The returned handle can be used to create a Java Session object.
#[no_mangle]
pub extern "system" fn Java_org_lance_Dataset_nativeGetSessionHandle(
    mut env: JNIEnv,
    java_dataset: JObject,
) -> jlong {
    ok_or_throw_with_return!(env, inner_get_session_handle(&mut env, java_dataset), 0)
}

fn inner_get_session_handle(env: &mut JNIEnv, java_dataset: JObject) -> Result<jlong> {
    let dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    let session = dataset_guard.inner.session();
    Ok(handle_from_session(session))
}
