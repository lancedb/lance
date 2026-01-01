// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Manifest-based namespace implementation
//!
//! This module provides a namespace implementation that uses a manifest table
//! to track tables and nested namespaces.

use arrow::array::{Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow_ipc::reader::StreamReader;
use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::StreamExt;
use lance::dataset::optimize::{compact_files, CompactionOptions};
use lance::dataset::{builder::DatasetBuilder, WriteParams};
use lance::session::Session;
use lance::{dataset::scanner::Scanner, Dataset};
use lance_core::{box_error, Error, Result};
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
use lance_index::traits::DatasetIndexExt;
use lance_index::IndexType;
use lance_io::object_store::{ObjectStore, ObjectStoreParams};
use lance_namespace::models::{
    CreateEmptyTableRequest, CreateEmptyTableResponse, CreateNamespaceRequest,
    CreateNamespaceResponse, CreateTableRequest, CreateTableResponse, DeclareTableRequest,
    DeclareTableResponse, DeregisterTableRequest, DeregisterTableResponse,
    DescribeNamespaceRequest, DescribeNamespaceResponse, DescribeTableRequest,
    DescribeTableResponse, DropNamespaceRequest, DropNamespaceResponse, DropTableRequest,
    DropTableResponse, ListNamespacesRequest, ListNamespacesResponse, ListTablesRequest,
    ListTablesResponse, NamespaceExistsRequest, RegisterTableRequest, RegisterTableResponse,
    TableExistsRequest,
};
use lance_namespace::schema::arrow_schema_to_json;
use lance_namespace::LanceNamespace;
use object_store::path::Path;
use snafu::location;
use std::io::Cursor;
use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
    ops::{Deref, DerefMut},
    sync::Arc,
};
use tokio::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

const MANIFEST_TABLE_NAME: &str = "__manifest";
const DELIMITER: &str = "$";

// Index names for the __manifest table
/// BTREE index on the object_id column for fast lookups
const OBJECT_ID_INDEX_NAME: &str = "object_id_btree";
/// Bitmap index on the object_type column for filtering by type
const OBJECT_TYPE_INDEX_NAME: &str = "object_type_bitmap";
/// LabelList index on the base_objects column for view dependencies
const BASE_OBJECTS_INDEX_NAME: &str = "base_objects_label_list";

/// Object types that can be stored in the manifest
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectType {
    Namespace,
    Table,
}

impl ObjectType {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Namespace => "namespace",
            Self::Table => "table",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "namespace" => Ok(Self::Namespace),
            "table" => Ok(Self::Table),
            _ => Err(Error::io(
                format!("Invalid object type: {}", s),
                location!(),
            )),
        }
    }
}

/// Information about a table stored in the manifest
#[derive(Debug, Clone)]
pub struct TableInfo {
    pub namespace: Vec<String>,
    pub name: String,
    pub location: String,
}

/// Information about a namespace stored in the manifest
#[derive(Debug, Clone)]
pub struct NamespaceInfo {
    pub namespace: Vec<String>,
    pub name: String,
    pub metadata: Option<HashMap<String, String>>,
}

/// A wrapper around a Dataset that provides concurrent access.
///
/// This can be cloned cheaply. It supports concurrent reads or exclusive writes.
/// The manifest dataset is always kept strongly consistent by reloading on each read.
#[derive(Debug, Clone)]
pub struct DatasetConsistencyWrapper(Arc<RwLock<Dataset>>);

impl DatasetConsistencyWrapper {
    /// Create a new wrapper with the given dataset.
    pub fn new(dataset: Dataset) -> Self {
        Self(Arc::new(RwLock::new(dataset)))
    }

    /// Get an immutable reference to the dataset.
    /// Always reloads to ensure strong consistency.
    pub async fn get(&self) -> Result<DatasetReadGuard<'_>> {
        self.reload().await?;
        Ok(DatasetReadGuard {
            guard: self.0.read().await,
        })
    }

    /// Get a mutable reference to the dataset.
    /// Always reloads to ensure strong consistency.
    pub async fn get_mut(&self) -> Result<DatasetWriteGuard<'_>> {
        self.reload().await?;
        Ok(DatasetWriteGuard {
            guard: self.0.write().await,
        })
    }

    /// Provide a known latest version of the dataset.
    ///
    /// This is usually done after some write operation, which inherently will
    /// have the latest version.
    pub async fn set_latest(&self, dataset: Dataset) {
        let mut write_guard = self.0.write().await;
        if dataset.manifest().version > write_guard.manifest().version {
            *write_guard = dataset;
        }
    }

    /// Reload the dataset to the latest version.
    async fn reload(&self) -> Result<()> {
        // First check if we need to reload (with read lock)
        let read_guard = self.0.read().await;
        let dataset_uri = read_guard.uri().to_string();
        let current_version = read_guard.version().version;
        log::debug!(
            "Reload starting for uri={}, current_version={}",
            dataset_uri,
            current_version
        );
        let latest_version = read_guard
            .latest_version_id()
            .await
            .map_err(|e| Error::IO {
                source: box_error(std::io::Error::other(format!(
                    "Failed to get latest version: {}",
                    e
                ))),
                location: location!(),
            })?;
        log::debug!(
            "Reload got latest_version={} for uri={}, current_version={}",
            latest_version,
            dataset_uri,
            current_version
        );
        drop(read_guard);

        // If already up-to-date, return early
        if latest_version == current_version {
            log::debug!("Already up-to-date for uri={}", dataset_uri);
            return Ok(());
        }

        // Need to reload, acquire write lock
        let mut write_guard = self.0.write().await;

        // Double-check after acquiring write lock (someone else might have reloaded)
        let latest_version = write_guard
            .latest_version_id()
            .await
            .map_err(|e| Error::IO {
                source: box_error(std::io::Error::other(format!(
                    "Failed to get latest version: {}",
                    e
                ))),
                location: location!(),
            })?;

        if latest_version != write_guard.version().version {
            write_guard.checkout_latest().await.map_err(|e| Error::IO {
                source: box_error(std::io::Error::other(format!(
                    "Failed to checkout latest: {}",
                    e
                ))),
                location: location!(),
            })?;
        }

        Ok(())
    }
}

pub struct DatasetReadGuard<'a> {
    guard: RwLockReadGuard<'a, Dataset>,
}

impl Deref for DatasetReadGuard<'_> {
    type Target = Dataset;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

pub struct DatasetWriteGuard<'a> {
    guard: RwLockWriteGuard<'a, Dataset>,
}

impl Deref for DatasetWriteGuard<'_> {
    type Target = Dataset;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl DerefMut for DatasetWriteGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard
    }
}

/// Manifest-based namespace implementation
///
/// Uses a special `__manifest` Lance table to track tables and nested namespaces.
#[derive(Debug)]
pub struct ManifestNamespace {
    root: String,
    storage_options: Option<HashMap<String, String>>,
    #[allow(dead_code)]
    session: Option<Arc<Session>>,
    #[allow(dead_code)]
    object_store: Arc<ObjectStore>,
    #[allow(dead_code)]
    base_path: Path,
    manifest_dataset: DatasetConsistencyWrapper,
    /// Whether directory listing is enabled in dual mode
    /// If true, root namespace tables use {table_name}.lance naming
    /// If false, they use namespace-prefixed names
    dir_listing_enabled: bool,
    /// Whether to perform inline optimization (compaction and indexing) on the __manifest table
    /// after every write. Defaults to true.
    inline_optimization_enabled: bool,
}

impl ManifestNamespace {
    /// Create a new ManifestNamespace from an existing DirectoryNamespace
    pub async fn from_directory(
        root: String,
        storage_options: Option<HashMap<String, String>>,
        session: Option<Arc<Session>>,
        object_store: Arc<ObjectStore>,
        base_path: Path,
        dir_listing_enabled: bool,
        inline_optimization_enabled: bool,
    ) -> Result<Self> {
        let manifest_dataset =
            Self::create_or_get_manifest(&root, &storage_options, session.clone()).await?;

        Ok(Self {
            root,
            storage_options,
            session,
            object_store,
            base_path,
            manifest_dataset,
            dir_listing_enabled,
            inline_optimization_enabled,
        })
    }

    /// Build object ID from namespace path and name
    pub fn build_object_id(namespace: &[String], name: &str) -> String {
        if namespace.is_empty() {
            name.to_string()
        } else {
            let mut id = namespace.join(DELIMITER);
            id.push_str(DELIMITER);
            id.push_str(name);
            id
        }
    }

    /// Parse object ID into namespace path and name
    pub fn parse_object_id(object_id: &str) -> (Vec<String>, String) {
        let parts: Vec<&str> = object_id.split(DELIMITER).collect();
        if parts.len() == 1 {
            (Vec::new(), parts[0].to_string())
        } else {
            let namespace = parts[..parts.len() - 1]
                .iter()
                .map(|s| s.to_string())
                .collect();
            let name = parts[parts.len() - 1].to_string();
            (namespace, name)
        }
    }

    /// Split an object ID (table_id as vec of strings) into namespace and table name
    fn split_object_id(table_id: &[String]) -> (Vec<String>, String) {
        if table_id.len() == 1 {
            (vec![], table_id[0].clone())
        } else {
            (
                table_id[..table_id.len() - 1].to_vec(),
                table_id[table_id.len() - 1].clone(),
            )
        }
    }

    /// Convert a table ID (vec of strings) to an object_id string
    fn str_object_id(table_id: &[String]) -> String {
        table_id.join(DELIMITER)
    }

    /// Generate a new directory name in format: <hash>_<object_id>
    /// The hash is used to (1) optimize object store throughput,
    /// (2) have high enough entropy in a short period of time to prevent issues like
    /// failed table creation, delete and create new table of the same name, etc.
    /// The object_id is added after the hash to ensure
    /// dir name uniqueness and make debugging easier.
    fn generate_dir_name(object_id: &str) -> String {
        // Generate a random number for uniqueness
        let random_num: u64 = rand::random();

        // Create hash from random number + object_id
        let mut hasher = DefaultHasher::new();
        random_num.hash(&mut hasher);
        object_id.hash(&mut hasher);
        let hash = hasher.finish();

        // Format as lowercase hex (8 characters - sufficient entropy for uniqueness)
        format!("{:08x}_{}", (hash & 0xFFFFFFFF) as u32, object_id)
    }

    /// Construct a full URI from root and relative location
    pub(crate) fn construct_full_uri(root: &str, relative_location: &str) -> Result<String> {
        let mut base_url = lance_io::object_store::uri_to_url(root)?;

        // Ensure the base URL has a trailing slash so that URL.join() appends
        // rather than replaces the last path segment.
        // Without this fix, "s3://bucket/path/subdir".join("table.lance")
        // would incorrectly produce "s3://bucket/path/table.lance" (missing subdir).
        if !base_url.path().ends_with('/') {
            base_url.set_path(&format!("{}/", base_url.path()));
        }

        let full_url = base_url
            .join(relative_location)
            .map_err(|e| Error::InvalidInput {
                source: format!(
                    "Failed to join URI '{}' with '{}': {:?}",
                    root, relative_location, e
                )
                .into(),
                location: location!(),
            })?;

        Ok(full_url.to_string())
    }

    /// Perform inline optimization on the __manifest table.
    ///
    /// This method:
    /// 1. Creates three indexes on the manifest table:
    ///    - BTREE index on object_id for fast lookups
    ///    - Bitmap index on object_type for filtering by type
    ///    - LabelList index on base_objects for view dependencies
    /// 2. Runs file compaction to merge small files
    /// 3. Optimizes existing indices
    ///
    /// This is called automatically after writes when inline_optimization_enabled is true.
    async fn run_inline_optimization(&self) -> Result<()> {
        if !self.inline_optimization_enabled {
            return Ok(());
        }

        // Get a mutable reference to the dataset to perform optimization
        let mut dataset_guard = self.manifest_dataset.get_mut().await?;
        let dataset: &mut Dataset = &mut dataset_guard;

        // Step 1: Create indexes if they don't already exist
        let indices = dataset.load_indices().await?;

        // Check which indexes already exist
        let has_object_id_index = indices.iter().any(|idx| idx.name == OBJECT_ID_INDEX_NAME);
        let has_object_type_index = indices.iter().any(|idx| idx.name == OBJECT_TYPE_INDEX_NAME);
        let has_base_objects_index = indices
            .iter()
            .any(|idx| idx.name == BASE_OBJECTS_INDEX_NAME);

        // Create BTREE index on object_id
        if !has_object_id_index {
            log::debug!(
                "Creating BTREE index '{}' on object_id for __manifest table",
                OBJECT_ID_INDEX_NAME
            );
            let params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
            if let Err(e) = dataset
                .create_index(
                    &["object_id"],
                    IndexType::BTree,
                    Some(OBJECT_ID_INDEX_NAME.to_string()),
                    &params,
                    true,
                )
                .await
            {
                log::warn!("Failed to create BTREE index on object_id for __manifest table: {:?}. Query performance may be impacted.", e);
            } else {
                log::info!(
                    "Created BTREE index '{}' on object_id for __manifest table",
                    OBJECT_ID_INDEX_NAME
                );
            }
        }

        // Create Bitmap index on object_type
        if !has_object_type_index {
            log::debug!(
                "Creating Bitmap index '{}' on object_type for __manifest table",
                OBJECT_TYPE_INDEX_NAME
            );
            let params = ScalarIndexParams::default();
            if let Err(e) = dataset
                .create_index(
                    &["object_type"],
                    IndexType::Bitmap,
                    Some(OBJECT_TYPE_INDEX_NAME.to_string()),
                    &params,
                    true,
                )
                .await
            {
                log::warn!("Failed to create Bitmap index on object_type for __manifest table: {:?}. Query performance may be impacted.", e);
            } else {
                log::info!(
                    "Created Bitmap index '{}' on object_type for __manifest table",
                    OBJECT_TYPE_INDEX_NAME
                );
            }
        }

        // Create LabelList index on base_objects
        if !has_base_objects_index {
            log::debug!(
                "Creating LabelList index '{}' on base_objects for __manifest table",
                BASE_OBJECTS_INDEX_NAME
            );
            let params = ScalarIndexParams::default();
            if let Err(e) = dataset
                .create_index(
                    &["base_objects"],
                    IndexType::LabelList,
                    Some(BASE_OBJECTS_INDEX_NAME.to_string()),
                    &params,
                    true,
                )
                .await
            {
                log::warn!("Failed to create LabelList index on base_objects for __manifest table: {:?}. Query performance may be impacted.", e);
            } else {
                log::info!(
                    "Created LabelList index '{}' on base_objects for __manifest table",
                    BASE_OBJECTS_INDEX_NAME
                );
            }
        }

        // Step 2: Run file compaction
        log::debug!("Running file compaction on __manifest table");
        match compact_files(dataset, CompactionOptions::default(), None).await {
            Ok(compaction_metrics) => {
                if compaction_metrics.fragments_removed > 0 {
                    log::info!(
                        "Compacted __manifest table: removed {} fragments, added {} fragments",
                        compaction_metrics.fragments_removed,
                        compaction_metrics.fragments_added
                    );
                }
            }
            Err(e) => {
                log::warn!("Failed to compact files for __manifest table: {:?}. Continuing with optimization.", e);
            }
        }

        // Step 3: Optimize indices
        log::debug!("Optimizing indices on __manifest table");
        match dataset.optimize_indices(&OptimizeOptions::default()).await {
            Ok(_) => {
                log::info!("Successfully optimized indices on __manifest table");
            }
            Err(e) => {
                log::warn!(
                    "Failed to optimize indices on __manifest table: {:?}. Continuing anyway.",
                    e
                );
            }
        }

        Ok(())
    }

    /// Get the manifest schema
    fn manifest_schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            Field::new("object_id", DataType::Utf8, false),
            Field::new("object_type", DataType::Utf8, false),
            Field::new("location", DataType::Utf8, true),
            Field::new("metadata", DataType::Utf8, true),
            Field::new(
                "base_objects",
                DataType::List(Arc::new(Field::new("object_id", DataType::Utf8, true))),
                true,
            ),
        ]))
    }

    /// Get a scanner for the manifest dataset
    async fn manifest_scanner(&self) -> Result<Scanner> {
        let dataset_guard = self.manifest_dataset.get().await?;
        Ok(dataset_guard.scan())
    }

    /// Helper to execute a scanner and collect results into a Vec
    async fn execute_scanner(scanner: Scanner) -> Result<Vec<RecordBatch>> {
        let mut stream = scanner.try_into_stream().await.map_err(|e| Error::IO {
            source: box_error(std::io::Error::other(format!(
                "Failed to create stream: {}",
                e
            ))),
            location: location!(),
        })?;

        let mut batches = Vec::new();
        while let Some(batch) = stream.next().await {
            batches.push(batch.map_err(|e| Error::IO {
                source: box_error(std::io::Error::other(format!(
                    "Failed to read batch: {}",
                    e
                ))),
                location: location!(),
            })?);
        }

        Ok(batches)
    }

    /// Helper to get a string column from a record batch
    fn get_string_column<'a>(batch: &'a RecordBatch, column_name: &str) -> Result<&'a StringArray> {
        let column = batch
            .column_by_name(column_name)
            .ok_or_else(|| Error::io(format!("Column '{}' not found", column_name), location!()))?;
        column
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                Error::io(
                    format!("Column '{}' is not a string array", column_name),
                    location!(),
                )
            })
    }

    /// Check if the manifest contains an object with the given ID
    async fn manifest_contains_object(&self, object_id: &str) -> Result<bool> {
        let filter = format!("object_id = '{}'", object_id);

        let dataset_guard = self.manifest_dataset.get().await?;
        let mut scanner = dataset_guard.scan();

        scanner.filter(&filter).map_err(|e| Error::IO {
            source: box_error(std::io::Error::other(format!("Failed to filter: {}", e))),
            location: location!(),
        })?;

        // Project no columns and enable row IDs for count_rows to work
        scanner.project::<&str>(&[]).map_err(|e| Error::IO {
            source: box_error(std::io::Error::other(format!("Failed to project: {}", e))),
            location: location!(),
        })?;

        scanner.with_row_id();

        let count = scanner.count_rows().await.map_err(|e| Error::IO {
            source: box_error(std::io::Error::other(format!(
                "Failed to count rows: {}",
                e
            ))),
            location: location!(),
        })?;

        Ok(count > 0)
    }

    /// Query the manifest for a table with the given object ID
    async fn query_manifest_for_table(&self, object_id: &str) -> Result<Option<TableInfo>> {
        let filter = format!("object_id = '{}' AND object_type = 'table'", object_id);
        let mut scanner = self.manifest_scanner().await?;
        scanner.filter(&filter).map_err(|e| Error::IO {
            source: box_error(std::io::Error::other(format!("Failed to filter: {}", e))),
            location: location!(),
        })?;
        scanner
            .project(&["object_id", "location"])
            .map_err(|e| Error::IO {
                source: box_error(std::io::Error::other(format!("Failed to project: {}", e))),
                location: location!(),
            })?;
        let batches = Self::execute_scanner(scanner).await?;

        let mut found_result: Option<TableInfo> = None;
        let mut total_rows = 0;

        for batch in batches {
            if batch.num_rows() == 0 {
                continue;
            }

            total_rows += batch.num_rows();
            if total_rows > 1 {
                return Err(Error::io(
                    format!(
                        "Expected exactly 1 table with id '{}', found {}",
                        object_id, total_rows
                    ),
                    location!(),
                ));
            }

            let object_id_array = Self::get_string_column(&batch, "object_id")?;
            let location_array = Self::get_string_column(&batch, "location")?;
            let location = location_array.value(0).to_string();
            let (namespace, name) = Self::parse_object_id(object_id_array.value(0));
            found_result = Some(TableInfo {
                namespace,
                name,
                location,
            });
        }

        Ok(found_result)
    }

    /// List all table locations in the manifest (for root namespace only)
    /// Returns a set of table locations (e.g., "table_name.lance")
    pub async fn list_manifest_table_locations(&self) -> Result<std::collections::HashSet<String>> {
        let filter = "object_type = 'table' AND NOT contains(object_id, '$')";
        let mut scanner = self.manifest_scanner().await?;
        scanner.filter(filter).map_err(|e| Error::IO {
            source: box_error(std::io::Error::other(format!("Failed to filter: {}", e))),
            location: location!(),
        })?;
        scanner.project(&["location"]).map_err(|e| Error::IO {
            source: box_error(std::io::Error::other(format!("Failed to project: {}", e))),
            location: location!(),
        })?;

        let batches = Self::execute_scanner(scanner).await?;
        let mut locations = std::collections::HashSet::new();

        for batch in batches {
            if batch.num_rows() == 0 {
                continue;
            }
            let location_array = Self::get_string_column(&batch, "location")?;
            for i in 0..location_array.len() {
                locations.insert(location_array.value(i).to_string());
            }
        }

        Ok(locations)
    }

    /// Insert an entry into the manifest table
    async fn insert_into_manifest(
        &self,
        object_id: String,
        object_type: ObjectType,
        location: Option<String>,
    ) -> Result<()> {
        self.insert_into_manifest_with_metadata(object_id, object_type, location, None, None)
            .await
    }

    /// Insert an entry into the manifest table with metadata and base_objects
    async fn insert_into_manifest_with_metadata(
        &self,
        object_id: String,
        object_type: ObjectType,
        location: Option<String>,
        metadata: Option<String>,
        base_objects: Option<Vec<String>>,
    ) -> Result<()> {
        use arrow::array::builder::{ListBuilder, StringBuilder};

        let schema = Self::manifest_schema();

        // Create base_objects array from the provided list
        let string_builder = StringBuilder::new();
        let mut list_builder = ListBuilder::new(string_builder).with_field(Arc::new(Field::new(
            "object_id",
            DataType::Utf8,
            true,
        )));

        match base_objects {
            Some(objects) => {
                for obj in objects {
                    list_builder.values().append_value(obj);
                }
                list_builder.append(true);
            }
            None => {
                list_builder.append_null();
            }
        }

        let base_objects_array = list_builder.finish();

        // Create arrays with optional values
        let location_array = match location {
            Some(loc) => Arc::new(StringArray::from(vec![Some(loc)])),
            None => Arc::new(StringArray::from(vec![None::<String>])),
        };

        let metadata_array = match metadata {
            Some(meta) => Arc::new(StringArray::from(vec![Some(meta)])),
            None => Arc::new(StringArray::from(vec![None::<String>])),
        };

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![object_id.as_str()])),
                Arc::new(StringArray::from(vec![object_type.as_str()])),
                location_array,
                metadata_array,
                Arc::new(base_objects_array),
            ],
        )
        .map_err(|e| {
            Error::io(
                format!("Failed to create manifest entry: {}", e),
                location!(),
            )
        })?;

        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

        // Use MergeInsert to ensure uniqueness on object_id
        let dataset_guard = self.manifest_dataset.get().await?;
        let dataset_arc = Arc::new(dataset_guard.clone());
        drop(dataset_guard); // Drop read guard before merge insert

        let mut merge_builder =
            lance::dataset::MergeInsertBuilder::try_new(dataset_arc, vec!["object_id".to_string()])
                .map_err(|e| Error::IO {
                    source: box_error(std::io::Error::other(format!(
                        "Failed to create merge builder: {}",
                        e
                    ))),
                    location: location!(),
                })?;

        merge_builder.when_matched(lance::dataset::WhenMatched::Fail);
        merge_builder.when_not_matched(lance::dataset::WhenNotMatched::InsertAll);

        let (new_dataset_arc, _merge_stats) = merge_builder
            .try_build()
            .map_err(|e| Error::IO {
                source: box_error(std::io::Error::other(format!(
                    "Failed to build merge: {}",
                    e
                ))),
                location: location!(),
            })?
            .execute_reader(Box::new(reader))
            .await
            .map_err(|e| {
                // Check if this is a "matched row" error from WhenMatched::Fail
                let error_msg = e.to_string();
                if error_msg.contains("matched")
                    || error_msg.contains("duplicate")
                    || error_msg.contains("already exists")
                {
                    Error::io(
                        format!("Object with id '{}' already exists in manifest", object_id),
                        location!(),
                    )
                } else {
                    Error::IO {
                        source: box_error(std::io::Error::other(format!(
                            "Failed to execute merge: {}",
                            e
                        ))),
                        location: location!(),
                    }
                }
            })?;

        let new_dataset = Arc::try_unwrap(new_dataset_arc).unwrap_or_else(|arc| (*arc).clone());
        self.manifest_dataset.set_latest(new_dataset).await;

        // Run inline optimization after write
        if let Err(e) = self.run_inline_optimization().await {
            log::warn!(
                "Unexpected failure when running inline optimization: {:?}",
                e
            );
        }

        Ok(())
    }

    /// Delete an entry from the manifest table
    pub async fn delete_from_manifest(&self, object_id: &str) -> Result<()> {
        {
            let predicate = format!("object_id = '{}'", object_id);
            let mut dataset_guard = self.manifest_dataset.get_mut().await?;
            dataset_guard
                .delete(&predicate)
                .await
                .map_err(|e| Error::IO {
                    source: box_error(std::io::Error::other(format!("Failed to delete: {}", e))),
                    location: location!(),
                })?;
        } // Drop the guard here

        self.manifest_dataset.reload().await?;

        // Run inline optimization after delete
        if let Err(e) = self.run_inline_optimization().await {
            log::warn!(
                "Unexpected failure when running inline optimization: {:?}",
                e
            );
        }

        Ok(())
    }

    /// Register a table in the manifest without creating the physical table (internal helper for migration)
    pub async fn register_table(&self, name: &str, location: String) -> Result<()> {
        let object_id = Self::build_object_id(&[], name);
        if self.manifest_contains_object(&object_id).await? {
            return Err(Error::io(
                format!("Table '{}' already exists", name),
                location!(),
            ));
        }

        self.insert_into_manifest(object_id, ObjectType::Table, Some(location))
            .await
    }

    /// Validate that all levels of a namespace path exist
    async fn validate_namespace_levels_exist(&self, namespace_path: &[String]) -> Result<()> {
        for i in 1..=namespace_path.len() {
            let partial_path = &namespace_path[..i];
            let object_id = partial_path.join(DELIMITER);
            if !self.manifest_contains_object(&object_id).await? {
                return Err(Error::Namespace {
                    source: format!("Parent namespace '{}' does not exist", object_id).into(),
                    location: location!(),
                });
            }
        }
        Ok(())
    }

    /// Query the manifest for a namespace with the given object ID
    async fn query_manifest_for_namespace(&self, object_id: &str) -> Result<Option<NamespaceInfo>> {
        let filter = format!("object_id = '{}' AND object_type = 'namespace'", object_id);
        let mut scanner = self.manifest_scanner().await?;
        scanner.filter(&filter).map_err(|e| Error::IO {
            source: box_error(std::io::Error::other(format!("Failed to filter: {}", e))),
            location: location!(),
        })?;
        scanner
            .project(&["object_id", "metadata"])
            .map_err(|e| Error::IO {
                source: box_error(std::io::Error::other(format!("Failed to project: {}", e))),
                location: location!(),
            })?;
        let batches = Self::execute_scanner(scanner).await?;

        let mut found_result: Option<NamespaceInfo> = None;
        let mut total_rows = 0;

        for batch in batches {
            if batch.num_rows() == 0 {
                continue;
            }

            total_rows += batch.num_rows();
            if total_rows > 1 {
                return Err(Error::io(
                    format!(
                        "Expected exactly 1 namespace with id '{}', found {}",
                        object_id, total_rows
                    ),
                    location!(),
                ));
            }

            let object_id_array = Self::get_string_column(&batch, "object_id")?;
            let metadata_array = Self::get_string_column(&batch, "metadata")?;

            let object_id_str = object_id_array.value(0);
            let metadata = if !metadata_array.is_null(0) {
                let metadata_str = metadata_array.value(0);
                match serde_json::from_str::<HashMap<String, String>>(metadata_str) {
                    Ok(map) => Some(map),
                    Err(e) => {
                        return Err(Error::io(
                            format!(
                                "Failed to deserialize metadata for namespace '{}': {}",
                                object_id, e
                            ),
                            location!(),
                        ));
                    }
                }
            } else {
                None
            };

            let (namespace, name) = Self::parse_object_id(object_id_str);
            found_result = Some(NamespaceInfo {
                namespace,
                name,
                metadata,
            });
        }

        Ok(found_result)
    }

    /// Create or get the manifest dataset
    async fn create_or_get_manifest(
        root: &str,
        storage_options: &Option<HashMap<String, String>>,
        session: Option<Arc<Session>>,
    ) -> Result<DatasetConsistencyWrapper> {
        let manifest_path = format!("{}/{}", root, MANIFEST_TABLE_NAME);
        log::debug!("Attempting to load manifest from {}", manifest_path);
        let mut builder = DatasetBuilder::from_uri(&manifest_path);

        if let Some(sess) = session.clone() {
            builder = builder.with_session(sess);
        }

        if let Some(opts) = storage_options {
            builder = builder.with_storage_options(opts.clone());
        }

        let dataset_result = builder.load().await;
        if let Ok(dataset) = dataset_result {
            Ok(DatasetConsistencyWrapper::new(dataset))
        } else {
            log::info!("Creating new manifest table at {}", manifest_path);
            let schema = Self::manifest_schema();
            let empty_batch = RecordBatch::new_empty(schema.clone());
            let reader = RecordBatchIterator::new(vec![Ok(empty_batch)], schema.clone());

            let write_params = WriteParams {
                session,
                store_params: storage_options.as_ref().map(|opts| ObjectStoreParams {
                    storage_options: Some(opts.clone()),
                    ..Default::default()
                }),
                ..Default::default()
            };

            let dataset = Dataset::write(Box::new(reader), &manifest_path, Some(write_params))
                .await
                .map_err(|e| Error::IO {
                    source: box_error(std::io::Error::other(format!(
                        "Failed to create manifest dataset: {}",
                        e
                    ))),
                    location: location!(),
                })?;

            log::info!(
                "Successfully created manifest table at {}, version={}, uri={}",
                manifest_path,
                dataset.version().version,
                dataset.uri()
            );
            Ok(DatasetConsistencyWrapper::new(dataset))
        }
    }
}

#[async_trait]
impl LanceNamespace for ManifestNamespace {
    fn namespace_id(&self) -> String {
        self.root.clone()
    }

    async fn list_tables(&self, request: ListTablesRequest) -> Result<ListTablesResponse> {
        let namespace_id = request.id.as_ref().ok_or_else(|| Error::InvalidInput {
            source: "Namespace ID is required".into(),
            location: location!(),
        })?;

        // Build filter to find tables in this namespace
        let filter = if namespace_id.is_empty() {
            // Root namespace: find tables without a namespace prefix
            "object_type = 'table' AND NOT contains(object_id, '$')".to_string()
        } else {
            // Namespaced: find tables that start with namespace$ but have no additional $
            let prefix = namespace_id.join(DELIMITER);
            format!(
                "object_type = 'table' AND starts_with(object_id, '{}{}') AND NOT contains(substring(object_id, {}), '$')",
                prefix, DELIMITER, prefix.len() + 2
            )
        };

        let mut scanner = self.manifest_scanner().await?;
        scanner.filter(&filter).map_err(|e| Error::IO {
            source: box_error(std::io::Error::other(format!("Failed to filter: {}", e))),
            location: location!(),
        })?;
        scanner.project(&["object_id"]).map_err(|e| Error::IO {
            source: box_error(std::io::Error::other(format!("Failed to project: {}", e))),
            location: location!(),
        })?;

        let batches = Self::execute_scanner(scanner).await?;

        let mut tables = Vec::new();
        for batch in batches {
            if batch.num_rows() == 0 {
                continue;
            }

            let object_id_array = Self::get_string_column(&batch, "object_id")?;
            for i in 0..batch.num_rows() {
                let object_id = object_id_array.value(i);
                let (_namespace, name) = Self::parse_object_id(object_id);
                tables.push(name);
            }
        }

        Ok(ListTablesResponse::new(tables))
    }

    async fn describe_table(&self, request: DescribeTableRequest) -> Result<DescribeTableResponse> {
        let table_id = request.id.as_ref().ok_or_else(|| Error::InvalidInput {
            source: "Table ID is required".into(),
            location: location!(),
        })?;

        if table_id.is_empty() {
            return Err(Error::InvalidInput {
                source: "Table ID cannot be empty".into(),
                location: location!(),
            });
        }

        let object_id = Self::str_object_id(table_id);
        let table_info = self.query_manifest_for_table(&object_id).await?;

        // Extract table name and namespace from table_id
        let table_name = table_id.last().cloned().unwrap_or_default();
        let namespace_id: Vec<String> = if table_id.len() > 1 {
            table_id[..table_id.len() - 1].to_vec()
        } else {
            vec![]
        };

        match table_info {
            Some(info) => {
                // Construct full URI from relative location
                let table_uri = Self::construct_full_uri(&self.root, &info.location)?;

                // Try to open the dataset to get version and schema
                match Dataset::open(&table_uri).await {
                    Ok(mut dataset) => {
                        // If a specific version is requested, checkout that version
                        if let Some(requested_version) = request.version {
                            dataset = dataset.checkout_version(requested_version as u64).await?;
                        }

                        let version = dataset.version().version;
                        let lance_schema = dataset.schema();
                        let arrow_schema: arrow_schema::Schema = lance_schema.into();
                        let json_schema = arrow_schema_to_json(&arrow_schema)?;

                        Ok(DescribeTableResponse {
                            table: Some(table_name.clone()),
                            namespace: Some(namespace_id.clone()),
                            version: Some(version as i64),
                            location: Some(table_uri.clone()),
                            table_uri: Some(table_uri),
                            schema: Some(Box::new(json_schema)),
                            storage_options: self.storage_options.clone(),
                            stats: None,
                        })
                    }
                    Err(_) => {
                        // If dataset can't be opened (e.g., empty table), return minimal info
                        Ok(DescribeTableResponse {
                            table: Some(table_name),
                            namespace: Some(namespace_id),
                            version: None,
                            location: Some(table_uri.clone()),
                            table_uri: Some(table_uri),
                            schema: None,
                            storage_options: self.storage_options.clone(),
                            stats: None,
                        })
                    }
                }
            }
            None => Err(Error::Namespace {
                source: format!("Table '{}' not found", object_id).into(),
                location: location!(),
            }),
        }
    }

    async fn table_exists(&self, request: TableExistsRequest) -> Result<()> {
        let table_id = request.id.as_ref().ok_or_else(|| Error::InvalidInput {
            source: "Table ID is required".into(),
            location: location!(),
        })?;

        if table_id.is_empty() {
            return Err(Error::InvalidInput {
                source: "Table ID cannot be empty".into(),
                location: location!(),
            });
        }

        let (namespace, table_name) = Self::split_object_id(table_id);
        let object_id = Self::build_object_id(&namespace, &table_name);
        let exists = self.manifest_contains_object(&object_id).await?;
        if exists {
            Ok(())
        } else {
            Err(Error::Namespace {
                source: format!("Table '{}' not found", table_name).into(),
                location: location!(),
            })
        }
    }

    async fn create_table(
        &self,
        request: CreateTableRequest,
        data: Bytes,
    ) -> Result<CreateTableResponse> {
        let table_id = request.id.as_ref().ok_or_else(|| Error::InvalidInput {
            source: "Table ID is required".into(),
            location: location!(),
        })?;

        if table_id.is_empty() {
            return Err(Error::InvalidInput {
                source: "Table ID cannot be empty".into(),
                location: location!(),
            });
        }

        let (namespace, table_name) = Self::split_object_id(table_id);
        let object_id = Self::build_object_id(&namespace, &table_name);

        // Check if table already exists in manifest
        if self.manifest_contains_object(&object_id).await? {
            return Err(Error::io(
                format!("Table '{}' already exists", table_name),
                location!(),
            ));
        }

        // Create the physical table location with hash-based naming
        // When dir_listing_enabled is true and it's a root table, use directory-style naming: {table_name}.lance
        // Otherwise, use hash-based naming: {hash}_{object_id}
        let dir_name = if namespace.is_empty() && self.dir_listing_enabled {
            // Root table with directory listing enabled: use {table_name}.lance
            format!("{}.lance", table_name)
        } else {
            // Child namespace table or dir listing disabled: use hash-based naming
            Self::generate_dir_name(&object_id)
        };
        let table_uri = Self::construct_full_uri(&self.root, &dir_name)?;

        // Validate that request_data is provided
        if data.is_empty() {
            return Err(Error::Namespace {
                source: "Request data (Arrow IPC stream) is required for create_table".into(),
                location: location!(),
            });
        }

        // Write the data using Lance Dataset
        let cursor = Cursor::new(data.to_vec());
        let stream_reader = StreamReader::try_new(cursor, None)
            .map_err(|e| Error::io(format!("Failed to read IPC stream: {}", e), location!()))?;

        let batches: Vec<RecordBatch> =
            stream_reader
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| Error::io(format!("Failed to collect batches: {}", e), location!()))?;

        if batches.is_empty() {
            return Err(Error::io(
                "No data provided for table creation",
                location!(),
            ));
        }

        let schema = batches[0].schema();
        let batch_results: Vec<std::result::Result<RecordBatch, arrow_schema::ArrowError>> =
            batches.into_iter().map(Ok).collect();
        let reader = RecordBatchIterator::new(batch_results, schema);

        let write_params = WriteParams::default();
        let _dataset = Dataset::write(Box::new(reader), &table_uri, Some(write_params))
            .await
            .map_err(|e| Error::IO {
                source: box_error(std::io::Error::other(format!(
                    "Failed to write dataset: {}",
                    e
                ))),
                location: location!(),
            })?;

        // Register in manifest (store dir_name, not full URI)
        self.insert_into_manifest(object_id, ObjectType::Table, Some(dir_name))
            .await?;

        Ok(CreateTableResponse {
            transaction_id: None,
            version: Some(1),
            location: Some(table_uri),
            storage_options: self.storage_options.clone(),
        })
    }

    async fn drop_table(&self, request: DropTableRequest) -> Result<DropTableResponse> {
        let table_id = request.id.as_ref().ok_or_else(|| Error::InvalidInput {
            source: "Table ID is required".into(),
            location: location!(),
        })?;

        if table_id.is_empty() {
            return Err(Error::InvalidInput {
                source: "Table ID cannot be empty".into(),
                location: location!(),
            });
        }

        let (namespace, table_name) = Self::split_object_id(table_id);
        let object_id = Self::build_object_id(&namespace, &table_name);

        // Query manifest for table location
        let table_info = self.query_manifest_for_table(&object_id).await?;

        match table_info {
            Some(info) => {
                // Delete from manifest first
                self.delete_from_manifest(&object_id).await?;

                // Delete physical data directory using the dir_name from manifest
                let table_path = self.base_path.child(info.location.as_str());
                let table_uri = Self::construct_full_uri(&self.root, &info.location)?;

                // Remove the table directory
                self.object_store
                    .remove_dir_all(table_path)
                    .await
                    .map_err(|e| Error::Namespace {
                        source: format!("Failed to delete table directory: {}", e).into(),
                        location: location!(),
                    })?;

                Ok(DropTableResponse {
                    id: request.id.clone(),
                    location: Some(table_uri),
                    properties: None,
                    transaction_id: None,
                })
            }
            None => Err(Error::Namespace {
                source: format!("Table '{}' not found", table_name).into(),
                location: location!(),
            }),
        }
    }

    async fn list_namespaces(
        &self,
        request: ListNamespacesRequest,
    ) -> Result<ListNamespacesResponse> {
        let parent_namespace = request.id.as_ref().ok_or_else(|| Error::InvalidInput {
            source: "Namespace ID is required".into(),
            location: location!(),
        })?;

        // Build filter to find direct child namespaces
        let filter = if parent_namespace.is_empty() {
            // Root namespace: find all namespaces without a parent
            "object_type = 'namespace' AND NOT contains(object_id, '$')".to_string()
        } else {
            // Non-root: find namespaces that start with parent$ but have no additional $
            let prefix = parent_namespace.join(DELIMITER);
            format!(
                "object_type = 'namespace' AND starts_with(object_id, '{}{}') AND NOT contains(substring(object_id, {}), '$')",
                prefix, DELIMITER, prefix.len() + 2
            )
        };

        let mut scanner = self.manifest_scanner().await?;
        scanner.filter(&filter).map_err(|e| Error::IO {
            source: box_error(std::io::Error::other(format!("Failed to filter: {}", e))),
            location: location!(),
        })?;
        scanner.project(&["object_id"]).map_err(|e| Error::IO {
            source: box_error(std::io::Error::other(format!("Failed to project: {}", e))),
            location: location!(),
        })?;

        let batches = Self::execute_scanner(scanner).await?;
        let mut namespaces = Vec::new();

        for batch in batches {
            if batch.num_rows() == 0 {
                continue;
            }

            let object_id_array = Self::get_string_column(&batch, "object_id")?;
            for i in 0..batch.num_rows() {
                let object_id = object_id_array.value(i);
                let (_namespace, name) = Self::parse_object_id(object_id);
                namespaces.push(name);
            }
        }

        Ok(ListNamespacesResponse::new(namespaces))
    }

    async fn describe_namespace(
        &self,
        request: DescribeNamespaceRequest,
    ) -> Result<DescribeNamespaceResponse> {
        let namespace_id = request.id.as_ref().ok_or_else(|| Error::InvalidInput {
            source: "Namespace ID is required".into(),
            location: location!(),
        })?;

        // Root namespace always exists
        if namespace_id.is_empty() {
            return Ok(DescribeNamespaceResponse {
                properties: Some(HashMap::new()),
            });
        }

        // Check if namespace exists in manifest
        let object_id = namespace_id.join(DELIMITER);
        let namespace_info = self.query_manifest_for_namespace(&object_id).await?;

        match namespace_info {
            Some(info) => Ok(DescribeNamespaceResponse {
                properties: info.metadata,
            }),
            None => Err(Error::Namespace {
                source: format!("Namespace '{}' not found", object_id).into(),
                location: location!(),
            }),
        }
    }

    async fn create_namespace(
        &self,
        request: CreateNamespaceRequest,
    ) -> Result<CreateNamespaceResponse> {
        let namespace_id = request.id.as_ref().ok_or_else(|| Error::InvalidInput {
            source: "Namespace ID is required".into(),
            location: location!(),
        })?;

        // Root namespace always exists and cannot be created
        if namespace_id.is_empty() {
            return Err(Error::Namespace {
                source: "Root namespace already exists and cannot be created".into(),
                location: location!(),
            });
        }

        // Validate parent namespaces exist (but not the namespace being created)
        if namespace_id.len() > 1 {
            self.validate_namespace_levels_exist(&namespace_id[..namespace_id.len() - 1])
                .await?;
        }

        let object_id = namespace_id.join(DELIMITER);
        if self.manifest_contains_object(&object_id).await? {
            return Err(Error::Namespace {
                source: format!("Namespace '{}' already exists", object_id).into(),
                location: location!(),
            });
        }

        // Serialize properties if provided
        let metadata = request.properties.as_ref().and_then(|props| {
            if props.is_empty() {
                None
            } else {
                Some(serde_json::to_string(props).ok()?)
            }
        });

        self.insert_into_manifest_with_metadata(
            object_id,
            ObjectType::Namespace,
            None,
            metadata,
            None,
        )
        .await?;

        Ok(CreateNamespaceResponse {
            transaction_id: None,
            properties: request.properties,
        })
    }

    async fn drop_namespace(&self, request: DropNamespaceRequest) -> Result<DropNamespaceResponse> {
        let namespace_id = request.id.as_ref().ok_or_else(|| Error::InvalidInput {
            source: "Namespace ID is required".into(),
            location: location!(),
        })?;

        // Root namespace always exists and cannot be dropped
        if namespace_id.is_empty() {
            return Err(Error::Namespace {
                source: "Root namespace cannot be dropped".into(),
                location: location!(),
            });
        }

        let object_id = namespace_id.join(DELIMITER);

        // Check if namespace exists
        if !self.manifest_contains_object(&object_id).await? {
            return Err(Error::Namespace {
                source: format!("Namespace '{}' not found", object_id).into(),
                location: location!(),
            });
        }

        // Check for child namespaces
        let prefix = format!("{}{}", object_id, DELIMITER);
        let filter = format!("starts_with(object_id, '{}')", prefix);
        let mut scanner = self.manifest_scanner().await?;
        scanner.filter(&filter).map_err(|e| Error::IO {
            source: box_error(std::io::Error::other(format!("Failed to filter: {}", e))),
            location: location!(),
        })?;
        scanner.project::<&str>(&[]).map_err(|e| Error::IO {
            source: box_error(std::io::Error::other(format!("Failed to project: {}", e))),
            location: location!(),
        })?;
        scanner.with_row_id();
        let count = scanner.count_rows().await.map_err(|e| Error::IO {
            source: box_error(std::io::Error::other(format!(
                "Failed to count rows: {}",
                e
            ))),
            location: location!(),
        })?;

        if count > 0 {
            return Err(Error::Namespace {
                source: format!(
                    "Namespace '{}' is not empty (contains {} child objects)",
                    object_id, count
                )
                .into(),
                location: location!(),
            });
        }

        self.delete_from_manifest(&object_id).await?;

        Ok(DropNamespaceResponse {
            properties: None,
            transaction_id: None,
        })
    }

    async fn namespace_exists(&self, request: NamespaceExistsRequest) -> Result<()> {
        let namespace_id = request.id.as_ref().ok_or_else(|| Error::InvalidInput {
            source: "Namespace ID is required".into(),
            location: location!(),
        })?;

        // Root namespace always exists
        if namespace_id.is_empty() {
            return Ok(());
        }

        let object_id = namespace_id.join(DELIMITER);
        if self.manifest_contains_object(&object_id).await? {
            Ok(())
        } else {
            Err(Error::Namespace {
                source: format!("Namespace '{}' not found", object_id).into(),
                location: location!(),
            })
        }
    }

    async fn create_empty_table(
        &self,
        request: CreateEmptyTableRequest,
    ) -> Result<CreateEmptyTableResponse> {
        let table_id = request.id.as_ref().ok_or_else(|| Error::InvalidInput {
            source: "Table ID is required".into(),
            location: location!(),
        })?;

        if table_id.is_empty() {
            return Err(Error::InvalidInput {
                source: "Table ID cannot be empty".into(),
                location: location!(),
            });
        }

        let (namespace, table_name) = Self::split_object_id(table_id);
        let object_id = Self::build_object_id(&namespace, &table_name);

        // Check if table already exists in manifest
        let existing = self.query_manifest_for_table(&object_id).await?;
        if existing.is_some() {
            return Err(Error::Namespace {
                source: format!("Table '{}' already exists", table_name).into(),
                location: location!(),
            });
        }

        // Create table location path with hash-based naming
        // When dir_listing_enabled is true and it's a root table, use directory-style naming: {table_name}.lance
        // Otherwise, use hash-based naming: {hash}_{object_id}
        let dir_name = if namespace.is_empty() && self.dir_listing_enabled {
            // Root table with directory listing enabled: use {table_name}.lance
            format!("{}.lance", table_name)
        } else {
            // Child namespace table or dir listing disabled: use hash-based naming
            Self::generate_dir_name(&object_id)
        };
        let table_path = self.base_path.child(dir_name.as_str());
        let table_uri = Self::construct_full_uri(&self.root, &dir_name)?;

        // Validate location if provided
        if let Some(req_location) = &request.location {
            let req_location = req_location.trim_end_matches('/');
            if req_location != table_uri {
                return Err(Error::Namespace {
                    source: format!(
                        "Cannot create table {} at location {}, must be at location {}",
                        table_name, req_location, table_uri
                    )
                    .into(),
                    location: location!(),
                });
            }
        }

        // Create the .lance-reserved file to mark the table as existing
        let reserved_file_path = table_path.child(".lance-reserved");

        self.object_store
            .create(&reserved_file_path)
            .await
            .map_err(|e| Error::Namespace {
                source: format!(
                    "Failed to create .lance-reserved file for table {}: {}",
                    table_name, e
                )
                .into(),
                location: location!(),
            })?
            .shutdown()
            .await
            .map_err(|e| Error::Namespace {
                source: format!(
                    "Failed to finalize .lance-reserved file for table {}: {}",
                    table_name, e
                )
                .into(),
                location: location!(),
            })?;

        // Add entry to manifest marking this as an empty table (store dir_name, not full path)
        self.insert_into_manifest(object_id, ObjectType::Table, Some(dir_name))
            .await?;

        log::info!(
            "Created empty table '{}' in manifest at {}",
            table_name,
            table_uri
        );

        Ok(CreateEmptyTableResponse {
            transaction_id: None,
            location: Some(table_uri),
            storage_options: self.storage_options.clone(),
        })
    }

    async fn declare_table(&self, request: DeclareTableRequest) -> Result<DeclareTableResponse> {
        let table_id = request.id.as_ref().ok_or_else(|| Error::InvalidInput {
            source: "Table ID is required".into(),
            location: location!(),
        })?;

        if table_id.is_empty() {
            return Err(Error::InvalidInput {
                source: "Table ID cannot be empty".into(),
                location: location!(),
            });
        }

        let (namespace, table_name) = Self::split_object_id(table_id);
        let object_id = Self::build_object_id(&namespace, &table_name);

        // Check if table already exists in manifest
        let existing = self.query_manifest_for_table(&object_id).await?;
        if existing.is_some() {
            return Err(Error::Namespace {
                source: format!("Table '{}' already exists", table_name).into(),
                location: location!(),
            });
        }

        // Create table location path with hash-based naming
        // When dir_listing_enabled is true and it's a root table, use directory-style naming: {table_name}.lance
        // Otherwise, use hash-based naming: {hash}_{object_id}
        let dir_name = if namespace.is_empty() && self.dir_listing_enabled {
            // Root table with directory listing enabled: use {table_name}.lance
            format!("{}.lance", table_name)
        } else {
            // Child namespace table or dir listing disabled: use hash-based naming
            Self::generate_dir_name(&object_id)
        };
        let table_path = self.base_path.child(dir_name.as_str());
        let table_uri = Self::construct_full_uri(&self.root, &dir_name)?;

        // Validate location if provided
        if let Some(req_location) = &request.location {
            let req_location = req_location.trim_end_matches('/');
            if req_location != table_uri {
                return Err(Error::Namespace {
                    source: format!(
                        "Cannot declare table {} at location {}, must be at location {}",
                        table_name, req_location, table_uri
                    )
                    .into(),
                    location: location!(),
                });
            }
        }

        // Create the .lance-reserved file to mark the table as existing
        let reserved_file_path = table_path.child(".lance-reserved");

        self.object_store
            .create(&reserved_file_path)
            .await
            .map_err(|e| Error::Namespace {
                source: format!(
                    "Failed to create .lance-reserved file for table {}: {}",
                    table_name, e
                )
                .into(),
                location: location!(),
            })?
            .shutdown()
            .await
            .map_err(|e| Error::Namespace {
                source: format!(
                    "Failed to finalize .lance-reserved file for table {}: {}",
                    table_name, e
                )
                .into(),
                location: location!(),
            })?;

        // Add entry to manifest marking this as a declared table (store dir_name, not full path)
        self.insert_into_manifest(object_id, ObjectType::Table, Some(dir_name))
            .await?;

        log::info!(
            "Declared table '{}' in manifest at {}",
            table_name,
            table_uri
        );

        Ok(DeclareTableResponse {
            transaction_id: None,
            location: Some(table_uri),
            storage_options: self.storage_options.clone(),
        })
    }

    async fn register_table(&self, request: RegisterTableRequest) -> Result<RegisterTableResponse> {
        let table_id = request.id.as_ref().ok_or_else(|| Error::InvalidInput {
            source: "Table ID is required".into(),
            location: location!(),
        })?;

        if table_id.is_empty() {
            return Err(Error::InvalidInput {
                source: "Table ID cannot be empty".into(),
                location: location!(),
            });
        }

        let location = request.location.clone();

        // Validate that location is a relative path within the root directory
        // We don't allow absolute URIs or paths that escape the root
        if location.contains("://") {
            return Err(Error::InvalidInput {
                source: format!(
                    "Absolute URIs are not allowed for register_table. Location must be a relative path within the root directory: {}",
                    location
                ).into(),
                location: location!(),
            });
        }

        if location.starts_with('/') {
            return Err(Error::InvalidInput {
                source: format!(
                    "Absolute paths are not allowed for register_table. Location must be a relative path within the root directory: {}",
                    location
                ).into(),
                location: location!(),
            });
        }

        // Check for path traversal attempts
        if location.contains("..") {
            return Err(Error::InvalidInput {
                source: format!(
                    "Path traversal is not allowed. Location must be a relative path within the root directory: {}",
                    location
                ).into(),
                location: location!(),
            });
        }

        let (namespace, table_name) = Self::split_object_id(table_id);
        let object_id = Self::build_object_id(&namespace, &table_name);

        // Validate that parent namespaces exist (if not root)
        if !namespace.is_empty() {
            self.validate_namespace_levels_exist(&namespace).await?;
        }

        // Check if table already exists
        if self.manifest_contains_object(&object_id).await? {
            return Err(Error::Namespace {
                source: format!("Table '{}' already exists", object_id).into(),
                location: location!(),
            });
        }

        // Register the table with its location in the manifest
        self.insert_into_manifest(object_id, ObjectType::Table, Some(location.clone()))
            .await?;

        Ok(RegisterTableResponse {
            transaction_id: None,
            location: Some(location),
            properties: None,
        })
    }

    async fn deregister_table(
        &self,
        request: DeregisterTableRequest,
    ) -> Result<DeregisterTableResponse> {
        let table_id = request.id.as_ref().ok_or_else(|| Error::InvalidInput {
            source: "Table ID is required".into(),
            location: location!(),
        })?;

        if table_id.is_empty() {
            return Err(Error::InvalidInput {
                source: "Table ID cannot be empty".into(),
                location: location!(),
            });
        }

        let (namespace, table_name) = Self::split_object_id(table_id);
        let object_id = Self::build_object_id(&namespace, &table_name);

        // Get table info before deleting
        let table_info = self.query_manifest_for_table(&object_id).await?;

        let table_uri = match table_info {
            Some(info) => {
                // Delete from manifest only (leave physical data intact)
                self.delete_from_manifest(&object_id).await?;
                Self::construct_full_uri(&self.root, &info.location)?
            }
            None => {
                return Err(Error::Namespace {
                    source: format!("Table '{}' not found", object_id).into(),
                    location: location!(),
                });
            }
        };

        Ok(DeregisterTableResponse {
            transaction_id: None,
            id: request.id.clone(),
            location: Some(table_uri),
            properties: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{DirectoryNamespaceBuilder, ManifestNamespace};
    use bytes::Bytes;
    use lance_core::utils::tempfile::TempStdDir;
    use lance_namespace::models::{
        CreateTableRequest, DescribeTableRequest, DropTableRequest, ListTablesRequest,
        TableExistsRequest,
    };
    use lance_namespace::LanceNamespace;
    use rstest::rstest;

    fn create_test_ipc_data() -> Vec<u8> {
        use arrow::array::{Int32Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::ipc::writer::StreamWriter;
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .unwrap();

        let mut buffer = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut buffer, &schema).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }
        buffer
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_manifest_namespace_basic_create_and_list(#[case] inline_optimization: bool) {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        // Create a DirectoryNamespace with manifest enabled (default)
        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Verify we can list tables (should be empty)
        let mut request = ListTablesRequest::new();
        request.id = Some(vec![]);
        let response = dir_namespace.list_tables(request).await.unwrap();
        assert_eq!(response.tables.len(), 0);

        // Create a test table
        let buffer = create_test_ipc_data();
        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["test_table".to_string()]);

        let _response = dir_namespace
            .create_table(create_request, Bytes::from(buffer))
            .await
            .unwrap();

        // List tables again - should see our new table
        let mut request = ListTablesRequest::new();
        request.id = Some(vec![]);
        let response = dir_namespace.list_tables(request).await.unwrap();
        assert_eq!(response.tables.len(), 1);
        assert_eq!(response.tables[0], "test_table");
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_manifest_namespace_table_exists(#[case] inline_optimization: bool) {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Check non-existent table
        let mut request = TableExistsRequest::new();
        request.id = Some(vec!["nonexistent".to_string()]);
        let result = dir_namespace.table_exists(request).await;
        assert!(result.is_err());

        // Create table
        let buffer = create_test_ipc_data();
        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["test_table".to_string()]);
        dir_namespace
            .create_table(create_request, Bytes::from(buffer))
            .await
            .unwrap();

        // Check existing table
        let mut request = TableExistsRequest::new();
        request.id = Some(vec!["test_table".to_string()]);
        let result = dir_namespace.table_exists(request).await;
        assert!(result.is_ok());
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_manifest_namespace_describe_table(#[case] inline_optimization: bool) {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Describe non-existent table
        let mut request = DescribeTableRequest::new();
        request.id = Some(vec!["nonexistent".to_string()]);
        let result = dir_namespace.describe_table(request).await;
        assert!(result.is_err());

        // Create table
        let buffer = create_test_ipc_data();
        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["test_table".to_string()]);
        dir_namespace
            .create_table(create_request, Bytes::from(buffer))
            .await
            .unwrap();

        // Describe existing table
        let mut request = DescribeTableRequest::new();
        request.id = Some(vec!["test_table".to_string()]);
        let response = dir_namespace.describe_table(request).await.unwrap();
        assert!(response.location.is_some());
        assert!(response.location.unwrap().contains("test_table"));
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_manifest_namespace_drop_table(#[case] inline_optimization: bool) {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Create table
        let buffer = create_test_ipc_data();
        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["test_table".to_string()]);
        dir_namespace
            .create_table(create_request, Bytes::from(buffer))
            .await
            .unwrap();

        // Verify table exists
        let mut request = ListTablesRequest::new();
        request.id = Some(vec![]);
        let response = dir_namespace.list_tables(request).await.unwrap();
        assert_eq!(response.tables.len(), 1);

        // Drop table
        let mut drop_request = DropTableRequest::new();
        drop_request.id = Some(vec!["test_table".to_string()]);
        let _response = dir_namespace.drop_table(drop_request).await.unwrap();

        // Verify table is gone
        let mut request = ListTablesRequest::new();
        request.id = Some(vec![]);
        let response = dir_namespace.list_tables(request).await.unwrap();
        assert_eq!(response.tables.len(), 0);
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_manifest_namespace_multiple_tables(#[case] inline_optimization: bool) {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Create multiple tables
        let buffer = create_test_ipc_data();
        for i in 1..=3 {
            let mut create_request = CreateTableRequest::new();
            create_request.id = Some(vec![format!("table{}", i)]);
            dir_namespace
                .create_table(create_request, Bytes::from(buffer.clone()))
                .await
                .unwrap();
        }

        // List all tables
        let mut request = ListTablesRequest::new();
        request.id = Some(vec![]);
        let response = dir_namespace.list_tables(request).await.unwrap();
        assert_eq!(response.tables.len(), 3);
        assert!(response.tables.contains(&"table1".to_string()));
        assert!(response.tables.contains(&"table2".to_string()));
        assert!(response.tables.contains(&"table3".to_string()));
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_directory_only_mode(#[case] inline_optimization: bool) {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        // Create a DirectoryNamespace with manifest disabled
        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(false)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Verify we can list tables (should be empty)
        let mut request = ListTablesRequest::new();
        request.id = Some(vec![]);
        let response = dir_namespace.list_tables(request).await.unwrap();
        assert_eq!(response.tables.len(), 0);

        // Create a test table
        let buffer = create_test_ipc_data();
        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["test_table".to_string()]);

        // Create table - this should use directory-only mode
        let _response = dir_namespace
            .create_table(create_request, Bytes::from(buffer))
            .await
            .unwrap();

        // List tables - should see our new table
        let mut request = ListTablesRequest::new();
        request.id = Some(vec![]);
        let response = dir_namespace.list_tables(request).await.unwrap();
        assert_eq!(response.tables.len(), 1);
        assert_eq!(response.tables[0], "test_table");
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_dual_mode_merge(#[case] inline_optimization: bool) {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        // Create a DirectoryNamespace with both manifest and directory enabled
        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(true)
            .dir_listing_enabled(true)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Create tables through manifest
        let buffer = create_test_ipc_data();
        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["table1".to_string()]);
        dir_namespace
            .create_table(create_request, Bytes::from(buffer))
            .await
            .unwrap();

        // List tables - should see table from both manifest and directory
        let mut request = ListTablesRequest::new();
        request.id = Some(vec![]);
        let response = dir_namespace.list_tables(request).await.unwrap();
        assert_eq!(response.tables.len(), 1);
        assert_eq!(response.tables[0], "table1");
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_manifest_only_mode(#[case] inline_optimization: bool) {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        // Create a DirectoryNamespace with only manifest enabled
        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(true)
            .dir_listing_enabled(false)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Create table
        let buffer = create_test_ipc_data();
        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["test_table".to_string()]);
        dir_namespace
            .create_table(create_request, Bytes::from(buffer))
            .await
            .unwrap();

        // List tables - should only use manifest
        let mut request = ListTablesRequest::new();
        request.id = Some(vec![]);
        let response = dir_namespace.list_tables(request).await.unwrap();
        assert_eq!(response.tables.len(), 1);
        assert_eq!(response.tables[0], "test_table");
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_drop_nonexistent_table(#[case] inline_optimization: bool) {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Try to drop non-existent table
        let mut drop_request = DropTableRequest::new();
        drop_request.id = Some(vec!["nonexistent".to_string()]);
        let result = dir_namespace.drop_table(drop_request).await;
        assert!(result.is_err());
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_create_duplicate_table_fails(#[case] inline_optimization: bool) {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Create table
        let buffer = create_test_ipc_data();
        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["test_table".to_string()]);
        dir_namespace
            .create_table(create_request, Bytes::from(buffer.clone()))
            .await
            .unwrap();

        // Try to create table with same name - should fail
        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["test_table".to_string()]);
        let result = dir_namespace
            .create_table(create_request, Bytes::from(buffer))
            .await;
        assert!(result.is_err());
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_create_child_namespace(#[case] inline_optimization: bool) {
        use lance_namespace::models::{
            CreateNamespaceRequest, ListNamespacesRequest, NamespaceExistsRequest,
        };

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Create a child namespace
        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["ns1".to_string()]);
        let result = dir_namespace.create_namespace(create_req).await;
        assert!(
            result.is_ok(),
            "Failed to create child namespace: {:?}",
            result.err()
        );

        // Verify namespace exists
        let exists_req = NamespaceExistsRequest {
            id: Some(vec!["ns1".to_string()]),
        };
        let result = dir_namespace.namespace_exists(exists_req).await;
        assert!(result.is_ok(), "Namespace should exist");

        // List child namespaces of root
        let list_req = ListNamespacesRequest {
            id: Some(vec![]),
            page_token: None,
            limit: None,
        };
        let result = dir_namespace.list_namespaces(list_req).await;
        assert!(result.is_ok());
        let namespaces = result.unwrap();
        assert_eq!(namespaces.namespaces.len(), 1);
        assert_eq!(namespaces.namespaces[0], "ns1");
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_create_nested_namespace(#[case] inline_optimization: bool) {
        use lance_namespace::models::{
            CreateNamespaceRequest, ListNamespacesRequest, NamespaceExistsRequest,
        };

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Create parent namespace
        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["parent".to_string()]);
        dir_namespace.create_namespace(create_req).await.unwrap();

        // Create nested child namespace
        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["parent".to_string(), "child".to_string()]);
        let result = dir_namespace.create_namespace(create_req).await;
        assert!(
            result.is_ok(),
            "Failed to create nested namespace: {:?}",
            result.err()
        );

        // Verify nested namespace exists
        let exists_req = NamespaceExistsRequest {
            id: Some(vec!["parent".to_string(), "child".to_string()]),
        };
        let result = dir_namespace.namespace_exists(exists_req).await;
        assert!(result.is_ok(), "Nested namespace should exist");

        // List child namespaces of parent
        let list_req = ListNamespacesRequest {
            id: Some(vec!["parent".to_string()]),
            page_token: None,
            limit: None,
        };
        let result = dir_namespace.list_namespaces(list_req).await;
        assert!(result.is_ok());
        let namespaces = result.unwrap();
        assert_eq!(namespaces.namespaces.len(), 1);
        assert_eq!(namespaces.namespaces[0], "child");
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_create_namespace_without_parent_fails(#[case] inline_optimization: bool) {
        use lance_namespace::models::CreateNamespaceRequest;

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Try to create nested namespace without parent
        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["nonexistent_parent".to_string(), "child".to_string()]);
        let result = dir_namespace.create_namespace(create_req).await;
        assert!(result.is_err(), "Should fail when parent doesn't exist");
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_drop_child_namespace(#[case] inline_optimization: bool) {
        use lance_namespace::models::{
            CreateNamespaceRequest, DropNamespaceRequest, NamespaceExistsRequest,
        };

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Create a child namespace
        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["ns1".to_string()]);
        dir_namespace.create_namespace(create_req).await.unwrap();

        // Drop the namespace
        let mut drop_req = DropNamespaceRequest::new();
        drop_req.id = Some(vec!["ns1".to_string()]);
        let result = dir_namespace.drop_namespace(drop_req).await;
        assert!(
            result.is_ok(),
            "Failed to drop namespace: {:?}",
            result.err()
        );

        // Verify namespace no longer exists
        let exists_req = NamespaceExistsRequest {
            id: Some(vec!["ns1".to_string()]),
        };
        let result = dir_namespace.namespace_exists(exists_req).await;
        assert!(result.is_err(), "Namespace should not exist after drop");
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_drop_namespace_with_children_fails(#[case] inline_optimization: bool) {
        use lance_namespace::models::{CreateNamespaceRequest, DropNamespaceRequest};

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Create parent and child namespaces
        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["parent".to_string()]);
        dir_namespace.create_namespace(create_req).await.unwrap();

        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["parent".to_string(), "child".to_string()]);
        dir_namespace.create_namespace(create_req).await.unwrap();

        // Try to drop parent namespace - should fail because it has children
        let mut drop_req = DropNamespaceRequest::new();
        drop_req.id = Some(vec!["parent".to_string()]);
        let result = dir_namespace.drop_namespace(drop_req).await;
        assert!(result.is_err(), "Should fail when namespace has children");
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_create_table_in_child_namespace(#[case] inline_optimization: bool) {
        use lance_namespace::models::{
            CreateNamespaceRequest, CreateTableRequest, ListTablesRequest,
        };

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Create a child namespace
        let mut create_ns_req = CreateNamespaceRequest::new();
        create_ns_req.id = Some(vec!["ns1".to_string()]);
        dir_namespace.create_namespace(create_ns_req).await.unwrap();

        // Create a table in the child namespace
        let buffer = create_test_ipc_data();
        let mut create_table_req = CreateTableRequest::new();
        create_table_req.id = Some(vec!["ns1".to_string(), "table1".to_string()]);
        let result = dir_namespace
            .create_table(create_table_req, Bytes::from(buffer))
            .await;
        assert!(
            result.is_ok(),
            "Failed to create table in child namespace: {:?}",
            result.err()
        );

        // List tables in the namespace
        let list_req = ListTablesRequest {
            id: Some(vec!["ns1".to_string()]),
            page_token: None,
            limit: None,
        };
        let result = dir_namespace.list_tables(list_req).await;
        assert!(result.is_ok());
        let tables = result.unwrap();
        assert_eq!(tables.tables.len(), 1);
        assert_eq!(tables.tables[0], "table1");
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_describe_child_namespace(#[case] inline_optimization: bool) {
        use lance_namespace::models::{CreateNamespaceRequest, DescribeNamespaceRequest};

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Create a child namespace with properties
        let mut properties = std::collections::HashMap::new();
        properties.insert("key1".to_string(), "value1".to_string());

        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["ns1".to_string()]);
        create_req.properties = Some(properties.clone());
        dir_namespace.create_namespace(create_req).await.unwrap();

        // Describe the namespace
        let describe_req = DescribeNamespaceRequest {
            id: Some(vec!["ns1".to_string()]),
        };
        let result = dir_namespace.describe_namespace(describe_req).await;
        assert!(
            result.is_ok(),
            "Failed to describe namespace: {:?}",
            result.err()
        );
        let response = result.unwrap();
        assert!(response.properties.is_some());
        assert_eq!(
            response.properties.unwrap().get("key1"),
            Some(&"value1".to_string())
        );
    }

    #[test]
    fn test_construct_full_uri_with_cloud_urls() {
        // Test S3-style URL with nested path (no trailing slash)
        let s3_result =
            ManifestNamespace::construct_full_uri("s3://bucket/path/subdir", "table.lance")
                .unwrap();
        assert_eq!(
            s3_result, "s3://bucket/path/subdir/table.lance",
            "S3 URL should correctly append table name to nested path"
        );

        // Test Azure-style URL with nested path (no trailing slash)
        let az_result =
            ManifestNamespace::construct_full_uri("az://container/path/subdir", "table.lance")
                .unwrap();
        assert_eq!(
            az_result, "az://container/path/subdir/table.lance",
            "Azure URL should correctly append table name to nested path"
        );

        // Test GCS-style URL with nested path (no trailing slash)
        let gs_result =
            ManifestNamespace::construct_full_uri("gs://bucket/path/subdir", "table.lance")
                .unwrap();
        assert_eq!(
            gs_result, "gs://bucket/path/subdir/table.lance",
            "GCS URL should correctly append table name to nested path"
        );

        // Test with deeper nesting
        let deep_result =
            ManifestNamespace::construct_full_uri("s3://bucket/a/b/c/d", "my_table.lance").unwrap();
        assert_eq!(
            deep_result, "s3://bucket/a/b/c/d/my_table.lance",
            "Deeply nested path should work correctly"
        );

        // Test with root-level path (single segment after bucket)
        let shallow_result =
            ManifestNamespace::construct_full_uri("s3://bucket", "table.lance").unwrap();
        assert_eq!(
            shallow_result, "s3://bucket/table.lance",
            "Single-level nested path should work correctly"
        );

        // Test that URLs with trailing slash already work (no regression)
        let trailing_slash_result =
            ManifestNamespace::construct_full_uri("s3://bucket/path/subdir/", "table.lance")
                .unwrap();
        assert_eq!(
            trailing_slash_result, "s3://bucket/path/subdir/table.lance",
            "URL with existing trailing slash should still work"
        );
    }
}
