// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Version Archive module
//!
//! This module provides version archive functionality for preserving version metadata
//! when manifests are cleaned up.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Array, BooleanArray, Int64Array, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::stream::StreamExt;
use lance_core::datatypes::Schema as LanceSchema;
use lance_core::{Error, Result};
use lance_file::previous::reader::FileReader as PreviousFileReader;
use lance_file::previous::writer::{
    FileWriter as PreviousFileWriter, FileWriterOptions as PreviousFileWriterOptions,
};
use lance_io::object_store::ObjectStore;
use lance_table::format::{ManifestSummary, SelfDescribingFileReader};
use lance_table::io::manifest::ManifestDescribing;
use object_store::path::Path;

pub const ARCHIVE_DIR: &str = "_archive";
const VERSION_ARCHIVE_FILE_SUFFIX: &str = ".lance";
const METADATA_KEY_DATASET_CREATED: &str = "lance:archive:dataset_created";
const METADATA_KEY_CREATED_AT: &str = "lance:archive:created_at";
const MAX_TRANSACTION_PROPERTIES_KEYS: usize = 10;
const MAX_TRANSACTION_PROPERTY_VALUE_BYTES: usize = 200;

/// Generate archive filename for a given version
fn archive_filename(version: u64) -> String {
    format!("{:020}{}", version, VERSION_ARCHIVE_FILE_SUFFIX)
}

fn archive_arrow_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("version", DataType::UInt64, false),
        Field::new("timestamp_millis", DataType::Int64, false),
        Field::new("total_fragments", DataType::UInt64, false),
        Field::new("total_data_files", DataType::UInt64, false),
        Field::new("total_files_size", DataType::UInt64, false),
        Field::new("total_deletion_files", DataType::UInt64, false),
        Field::new("total_data_file_rows", DataType::UInt64, false),
        Field::new("total_deletion_file_rows", DataType::UInt64, false),
        Field::new("total_rows", DataType::UInt64, false),
        Field::new("is_tagged", DataType::Boolean, false),
        Field::new("transaction_uuid", DataType::Utf8, true),
        Field::new("read_version", DataType::UInt64, true),
        Field::new("operation_type", DataType::Utf8, true),
        Field::new("transaction_properties", DataType::Utf8, true),
    ]))
}

fn archive_lance_schema() -> Result<LanceSchema> {
    let arrow_schema = archive_arrow_schema();
    LanceSchema::try_from(arrow_schema.as_ref())
        .map_err(|e| Error::invalid_input(format!("Failed to create Lance schema: {}", e)))
}

fn truncate_value(value: &str, max_bytes: usize) -> &str {
    let mut end = value.len().min(max_bytes);
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

fn normalize_transaction_properties(props: &HashMap<String, String>) -> HashMap<String, String> {
    let mut normalized = HashMap::new();
    let mut keys = props.keys().collect::<Vec<_>>();
    keys.sort();

    let mut truncated = false;
    for key in keys.iter().take(MAX_TRANSACTION_PROPERTIES_KEYS) {
        let value = &props[*key];
        if value.len() > MAX_TRANSACTION_PROPERTY_VALUE_BYTES {
            truncated = true;
            normalized.insert(
                (*key).clone(),
                truncate_value(value, MAX_TRANSACTION_PROPERTY_VALUE_BYTES).to_string(),
            );
            normalized.insert(format!("_truncated_value:{}", key), value.len().to_string());
        } else {
            normalized.insert((*key).clone(), value.clone());
        }
    }

    if props.len() > MAX_TRANSACTION_PROPERTIES_KEYS {
        truncated = true;
        normalized.insert(
            "_dropped_key_count".to_string(),
            (props.len() - MAX_TRANSACTION_PROPERTIES_KEYS).to_string(),
        );
    }

    if truncated {
        normalized.insert("_truncated".to_string(), "true".to_string());
        normalized.insert("_original_key_count".to_string(), props.len().to_string());
    }

    normalized
}

fn encode_transaction_properties(props: &HashMap<String, String>) -> Result<Option<String>> {
    if props.is_empty() {
        return Ok(None);
    }

    let normalized = normalize_transaction_properties(props);
    match serde_json::to_string(&normalized) {
        Ok(json) => Ok(Some(json)),
        Err(e) => {
            tracing::warn!("Failed to serialize transaction_properties: {}", e);
            Ok(None)
        }
    }
}

fn decode_transaction_properties(value: Option<&str>) -> HashMap<String, String> {
    match value {
        Some(value) => match serde_json::from_str(value) {
            Ok(props) => props,
            Err(e) => {
                tracing::warn!(
                    "Failed to parse transaction_properties: {}, raw value: {}",
                    e,
                    value
                );
                HashMap::from([("transaction_properties".to_string(), value.to_string())])
            }
        },
        None => HashMap::new(),
    }
}

fn archived_versions_to_record_batch(entries: &[ArchivedVersion]) -> Result<RecordBatch> {
    let schema = archive_arrow_schema();
    let capacity = entries.len();

    // Pre-allocate all vectors with known capacity for better performance
    let mut versions = Vec::with_capacity(capacity);
    let mut timestamps = Vec::with_capacity(capacity);
    let mut total_fragments = Vec::with_capacity(capacity);
    let mut total_data_files = Vec::with_capacity(capacity);
    let mut total_files_size = Vec::with_capacity(capacity);
    let mut total_deletion_files = Vec::with_capacity(capacity);
    let mut total_data_file_rows = Vec::with_capacity(capacity);
    let mut total_deletion_file_rows = Vec::with_capacity(capacity);
    let mut total_rows = Vec::with_capacity(capacity);
    let mut is_tagged = Vec::with_capacity(capacity);
    let mut transaction_uuid = Vec::with_capacity(capacity);
    let mut read_version = Vec::with_capacity(capacity);
    let mut operation_type = Vec::with_capacity(capacity);
    let mut transaction_properties = Vec::with_capacity(capacity);

    // Single pass through entries - more cache friendly
    for s in entries {
        versions.push(Some(s.version));
        timestamps.push(Some(s.timestamp_millis));
        total_fragments.push(Some(s.manifest_summary.total_fragments));
        total_data_files.push(Some(s.manifest_summary.total_data_files));
        total_files_size.push(Some(s.manifest_summary.total_files_size));
        total_deletion_files.push(Some(s.manifest_summary.total_deletion_files));
        total_data_file_rows.push(Some(s.manifest_summary.total_data_file_rows));
        total_deletion_file_rows.push(Some(s.manifest_summary.total_deletion_file_rows));
        total_rows.push(Some(s.manifest_summary.total_rows));
        is_tagged.push(Some(s.is_tagged));
        transaction_uuid.push(s.transaction_uuid.as_deref());
        read_version.push(s.read_version);
        operation_type.push(s.operation_type.as_deref());

        transaction_properties.push(encode_transaction_properties(&s.transaction_properties)?);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from(versions)),
            Arc::new(Int64Array::from(timestamps)),
            Arc::new(UInt64Array::from(total_fragments)),
            Arc::new(UInt64Array::from(total_data_files)),
            Arc::new(UInt64Array::from(total_files_size)),
            Arc::new(UInt64Array::from(total_deletion_files)),
            Arc::new(UInt64Array::from(total_data_file_rows)),
            Arc::new(UInt64Array::from(total_deletion_file_rows)),
            Arc::new(UInt64Array::from(total_rows)),
            Arc::new(BooleanArray::from(is_tagged)),
            Arc::new(StringArray::from(transaction_uuid)),
            Arc::new(UInt64Array::from(read_version)),
            Arc::new(StringArray::from(operation_type)),
            Arc::new(StringArray::from(transaction_properties)),
        ],
    )
    .map_err(|e| Error::invalid_input(format!("Failed to create RecordBatch: {}", e)))
}

/// Macro to extract and downcast a column from a RecordBatch
macro_rules! get_column {
    ($batch:expr, $name:expr, $type:ty, $type_str:expr) => {
        $batch
            .column_by_name($name)
            .ok_or_else(|| Error::invalid_input(concat!($name, " column not found")))?
            .as_any()
            .downcast_ref::<$type>()
            .ok_or_else(|| Error::invalid_input(concat!($name, " column is not ", $type_str)))?
    };
}

fn record_batch_to_archived_versions(batch: &RecordBatch) -> Result<Vec<ArchivedVersion>> {
    let mut entries = Vec::with_capacity(batch.num_rows());

    let version_col = get_column!(batch, "version", UInt64Array, "UInt64");
    let timestamp_col = get_column!(batch, "timestamp_millis", Int64Array, "Int64");
    let total_fragments_col = get_column!(batch, "total_fragments", UInt64Array, "UInt64");
    let total_data_files_col = get_column!(batch, "total_data_files", UInt64Array, "UInt64");
    let total_files_size_col = get_column!(batch, "total_files_size", UInt64Array, "UInt64");
    let total_deletion_files_col =
        get_column!(batch, "total_deletion_files", UInt64Array, "UInt64");
    let total_data_file_rows_col =
        get_column!(batch, "total_data_file_rows", UInt64Array, "UInt64");
    let total_deletion_file_rows_col =
        get_column!(batch, "total_deletion_file_rows", UInt64Array, "UInt64");
    let total_rows_col = get_column!(batch, "total_rows", UInt64Array, "UInt64");
    let is_tagged_col = get_column!(batch, "is_tagged", BooleanArray, "Boolean");
    let transaction_uuid_col = get_column!(batch, "transaction_uuid", StringArray, "String");
    let read_version_col = get_column!(batch, "read_version", UInt64Array, "UInt64");
    let operation_type_col = get_column!(batch, "operation_type", StringArray, "String");
    let transaction_properties_col =
        get_column!(batch, "transaction_properties", StringArray, "String");

    for i in 0..batch.num_rows() {
        let transaction_properties = decode_transaction_properties(
            transaction_properties_col
                .is_valid(i)
                .then(|| transaction_properties_col.value(i)),
        );

        entries.push(ArchivedVersion {
            version: version_col.value(i),
            timestamp_millis: timestamp_col.value(i),
            manifest_summary: ManifestSummary {
                total_fragments: total_fragments_col.value(i),
                total_data_files: total_data_files_col.value(i),
                total_files_size: total_files_size_col.value(i),
                total_deletion_files: total_deletion_files_col.value(i),
                total_data_file_rows: total_data_file_rows_col.value(i),
                total_deletion_file_rows: total_deletion_file_rows_col.value(i),
                total_rows: total_rows_col.value(i),
            },
            is_tagged: is_tagged_col.value(i),
            transaction_uuid: transaction_uuid_col
                .is_valid(i)
                .then(|| transaction_uuid_col.value(i).to_string()),
            read_version: read_version_col
                .is_valid(i)
                .then(|| read_version_col.value(i)),
            operation_type: operation_type_col
                .is_valid(i)
                .then(|| operation_type_col.value(i).to_string()),
            transaction_properties,
        });
    }

    Ok(entries)
}

/// Archive configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VersionArchiveConfig {
    /// Whether VersionArchive is enabled
    pub enabled: bool,

    /// Maximum number of version entries to retain in a single archive file
    pub max_entries: usize,

    /// Maximum number of archive files to retain
    pub max_archive_files: usize,
}

impl Default for VersionArchiveConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_entries: 10000,
            max_archive_files: 2,
        }
    }
}

impl VersionArchiveConfig {
    /// Read configuration from manifest config
    ///
    /// Ensures that `max_entries` and `max_archive_files` are at least 1
    /// to prevent data loss during archive finalization.
    pub fn from_config(config: &HashMap<String, String>) -> Self {
        let max_entries = config
            .get("lance.version_archive.max_entries")
            .and_then(|v| v.parse().ok())
            .unwrap_or(10000)
            .max(1);

        let max_archive_files = config
            .get("lance.version_archive.max_archive_files")
            .and_then(|v| v.parse().ok())
            .unwrap_or(2)
            .max(1);

        Self {
            enabled: config
                .get("lance.version_archive.enabled")
                .and_then(|v| v.parse().ok())
                .unwrap_or(true),
            max_entries,
            max_archive_files,
        }
    }
}

/// Archived dataset version with flattened manifest statistics
///
/// This structure contains metadata about a specific version of the dataset,
/// including statistics about fragments, data files, deletion files, and rows.
/// It is used for archive preservation when manifests are cleaned up.
#[derive(Debug, Clone, PartialEq)]
pub struct ArchivedVersion {
    /// The version number of this dataset version
    pub version: u64,
    /// Timestamp when this version was created (milliseconds since epoch)
    pub timestamp_millis: i64,
    /// Summary statistics about the manifest for this version
    pub manifest_summary: ManifestSummary,
    /// Whether this version is tagged (should be retained)
    pub is_tagged: bool,
    /// Unique identifier for the transaction that created this version
    pub transaction_uuid: Option<String>,
    /// The version that was read when this transaction was committed
    pub read_version: Option<u64>,
    /// The type of operation that created this version (e.g., "append", "overwrite")
    pub operation_type: Option<String>,
    /// Additional properties from the transaction
    pub transaction_properties: HashMap<String, String>,
}

/// Version archive with persistence capability
///
/// This structure maintains version metadata for dataset versions, preserving
/// information about versions even after their manifests have been cleaned up.
/// Archives are stored in Lance format and support automatic cleanup of
/// old archive files based on configuration.
#[derive(Debug, Clone)]
pub struct VersionArchive {
    /// Archived versions stored in this archive
    pub versions: Vec<ArchivedVersion>,
    /// The highest version number in this archive
    pub latest_version_number: u64,
    /// Timestamp when the dataset was created (milliseconds since epoch)
    pub dataset_created_millis: i64,
    /// Timestamp when this archive was created (milliseconds since epoch)
    pub created_at_millis: i64,
    /// Configuration for archive behavior
    config: VersionArchiveConfig,
    /// Base path for the dataset
    base: Path,
    /// Object store for persistence
    object_store: Arc<ObjectStore>,
}

impl VersionArchive {
    pub fn archive_dir(&self) -> Path {
        self.base.clone().join(ARCHIVE_DIR)
    }

    async fn list_archive_files(
        object_store: &ObjectStore,
        archive_dir: &Path,
    ) -> Result<Vec<(u64, Path)>> {
        let mut archives = Vec::new();
        let mut stream = object_store.list(Some(archive_dir.clone()));
        while let Some(meta) = stream.next().await {
            let meta = meta?;
            if let Some(filename) = meta.location.filename()
                && let Some(version) = filename
                    .strip_suffix(VERSION_ARCHIVE_FILE_SUFFIX)
                    .and_then(|s| s.parse::<u64>().ok())
            {
                archives.push((version, meta.location));
            }
        }

        archives.sort_by(|a, b| b.0.cmp(&a.0));
        Ok(archives)
    }

    /// Private helper to write an archive in Lance format
    async fn write_archive(&self) -> Result<()> {
        let archive_dir = self.archive_dir();
        let filename = archive_filename(self.latest_version_number);
        let path = archive_dir.join(filename);

        let batch = archived_versions_to_record_batch(&self.versions)?;
        let mut lance_schema = archive_lance_schema()?;
        lance_schema.metadata.insert(
            METADATA_KEY_DATASET_CREATED.to_string(),
            self.dataset_created_millis.to_string(),
        );
        lance_schema.metadata.insert(
            METADATA_KEY_CREATED_AT.to_string(),
            self.created_at_millis.to_string(),
        );

        let options = PreviousFileWriterOptions::default();
        let mut writer = PreviousFileWriter::<ManifestDescribing>::try_new(
            &self.object_store,
            &path,
            lance_schema,
            &options,
        )
        .await?;

        writer.write(&[batch]).await?;
        writer.finish().await?;

        Ok(())
    }

    /// Private helper to read an archive from Lance format
    async fn read_archive(
        base: &Path,
        object_store: Arc<ObjectStore>,
        path: &Path,
        config: VersionArchiveConfig,
    ) -> Result<Self> {
        let reader = object_store.open(path).await?;
        let reader =
            PreviousFileReader::try_new_self_described_from_reader(reader.into(), None).await?;
        let num_batches = reader.num_batches();

        let mut all_entries = Vec::new();
        for i in 0..num_batches {
            let batch = reader
                .read_batch(
                    i as i32,
                    lance_io::ReadBatchParams::RangeFull,
                    reader.schema(),
                )
                .await?;
            all_entries.extend(record_batch_to_archived_versions(&batch)?);
        }

        let latest_version_number = all_entries.iter().map(|s| s.version).max().unwrap_or(0);

        let schema = reader.schema();
        let dataset_created_millis = Self::parse_metadata(
            &schema.metadata,
            METADATA_KEY_DATASET_CREATED,
            0i64,
            path,
            "i64",
        );
        let created_at_millis = Self::parse_metadata(
            &schema.metadata,
            METADATA_KEY_CREATED_AT,
            chrono::Utc::now().timestamp_millis(),
            path,
            "i64",
        );

        Ok(Self {
            versions: all_entries,
            latest_version_number,
            dataset_created_millis,
            created_at_millis,
            config,
            base: base.clone(),
            object_store,
        })
    }

    /// Load archive files from storage
    ///
    /// Attempts to load archive files in order (newest first) and returns
    /// the first successfully loaded archive.
    async fn load_from_files(
        base: &Path,
        object_store: Arc<ObjectStore>,
        config: VersionArchiveConfig,
    ) -> Result<Option<Self>> {
        let archive_dir = base.clone().join(ARCHIVE_DIR);
        let archives = Self::list_archive_files(&object_store, &archive_dir).await?;

        for (_, path) in archives {
            match Self::read_archive(base, object_store.clone(), &path, config).await {
                Ok(archive) => return Ok(Some(archive)),
                Err(e) => {
                    tracing::warn!("Failed to load archive file {}: {}", path, e);
                }
            }
        }
        Ok(None)
    }

    /// Load the latest archive from storage, or create a new empty one
    ///
    /// Tries to load from the newest archive file. If corrupted, tries older files.
    /// If no valid archive exists, creates a new empty one.
    pub async fn load_or_new(
        base: Path,
        object_store: Arc<ObjectStore>,
        config: VersionArchiveConfig,
    ) -> Result<Self> {
        if let Some(archive) = Self::load_from_files(&base, object_store.clone(), config).await? {
            return Ok(archive);
        }

        Ok(Self {
            versions: Vec::new(),
            latest_version_number: 0,
            dataset_created_millis: 0,
            created_at_millis: chrono::Utc::now().timestamp_millis(),
            config,
            base,
            object_store,
        })
    }

    /// Load the latest archive from storage
    pub async fn load_latest(
        base: Path,
        object_store: Arc<ObjectStore>,
        config: VersionArchiveConfig,
    ) -> Result<Option<Self>> {
        Self::load_from_files(&base, object_store, config).await
    }

    pub fn add(&mut self, entries: &[ArchivedVersion]) {
        if entries.is_empty() {
            return;
        }
        self.versions.extend(entries.iter().cloned());
    }

    /// Finalize the archive before flushing
    fn finalize_entries(&mut self) {
        if self.versions.is_empty() {
            return;
        }

        self.versions.sort_by_key(|v| v.version);
        if self.dataset_created_millis == 0 {
            self.dataset_created_millis = self
                .versions
                .first()
                .map(|v| v.timestamp_millis)
                .unwrap_or(0);
        }

        if self.versions.len() > self.config.max_entries {
            let remove_count = self.versions.len() - self.config.max_entries;
            self.versions.drain(0..remove_count);
        }

        self.latest_version_number = self.versions.iter().map(|v| v.version).max().unwrap_or(0);
        self.created_at_millis = chrono::Utc::now().timestamp_millis();
    }

    /// Flush the archive to storage
    pub async fn flush(&mut self) -> Result<()> {
        self.finalize_entries();

        if self.versions.is_empty() {
            return Ok(());
        }

        self.write_archive().await?;
        self.cleanup_old_archives().await?;

        Ok(())
    }

    async fn cleanup_old_archives(&self) -> Result<()> {
        let archive_dir = self.archive_dir();
        let archives = Self::list_archive_files(&self.object_store, &archive_dir).await?;

        if archives.len() > self.config.max_archive_files {
            // archives is sorted in descending order (newest first)
            // skip the newest max_archive_files, delete the rest (older ones)
            for (_, path) in archives.iter().skip(self.config.max_archive_files) {
                if let Err(e) = self.object_store.delete(path).await {
                    tracing::warn!("Failed to delete old archive file {}: {}", path, e);
                }
            }
        }

        Ok(())
    }

    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Parse typed metadata from schema with default value on error
    ///
    /// This generic function attempts to parse a metadata value as the specified type T.
    /// If the key is missing or parsing fails, it logs a warning and returns the default value.
    ///
    /// # Type Parameters
    /// * `T` - The target type to parse the metadata value as (must implement FromStr and Default)
    ///
    /// # Arguments
    /// * `metadata` - The metadata HashMap to read from
    /// * `key` - The metadata key to look up
    /// * `default` - The default value to use if the key is missing or parsing fails
    /// * `path` - The archive file path for error reporting
    /// * `type_name` - Human-readable type name for error messages (e.g., "i64", "u64")
    ///
    /// # Returns
    /// The parsed value or the default value
    ///
    /// # Example
    /// ```ignore
    /// let value = Self::parse_metadata(
    ///     &schema.metadata,
    ///     "lance:archive:dataset_created",
    ///     0,
    ///     &path,
    ///     "i64"
    /// );
    /// ```
    fn parse_metadata<T: std::str::FromStr + Default + std::fmt::Debug>(
        metadata: &HashMap<String, String>,
        key: &str,
        default: T,
        path: &Path,
        type_name: &str,
    ) -> T
    where
        <T as std::str::FromStr>::Err: std::fmt::Display,
    {
        match metadata.get(key) {
            Some(metadata_value) => match metadata_value.parse::<T>() {
                Ok(parsed_value) => parsed_value,
                Err(parse_error) => {
                    tracing::warn!(
                        "Failed to parse {} metadata '{}' as {}: {}, using default {:?}",
                        key,
                        metadata_value,
                        type_name,
                        parse_error,
                        default
                    );
                    default
                }
            },
            None => {
                tracing::warn!(
                    "Missing {} metadata in archive file {}, using default {:?}",
                    key,
                    path,
                    default
                );
                default
            }
        }
    }

    #[cfg(test)]
    fn config(&self) -> &VersionArchiveConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use lance_io::object_store::ObjectStore;
    use lance_table::format::ManifestSummary;

    use super::*;

    fn archived_version(version: u64) -> ArchivedVersion {
        ArchivedVersion {
            version,
            timestamp_millis: version as i64 * 1000,
            manifest_summary: ManifestSummary {
                total_fragments: version,
                total_data_files: version,
                total_files_size: version * 100,
                total_deletion_files: 0,
                total_data_file_rows: version * 100,
                total_deletion_file_rows: 0,
                total_rows: version * 100,
            },
            is_tagged: false,
            transaction_uuid: None,
            read_version: None,
            operation_type: None,
            transaction_properties: HashMap::new(),
        }
    }

    struct ArchiveTestFixture {
        archive: VersionArchive,
    }

    impl ArchiveTestFixture {
        async fn new() -> Self {
            Self::new_with_config(VersionArchiveConfig::default()).await
        }

        async fn new_with_config(config: VersionArchiveConfig) -> Self {
            let object_store = ObjectStore::new(
                Arc::new(object_store::memory::InMemory::new()),
                url::Url::parse("memory://").unwrap(),
                None,
                None,
                false,
                true,
                1,
                3,
                None,
            );
            let archive =
                VersionArchive::load_or_new(Path::from("test"), Arc::new(object_store), config)
                    .await
                    .unwrap();
            Self { archive }
        }

        async fn load(&self) -> VersionArchive {
            VersionArchive::load_or_new(
                self.archive.base.clone(),
                self.archive.object_store.clone(),
                *self.archive.config(),
            )
            .await
            .unwrap()
        }
    }

    #[tokio::test]
    async fn test_archive_add_and_flush() {
        let mut fixture = ArchiveTestFixture::new().await;
        assert!(fixture.archive.versions.is_empty());
        assert_eq!(fixture.archive.latest_version_number, 0);
        assert!(
            fixture.archive.is_enabled(),
            "Archive should be enabled by default"
        );

        fixture
            .archive
            .add(&[archived_version(1), archived_version(2)]);
        fixture.archive.flush().await.unwrap();

        assert_eq!(fixture.archive.versions.len(), 2);
        assert_eq!(fixture.archive.latest_version_number, 2);

        // Test disabled archive - is_enabled() returns false
        let disabled_fixture = ArchiveTestFixture::new_with_config(VersionArchiveConfig {
            enabled: false,
            ..Default::default()
        })
        .await;
        assert!(
            !disabled_fixture.archive.is_enabled(),
            "Archive should be disabled when enabled=false"
        );
    }

    #[tokio::test]
    async fn test_load_sorts_and_preserves_tag() {
        let mut fixture = ArchiveTestFixture::new().await;
        let mut tagged = archived_version(2);
        tagged.is_tagged = true;
        fixture
            .archive
            .add(&[archived_version(3), archived_version(1), tagged]);
        fixture.archive.flush().await.unwrap();

        let loaded = fixture.load().await;
        assert_eq!(loaded.versions.len(), 3);
        assert_eq!(loaded.versions[0].version, 1);
        assert_eq!(loaded.versions[1].version, 2);
        assert_eq!(loaded.versions[2].version, 3);
        assert!(loaded.versions[1].is_tagged);
        assert_eq!(loaded.latest_version_number, 3);
    }

    #[tokio::test]
    async fn test_archive_truncation() {
        let mut fixture = ArchiveTestFixture::new_with_config(VersionArchiveConfig {
            max_entries: 2,
            ..Default::default()
        })
        .await;

        fixture.archive.add(&[
            archived_version(1),
            archived_version(2),
            archived_version(3),
        ]);
        fixture.archive.flush().await.unwrap();

        assert_eq!(fixture.archive.versions.len(), 2);
        assert_eq!(fixture.archive.versions[0].version, 2);
        assert_eq!(fixture.archive.versions[1].version, 3);
    }

    #[tokio::test]
    async fn test_archive_corruption_graceful_degradation() {
        let mut fixture = ArchiveTestFixture::new().await;
        fixture.archive.add(&[archived_version(1)]);
        fixture.archive.flush().await.unwrap();

        let archive_dir = fixture.archive.archive_dir();

        // Corrupt the archive file
        let lance_path = archive_dir.join(archive_filename(1));
        fixture
            .archive
            .object_store
            .put(&lance_path, b"corrupted data")
            .await
            .unwrap();

        let loaded = VersionArchive::load_or_new(
            fixture.archive.base.clone(),
            fixture.archive.object_store.clone(),
            *fixture.archive.config(),
        )
        .await
        .unwrap();

        assert!(loaded.versions.is_empty());
    }

    #[test]
    fn test_config_from_config() {
        assert_eq!(
            VersionArchiveConfig::from_config(&HashMap::new()),
            VersionArchiveConfig::default()
        );

        let mut config = HashMap::new();
        config.insert(
            "lance.version_archive.enabled".to_string(),
            "false".to_string(),
        );
        config.insert(
            "lance.version_archive.max_entries".to_string(),
            "100".to_string(),
        );
        config.insert(
            "lance.version_archive.max_archive_files".to_string(),
            "5".to_string(),
        );

        let archive_config = VersionArchiveConfig::from_config(&config);
        assert!(!archive_config.enabled);
        assert_eq!(archive_config.max_entries, 100);
        assert_eq!(archive_config.max_archive_files, 5);
    }

    #[tokio::test]
    async fn test_max_archive_files_cleanup() {
        let mut fixture = ArchiveTestFixture::new_with_config(VersionArchiveConfig {
            max_archive_files: 2,
            ..Default::default()
        })
        .await;

        for i in 1..=4 {
            fixture.archive.add(&[archived_version(i)]);
            fixture.archive.flush().await.unwrap();
        }

        let archive_dir = fixture.archive.archive_dir();
        let mut versions = Vec::new();
        let mut stream = fixture.archive.object_store.list(Some(archive_dir));
        while let Some(meta) = stream.next().await.transpose().unwrap() {
            if let Some(filename) = meta.location.filename()
                && let Some(version) = filename
                    .strip_suffix(VERSION_ARCHIVE_FILE_SUFFIX)
                    .and_then(|s| s.parse::<u64>().ok())
            {
                versions.push(version);
            }
        }

        // Verify only 2 files remain
        assert_eq!(versions.len(), 2, "Expected 2 archive files");

        // Verify the newest files are retained (v3 and v4)
        versions.sort();
        assert_eq!(
            versions,
            vec![3, 4],
            "Expected to retain the newest archive files (v3 and v4)"
        );
    }

    #[tokio::test]
    async fn test_load_newest_valid_archive() {
        let mut fixture = ArchiveTestFixture::new().await;

        fixture.archive.add(&[archived_version(1)]);
        fixture.archive.flush().await.unwrap();

        fixture.archive.add(&[archived_version(2)]);
        fixture.archive.flush().await.unwrap();

        let archive_dir = fixture.archive.archive_dir();

        // Corrupt v2 archive to ensure we fall back to v1
        let v2_path = archive_dir.join(archive_filename(2));
        fixture
            .archive
            .object_store
            .put(&v2_path, b"corrupted")
            .await
            .unwrap();

        let loaded = VersionArchive::load_or_new(
            fixture.archive.base.clone(),
            fixture.archive.object_store.clone(),
            *fixture.archive.config(),
        )
        .await
        .unwrap();

        assert_eq!(loaded.latest_version_number, 1);
    }

    #[tokio::test]
    async fn test_transaction_properties_limits() {
        let mut fixture = ArchiveTestFixture::new().await;

        let mut small = archived_version(1);
        small
            .transaction_properties
            .insert("key".to_string(), "value".to_string());

        let mut limited = archived_version(2);
        for i in 0..12 {
            limited
                .transaction_properties
                .insert(format!("key_{:02}", i), format!("value_{:02}", i));
        }
        limited
            .transaction_properties
            .insert("key_05".to_string(), "x".repeat(250));

        fixture.archive.add(&[small, limited]);
        fixture.archive.flush().await.unwrap();

        let loaded = fixture.load().await;

        assert_eq!(loaded.versions.len(), 2);
        assert_eq!(
            loaded.versions[0].transaction_properties.get("key"),
            Some(&"value".to_string())
        );

        let limited = &loaded.versions[1].transaction_properties;
        assert_eq!(limited.get("_truncated"), Some(&"true".to_string()));
        assert_eq!(limited.get("_original_key_count"), Some(&"12".to_string()));
        assert_eq!(limited.get("_dropped_key_count"), Some(&"2".to_string()));
        assert_eq!(limited.get("key_00"), Some(&"value_00".to_string()));
        assert_eq!(
            limited.get("key_05").map(|value| value.len()),
            Some(MAX_TRANSACTION_PROPERTY_VALUE_BYTES)
        );
        assert_eq!(limited.get("key_09"), Some(&"value_09".to_string()));
        assert!(!limited.contains_key("key_10"));
        assert!(!limited.contains_key("transaction_properties"));
        assert_eq!(
            limited.get("_truncated_value:key_05"),
            Some(&"250".to_string())
        );

        let encoded = encode_transaction_properties(limited).unwrap().unwrap();
        let decoded: HashMap<String, String> = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.get("key_05"), limited.get("key_05"));
    }
}
