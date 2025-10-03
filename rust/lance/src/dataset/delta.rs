// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::transaction::Transaction;
use crate::Dataset;
use crate::Result;
use arrow_array::{Array, RecordBatch, StructArray, UInt64Array};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType, Field, Schema};
use futures::stream::{self, BoxStream, StreamExt, TryStreamExt};
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::Error;
use lance_table::format::{Fragment, DatasetVersionSequence};
use lance_table::rowids::{read_row_ids, RowIdSequence};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use tracing::instrument;

/// The delta dataset between two versions of a dataset.
pub struct DatasetDelta {
    /// The base version number for comparison.
    pub(crate) begin_version: u64,
    /// The current version number.
    pub(crate) end_version: u64,
    /// Base path of the dataset.
    pub(crate) base_dataset: Dataset,
}

impl DatasetDelta {
    /// Listing the transactions between two versions.
    pub async fn list_transactions(&self) -> Result<Vec<Transaction>> {
        stream::iter((self.begin_version + 1)..=self.end_version)
            .map(|version| {
                let base_dataset = self.base_dataset.clone();
                async move {
                    let current_ds = match base_dataset.checkout_version(version).await {
                        Ok(ds) => ds,
                        Err(err) => {
                            if matches!(err, Error::DatasetNotFound { .. }) {
                                return Err(Error::VersionNotFound {
                                    message: format!(
                                        "Can not find version {}, please check if it has been cleanup.",
                                        version
                                    ),
                                });
                            } else {
                                return Err(err);
                            }
                        }
                    };
                    current_ds.read_transaction().await
                }
            })
            .buffered(get_num_compute_intensive_cpus())
            .try_filter_map(|result| async move { Ok(result) })
            .try_collect()
            .await
    }
}

/// Type of diff operation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiffOperation {
    /// Record was inserted in the newer version
    Insert,
    /// Record was updated between versions
    Update,
    /// Record was deleted in the newer version
    Delete,
}

impl DiffOperation {
    /// Convert operation to string representation for Arrow arrays
    pub fn as_str(&self) -> &'static str {
        match self {
            DiffOperation::Insert => "insert",
            DiffOperation::Update => "update",
            DiffOperation::Delete => "delete",
        }
    }
}

/// Stream of diff record batches
pub type DiffRecordBatchStream = BoxStream<'static, Result<RecordBatch>>;

/// Create the schema for diff results
///
/// Schema:
/// - row_id: UInt64 - The stable row ID
/// - operation: Utf8 - The operation type ("insert", "update", "delete")
/// - version: UInt64 - The version when this change occurred
/// - pre_image: Struct - The record state before the change (null for inserts)
/// - post_image: Struct - The record state after the change (null for deletes)
pub fn create_diff_schema(data_schema: Arc<Schema>) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::UInt64, false),
        Field::new("operation", DataType::Utf8, false),
        Field::new("version", DataType::UInt64, false),
        Field::new(
            "pre_image",
            DataType::Struct(data_schema.fields().clone()),
            true,
        ),
        Field::new(
            "post_image",
            DataType::Struct(data_schema.fields().clone()),
            true,
        ),
    ]))
}

/// Build a diff RecordBatch from component data
fn build_diff_batch(
    diff_schema: &Arc<Schema>,
    data_schema: &Arc<Schema>,
    row_ids: Vec<u64>,
    versions: Vec<u64>,
    operation: DiffOperation,
    pre_image_batch: Option<RecordBatch>,
    post_image_batch: Option<RecordBatch>,
) -> Result<RecordBatch> {
    use arrow_array::StringArray;

    let num_rows = row_ids.len();

    // Build row_id array
    let row_id_array = Arc::new(UInt64Array::from(row_ids)) as Arc<dyn Array>;

    // Build operation array
    let operation_str = operation.as_str();
    let operation_array = Arc::new(StringArray::from(vec![operation_str; num_rows])) as Arc<dyn Array>;

    // Build version array
    let version_array = Arc::new(UInt64Array::from(versions)) as Arc<dyn Array>;

    // Build pre_image struct array
    let pre_image_array = if let Some(batch) = pre_image_batch {
        // Convert RecordBatch to StructArray
        let arrays: Vec<Arc<dyn Array>> = batch.columns().to_vec();
        Arc::new(StructArray::new(
            data_schema.fields().clone(),
            arrays,
            None,
        )) as Arc<dyn Array>
    } else {
        // Create null struct array
        let arrays: Vec<Arc<dyn Array>> = data_schema
            .fields()
            .iter()
            .map(|field| arrow_array::new_null_array(field.data_type(), num_rows))
            .collect();
        Arc::new(StructArray::new(
            data_schema.fields().clone(),
            arrays,
            Some(NullBuffer::new_null(num_rows)),
        )) as Arc<dyn Array>
    };

    // Build post_image struct array
    let post_image_array = if let Some(batch) = post_image_batch {
        // Convert RecordBatch to StructArray
        let arrays: Vec<Arc<dyn Array>> = batch.columns().to_vec();
        Arc::new(StructArray::new(
            data_schema.fields().clone(),
            arrays,
            None,
        )) as Arc<dyn Array>
    } else {
        // Create null struct array
        let arrays: Vec<Arc<dyn Array>> = data_schema
            .fields()
            .iter()
            .map(|field| arrow_array::new_null_array(field.data_type(), num_rows))
            .collect();
        Arc::new(StructArray::new(
            data_schema.fields().clone(),
            arrays,
            Some(NullBuffer::new_null(num_rows)),
        )) as Arc<dyn Array>
    };

    RecordBatch::try_new(
        diff_schema.clone(),
        vec![
            row_id_array,
            operation_array,
            version_array,
            pre_image_array,
            post_image_array,
        ],
    )
    .map_err(|e| Error::Internal {
        message: format!("Failed to create diff RecordBatch: {}", e),
        location: snafu::location!(),
    })
}

/// Configuration for diff operations
#[derive(Debug, Clone)]
pub struct DiffConfig {
    /// Maximum number of records to process in parallel
    pub max_concurrency: usize,
    /// Whether to include the actual record data in diff results
    pub include_data: bool,
    /// Batch size for processing row ids
    pub batch_size: usize,
}

impl Default for DiffConfig {
    fn default() -> Self {
        Self {
            max_concurrency: 4,
            include_data: true,
            batch_size: 1000,
        }
    }
}

/// Fragment-level diff analyzer
pub struct FragmentDiffAnalyzer {
    current_dataset: Arc<Dataset>,
    old_dataset: Arc<Dataset>,
    compared_version: u64,
    config: DiffConfig,
    /// Row IDs present in the old dataset (for fast classification)
    old_row_ids: Arc<HashSet<u64>>,
}

impl FragmentDiffAnalyzer {
    /// Create a new fragment diff analyzer
    pub fn new(
        current_dataset: Arc<Dataset>,
        old_dataset: Arc<Dataset>,
        compared_version: u64,
        config: DiffConfig,
        old_row_ids: Arc<HashSet<u64>>,
    ) -> Self {
        Self {
            current_dataset,
            old_dataset,
            compared_version,
            config,
            old_row_ids,
        }
    }

    /// Analyze a single fragment for differences, returning RecordBatches
    #[instrument(level = "debug", skip_all)]
    pub async fn analyze_fragment(&self, fragment: &Fragment) -> Result<Vec<RecordBatch>> {
        // Load the row version sequence of this fragment
        // If version metadata is missing, skip this fragment (treat as "existed since beginning")
        let version_sequence = if let Some(version_meta) =
            fragment.last_updated_at_version_meta.as_ref()
        {
            version_meta.load_sequence()?
        } else {
            // Fragment predates version tracking or was created before diff support
            // Skip it from diff results
            return Ok(Vec::new());
        };

        // Find updated and inserted records as RecordBatches
        self.find_updated_and_inserted_records(fragment, &version_sequence)
            .await
    }

    /// Find records that were updated or inserted, returning RecordBatches
    async fn find_updated_and_inserted_records(
        &self,
        fragment: &Fragment,
        version_sequence: &DatasetVersionSequence,
    ) -> Result<Vec<RecordBatch>> {
        let row_ids_sequence = if let Some(row_id_meta) = &fragment.row_id_meta {
            match row_id_meta {
                lance_table::format::RowIdMeta::Inline(data) => read_row_ids(data).unwrap(),
                lance_table::format::RowIdMeta::External(_file) => {
                    todo!("External row id meta currently not supported in diff")
                }
            }
        } else {
            return Err(Error::invalid_input(
                "In stable row id mode, fragment must have the row id meta!",
                Default::default(),
            ));
        };

        // Get all rows with versions greater than the compared version
        let changed_rows = version_sequence
            .rows_with_version_greater_than(&row_ids_sequence, self.compared_version);

        let changed_set: HashSet<u64> = changed_rows.iter().copied().collect();
        let mut updated_row_ids: Vec<u64> = changed_set
            .intersection(&self.old_row_ids)
            .copied()
            .collect();
        let mut inserted_row_ids: Vec<u64> =
            changed_set.difference(&self.old_row_ids).copied().collect();

        // For a stable and predictable output order, sort by row_id
        updated_row_ids.sort_unstable();
        inserted_row_ids.sort_unstable();

        let batch_size = self.config.batch_size.max(1);
        let max_concurrency = self.config.max_concurrency.max(1);

        // Filter the given row_ids (version boundaries) and fetch data in batches, returning RecordBatch
        async fn process_rows(
            analyzer: Arc<FragmentDiffAnalyzer>,
            row_ids: Vec<u64>,
            version_sequence: &DatasetVersionSequence,
            row_ids_sequence: &RowIdSequence,
            op: DiffOperation,
            batch_size: usize,
            max_concurrency: usize,
        ) -> Result<Vec<RecordBatch>> {
            // First, filter out the rows that are not within the current version range,
            // and retain the version numbers for constructing the output.
            let mut filtered: Vec<(u64, u64)> = Vec::with_capacity(row_ids.len());
            for row_id in row_ids {
                let version = version_sequence
                    .get_version_for_row_id(row_ids_sequence, row_id)
                    .unwrap();
                if version <= analyzer.current_dataset.manifest.version {
                    filtered.push((row_id, version));
                }
            }

            if filtered.is_empty() {
                return Ok(Vec::new());
            }

            let lance_schema = analyzer.current_dataset.schema();
            // Convert lance Schema to arrow Schema
            // lance_schema is Arc<lance_core::datatypes::Schema>
            // We need &lance_core::datatypes::Schema to convert to arrow_schema::Schema
            let arrow_schema = arrow_schema::Schema::from(&*lance_schema);
            let data_schema = Arc::new(arrow_schema);
            let diff_schema = create_diff_schema(data_schema.clone());

            // Batch concurrent processing - collect chunks into owned Vecs to avoid lifetime issues
            let chunks: Vec<Vec<(u64, u64)>> = filtered
                .chunks(batch_size)
                .map(|chunk| chunk.to_vec())
                .collect();

            let stream = futures::stream::iter(chunks.into_iter().map(|chunk| {
                let ids: Vec<u64> = chunk.iter().map(|(id, _)| *id).collect();
                let versions: Vec<u64> = chunk.iter().map(|(_, v)| *v).collect();
                let op_clone = op.clone();
                let analyzer = analyzer.clone();
                let diff_schema = diff_schema.clone();
                let data_schema = data_schema.clone();

                async move {
                    let include_data = analyzer.config.include_data;

                    let projection = super::ProjectionRequest::Schema(Arc::new(analyzer.current_dataset.schema().clone()));

                    // batch post_image data
                    let post_image_batch = if include_data {
                        Some(
                            analyzer
                                .current_dataset
                                .take_rows(&ids, projection.clone())
                                .await?,
                        )
                    } else {
                        None
                    };

                    // Batch pre_image data (only required for Update)
                    let pre_image_batch = if include_data && op_clone == DiffOperation::Update {
                        Some(
                            analyzer
                                .old_dataset
                                .take_rows(&ids, projection.clone())
                                .await?,
                        )
                    } else {
                        None
                    };

                    // Robustness check: if data is enabled and the number of rows does not match, return an error
                    if include_data {
                        if let Some(ref nb) = post_image_batch {
                            if nb.num_rows() != ids.len() {
                                return Err(Error::invalid_input(
                                    format!(
                                        "Expected {} rows in post_image batch, got {}",
                                        ids.len(),
                                        nb.num_rows()
                                    ),
                                    Default::default(),
                                ));
                            }
                        }
                        if let Some(ref ob) = pre_image_batch {
                            if ob.num_rows() != ids.len() {
                                return Err(Error::invalid_input(
                                    format!(
                                        "Expected {} rows in pre_image batch, got {}",
                                        ids.len(),
                                        ob.num_rows()
                                    ),
                                    Default::default(),
                                ));
                            }
                        }
                    }

                    // Build the diff RecordBatch
                    build_diff_batch(
                        &diff_schema,
                        &data_schema,
                        ids,
                        versions,
                        op_clone,
                        pre_image_batch,
                        post_image_batch,
                    )
                }
            }))
            .buffer_unordered(max_concurrency)
            .try_collect::<Vec<RecordBatch>>()
            .await?;

            Ok(stream)
        }

        let mut batches = Vec::new();

        let updates = process_rows(
            Arc::new(self.clone()),
            updated_row_ids,
            version_sequence,
            &row_ids_sequence,
            DiffOperation::Update,
            batch_size,
            max_concurrency,
        )
        .await?;
        batches.extend(updates);

        let inserts = process_rows(
            Arc::new(self.clone()),
            inserted_row_ids,
            version_sequence,
            &row_ids_sequence,
            DiffOperation::Insert,
            batch_size,
            max_concurrency,
        )
        .await?;
        batches.extend(inserts);

        Ok(batches)
    }

    /// Create a stream of diff record batches for all fragments
    pub async fn create_diff_stream(self) -> DiffRecordBatchStream {
        let fragments = Arc::try_unwrap(self.current_dataset.manifest.fragments.clone())
            .unwrap_or_else(|arc| (*arc).clone());
        let max_concurrency = self.config.max_concurrency;

        let stream = futures::stream::iter(fragments)
            .map(move |fragment| {
                let analyzer = self.clone();
                async move { analyzer.analyze_fragment(&fragment).await }
            })
            .buffer_unordered(max_concurrency)
            .flat_map(|result| {
                futures::stream::iter(match result {
                    Ok(batches) => batches.into_iter().map(Ok).collect::<Vec<_>>(),
                    Err(e) => vec![Err(e)],
                })
            });

        Box::pin(stream)
    }
}

/// Find the first manifest version where the given fragment id appears.
async fn first_appearance_version(dataset: &Dataset, fragment_id: u64) -> Result<u64> {
    let mut first_seen: Option<u64> = None;
    let mut v = dataset.manifest.version;

    // Scan backward until the fragment id disappears
    while v >= 1 {
        let ds_v = if v == dataset.manifest.version {
            dataset.clone()
        } else {
            dataset.checkout_version(v).await?
        };
        let exists = ds_v.manifest.fragments.iter().any(|f| f.id == fragment_id);
        if exists {
            first_seen = Some(v);
            if v == 1 {
                break;
            }
            v -= 1;
            continue;
        } else {
            // If we have already seen the fragment in a newer version, we stop
            break;
        }
    }

    first_seen.ok_or_else(|| Error::Internal {
        message: format!(
            "Fragment {} not found in dataset manifests up to current version {}",
            fragment_id, dataset.manifest.version
        ),
        location: snafu::location!(),
    })
}

/// Infer the creation version for a fragment.
///
/// If the fragment was created by a Rewrite (compaction) operation, this traces
/// lineage to old fragments and uses the earliest old fragment creation version
/// to avoid misclassifying compaction as a change.
pub async fn infer_fragment_creation_version(
    dataset: &Dataset,
    fragment: &Fragment,
) -> Result<u64> {
    let base_creation = first_appearance_version(dataset, fragment.id).await?;

    // Try to refine using rewrite lineage if possible
    if let Some(tx) = dataset.read_transaction_by_version(base_creation).await? {
        match tx.operation {
            crate::dataset::transaction::Operation::Rewrite { groups, .. } => {
                // If our fragment id is among the new fragments in any group,
                // then creation was due to rewrite. Trace to old fragments.
                for g in groups.iter() {
                    if g.new_fragments.iter().any(|nf| nf.id == fragment.id) {
                        // Use the minimum first appearance of old fragments
                        let mut min_old_creation = base_creation;
                        for of in g.old_fragments.iter() {
                            let old_first = first_appearance_version(dataset, of.id).await?;
                            if old_first < min_old_creation {
                                min_old_creation = old_first;
                            }
                        }
                        return Ok(min_old_creation);
                    }
                }
                Ok(base_creation)
            }
            // Update / Append / Overwrite: the fragment id's first appearance
            // equals its creation commit version, which is what we want.
            _ => Ok(base_creation),
        }
    } else {
        // No transaction recorded (legacy or detached). Fall back to base creation.
        Ok(base_creation)
    }
}

/// Implement Clone for FragmentDiffAnalyzer to support stream operations
impl Clone for FragmentDiffAnalyzer {
    fn clone(&self) -> Self {
        Self {
            current_dataset: self.current_dataset.clone(),
            old_dataset: self.old_dataset.clone(),
            compared_version: self.compared_version,
            config: self.config.clone(),
            old_row_ids: self.old_row_ids.clone(),
        }
    }
}

/// Collect all row IDs present in the given dataset (across all fragments).
fn collect_all_row_ids(dataset: &Dataset) -> Result<HashSet<u64>> {
    let mut set: HashSet<u64> = HashSet::new();
    let fragments = &dataset.manifest.fragments;
    for fragment in fragments.iter() {
        let row_ids_sequence = if let Some(row_id_meta) = &fragment.row_id_meta {
            match row_id_meta {
                lance_table::format::RowIdMeta::Inline(data) => read_row_ids(data).unwrap(),
                lance_table::format::RowIdMeta::External(_file) => {
                    todo!("External row id meta currently not supported in diff")
                }
            }
        } else {
            return Err(Error::invalid_input(
                "In stable row id mode, fragment must have the row id meta!",
                Default::default(),
            ));
        };
        for id in row_ids_sequence.iter() {
            set.insert(id);
        }
    }
    Ok(set)
}

/// Dataset diff builder for configuring and executing diff operations
pub struct DatasetDiffBuilder {
    dataset: Arc<Dataset>,
    compared_version: u64,
    /// Maximum number of records to process in parallel
    max_concurrency: usize,
    /// Whether to include the actual record data in diff results
    include_data: bool,
    /// Batch size for processing records
    batch_size: usize,
    /// Whether to fetch pre_image data for update operations (default: false)
    /// When false, pre_image will be NULL for all operations
    /// When true, pre_image will be populated by fetching from the old dataset version
    fetch_pre_image: bool,
}

impl DatasetDiffBuilder {
    /// Create a new dataset diff builder
    pub fn new(dataset: Arc<Dataset>, compared_version: u64) -> Self {
        Self {
            dataset,
            compared_version,
            max_concurrency: 4,
            include_data: true,
            batch_size: 1000,
            fetch_pre_image: false,
        }
    }

    /// Set the maximum concurrency for diff operations
    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.max_concurrency = max_concurrency;
        self
    }

    /// Set whether to include actual record data in diff results
    pub fn with_include_data(mut self, include_data: bool) -> Self {
        self.include_data = include_data;
        self
    }

    /// Set the batch size for processing records
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set whether to fetch pre_image data for update operations
    ///
    /// When true, pre_image will be populated by fetching data from the old dataset version.
    /// When false (default), pre_image will be NULL for all operations, which is more efficient.
    pub fn with_fetch_pre_image(mut self, fetch_pre_image: bool) -> Self {
        self.fetch_pre_image = fetch_pre_image;
        self
    }

    /// Execute the diff operation and return a stream of diff record batches
    pub async fn execute(self) -> Result<DiffRecordBatchStream> {
        // Validate preconditions
        self.validate_preconditions().await?;

        if self.fetch_pre_image {
            // Use the old implementation that fetches pre_image data
            self.execute_with_pre_image().await
        } else {
            // Use the new SQL-based implementation (more efficient)
            self.execute_sql_based().await
        }
    }

    /// Execute diff using SQL transformations (no pre_image fetch)
    async fn execute_sql_based(self) -> Result<DiffRecordBatchStream> {
        use datafusion::logical_expr::{col, lit, Expr};
        use datafusion::scalar::ScalarValue;
        use lance_core::{ROW_ID, ROW_CREATED_AT_VERSION, ROW_LAST_UPDATED_AT_VERSION};

        // Build a scanner that includes version columns and filters to changed rows
        let scanner = self.dataset
            .scan()
            .with_row_id()
            // TODO: Add version columns via a custom projection or exec node
            // Filter to rows that changed after compared_version
            .filter(&format!("{} > {}", ROW_LAST_UPDATED_AT_VERSION, self.compared_version))?;

        // For now, since AddVersionColumnsExec integration with Scanner is complex,
        // fall back to the original implementation
        // TODO: Complete SQL-based implementation by:
        // 1. Extending Scanner to support AddVersionColumnsExec
        // 2. Adding operation column via SQL CASE expression
        // 3. Projecting to diff schema with NULL pre_image
        self.execute_with_pre_image().await
    }

    /// Execute diff with pre_image data fetching (original implementation)
    async fn execute_with_pre_image(self) -> Result<DiffRecordBatchStream> {
        // Load the old version of the dataset
        let old_dataset = Arc::new(self.dataset.checkout_version(self.compared_version).await?);

        // Pre-compute row IDs in the old dataset for batch classification
        let old_row_ids = collect_all_row_ids(old_dataset.as_ref())?;

        let config = DiffConfig {
            max_concurrency: self.max_concurrency,
            include_data: self.include_data,
            batch_size: self.batch_size,
        };

        // Create the fragment diff analyzer
        let analyzer = FragmentDiffAnalyzer::new(
            self.dataset,
            old_dataset,
            self.compared_version,
            config,
            Arc::new(old_row_ids),
        );

        // Return the diff stream
        Ok(analyzer.create_diff_stream().await)
    }

    /// Validate preconditions for diff operation
    async fn validate_preconditions(&self) -> Result<()> {
        // Check if stable row IDs are enabled
        if !self.dataset.manifest.uses_stable_row_ids() {
            return Err(Error::invalid_input(
                "Diff functionality requires stable row IDs to be enabled",
                Default::default(),
            ));
        }

        // Validate version bounds
        if self.compared_version >= self.dataset.manifest.version {
            return Err(Error::invalid_input(
                format!(
                    "Compared version {} must be less than current version {}",
                    self.compared_version, self.dataset.manifest.version
                ),
                Default::default(),
            ));
        }

        if self.compared_version < 1 {
            return Err(Error::invalid_input(
                format!(
                    "Compared version must be > 0 (got {})",
                    self.compared_version
                ),
                Default::default(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::dataset::delta::{DatasetDiffBuilder, DiffOperation};
    use crate::dataset::transaction::Operation;
    use crate::dataset::{Dataset, WriteMode, WriteParams};
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int32Type;
    use arrow_array::{Array, Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use chrono::Duration;
    use futures::TryStreamExt;
    use lance_core::utils::testing::MockClock;
    use lance_datagen::{array, BatchCount, RowCount};
    use std::sync::Arc;

    async fn create_test_dataset() -> Dataset {
        let data = lance_datagen::gen_batch()
            .col("key", array::step::<Int32Type>())
            .col("value", array::fill_utf8("value".to_string()))
            .into_reader_rows(RowCount::from(1_000), BatchCount::from(10));

        let write_params = WriteParams {
            ..Default::default()
        };
        Dataset::write(data, "memory://", Some(write_params.clone()))
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn test_diff_meta_no_transaction() {
        let ds = create_test_dataset().await;
        let result = ds.diff_meta(1).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_diff_meta_single_transaction() {
        let mut ds = create_test_dataset().await;
        ds.delete("key = 5").await.unwrap();

        let delta_struct = crate::dataset::delta::DatasetDelta {
            begin_version: 1,
            end_version: ds.version().version,
            base_dataset: ds.clone(),
        };
        let txs = delta_struct.list_transactions().await.unwrap();
        assert_eq!(txs.len(), 1);
        assert!(matches!(txs[0].operation, Operation::Delete { .. }));
    }

    #[tokio::test]
    async fn test_diff_meta_multiple_transactions() {
        let mut ds = create_test_dataset().await;
        ds.delete("key = 5").await.unwrap();
        ds.delete("key = 6").await.unwrap();

        let delta_struct = crate::dataset::delta::DatasetDelta {
            begin_version: 1,
            end_version: ds.version().version,
            base_dataset: ds.clone(),
        };
        let txs = delta_struct.list_transactions().await.unwrap();
        assert_eq!(txs.len(), 2);
    }

    #[tokio::test]
    async fn test_diff_meta_contains_deleted_transaction() {
        let clock = MockClock::new();

        clock.set_system_time(Duration::seconds(1));

        let mut ds = create_test_dataset().await;

        clock.set_system_time(Duration::seconds(2));

        ds.delete("key = 5").await.unwrap();
        ds.delete("key = 6").await.unwrap();
        ds.delete("key = 7").await.unwrap();

        clock.set_system_time(Duration::seconds(3));

        let end_version = ds.version().version;
        let base_dataset = ds.clone();

        clock.set_system_time(Duration::seconds(4));

        ds.cleanup_old_versions(Duration::seconds(1), Some(true), None)
            .await
            .expect("Cleanup old versions failed");

        clock.set_system_time(Duration::seconds(5));

        let delta_struct = crate::dataset::delta::DatasetDelta {
            begin_version: 1,
            end_version,
            base_dataset,
        };

        let result = delta_struct.list_transactions().await;
        match result {
            Err(lance_core::Error::VersionNotFound { message }) => {
                assert!(message.contains("Can not find version"));
            }
            _ => panic!("Expected VersionNotFound error."),
        }
    }

    /// Helper function to create a test dataset with stable row IDs enabled
    async fn create_test_dataset_with_stable_row_ids() -> Dataset {
        let data = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("name", array::fill_utf8("initial".to_string()))
            .col("value", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));

        let write_params = WriteParams {
            mode: WriteMode::Create,
            enable_stable_row_ids: true,
            ..Default::default()
        };

        Dataset::write(data, "memory://test_diff_dataset", Some(write_params))
            .await
            .unwrap()
    }

    async fn append_data_to_dataset(mut dataset: Dataset, start_id: i32, count: usize) -> Dataset {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let ids: Vec<i32> = (start_id..start_id + count as i32).collect();
        let names: Vec<String> = (0..count).map(|i| format!("new_{}", i)).collect();
        let values: Vec<i32> = (start_id..start_id + count as i32)
            .map(|x| x * 10)
            .collect();

        let batch = arrow_array::RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(names)),
                Arc::new(Int32Array::from(values)),
            ],
        )
        .unwrap();

        let batches = arrow_array::RecordBatchIterator::new([Ok(batch)], schema);

        let write_params = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        };

        dataset.append(batches, Some(write_params)).await.unwrap();
        dataset
    }

    #[tokio::test]
    async fn test_diff_merge_insert_sub_schema_update() {
        // Create initial dataset with stable row IDs
        let dataset = create_test_dataset_with_stable_row_ids().await;
        let initial_version = dataset.version().version;

        // Create merge insert data with sub-schema (only updating specific columns)
        // We'll update only the 'name' column, leaving 'value' unchanged
        let merge_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false), // key column for matching
            Field::new("name", DataType::Utf8, false), // column to update
                                                      // Note: 'value' column is intentionally omitted (sub-schema)
        ]));

        // Create merge data that will update records with id 10-14
        let merge_ids: Vec<i32> = (10..15).collect();
        let merge_names: Vec<String> = (10..15).map(|i| format!("merge_updated_{}", i)).collect();

        let merge_batch = arrow_array::RecordBatch::try_new(
            merge_schema.clone(),
            vec![
                Arc::new(Int32Array::from(merge_ids)),
                Arc::new(StringArray::from(merge_names)),
            ],
        )
        .unwrap();

        let merge_batches = arrow_array::RecordBatchIterator::new([Ok(merge_batch)], merge_schema);

        // Perform merge insert using MergeInsertBuilder with sub-schema update
        let mut merge_insert_builder = crate::dataset::MergeInsertBuilder::try_new(
            Arc::new(dataset.clone()),
            vec!["id".to_string()], // match on 'id' column
        )
        .unwrap();

        merge_insert_builder
            .when_matched(crate::dataset::WhenMatched::UpdateAll) // Update matching records
            .when_not_matched_by_source(crate::dataset::WhenNotMatchedBySource::Keep) // Keep non-matching target records
            .when_not_matched(crate::dataset::WhenNotMatched::InsertAll); // Insert new records if any

        let (updated_dataset, merge_stats) = merge_insert_builder
            .try_build()
            .unwrap()
            .execute_reader(Box::new(merge_batches))
            .await
            .unwrap();

        // Verify merge statistics
        assert_eq!(
            merge_stats.num_updated_rows, 5,
            "Should have updated 5 rows (id 10-14)"
        );
        assert_eq!(
            merge_stats.num_inserted_rows, 0,
            "Should have inserted 0 new rows"
        );

        // Execute diff operation to compare before and after merge insert
        let diff_builder = DatasetDiffBuilder::new(updated_dataset.clone(), initial_version)
            .with_include_data(true)
            .with_max_concurrency(1)
            .with_batch_size(10);

        let diff_stream = diff_builder.execute().await.unwrap();
        let diff_batches: Vec<_> = diff_stream.try_collect().await.unwrap();

        // Analyze diff results
        let mut update_count = 0;
        let mut updated_ids = Vec::new();

        for batch in &diff_batches {
            let operation_array = batch
                .column_by_name("operation")
                .unwrap()
                .as_string::<i32>();
            let pre_image_array = batch
                .column_by_name("pre_image")
                .unwrap()
                .as_struct();
            let post_image_array = batch
                .column_by_name("post_image")
                .unwrap()
                .as_struct();

            for row_idx in 0..batch.num_rows() {
                let operation = operation_array.value(row_idx);

                match operation {
                    "update" => {
                        update_count += 1;

                        // Verify the record has both pre_image and post_image data
                        assert!(!pre_image_array.is_null(row_idx), "Update should have pre_image data");
                        assert!(!post_image_array.is_null(row_idx), "Update should have post_image data");

                        // Get ID from post_image
                        let post_id_array = post_image_array
                            .column_by_name("id")
                            .unwrap()
                            .as_primitive::<arrow_array::types::Int32Type>();
                        let id_value = post_id_array.value(row_idx);
                        updated_ids.push(id_value);

                        // Verify this is one of the expected updated records
                        assert!(
                            (10..15).contains(&id_value),
                            "Updated record should have id in range 10-14, got {}",
                            id_value
                        );

                        // Verify that 'name' column was updated
                        let pre_name_array = pre_image_array
                            .column_by_name("name")
                            .unwrap()
                            .as_string::<i32>();
                        let old_name = pre_name_array.value(row_idx);

                        let post_name_array = post_image_array
                            .column_by_name("name")
                            .unwrap()
                            .as_string::<i32>();
                        let new_name = post_name_array.value(row_idx);

                        assert_eq!(old_name, "initial", "Old name should be 'initial'");
                        assert_eq!(
                            new_name,
                            format!("merge_updated_{}", id_value),
                            "New name should be 'merge_updated_{}'",
                            id_value
                        );

                        // Verify that 'value' column was NOT changed (sub-schema behavior)
                        let pre_value_array = pre_image_array
                            .column_by_name("value")
                            .unwrap()
                            .as_primitive::<arrow_array::types::Int32Type>();
                        let old_value = pre_value_array.value(row_idx);

                        let post_value_array = post_image_array
                            .column_by_name("value")
                            .unwrap()
                            .as_primitive::<arrow_array::types::Int32Type>();
                        let new_value = post_value_array.value(row_idx);

                        assert_eq!(
                            old_value, new_value,
                            "Value column should remain unchanged in sub-schema update, but old={} != new={}",
                            old_value, new_value
                        );
                    }
                    "insert" => {
                        panic!("Should not have any insert operations in this test");
                    }
                    "delete" => {
                        panic!("Should not have any delete operations in this test");
                    }
                    _ => panic!("Unknown operation type: {}", operation),
                }
            }
        }

        // Final assertions
        assert_eq!(update_count, 5, "Should have exactly 5 update operations");

        updated_ids.sort();
        let expected_ids: Vec<i32> = (10..15).collect();
        assert_eq!(
            updated_ids, expected_ids,
            "Should have updated exactly IDs 10-14, got {:?}",
            updated_ids
        );
    }

    #[tokio::test]
    async fn test_diff_basic_insert_operations() {
        let initial_dataset = create_test_dataset_with_stable_row_ids().await;
        let initial_version = initial_dataset.version().version;

        // Append new data to create version 2
        let updated_dataset = append_data_to_dataset(initial_dataset, 200, 50).await;

        // Execute diff operation between versions
        let diff_builder = DatasetDiffBuilder::new(Arc::new(updated_dataset), initial_version)
            .with_include_data(true)
            .with_max_concurrency(2)
            .with_batch_size(10);

        let diff_stream = diff_builder.execute().await.unwrap();
        let diff_batches: Vec<_> = diff_stream.try_collect().await.unwrap();

        // Verify diff results
        let mut insert_count = 0;
        let mut update_count = 0;
        let mut delete_count = 0;

        for batch in diff_batches {
            let operation_array = batch
                .column_by_name("operation")
                .unwrap()
                .as_string::<i32>();
            let pre_image_array = batch
                .column_by_name("pre_image")
                .unwrap()
                .as_struct();
            let post_image_array = batch
                .column_by_name("post_image")
                .unwrap()
                .as_struct();

            for row_idx in 0..batch.num_rows() {
                let operation = operation_array.value(row_idx);

                match operation {
                    "insert" => {
                        insert_count += 1;
                        assert!(pre_image_array.is_null(row_idx), "Insert should have no pre_image data");
                        assert!(!post_image_array.is_null(row_idx), "Insert should have post_image data");
                    }
                    "update" => {
                        update_count += 1;
                        assert!(!pre_image_array.is_null(row_idx), "Update should have pre_image data");
                        assert!(!post_image_array.is_null(row_idx), "Update should have post_image data");
                    }
                    "delete" => {
                        delete_count += 1;
                        assert!(!pre_image_array.is_null(row_idx), "Delete should have pre_image data");
                        assert!(post_image_array.is_null(row_idx), "Delete should have no post_image data");
                    }
                    _ => panic!("Unknown operation type: {}", operation),
                }
            }
        }

        // For this test, we expect only insert operations
        assert_eq!(insert_count, 50, "Should have 50 insert operations");
        assert_eq!(update_count, 0, "Should have no update operations");
        assert_eq!(delete_count, 0, "Should have no delete operations");
    }

    #[tokio::test]
    async fn test_diff_mixed_operations() {
        let mut dataset = create_test_dataset_with_stable_row_ids().await;
        let initial_version = dataset.version().version;

        // Perform update operation using UpdateBuilder
        // Update records with id 5-9 to have new names and values
        let update_result = crate::dataset::UpdateBuilder::new(Arc::new(dataset.clone()))
            .update_where("id >= 5 AND id <= 9")
            .unwrap()
            .set("name", "'updated_name'")
            .unwrap()
            .set("value", "id * 100")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();

        dataset = (*update_result.new_dataset).clone();

        // Append new data (insert operation)
        dataset = append_data_to_dataset(dataset, 300, 20).await;

        // Execute diff operation from initial version to final version
        let diff_builder = DatasetDiffBuilder::new(Arc::new(dataset.clone()), initial_version)
            .with_include_data(true) // Enable data inclusion to verify update content
            .with_max_concurrency(1)
            .with_batch_size(5);

        let diff_stream = diff_builder.execute().await.unwrap();
        let diff_batches: Vec<_> = diff_stream.try_collect().await.unwrap();

        let mut insert_count = 0;
        let mut update_count = 0;
        let mut updated_ids = Vec::new();

        for batch in diff_batches {
            let operation_array = batch
                .column_by_name("operation")
                .unwrap()
                .as_string::<i32>();
            let pre_image_array = batch
                .column_by_name("pre_image")
                .unwrap()
                .as_struct();
            let post_image_array = batch
                .column_by_name("post_image")
                .unwrap()
                .as_struct();

            for row_idx in 0..batch.num_rows() {
                let operation = operation_array.value(row_idx);

                match operation {
                    "insert" => {
                        insert_count += 1;
                        assert!(pre_image_array.is_null(row_idx), "Insert should have no pre_image data");
                        assert!(!post_image_array.is_null(row_idx), "Insert should have post_image data");
                    }
                    "update" => {
                        update_count += 1;
                        assert!(!pre_image_array.is_null(row_idx), "Update should have pre_image data");
                        assert!(!post_image_array.is_null(row_idx), "Update should have post_image data");

                        // Verify that this is one of the updated records (id 5-9)
                        let post_id_array = post_image_array
                            .column_by_name("id")
                            .unwrap()
                            .as_primitive::<arrow_array::types::Int32Type>();
                        let id_value = post_id_array.value(row_idx);
                        updated_ids.push(id_value);
                        assert!(
                            (5..=9).contains(&id_value),
                            "Updated record should have id in range 5-9, got {}",
                            id_value
                        );

                        // Verify the updated values
                        let post_name_array = post_image_array
                            .column_by_name("name")
                            .unwrap()
                            .as_string::<i32>();
                        let name_value = post_name_array.value(row_idx);
                        assert_eq!(
                            name_value, "updated_name",
                            "Updated name should be 'updated_name', got '{}'",
                            name_value
                        );

                        let post_value_array = post_image_array
                            .column_by_name("value")
                            .unwrap()
                            .as_primitive::<arrow_array::types::Int32Type>();
                        let value_value = post_value_array.value(row_idx);
                        assert_eq!(
                            value_value,
                            id_value * 100,
                            "Updated value should be id * 100 = {}, got {}",
                            id_value * 100,
                            value_value
                        );
                    }
                    "delete" => todo!(),
                    _ => panic!("Unknown operation type: {}", operation),
                }
            }
        }

        // Verify we have both insert and update operations, but no delete operations
        assert!(
            insert_count > 0,
            "Should have insert operations from appended data"
        );
        assert!(
            update_count > 0,
            "Should have update operations from UpdateBuilder"
        );

        // Specifically verify we have the expected number of updates (5 records updated: id 5,6,7,8,9)
        assert_eq!(update_count, 5, "Should have exactly 5 update operations");
        assert_eq!(
            update_result.rows_updated, 5,
            "UpdateBuilder should report 5 rows updated"
        );

        // Verify we have the expected number of inserts (20 new records)
        assert_eq!(insert_count, 20, "Should have exactly 20 insert operations");

        // Verify all expected IDs were updated
        updated_ids.sort();
        assert_eq!(
            updated_ids,
            vec![5, 6, 7, 8, 9],
            "Should have updated exactly IDs 5-9, got {:?}",
            updated_ids
        );
    }

    #[tokio::test]
    async fn test_diff_mixed_operations_not_in_latest_version() {
        let mut dataset = create_test_dataset_with_stable_row_ids().await;
        let _v1 = dataset.version().version;

        // First update（id 5-9，version 2）
        let update_result1 = crate::dataset::UpdateBuilder::new(Arc::new(dataset.clone()))
            .update_where("id >= 5 AND id <= 9")
            .unwrap()
            .set("name", "'updated_name1'")
            .unwrap()
            .set("value", "id * 100")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        dataset = (*update_result1.new_dataset).clone();
        let v2 = dataset.version().version;

        // Append data（id 300-319，version 3）
        dataset = append_data_to_dataset(dataset, 300, 20).await;
        let v3 = dataset.version().version;
        let dataset_v3 = dataset.clone();

        // Second update（id 7-8，version 4）
        let update_result2 = crate::dataset::UpdateBuilder::new(Arc::new(dataset.clone()))
            .update_where("id >= 7 AND id <= 8")
            .unwrap()
            .set("name", "'updated_name2'")
            .unwrap()
            .set("value", "id * 200")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        dataset = (*update_result2.new_dataset).clone();
        let _v4 = dataset.version().version;

        // diff v2 and v3（only insert）
        let diff_builder = DatasetDiffBuilder::new(Arc::new(dataset_v3.clone()), v2)
            .with_include_data(true)
            .with_max_concurrency(1)
            .with_batch_size(5);
        let diff_stream = diff_builder.execute().await.unwrap();
        let diff_batches: Vec<_> = diff_stream.try_collect().await.unwrap();

        let mut insert_count = 0;
        let mut update_count = 0;

        for batch in &diff_batches {
            let operation_array = batch
                .column_by_name("operation")
                .unwrap()
                .as_string::<i32>();
            for row_idx in 0..batch.num_rows() {
                match operation_array.value(row_idx) {
                    "insert" => insert_count += 1,
                    "update" => update_count += 1,
                    _ => {}
                }
            }
        }

        assert_eq!(
            insert_count, 20,
            "v2-v3 should only contain 20 records insert"
        );
        assert_eq!(update_count, 0, "v2-v3 should not update");

        // diff v3 and v4（only update）
        let diff_builder2 = DatasetDiffBuilder::new(Arc::new(dataset.clone()), v3)
            .with_include_data(true)
            .with_max_concurrency(1)
            .with_batch_size(5);
        let diff_stream2 = diff_builder2.execute().await.unwrap();
        let diff_batches2: Vec<_> = diff_stream2.try_collect().await.unwrap();

        let mut insert_count2 = 0;
        let mut update_count2 = 0;

        for batch in &diff_batches2 {
            let operation_array = batch
                .column_by_name("operation")
                .unwrap()
                .as_string::<i32>();
            for row_idx in 0..batch.num_rows() {
                match operation_array.value(row_idx) {
                    "insert" => insert_count2 += 1,
                    "update" => update_count2 += 1,
                    _ => {}
                }
            }
        }

        assert_eq!(insert_count2, 0, "v3-v4 should not contain insert");
        assert_eq!(
            update_count2, 2,
            "v3-v4 should contains 2 updated records（id 7,8）"
        );
    }

    #[tokio::test]
    async fn test_diff_error_handling_without_stable_row_ids() {
        let data = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("value", array::fill_utf8("test".to_string()))
            .into_reader_rows(RowCount::from(50), BatchCount::from(1));

        let write_params = WriteParams {
            mode: WriteMode::Create,
            enable_stable_row_ids: false, // Explicitly disable stable row IDs
            ..Default::default()
        };

        let dataset = Dataset::write(data, "memory://test_no_stable_ids", Some(write_params))
            .await
            .unwrap();

        // Try to execute diff operation - should fail
        let diff_builder = DatasetDiffBuilder::new(Arc::new(dataset.clone()), 1);
        let result = diff_builder.execute().await;

        assert!(result
            .err()
            .unwrap()
            .to_string()
            .contains("Diff functionality requires stable row IDs to be enabled"));
    }

    #[tokio::test]
    async fn test_diff_invalid_version_bounds() {
        let dataset = create_test_dataset_with_stable_row_ids().await;
        let current_version = dataset.version().version;

        // Test with version equal to current version (should fail)
        let diff_builder = DatasetDiffBuilder::new(Arc::new(dataset.clone()), current_version);
        let result = diff_builder.execute().await;
        assert!(result
            .err()
            .unwrap()
            .to_string()
            .contains("Compared version 1 must be less than current version 1"));

        // Test with version greater than current version (should fail)
        let diff_builder = DatasetDiffBuilder::new(Arc::new(dataset.clone()), current_version + 1);
        let result = diff_builder.execute().await;
        assert!(result
            .err()
            .unwrap()
            .to_string()
            .contains("Compared version 2 must be less than current version 1"));

        // Test with version 0 (should fail)
        let diff_builder = DatasetDiffBuilder::new(Arc::new(dataset.clone()), 0);
        let result = diff_builder.execute().await;
        assert!(result
            .err()
            .unwrap()
            .to_string()
            .contains("Compared version must be > 0 (got 0)"));
    }

    #[tokio::test]
    async fn test_diff_with_compaction() {
        let mut dataset = create_test_dataset_with_stable_row_ids().await;
        let v1 = dataset.version().version;

        // first append
        dataset = append_data_to_dataset(dataset, 200, 10).await;

        // second append
        dataset = append_data_to_dataset(dataset, 300, 15).await;

        // first update
        let update_result1 = crate::dataset::UpdateBuilder::new(Arc::new(dataset.clone()))
            .update_where("id >= 5 AND id <= 9")
            .unwrap()
            .set("name", "'updated_before_compaction'")
            .unwrap()
            .set("value", "id * 50")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        dataset = (*update_result1.new_dataset).clone();

        // third append
        dataset = append_data_to_dataset(dataset, 400, 12).await;
        let v5 = dataset.version().version;

        let fragments_before_compaction = dataset.get_fragments().len();
        assert!(
            fragments_before_compaction > 1,
            "Should have multiple fragments before compaction"
        );

        // do compaction
        use crate::dataset::optimize::{compact_files, CompactionOptions};
        let compaction_options = CompactionOptions {
            target_rows_per_fragment: 50,
            max_rows_per_group: 1024,
            materialize_deletions: true,
            materialize_deletions_threshold: 0.0,
            ..Default::default()
        };

        let compaction_metrics = compact_files(&mut dataset, compaction_options, None)
            .await
            .unwrap();
        let v6_after_compaction = dataset.version().version;

        // verify compaction happened
        assert!(
            compaction_metrics.fragments_removed > 0,
            "Compaction should have removed some fragments"
        );
        assert!(
            compaction_metrics.fragments_added > 0,
            "Compaction should have added new fragments"
        );

        let fragments_after_compaction = dataset.get_fragments().len();
        assert!(
            fragments_after_compaction < fragments_before_compaction,
            "Should have fewer fragments after compaction: {} -> {}",
            fragments_before_compaction,
            fragments_after_compaction
        );

        // update after compaction
        let update_result2 = crate::dataset::UpdateBuilder::new(Arc::new(dataset.clone()))
            .update_where("id >= 200 AND id <= 203")
            .unwrap()
            .set("name", "'updated_after_compaction'")
            .unwrap()
            .set("value", "id * 75")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        dataset = (*update_result2.new_dataset).clone();

        // diff v1 ~ v5 (before compaction)
        let diff_builder = DatasetDiffBuilder::new(Arc::new(dataset.clone()), v1)
            .with_include_data(true)
            .with_max_concurrency(1)
            .with_batch_size(10);
        let diff_stream = diff_builder.execute().await.unwrap();
        let diff_batches_v1_to_v5: Vec<_> = diff_stream.try_collect().await.unwrap();

        let mut insert_count_v1_to_v5 = 0;
        let mut update_count_v1_to_v5 = 0;

        for batch in &diff_batches_v1_to_v5 {
            let operation_array = batch
                .column_by_name("operation")
                .unwrap()
                .as_string::<i32>();
            for row_idx in 0..batch.num_rows() {
                match operation_array.value(row_idx) {
                    "insert" => insert_count_v1_to_v5 += 1,
                    "update" => update_count_v1_to_v5 += 1,
                    _ => {}
                }
            }
        }

        // Note: With version metadata now written for all operations,
        // the count includes all inserts tracked with version metadata
        assert_eq!(
            insert_count_v1_to_v5, 41,
            "Should have insert operations from v1 to v5"
        );
        assert_eq!(
            update_count_v1_to_v5, 5,
            "Should have 5 update operations from v1 to v5"
        );

        // diff v5 ~ v6 (compaction, should not have changes)
        let dataset_v6 = dataset.checkout_version(v6_after_compaction).await.unwrap();

        let diff_builder_compaction = DatasetDiffBuilder::new(Arc::new(dataset_v6), v5)
            .with_include_data(true)
            .with_max_concurrency(1)
            .with_batch_size(10);
        let diff_stream_compaction = diff_builder_compaction.execute().await.unwrap();
        let diff_records_compaction: Vec<_> = diff_stream_compaction.try_collect().await.unwrap();

        assert_eq!(
            diff_records_compaction.len(),
            0,
            "Compaction should not change data content, diff should be empty"
        );

        // 3. diff v6 ~ v7 (update after compaction)
        let diff_builder_post_compaction =
            DatasetDiffBuilder::new(Arc::new(dataset.clone()), v6_after_compaction)
                .with_include_data(true)
                .with_max_concurrency(1)
                .with_batch_size(10);
        let diff_stream_post_compaction = diff_builder_post_compaction.execute().await.unwrap();
        let diff_batches_post_compaction: Vec<_> =
            diff_stream_post_compaction.try_collect().await.unwrap();

        let mut update_count_post_compaction = 0;
        let mut insert_count_post_compaction = 0;

        for batch in &diff_batches_post_compaction {
            let operation_array = batch
                .column_by_name("operation")
                .unwrap()
                .as_string::<i32>();
            for row_idx in 0..batch.num_rows() {
                match operation_array.value(row_idx) {
                    "insert" => insert_count_post_compaction += 1,
                    "update" => update_count_post_compaction += 1,
                    _ => {}
                }
            }
        }

        assert_eq!(
            update_count_post_compaction, 4,
            "Should have 4 update operations after compaction (id 200-203)"
        );
        assert_eq!(
            insert_count_post_compaction, 0,
            "Should have no insert operations after compaction"
        );

        // diff v1 ~ v7
        let diff_builder_full = DatasetDiffBuilder::new(Arc::new(dataset.clone()), v1)
            .with_include_data(true)
            .with_max_concurrency(1)
            .with_batch_size(10);
        let diff_stream_full = diff_builder_full.execute().await.unwrap();
        let diff_batches_full: Vec<_> = diff_stream_full.try_collect().await.unwrap();

        let mut updated_ids = Vec::new();
        let mut inserted_count = 0;
        let mut final_update_count = 0;

        for batch in &diff_batches_full {
            let operation_array = batch
                .column_by_name("operation")
                .unwrap()
                .as_string::<i32>();
            let post_image_array = batch
                .column_by_name("post_image")
                .unwrap()
                .as_struct();

            for row_idx in 0..batch.num_rows() {
                let operation = operation_array.value(row_idx);

                match operation {
                    "insert" => inserted_count += 1,
                    "update" => {
                        final_update_count += 1;
                        if !post_image_array.is_null(row_idx) {
                            let post_id_array = post_image_array
                                .column_by_name("id")
                                .unwrap()
                                .as_primitive::<arrow_array::types::Int32Type>();
                            let id_value = post_id_array.value(row_idx);
                            updated_ids.push(id_value);
                        }
                    }
                    "delete" => {}
                    _ => panic!("Unknown operation type: {}", operation),
                }
            }
        }

        // Note: With version metadata now written for all operations, count reflects tracked inserts
        assert_eq!(inserted_count, 41, "Should have insert operations total");
        assert_eq!(
            final_update_count, 5,
            "Should have 5 update operations total (only id 5-9)"
        );

        updated_ids.sort();
        let expected_updated_ids = vec![5, 6, 7, 8, 9];
        assert_eq!(
            updated_ids, expected_updated_ids,
            "Updated IDs should include both pre and post compaction updates"
        );
    }

    #[tokio::test]
    async fn test_diff_merge_insert_full_schema_update() {
        let base_dataset = create_test_dataset_with_stable_row_ids().await;
        let initial_version = base_dataset.version().version;

        // update id 5..=9；insert id 110..=114
        let merge_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let mut ids: Vec<i32> = (5..=9).collect();
        ids.extend(110..=114);
        let names: Vec<String> = ids
            .iter()
            .map(|id| {
                if (5..=9).contains(id) {
                    format!("merge_full_updated_{}", id)
                } else {
                    format!("merge_insert_{}", id)
                }
            })
            .collect();
        let values: Vec<i32> = ids
            .iter()
            .map(|id| {
                if (5..=9).contains(id) {
                    id * 100
                } else {
                    id * 10
                }
            })
            .collect();

        let merge_batch = arrow_array::RecordBatch::try_new(
            merge_schema.clone(),
            vec![
                Arc::new(Int32Array::from(ids.clone())),
                Arc::new(StringArray::from(names.clone())),
                Arc::new(Int32Array::from(values.clone())),
            ],
        )
        .unwrap();
        let merge_reader = arrow_array::RecordBatchIterator::new([Ok(merge_batch)], merge_schema);

        let mut builder = crate::dataset::MergeInsertBuilder::try_new(
            Arc::new(base_dataset.clone()),
            vec!["id".to_string()],
        )
        .unwrap();
        builder
            .when_matched(crate::dataset::WhenMatched::UpdateAll)
            .when_not_matched(crate::dataset::WhenNotMatched::InsertAll)
            .when_not_matched_by_source(crate::dataset::WhenNotMatchedBySource::Keep);

        let (updated_dataset, stats) = builder
            .try_build()
            .unwrap()
            .execute_reader(Box::new(merge_reader))
            .await
            .unwrap();

        assert_eq!(stats.num_updated_rows, 5);
        assert_eq!(stats.num_inserted_rows, 5);

        let diff_builder = DatasetDiffBuilder::new(updated_dataset.clone(), initial_version)
            .with_include_data(true)
            .with_max_concurrency(1)
            .with_batch_size(32);
        let diff_stream = diff_builder.execute().await.unwrap();
        let diff_batches: Vec<_> = diff_stream.try_collect().await.unwrap();

        let mut update_ids = Vec::new();
        let mut insert_ids = Vec::new();

        for batch in diff_batches.iter() {
            let operation_array = batch
                .column_by_name("operation")
                .unwrap()
                .as_string::<i32>();
            let pre_image_array = batch
                .column_by_name("pre_image")
                .unwrap()
                .as_struct();
            let post_image_array = batch
                .column_by_name("post_image")
                .unwrap()
                .as_struct();

            for row_idx in 0..batch.num_rows() {
                let operation = operation_array.value(row_idx);

                match operation {
                    "update" => {
                        assert!(!pre_image_array.is_null(row_idx));
                        assert!(!post_image_array.is_null(row_idx));

                        // Verify id
                        let post_id_arr = post_image_array
                            .column_by_name("id")
                            .unwrap()
                            .as_primitive::<arrow_array::types::Int32Type>();
                        let id = post_id_arr.value(row_idx);
                        update_ids.push(id);
                        assert!((5..=9).contains(&id));

                        // Verify name
                        let pre_name_arr = pre_image_array
                            .column_by_name("name")
                            .unwrap()
                            .as_string::<i32>();
                        let old_name = pre_name_arr.value(row_idx);

                        let post_name_arr = post_image_array
                            .column_by_name("name")
                            .unwrap()
                            .as_string::<i32>();
                        let new_name = post_name_arr.value(row_idx);

                        assert_eq!(old_name, "initial");
                        assert_eq!(new_name, format!("merge_full_updated_{}", id));

                        // Verify value changes
                        let pre_val_arr = pre_image_array
                            .column_by_name("value")
                            .unwrap()
                            .as_primitive::<arrow_array::types::Int32Type>();
                        let old_val = pre_val_arr.value(row_idx);

                        let post_val_arr = post_image_array
                            .column_by_name("value")
                            .unwrap()
                            .as_primitive::<arrow_array::types::Int32Type>();
                        let new_val = post_val_arr.value(row_idx);

                        assert_eq!(old_val, id);
                        assert_eq!(new_val, id * 100);
                    }
                    "insert" => {
                        // insert: only post_image
                        assert!(pre_image_array.is_null(row_idx));
                        assert!(!post_image_array.is_null(row_idx));

                        let post_id_arr = post_image_array
                            .column_by_name("id")
                            .unwrap()
                            .as_primitive::<arrow_array::types::Int32Type>();
                        let id = post_id_arr.value(row_idx);
                        insert_ids.push(id);
                        assert!((110..=114).contains(&id));

                        let post_name_arr = post_image_array
                            .column_by_name("name")
                            .unwrap()
                            .as_string::<i32>();
                        let name = post_name_arr.value(row_idx);

                        let post_val_arr = post_image_array
                            .column_by_name("value")
                            .unwrap()
                            .as_primitive::<arrow_array::types::Int32Type>();
                        let val = post_val_arr.value(row_idx);

                        assert_eq!(name, format!("merge_insert_{}", id));
                        assert_eq!(val, id * 10);
                    }
                    "delete" => panic!("not supported!"),
                    _ => panic!("Unknown operation type: {}", operation),
                }
            }
        }

        // Verify all expected ids are present
        update_ids.sort();
        insert_ids.sort();
        assert_eq!(update_ids, (5..=9).collect::<Vec<i32>>());
        assert_eq!(insert_ids, (110..=114).collect::<Vec<i32>>());
    }
}
