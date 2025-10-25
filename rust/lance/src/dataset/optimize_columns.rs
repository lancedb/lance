// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Column layout optimization for Lance datasets.
//!
//! This module provides functionality to reorganize how columns are stored within
//! data files. By grouping frequently-accessed columns together, query performance
//! can be significantly improved for workloads that only read a subset of columns.
//!
//! The optimization works by:
//! 1. Analyzing the current column layout within each fragment
//! 2. Rewriting data files to match the specified column groupings
//! 3. Replacing fragments in-place (preserving fragment IDs and row counts)
//!
//! OptimizeColumns only reorganizes column storage within existing fragments.

use crate::dataset::fragment::FileFragment;
use crate::dataset::transaction::{Operation, Transaction};
use crate::dataset::write::{open_writer, GenericWriter};
use crate::Dataset;
use arrow_array::RecordBatch;
use datafusion::execution::SendableRecordBatchStream;
use futures::future::try_join_all;
use futures::{StreamExt, TryStreamExt};
use lance_core::datatypes::Schema;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::utils::tracing::{DATASET_OPTIMIZING_COLUMNS_EVENT, TRACE_DATASET_EVENTS};
use lance_core::{Error, Result};
use lance_table::format::{DataFile, Fragment};
use serde::{Deserialize, Serialize};
use snafu::location;
use std::collections::{HashMap, HashSet};
use std::ops::AddAssign;
use tracing::info;

/// A group of columns that should be stored together in a single data file.
///
/// When optimizing column layout, each `OptimizeGroups` will result in one data
/// file per fragment containing the specified columns. If a fragment doesn't contain
/// all columns from a group, that group is skipped for that fragment.
#[derive(Debug, Clone)]
pub struct OptimizeGroups {
    /// The names of columns to group together.
    pub columns: Vec<String>,
}

impl OptimizeGroups {
    pub fn new(columns: Vec<String>) -> Self {
        Self { columns }
    }
}

/// Configuration options for Optimize Columns
///
/// This struct controls how columns are regrouped within fragments.
#[derive(Debug, Clone, Default)]
pub struct OptimizeColumnsOptions {
    /// Specifies how columns should be grouped into files.
    /// Each [`OptimizeGroups`] will result in one data file per fragment.
    /// If `None`, all columns will be placed in a single file (the default).
    pub optimize_groups: Option<Vec<OptimizeGroups>>,
    /// The number of threads to use (how many compaction tasks to run in parallel).
    /// Defaults to the number of compute-intensive CPUs.
    pub num_threads: Option<usize>,
}

struct OptimizationPlan {
    kept_files: Vec<DataFile>,
    write_tasks: Vec<WriteTask>,
}

struct WriteTask {
    field_ids: Vec<i32>,
    dataset_field_indices: Vec<usize>,
}

struct OptimizeColumnsResult {
    pub metrics: OptimizeColumnsMetrics,
    /// The new fragment with reorganized files
    pub new_fragment: Fragment,
    /// The original fragment metadata prior to optimization
    pub old_fragment: Fragment,
}

struct StreamProjection {
    column_names: Vec<String>,
    per_task_indices: Vec<Vec<usize>>,
}

/// Metrics returned by [optimize_columns] operation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizeColumnsMetrics {
    /// The number of data files that have been removed during reorganization.
    pub files_removed: usize,
    /// The number of data files that have been added during reorganization.
    pub files_added: usize,
}

impl AddAssign for OptimizeColumnsMetrics {
    fn add_assign(&mut self, rhs: Self) {
        self.files_removed += rhs.files_removed;
        self.files_added += rhs.files_added;
    }
}

/// Reorganizes column storage within fragments to optimize read patterns.
///
/// This operation rewrites data files to group columns according to the specified
/// optimize groups, without changing row content or fragment structure. It's useful
/// for optimizing query performance when certain columns are frequently accessed together.
///
/// # Arguments
///
/// * `dataset` - The dataset to optimize
/// * `options` - Configuration for the optimization, including column groupings
///
/// # Returns
///
/// Returns metrics about the optimization operation, including the number of files
/// added and removed.
///
/// # Example
///
/// ```no_run
/// # use lance::Dataset;
/// # use lance::dataset::optimize_columns::{optimize_columns, OptimizeColumnsOptions, OptimizeGroups};
/// # use lance_datagen::{array, BatchCount, RowCount};
/// # use arrow_schema::DataType;
/// # async fn example() -> lance::Result<()> {
/// # // Create a test dataset
/// # let mut dataset = lance_datagen::gen_batch()
/// #     .col("id", array::step::<arrow::datatypes::Int32Type>())
/// #     .col("data", array::step::<arrow::datatypes::Int32Type>())
/// #     .into_reader_rows(RowCount::from(100), BatchCount::from(1))
/// #     .into_dataset("memory://test", None).await?;
/// #
/// let options = OptimizeColumnsOptions {
///     optimize_groups: Some(vec![
///         OptimizeGroups::new(vec!["id".to_string()]),
///         OptimizeGroups::new(vec!["data".to_string()]),
///     ]),
///     num_threads: Some(4),
/// };
///
/// let metrics = optimize_columns(&mut dataset, options).await?;
/// println!("Added {} files, removed {} files",
///          metrics.files_added, metrics.files_removed);
/// # Ok(())
/// # }
/// ```
pub async fn optimize_columns(
    dataset: &mut Dataset,
    options: OptimizeColumnsOptions,
) -> Result<OptimizeColumnsMetrics> {
    info!(target: TRACE_DATASET_EVENTS, event=DATASET_OPTIMIZING_COLUMNS_EVENT, uri = &dataset.uri);

    // If no optimize groups specified, create one group with all columns
    let optimize_groups = options.optimize_groups.unwrap_or_else(|| {
        vec![OptimizeGroups {
            columns: dataset
                .schema()
                .fields
                .iter()
                .map(|f| f.name.clone())
                .collect(),
        }]
    });

    validate_optimize_groups(dataset.schema(), &optimize_groups)?;

    let tasks = build_optimize_column_tasks(dataset, &optimize_groups)?;
    if tasks.is_empty() {
        return Ok(OptimizeColumnsMetrics::default());
    }

    let num_threads = options
        .num_threads
        .unwrap_or_else(get_num_compute_intensive_cpus);

    let optimize_columns_results_future = futures::stream::iter(tasks)
        .map(|(fragment, plan)| {
            let dataset_ref = dataset.clone();
            async move { execute_fragment_optimize_columns(&dataset_ref, fragment, plan).await }
        })
        .buffer_unordered(num_threads)
        .try_collect::<Vec<_>>()
        .await?;

    commit_optimize_columns(dataset, optimize_columns_results_future).await
}

/// Validates that optimize groups of columns cover the entire schema correctly.
///
/// Ensures that:
/// - No column appears in multiple groups
/// - All schema columns are assigned to a group
/// - No groups are empty
/// - All referenced columns exist in the schema
fn validate_optimize_groups(schema: &Schema, optimize_groups: &[OptimizeGroups]) -> Result<()> {
    let mut seen_columns = HashSet::new();

    for (i, group) in optimize_groups.iter().enumerate() {
        if group.columns.is_empty() {
            return Err(Error::InvalidInput {
                source: format!("Optimize group {} is empty", i).into(),
                location: location!(),
            });
        }
        for col in &group.columns {
            if col.contains('.') {
                return Err(Error::InvalidInput {
                    source: format!("Nested column references not supported: '{}'", col).into(),
                    location: location!(),
                });
            }

            if !schema.fields.iter().any(|f| f.name == col.as_str()) {
                return Err(Error::InvalidInput {
                    source: format!("Column '{}' in group {} not found in schema", col, i).into(),
                    location: location!(),
                });
            }
            if !seen_columns.insert(col.clone()) {
                return Err(Error::InvalidInput {
                    source: format!("Column '{}' appears in multiple groups", col).into(),
                    location: location!(),
                });
            }
        }
    }

    // Verify all columns are assigned
    let schema_cols: HashSet<_> = schema.fields.iter().map(|f| f.name.clone()).collect();
    if seen_columns != schema_cols {
        let missing: Vec<_> = schema_cols.difference(&seen_columns).collect();
        return Err(Error::InvalidInput {
            source: format!("Columns not included in any group: {:?}", missing).into(),
            location: location!(),
        });
    }
    Ok(())
}

fn build_optimize_column_tasks(
    dataset: &Dataset,
    column_groups: &[OptimizeGroups],
) -> Result<Vec<(FileFragment, OptimizationPlan)>> {
    let mut tasks = Vec::new();
    for fragment in dataset.get_fragments() {
        let plan = plan_optimization(&fragment, column_groups)?;
        if !plan.write_tasks.is_empty() {
            tasks.push((fragment.clone(), plan));
        }
    }
    Ok(tasks)
}

fn plan_optimization(
    fragment: &FileFragment,
    column_groups: &[OptimizeGroups],
) -> Result<OptimizationPlan> {
    let frag_ids: HashSet<_> = fragment
        .metadata
        .files
        .iter()
        .flat_map(|f| f.fields.iter().copied())
        .collect();

    // col name -> (field_id, schema_index)
    let column_to_field_id: HashMap<String, (i32, usize)> = fragment
        .schema()
        .fields
        .iter()
        .enumerate()
        .map(|(idx, f)| (f.name.clone(), (f.id, idx)))
        .collect();

    let top_level_field_ids = fragment.schema().top_level_field_ids();

    // For each data file, collect only its top-level field IDs.
    // We ignore nested field IDs since optimize groups operate on top-level columns only.
    let files_field_ids: Vec<HashSet<i32>> = fragment
        .metadata
        .files
        .iter()
        .map(|f| {
            f.fields
                .iter()
                .copied()
                .filter(|id| top_level_field_ids.contains(id))
                .collect::<HashSet<_>>()
        })
        .collect();

    let mut kept_files = Vec::new();
    let mut write_tasks = Vec::new();
    for group in column_groups {
        // Filter to columns that exist in this fragment and preserve order
        let cols: Vec<(i32, usize)> = group
            .columns
            .iter()
            .filter_map(|name| column_to_field_id.get(name.as_str()).copied())
            .filter(|(id, _)| frag_ids.contains(id))
            .collect();

        if cols.is_empty() {
            continue;
        }
        let group_field_ids: HashSet<i32> = cols.iter().map(|(id, _)| *id).collect();
        // If any existing file has exactly this set of field ids
        if let Some(idx) = files_field_ids.iter().position(|s| *s == group_field_ids) {
            kept_files.push(fragment.metadata.files[idx].clone());
        } else {
            write_tasks.push(WriteTask {
                field_ids: cols.iter().map(|(id, _)| *id).collect(),
                dataset_field_indices: cols.iter().map(|(_, idx)| *idx).collect(),
            });
        }
    }

    Ok(OptimizationPlan {
        kept_files,
        write_tasks,
    })
}

fn build_stream_projection(
    plan: &OptimizationPlan,
    schema: &Schema,
) -> Result<Option<StreamProjection>> {
    let mut projection_indices: Vec<usize> = plan
        .write_tasks
        .iter()
        .flat_map(|task| task.dataset_field_indices.iter().copied())
        .collect();

    // Sort to maintain consistent order matching the schema.
    projection_indices.sort_unstable();

    if projection_indices.is_empty() {
        return Ok(None);
    }

    let column_names: Vec<String> = projection_indices
        .iter()
        .map(|idx| schema.fields[*idx].name.clone())
        .collect();

    let index_map: HashMap<usize, usize> = projection_indices
        .iter()
        .enumerate()
        // Map from original dataset index to position in the projected stream
        .map(|(stream_idx, original_idx)| (*original_idx, stream_idx))
        .collect();

    let mut per_task_indices = Vec::with_capacity(plan.write_tasks.len());
    for task in &plan.write_tasks {
        let mut stream_indices = Vec::with_capacity(task.dataset_field_indices.len());
        for idx in &task.dataset_field_indices {
            let Some(stream_idx) = index_map.get(idx) else {
                return Err(Error::Internal {
                    message: format!("Missing projection index for field {}", idx),
                    location: location!(),
                });
            };
            stream_indices.push(*stream_idx);
        }
        per_task_indices.push(stream_indices);
    }

    Ok(Some(StreamProjection {
        column_names,
        per_task_indices,
    }))
}

async fn execute_fragment_optimize_columns(
    dataset: &Dataset,
    fragment: FileFragment,
    plan: OptimizationPlan,
) -> Result<OptimizeColumnsResult> {
    let mut scanner = dataset.scan();
    scanner.with_fragments(vec![fragment.metadata.clone()]);
    scanner.scan_in_order(true);
    // Preserve deleted rows to avoid invalidating indices.
    scanner.include_deleted_rows();
    // Indices reference row IDs, so we must maintain the exact row layout.
    scanner.with_row_id();

    let original_fragment = fragment.metadata().clone();
    let stream_projection = build_stream_projection(&plan, fragment.schema())?;
    let per_task_indices = if let Some(projection) = stream_projection {
        scanner.project(&projection.column_names)?;
        projection.per_task_indices
    } else {
        plan.write_tasks
            .iter()
            .map(|task| task.dataset_field_indices.clone())
            .collect()
    };

    let data_stream = SendableRecordBatchStream::from(scanner.try_into_stream().await?);

    let mut new_files = write_optimized_files_streaming(
        data_stream,
        &plan.write_tasks,
        &per_task_indices,
        fragment.dataset(),
    )
    .await?;
    let newly_written_file_count = new_files.len();
    new_files.extend(plan.kept_files.clone());
    let mut optimized_fragment = original_fragment.clone();
    optimized_fragment.files = new_files;

    let metrics = OptimizeColumnsMetrics {
        files_added: newly_written_file_count,
        files_removed: fragment.metadata().files.len() - plan.kept_files.len(),
    };

    Ok(OptimizeColumnsResult {
        metrics,
        new_fragment: optimized_fragment,
        old_fragment: original_fragment,
    })
}

async fn write_optimized_files_streaming(
    data_stream: SendableRecordBatchStream,
    write_tasks: &[WriteTask],
    stream_indices: &[Vec<usize>],
    dataset: &Dataset,
) -> Result<Vec<DataFile>> {
    if write_tasks.is_empty() {
        return Ok(Vec::new());
    }

    debug_assert_eq!(
        write_tasks.len(),
        stream_indices.len(),
        "write task count must match projection mapping count"
    );

    // fan out write to all writers
    let mut writers = create_all_writers(write_tasks, dataset).await?;
    let mut data_stream = data_stream.map_err(|e| Error::Internal {
        message: format!("Error reading stream: {}", e),
        location: location!(),
    });

    while let Some(batch) = data_stream.try_next().await? {
        write_batch_to_all_writers(&batch, &mut writers, stream_indices).await?;
    }
    finish_all_writers(writers).await
}

async fn create_all_writers(
    write_tasks: &[WriteTask],
    dataset: &Dataset,
) -> Result<Vec<Box<dyn GenericWriter>>> {
    // preserve storage format
    let data_storage_version = dataset
        .manifest()
        .data_storage_format
        .lance_file_version()?;

    let write_futures: Vec<_> = write_tasks
        .iter()
        .map(|task| async {
            let projected_schema = dataset.schema().project_by_ids(&task.field_ids, true);
            open_writer(
                &dataset.object_store,
                &projected_schema,
                &dataset.base,
                data_storage_version,
            )
            .await
        })
        .collect();

    try_join_all(write_futures).await
}

async fn finish_all_writers(writers: Vec<Box<dyn GenericWriter>>) -> Result<Vec<DataFile>> {
    let finish_futures: Vec<_> = writers
        .into_iter()
        .map(|mut writer| async move {
            let (_, data_file) = writer.finish().await?;
            Ok(data_file)
        })
        .collect();
    try_join_all(finish_futures).await
}

async fn write_batch_to_all_writers(
    batch: &RecordBatch,
    writers: &mut [Box<dyn GenericWriter>],
    stream_indices: &[Vec<usize>],
) -> Result<()> {
    let write_futures: Vec<_> = writers
        .iter_mut()
        .zip(stream_indices.iter())
        .map(|(writer, indices)| async move {
            let projected_batch = batch.project(indices)?;
            writer.write(&[projected_batch]).await
        })
        .collect();

    try_join_all(write_futures).await?;
    Ok(())
}

async fn commit_optimize_columns(
    dataset: &mut Dataset,
    completed_results: Vec<OptimizeColumnsResult>,
) -> Result<OptimizeColumnsMetrics> {
    if completed_results.is_empty() {
        return Ok(OptimizeColumnsMetrics::default());
    }

    let mut new_fragments = Vec::with_capacity(completed_results.len());
    let mut old_fragments = Vec::with_capacity(completed_results.len());
    let mut total_metrics = OptimizeColumnsMetrics::default();

    for result in completed_results {
        total_metrics += result.metrics;
        old_fragments.push(result.old_fragment);
        new_fragments.push(result.new_fragment);
    }

    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::OptimizeColumns {
            old_fragments,
            new_fragments,
        },
        None,
        None,
    );

    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await?;

    Ok(total_metrics)
}

#[cfg(test)]
mod tests {
    use crate::dataset::optimize_columns::{
        optimize_columns, OptimizeColumnsMetrics, OptimizeColumnsOptions, OptimizeGroups,
    };
    use crate::dataset::transaction::Operation;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use crate::Dataset;
    use arrow_array::types::{Float32Type, Int32Type, Int64Type};
    use arrow_array::RecordBatch;
    use arrow_schema::{DataType, Field, Fields};
    use arrow_select::concat::concat_batches;
    use futures::TryStreamExt;
    use lance_core::Error;
    use lance_datagen::Dimension;
    use lance_index::{scalar::ScalarIndexParams, DatasetIndexExt, IndexType};
    use std::collections::BTreeSet;

    #[tokio::test]
    async fn test_basic_optimize_columns() {
        let mut dataset = create_test_set().await;
        let original_data = collect_data(&dataset).await;

        let options = OptimizeColumnsOptions {
            optimize_groups: Some(vec![
                OptimizeGroups::new(vec!["vector".to_string()]),
                OptimizeGroups::new(vec!["labels".to_string()]),
                OptimizeGroups::new(vec!["meta".to_string()]),
            ]),
            ..Default::default()
        };

        let _metrics = optimize_columns(&mut dataset, options.clone())
            .await
            .unwrap();

        let new_data = collect_data(&dataset).await;

        assert_eq!(
            dataset.manifest.version, 2,
            "Manifest should track version 2 after optimize operation"
        );
        assert_eq!(
            dataset.fragments()[0].files.len(),
            3,
            "Should have 3 files per fragment now"
        );
        assert_eq!(original_data, new_data, "Data should remain unchanged");
    }

    #[tokio::test]
    async fn test_optimize_columns_records_transaction() {
        let mut dataset = create_test_set().await;
        let original_fragments: Vec<_> = dataset.fragments().iter().cloned().collect();

        let groups = vec![
            OptimizeGroups::new(vec!["vector".to_string()]),
            OptimizeGroups::new(vec!["labels".to_string(), "meta".to_string()]),
        ];

        let metrics = optimize_columns(
            &mut dataset,
            OptimizeColumnsOptions {
                optimize_groups: Some(groups.clone()),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert!(metrics.files_added > 0);

        let transaction = dataset.read_transaction().await.unwrap().unwrap();
        match transaction.operation {
            Operation::OptimizeColumns {
                old_fragments,
                new_fragments,
            } => {
                let original_ids: BTreeSet<_> = original_fragments.iter().map(|f| f.id).collect();
                let old_ids: BTreeSet<_> = old_fragments.iter().map(|f| f.id).collect();
                let new_ids: BTreeSet<_> = new_fragments.iter().map(|f| f.id).collect();
                assert_eq!(original_ids, old_ids);
                assert_eq!(original_ids, new_ids);
            }
            _ => panic!("expected OptimizeColumns operation"),
        }
    }

    #[tokio::test]
    async fn test_optimize_columns_no_tasks() {
        let mut dataset = create_test_set().await;

        let metrics = optimize_columns(&mut dataset, OptimizeColumnsOptions::default())
            .await
            .unwrap();

        assert_eq!(
            metrics,
            OptimizeColumnsMetrics::default(),
            "Should have default metrics no operation"
        );
        assert_eq!(
            dataset.manifest.version, 1,
            "Shouldn't bump version as it's no-op"
        );
    }

    #[tokio::test]
    async fn test_validate_groups_rejects_empty_group() {
        let dataset = create_test_set().await;

        let groups = vec![
            OptimizeGroups::new(vec![]), // empty
            OptimizeGroups::new(vec!["vector".to_string()]),
        ];
        let err = super::validate_optimize_groups(dataset.schema(), &groups).unwrap_err();

        assert!(
            matches!(err, Error::InvalidInput { .. }),
            "Should throw invalid input Error for empty optimize group"
        );
    }

    #[tokio::test]
    async fn test_validate_groups_rejects_duplicate_column() {
        let dataset = create_test_set().await;

        // duplicate vector optimize group
        let groups = vec![
            OptimizeGroups::new(vec!["vector".to_string()]),
            OptimizeGroups::new(vec![
                "vector".to_string(),
                "labels".to_string(),
                "meta".to_string(),
            ]),
        ];
        let err = super::validate_optimize_groups(dataset.schema(), &groups).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput { .. }),
            "Should throw invalid input error for duplicate column"
        );
    }

    #[tokio::test]
    async fn test_validate_groups_rejects_nested_column() {
        let dataset = lance_datagen::gen_batch()
            .col(
                "id",
                lance_datagen::array::rand_vec::<Int32Type>(Dimension::from(4)),
            )
            .col(
                "data",
                lance_datagen::array::rand_struct(Fields::from(vec![Field::new(
                    "nested",
                    DataType::Int32,
                    false,
                )])),
            )
            .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(8))
            .await
            .unwrap();

        let groups = vec![
            OptimizeGroups::new(vec!["id".to_string()]),
            OptimizeGroups::new(vec!["data.nested".to_string()]),
        ];
        let err = super::validate_optimize_groups(dataset.schema(), &groups).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput { .. }),
            "Should through invalid input Error for nested column"
        );
    }

    #[tokio::test]
    async fn test_validate_groups_require_covering_all_columns() {
        let dataset = create_test_set().await;

        let groups = vec![OptimizeGroups::new(vec!["vector".to_string()])];
        let err = super::validate_optimize_groups(dataset.schema(), &groups).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput { .. }),
            "Should through invalid input error for not providing entire top level schema"
        );
    }

    #[tokio::test]
    async fn test_optimize_columns_metrics_expected_counts() {
        let frag_count: usize = 3;
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "id",
                lance_datagen::array::rand_vec::<Int64Type>(Dimension::from(4)),
            )
            .col(
                "data",
                lance_datagen::array::rand_vec::<Int64Type>(Dimension::from(4)),
            )
            .into_ram_dataset(
                FragmentCount::from(frag_count as u32),
                FragmentRowCount::from(64),
            )
            .await
            .unwrap();

        let opts = OptimizeColumnsOptions {
            optimize_groups: Some(vec![
                OptimizeGroups::new(vec!["id".to_string()]),
                OptimizeGroups::new(vec!["data".to_string()]),
            ]),
            ..Default::default()
        };

        let metrics = optimize_columns(&mut dataset, opts).await.unwrap();
        assert_eq!(
            metrics.files_added,
            2 * frag_count,
            "Should have 2 new files for each fragment"
        );
        assert_eq!(
            metrics.files_removed,
            1 * frag_count,
            "Should remove original file"
        );
        assert_eq!(
            dataset.fragments()[0].files.len(),
            2,
            "Should have 2 files for each optimize group"
        );
    }

    #[tokio::test]
    async fn test_optimize_columns_preserves_scalar_index() {
        let mut dataset = lance_datagen::gen_batch()
            .col("id", lance_datagen::array::rand::<Int64Type>())
            .col("data", lance_datagen::array::rand::<Int64Type>())
            .into_ram_dataset(FragmentCount::from(4), FragmentRowCount::from(128))
            .await
            .unwrap();

        dataset
            .create_index(
                &["id"],
                IndexType::Scalar,
                Some("id_idx".into()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        let original_index = dataset
            .load_index_by_name("id_idx")
            .await
            .unwrap()
            .expect("index should exist after creation");

        let metrics = optimize_columns(
            &mut dataset,
            OptimizeColumnsOptions {
                optimize_groups: Some(vec![
                    OptimizeGroups::new(vec!["id".to_string()]),
                    OptimizeGroups::new(vec!["data".to_string()]),
                ]),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert!(
            metrics.files_added > 0,
            "Should have added files for each optimize group"
        );

        let index_after = dataset
            .load_index_by_name("id_idx")
            .await
            .unwrap()
            .expect("index should still exist after optimize");
        assert_eq!(index_after.uuid, original_index.uuid);

        let mut scanner = dataset.scan();
        scanner.filter("id = 1").unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let plan = scanner.explain_plan(false).await.unwrap();
        assert!(
            plan.contains("ScalarIndexQuery"),
            "Should have scalar index plan: {}",
            plan
        );
    }

    #[tokio::test]
    async fn test_optimize_columns_idempotent_second_run() {
        let mut dataset = create_test_set().await;

        let groups = vec![
            OptimizeGroups::new(vec!["vector".to_string()]),
            OptimizeGroups::new(vec!["labels".to_string()]),
            OptimizeGroups::new(vec!["meta".to_string()]),
        ];

        // First run: should rewrite layout and bump version.
        let metrics1 = optimize_columns(
            &mut dataset,
            OptimizeColumnsOptions {
                optimize_groups: Some(groups.clone()),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert!(
            metrics1.files_added > 0,
            "Should have added files for each optimize group"
        );
        let version_after_first = dataset.manifest.version;

        // Second run with the same grouping
        let metrics2 = optimize_columns(
            &mut dataset,
            OptimizeColumnsOptions {
                optimize_groups: Some(groups),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(
            metrics2,
            OptimizeColumnsMetrics::default(),
            "Second run should have default metrics"
        );
        assert_eq!(
            dataset.manifest.version, version_after_first,
            "Second run shouldn't bump manifest version"
        );
    }

    #[tokio::test]
    async fn test_build_stream_projection_mapping() {
        let dataset = create_test_set().await;

        // Split as [a,c] and [b] order to demonstrate write order within fragments
        let column_groups = vec![
            OptimizeGroups::new(vec!["vector".to_string(), "meta".to_string()]),
            OptimizeGroups::new(vec!["labels".to_string()]),
        ];

        let fragment = dataset.get_fragments().first().unwrap().clone();
        let plan = super::plan_optimization(&fragment, &column_groups).unwrap();

        assert_eq!(
            plan.kept_files.len(),
            0,
            "initial layout should require writes"
        );
        assert_eq!(
            plan.write_tasks.len(),
            2,
            "Should have same write task as optimize groups"
        );

        let schema = fragment.schema();
        let projection = super::build_stream_projection(&plan, &schema)
            .unwrap()
            .expect("projection must exist when there are write tasks");

        assert_eq!(
            projection.column_names,
            vec![
                "vector".to_string(),
                "labels".to_string(),
                "meta".to_string()
            ],
            "Should have projections for each column"
        );
        assert_eq!(
            projection.per_task_indices[0],
            vec![0, 2],
            "first task should be ordered by schema [a,c]"
        );
        assert_eq!(
            projection.per_task_indices[1],
            vec![1],
            "first task should be ordered by schema [b]"
        );
    }

    async fn create_test_set() -> Dataset {
        let dataset = lance_datagen::gen_batch()
            .col(
                "vector",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col(
                "labels",
                lance_datagen::array::rand_list_any(
                    lance_datagen::array::cycle::<Int64Type>(vec![1, 2, 3, 4, 5]),
                    false,
                ),
            )
            .col(
                "meta",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(8)),
            )
            .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
            .await
            .unwrap();
        dataset
    }

    async fn collect_data(dataset: &Dataset) -> RecordBatch {
        let batches: Vec<RecordBatch> = dataset
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        concat_batches(&batches[0].schema(), &batches).unwrap()
    }
}
