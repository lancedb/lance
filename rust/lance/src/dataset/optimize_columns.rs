// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Physical column-layout compaction without changing fragment row layout.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::RecordBatch;
use futures::{StreamExt, stream};
use lance_core::datatypes::Schema;
use lance_file::version::ConcreteFileVersion;
use lance_table::format::{DataFile, Fragment};
use serde::{Deserialize, Serialize};

use super::transaction::{Operation, OptimizeColumnsGroup, Transaction};
use super::{CommitBuilder, Dataset};
use crate::{Error, Result};

/// One requested physical data-file grouping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnGroup {
    /// Top-level logical fields to place in the same data file.
    pub fields: Vec<String>,
}

/// Options for [`optimize_columns`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OptimizeColumnsOptions {
    /// Explicit target file groups. Fields absent from all groups are not rewritten.
    pub groups: Vec<ColumnGroup>,
    /// Fragment ids to consider, or every current fragment when absent.
    pub fragment_ids: Option<Vec<u64>>,
    /// Maximum number of group files staged concurrently.
    pub max_concurrency: Option<usize>,
}

/// Metrics returned by [`optimize_columns`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizeColumnsMetrics {
    /// Number of selected fragments inspected by the planner.
    pub fragments_examined: usize,
    /// Number of fragments whose provider layout was rewritten.
    pub fragments_rewritten: usize,
    /// Number of new data files installed.
    pub files_added: usize,
    /// Number of old data files removed from the latest fragment descriptors.
    pub files_removed: usize,
    /// Number of mixed old files retained after selected fields were tombstoned.
    pub mixed_files_retained: usize,
    /// Bytes read while staging and committing the operation.
    pub bytes_read: u64,
    /// Bytes written while staging and committing the operation.
    pub bytes_written: u64,
}

/// Current physical column-layout statistics for one fragment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FragmentColumnLayoutStats {
    /// Fragment id.
    pub fragment_id: u64,
    /// Data files with at least one live field provider.
    pub live_file_count: usize,
    /// Known byte size for each data-file descriptor.
    pub file_sizes: Vec<Option<u64>>,
    /// Number of live physical fields in each data-file descriptor.
    pub fields_per_file: Vec<usize>,
    /// Number of overlay descriptors attached to the fragment.
    pub overlay_count: usize,
    /// Fraction of data-file field slots that are tombstoned.
    pub tombstoned_field_ratio: f64,
}

#[derive(Debug, Clone)]
struct PlannedGroup {
    fragment: Fragment,
    schema: Schema,
    field_names: Vec<String>,
    field_ids: Vec<u32>,
}

#[derive(Debug)]
struct StagedGroup {
    fragment_id: u64,
    field_ids: Vec<u32>,
    physical_rows: u64,
    data_file: DataFile,
}

impl Dataset {
    /// Reorganize selected top-level fields into explicit per-fragment data files.
    ///
    /// Logical values, fragment ids, physical row order, deletion files, stable
    /// row ids, and row-version metadata are preserved. Fields not named by any
    /// group keep their current providers.
    ///
    /// # Example
    ///
    /// ```
    /// # use lance::Dataset;
    /// # use lance::dataset::optimize_columns::{ColumnGroup, OptimizeColumnsOptions};
    /// # async fn compact_features(dataset: &mut Dataset) -> lance::Result<()> {
    /// let metrics = dataset
    ///     .optimize_columns(OptimizeColumnsOptions {
    ///         groups: vec![ColumnGroup {
    ///             fields: vec!["feature_a".into(), "feature_b".into()],
    ///         }],
    ///         fragment_ids: None,
    ///         max_concurrency: Some(8),
    ///     })
    ///     .await?;
    /// assert!(metrics.files_added <= metrics.fragments_examined);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn optimize_columns(
        &mut self,
        options: OptimizeColumnsOptions,
    ) -> Result<OptimizeColumnsMetrics> {
        optimize_columns(self, options).await
    }

    /// Return descriptor-only physical column-layout statistics.
    pub fn column_layout_stats(&self) -> Vec<FragmentColumnLayoutStats> {
        self.manifest
            .fragments
            .iter()
            .map(|fragment| {
                let fields_per_file = fragment
                    .files
                    .iter()
                    .map(|file| file.fields.iter().filter(|field| **field >= 0).count())
                    .collect::<Vec<_>>();
                let live_file_count = fields_per_file.iter().filter(|count| **count > 0).count();
                let file_sizes = fragment
                    .files
                    .iter()
                    .map(|file| file.file_size_bytes.get().map(|size| size.get()))
                    .collect();
                let field_slots = fragment
                    .files
                    .iter()
                    .map(|file| file.fields.len())
                    .sum::<usize>();
                let tombstones = fragment
                    .files
                    .iter()
                    .flat_map(|file| file.fields.iter())
                    .filter(|field| **field < 0)
                    .count();

                FragmentColumnLayoutStats {
                    fragment_id: fragment.id,
                    live_file_count,
                    file_sizes,
                    fields_per_file,
                    overlay_count: fragment.overlays.len(),
                    tombstoned_field_ratio: if field_slots == 0 {
                        0.0
                    } else {
                        tombstones as f64 / field_slots as f64
                    },
                }
            })
            .collect()
    }
}

/// Reorganize selected top-level fields into explicit per-fragment data files.
pub async fn optimize_columns(
    dataset: &mut Dataset,
    options: OptimizeColumnsOptions,
) -> Result<OptimizeColumnsMetrics> {
    let (selected_fragments, resolved_groups, max_concurrency) =
        validate_and_resolve(dataset, &options)?;
    let mut metrics = OptimizeColumnsMetrics {
        fragments_examined: selected_fragments.len(),
        ..Default::default()
    };

    let mut plans = Vec::new();
    let live_schema_ids = dataset
        .schema()
        .fields_pre_order()
        .map(|field| field.id)
        .collect::<HashSet<_>>();
    for fragment in &selected_fragments {
        let materialized_fields = fragment
            .files
            .iter()
            .flat_map(|file| file.fields.iter().copied())
            .chain(
                fragment
                    .overlays
                    .iter()
                    .flat_map(|overlay| overlay.data_file.fields.iter().copied()),
            )
            .filter(|field| *field >= 0)
            .map(|field| field as u32)
            .collect::<HashSet<_>>();

        let mut rewritten_fields = HashSet::new();
        for (schema, field_names, requested_ids) in &resolved_groups {
            let field_ids = requested_ids
                .iter()
                .copied()
                .filter(|field| materialized_fields.contains(field))
                .collect::<Vec<_>>();
            if field_ids.is_empty() || group_is_already_optimized(fragment, &field_ids) {
                continue;
            }
            rewritten_fields.extend(field_ids.iter().copied());
            plans.push(PlannedGroup {
                fragment: fragment.clone(),
                schema: schema.clone(),
                field_names: field_names.clone(),
                field_ids,
            });
        }

        if !rewritten_fields.is_empty() {
            metrics.fragments_rewritten += 1;
            for file in &fragment.files {
                let selected_in_file = file
                    .fields
                    .iter()
                    .any(|field| *field >= 0 && rewritten_fields.contains(&(*field as u32)));
                if !selected_in_file {
                    continue;
                }
                let retains_live_field = file.fields.iter().any(|field| {
                    live_schema_ids.contains(field) && !rewritten_fields.contains(&(*field as u32))
                });
                if retains_live_field {
                    metrics.mixed_files_retained += 1;
                } else {
                    metrics.files_removed += 1;
                }
            }
        }
    }

    if plans.is_empty() {
        return Ok(metrics);
    }

    let io_before = dataset.object_store.io_stats_snapshot();
    let dataset_snapshot = Arc::new(dataset.clone());
    let staged_results = stream::iter(plans.into_iter().map(|plan| {
        let dataset = dataset_snapshot.clone();
        async move { stage_group(dataset, plan).await }
    }))
    .buffer_unordered(max_concurrency)
    .collect::<Vec<_>>()
    .await;

    let mut staged = Vec::with_capacity(staged_results.len());
    let mut first_error = None;
    for result in staged_results {
        match result {
            Ok(group) => staged.push(group),
            Err(error) if first_error.is_none() => first_error = Some(error),
            Err(_) => {}
        }
    }
    if let Some(error) = first_error {
        discard_staged_groups(dataset_snapshot.as_ref(), &staged).await;
        return Err(error);
    }

    metrics.files_added = staged.len();
    let mut groups_by_fragment: HashMap<u64, OptimizeColumnsGroup> = HashMap::new();
    for staged_group in staged {
        let group = groups_by_fragment
            .entry(staged_group.fragment_id)
            .or_insert_with(|| OptimizeColumnsGroup {
                fragment_id: staged_group.fragment_id,
                field_ids: Vec::new(),
                new_files: Vec::new(),
                physical_rows: staged_group.physical_rows,
            });
        group.field_ids.extend(staged_group.field_ids);
        group.new_files.push(staged_group.data_file);
    }

    let materialized_through_version = dataset.manifest.version;
    let transaction = Transaction::new(
        materialized_through_version,
        Operation::OptimizeColumns {
            materialized_through_version,
            groups: groups_by_fragment.into_values().collect(),
        },
        None,
    );
    let committed = CommitBuilder::new(dataset_snapshot)
        .execute(transaction)
        .await?;
    *dataset = committed;

    let io_after = dataset.object_store.io_stats_snapshot();
    metrics.bytes_read = io_after.read_bytes.saturating_sub(io_before.read_bytes);
    metrics.bytes_written = io_after
        .written_bytes
        .saturating_sub(io_before.written_bytes);
    Ok(metrics)
}

type ResolvedGroup = (Schema, Vec<String>, Vec<u32>);

fn validate_and_resolve(
    dataset: &Dataset,
    options: &OptimizeColumnsOptions,
) -> Result<(Vec<Fragment>, Vec<ResolvedGroup>, usize)> {
    if options.groups.is_empty() {
        return Err(Error::invalid_input(
            "OptimizeColumnsOptions.groups must not be empty",
        ));
    }
    let max_concurrency = options
        .max_concurrency
        .unwrap_or_else(|| dataset.object_store.io_parallelism());
    if max_concurrency == 0 {
        return Err(Error::invalid_input(
            "OptimizeColumnsOptions.max_concurrency must be greater than zero",
        ));
    }

    let mut named_fields = HashSet::new();
    let mut resolved_groups = Vec::with_capacity(options.groups.len());
    for group in &options.groups {
        if group.fields.is_empty() {
            return Err(Error::invalid_input(
                "OptimizeColumns ColumnGroup.fields must not be empty",
            ));
        }
        let mut fields = Vec::with_capacity(group.fields.len());
        for name in &group.fields {
            if !named_fields.insert(name.as_str()) {
                return Err(Error::invalid_input(format!(
                    "OptimizeColumns field '{name}' appears in more than one group"
                )));
            }
            let field = dataset
                .schema()
                .fields
                .iter()
                .find(|field| field.name == *name)
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "OptimizeColumns field '{name}' is not a top-level dataset field"
                    ))
                })?;
            fields.push(field.clone());
        }
        let schema = Schema {
            fields,
            metadata: dataset.schema().metadata.clone(),
        };
        let field_ids: Vec<u32> = schema
            .fields_pre_order()
            .map(|field| field.id as u32)
            .collect();
        resolved_groups.push((schema, group.fields.clone(), field_ids));
    }

    let selected_fragments = if let Some(fragment_ids) = &options.fragment_ids {
        let unique = fragment_ids.iter().copied().collect::<HashSet<_>>();
        if unique.len() != fragment_ids.len() {
            return Err(Error::invalid_input(
                "OptimizeColumnsOptions.fragment_ids contains duplicates",
            ));
        }
        let known = dataset
            .manifest
            .fragments
            .iter()
            .map(|fragment| fragment.id)
            .collect::<HashSet<_>>();
        if let Some(missing) = fragment_ids.iter().find(|id| !known.contains(id)) {
            return Err(Error::invalid_input(format!(
                "OptimizeColumns fragment id {missing} does not exist"
            )));
        }
        dataset
            .manifest
            .fragments
            .iter()
            .filter(|fragment| unique.contains(&fragment.id))
            .cloned()
            .collect::<Vec<_>>()
    } else {
        dataset.manifest.fragments.as_ref().clone()
    };

    for fragment in &selected_fragments {
        for file in &fragment.files {
            if file.file_version()? == ConcreteFileVersion::V1
                && resolved_groups.iter().any(|(_, _, fields)| {
                    file.fields
                        .iter()
                        .any(|field| *field >= 0 && fields.contains(&(*field as u32)))
                })
            {
                return Err(Error::not_supported(format!(
                    "OptimizeColumns does not support legacy fragment {}",
                    fragment.id
                )));
            }
        }
    }

    Ok((selected_fragments, resolved_groups, max_concurrency))
}

fn group_is_already_optimized(fragment: &Fragment, field_ids: &[u32]) -> bool {
    let target = field_ids.iter().copied().collect::<HashSet<_>>();
    let has_materializable_overlay = fragment.overlays.iter().any(|overlay| {
        overlay
            .data_file
            .fields
            .iter()
            .any(|field| *field >= 0 && target.contains(&(*field as u32)))
    });
    if has_materializable_overlay {
        return false;
    }

    let providers = fragment
        .files
        .iter()
        .filter(|file| {
            file.fields
                .iter()
                .any(|field| *field >= 0 && target.contains(&(*field as u32)))
        })
        .collect::<Vec<_>>();
    if providers.len() != 1 {
        return false;
    }
    let provided = providers[0]
        .fields
        .iter()
        .filter(|field| **field >= 0)
        .map(|field| *field as u32)
        .collect::<HashSet<_>>();
    provided == target
}

async fn stage_group(dataset: Arc<Dataset>, plan: PlannedGroup) -> Result<StagedGroup> {
    let physical_rows = plan.fragment.physical_rows.ok_or_else(|| {
        Error::invalid_input(format!(
            "OptimizeColumns target fragment {} has no physical row count",
            plan.fragment.id
        ))
    })? as u64;
    let mut scanner = dataset.scan();
    let names = plan
        .field_names
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    scanner
        .project(&names)?
        .with_fragments(vec![plan.fragment.clone()])
        .with_row_id()
        .include_deleted_rows();
    let field_names = plan.field_names.clone();
    let data = scanner.try_into_stream().await?.map(move |batch| {
        let batch = batch?;
        project_user_fields(&batch, &field_names)
    });

    let fragment = super::FileFragment::new(dataset, plan.fragment);
    let super::transaction::DataReplacementGroup(_, mut data_file) =
        fragment.write_column(data, &plan.schema).await?;
    let selected = plan.field_ids.iter().copied().collect::<HashSet<_>>();
    data_file.fields = data_file
        .fields
        .iter()
        .map(|field| {
            if *field >= 0 && !selected.contains(&(*field as u32)) {
                lance_table::format::overlay::TOMBSTONE_FIELD_ID
            } else {
                *field
            }
        })
        .collect::<Vec<_>>()
        .into();

    Ok(StagedGroup {
        fragment_id: fragment.id() as u64,
        field_ids: plan.field_ids,
        physical_rows,
        data_file,
    })
}

fn project_user_fields(batch: &RecordBatch, field_names: &[String]) -> Result<RecordBatch> {
    let indices = field_names
        .iter()
        .map(|name| {
            batch.schema().index_of(name).map_err(|error| {
                Error::internal(format!(
                    "OptimizeColumns scan did not return projected field '{name}': {error}"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(batch.project(&indices)?)
}

async fn discard_staged_groups(dataset: &Dataset, groups: &[StagedGroup]) {
    for group in groups {
        let path = dataset.data_dir().join(group.data_file.path.as_str());
        if let Some(stem) = path
            .filename()
            .and_then(|name| name.strip_suffix(".lance"))
            .map(|stem| dataset.data_dir().join(stem))
            && let Err(error) = dataset.object_store.remove_dir_all(stem.clone()).await
        {
            log::warn!("failed to delete staged OptimizeColumns blob sidecars '{stem}': {error}");
        }
        if let Err(error) = dataset.object_store.delete(&path).await {
            log::warn!(
                "failed to delete staged OptimizeColumns file '{}': {error}",
                group.data_file.path
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use futures::stream;
    use lance_index::{IndexType, scalar::ScalarIndexParams};

    use super::*;
    use crate::dataset::NewColumnTransform;
    use crate::dataset::write::WriteParams;
    use crate::index::DatasetIndexExt;

    async fn wide_dataset() -> Dataset {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
            Field::new("c", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
                Arc::new(Int32Array::from(vec![10, 20, 30, 40])),
                Arc::new(Int32Array::from(vec![100, 200, 300, 400])),
            ],
        )
        .unwrap();
        Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            "memory://",
            Some(WriteParams {
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap()
    }

    async fn split_column(dataset: Dataset, name: &str) -> Dataset {
        let mut scanner = dataset.scan();
        scanner.project(&[name]).unwrap();
        let batch = scanner.try_into_batch().await.unwrap();
        let schema = Schema {
            fields: vec![dataset.schema().field(name).unwrap().clone()],
            metadata: Default::default(),
        };
        let replacement = dataset.get_fragments()[0]
            .write_column(stream::iter([Ok(batch)]), &schema)
            .await
            .unwrap();
        let read_version = dataset.manifest.version;
        CommitBuilder::new(Arc::new(dataset))
            .execute(Transaction::new(
                read_version,
                Operation::DataReplacement {
                    replacements: vec![replacement],
                },
                None,
            ))
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn test_optimize_columns_preserves_values_and_row_layout() {
        let dataset = split_column(wide_dataset().await, "b").await;
        let mut dataset = split_column(dataset, "c").await;
        dataset.delete("a = 2").await.unwrap();

        let version_before = dataset.manifest.version;
        let fragment_before = dataset.manifest.fragments[0].clone();
        let batch_before = dataset.scan().try_into_batch().await.unwrap();
        assert_eq!(dataset.column_layout_stats()[0].live_file_count, 3);

        let metrics = dataset
            .optimize_columns(OptimizeColumnsOptions {
                groups: vec![ColumnGroup {
                    fields: vec!["b".to_string(), "c".to_string()],
                }],
                fragment_ids: None,
                max_concurrency: Some(2),
            })
            .await
            .unwrap();

        assert_eq!(metrics.fragments_examined, 1);
        assert_eq!(metrics.fragments_rewritten, 1);
        assert_eq!(metrics.files_added, 1);
        assert_eq!(metrics.files_removed, 2);
        assert_eq!(metrics.mixed_files_retained, 0);
        assert!(metrics.bytes_read > 0);
        assert!(metrics.bytes_written > 0);
        assert_eq!(dataset.column_layout_stats()[0].live_file_count, 2);

        let fragment_after = &dataset.manifest.fragments[0];
        assert_eq!(fragment_after.id, fragment_before.id);
        assert_eq!(fragment_after.physical_rows, fragment_before.physical_rows);
        assert_eq!(fragment_after.deletion_file, fragment_before.deletion_file);
        assert_eq!(fragment_after.row_id_meta, fragment_before.row_id_meta);
        assert_eq!(
            fragment_after.created_at_version_meta,
            fragment_before.created_at_version_meta
        );
        assert_eq!(
            fragment_after.last_updated_at_version_meta,
            fragment_before.last_updated_at_version_meta
        );
        assert_eq!(dataset.scan().try_into_batch().await.unwrap(), batch_before);
        assert_eq!(
            dataset
                .checkout_version(version_before)
                .await
                .unwrap()
                .scan()
                .try_into_batch()
                .await
                .unwrap(),
            batch_before
        );

        let optimized_version = dataset.manifest.version;
        let no_op = dataset
            .optimize_columns(OptimizeColumnsOptions {
                groups: vec![ColumnGroup {
                    fields: vec!["b".to_string(), "c".to_string()],
                }],
                fragment_ids: None,
                max_concurrency: None,
            })
            .await
            .unwrap();
        assert_eq!(no_op.fragments_examined, 1);
        assert_eq!(no_op.fragments_rewritten, 0);
        assert_eq!(dataset.manifest.version, optimized_version);
    }

    #[tokio::test]
    async fn test_optimize_columns_validates_scope_before_writing() {
        let mut dataset = wide_dataset().await;
        let version = dataset.manifest.version;
        let error = dataset
            .optimize_columns(OptimizeColumnsOptions {
                groups: vec![
                    ColumnGroup {
                        fields: vec!["b".to_string()],
                    },
                    ColumnGroup {
                        fields: vec!["b".to_string()],
                    },
                ],
                fragment_ids: None,
                max_concurrency: Some(1),
            })
            .await
            .unwrap_err();
        assert!(error.to_string().contains("more than one group"), "{error}");
        assert_eq!(dataset.manifest.version, version);
    }

    #[tokio::test]
    async fn test_optimize_columns_keeps_fileless_fields_fileless() {
        let mut dataset = wide_dataset().await;
        dataset
            .add_columns(
                NewColumnTransform::AllNulls(Arc::new(ArrowSchema::new(vec![Field::new(
                    "d",
                    DataType::Int32,
                    true,
                )]))),
                None,
                None,
            )
            .await
            .unwrap();
        let fileless_id = dataset.schema().field("d").unwrap().id;

        let metrics = dataset
            .optimize_columns(OptimizeColumnsOptions {
                groups: vec![ColumnGroup {
                    fields: vec!["b".to_string(), "d".to_string()],
                }],
                fragment_ids: None,
                max_concurrency: Some(1),
            })
            .await
            .unwrap();

        assert_eq!(metrics.files_added, 1);
        assert!(
            dataset.manifest.fragments[0]
                .files
                .iter()
                .all(|file| !file.fields.contains(&fileless_id))
        );
        let batch = dataset.scan().try_into_batch().await.unwrap();
        assert_eq!(batch.column_by_name("d").unwrap().null_count(), 4);
    }

    #[tokio::test]
    async fn test_optimize_columns_prunes_index_coverage_without_replacing_index() {
        let dataset = split_column(wide_dataset().await, "b").await;
        let mut dataset = split_column(dataset, "c").await;
        dataset
            .create_index(
                &["b"],
                IndexType::BTree,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let before = dataset.load_indices().await.unwrap();
        let before = before.iter().find(|index| index.name == "b_idx").unwrap();
        let index_uuid = before.uuid;
        assert!(before.fragment_bitmap.as_ref().unwrap().contains(0));

        dataset
            .optimize_columns(OptimizeColumnsOptions {
                groups: vec![ColumnGroup {
                    fields: vec!["b".to_string(), "c".to_string()],
                }],
                fragment_ids: None,
                max_concurrency: Some(1),
            })
            .await
            .unwrap();

        let after = dataset.load_indices().await.unwrap();
        let after = after.iter().find(|index| index.name == "b_idx").unwrap();
        assert_eq!(after.uuid, index_uuid);
        assert!(!after.fragment_bitmap.as_ref().unwrap().contains(0));
    }
}
