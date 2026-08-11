// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Module for statistics related to the dataset.

use std::{collections::HashMap, future::Future, sync::Arc};

use datafusion::scalar::ScalarValue;
use futures::{StreamExt, TryStreamExt};
use lance_core::{Error, Result};
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::zonemap::ZoneMapIndex;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use roaring::RoaringBitmap;

use super::overlay::{collect_overlay_stale_frags, overlaid_fragments};
use super::{Dataset, fragment::FileFragment, versions};
use crate::index::{DatasetIndexExt, DatasetIndexInternalExt};

/// Aggregate statistics for the fragments in a dataset version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FragmentSummary {
    /// Number of fragments.
    pub fragment_count: u64,
    /// Minimum number of live rows in a fragment, or 0 when the dataset has no fragments.
    pub min_rows_per_fragment: u64,
    /// Maximum number of live rows in a fragment, or 0 when the dataset has no fragments.
    pub max_rows_per_fragment: u64,
    /// Minimum number of data files in a fragment, or 0 when the dataset has no fragments.
    pub min_data_files_per_fragment: u64,
    /// Maximum number of data files in a fragment, or 0 when the dataset has no fragments.
    pub max_data_files_per_fragment: u64,
}

impl Dataset {
    /// Aggregate fragment statistics from the loaded manifest in one pass.
    ///
    /// Returns an error for legacy fragments that do not contain enough metadata to determine
    /// their live row count.
    pub fn fragment_summary(&self) -> Result<FragmentSummary> {
        summarize_fragments(self.fragments())
    }
}

fn summarize_fragments(fragments: &[lance_table::format::Fragment]) -> Result<FragmentSummary> {
    let mut min_rows_per_fragment = u64::MAX;
    let mut max_rows_per_fragment = 0;
    let mut min_data_files_per_fragment = u64::MAX;
    let mut max_data_files_per_fragment = 0;

    for fragment in fragments {
        let live_rows = fragment.num_rows().ok_or_else(|| {
            Error::internal(format!(
                "Fragment summary requires physical row count and deletion count in fragment metadata, but fragment {} is missing required row-count metadata. Rewrite the dataset with a current Lance version to populate it",
                fragment.id
            ))
        })? as u64;
        min_rows_per_fragment = min_rows_per_fragment.min(live_rows);
        max_rows_per_fragment = max_rows_per_fragment.max(live_rows);
        let data_file_count = fragment.files.len() as u64;
        min_data_files_per_fragment = min_data_files_per_fragment.min(data_file_count);
        max_data_files_per_fragment = max_data_files_per_fragment.max(data_file_count);
    }

    if fragments.is_empty() {
        min_rows_per_fragment = 0;
        min_data_files_per_fragment = 0;
    }

    Ok(FragmentSummary {
        fragment_count: fragments.len() as u64,
        min_rows_per_fragment,
        max_rows_per_fragment,
        min_data_files_per_fragment,
        max_data_files_per_fragment,
    })
}

#[cfg(test)]
mod fragment_summary_tests {
    use lance_core::Error;
    use lance_file::version::ConcreteFileVersion;
    use lance_table::format::{DeletionFile, DeletionFileType, Fragment};

    use super::summarize_fragments;

    fn fragment(id: u64, rows: Option<usize>, file_count: usize) -> Fragment {
        let mut fragment = Fragment::new(id);
        fragment.physical_rows = rows;
        for file_idx in 0..file_count {
            fragment.add_file(
                format!("{id}-{file_idx}.lance"),
                vec![file_idx as i32],
                vec![file_idx as i32],
                ConcreteFileVersion::V2_1,
                None,
            );
        }
        fragment
    }

    #[test]
    fn test_fragment_summary_rejects_unknown_row_counts() {
        let known = fragment(0, Some(10), 1);
        let unknown_physical_rows = fragment(1, None, 2);
        let mut unknown_deletions = fragment(2, Some(30), 3);
        unknown_deletions.deletion_file = Some(DeletionFile {
            read_version: 1,
            id: 1,
            file_type: DeletionFileType::Array,
            num_deleted_rows: None,
            base_id: None,
        });

        let physical_rows_error =
            summarize_fragments(&[known.clone(), unknown_physical_rows]).unwrap_err();
        assert!(matches!(&physical_rows_error, Error::Internal { .. }));
        assert!(physical_rows_error.to_string().contains("fragment 1"));

        let deletion_count_error = summarize_fragments(&[known, unknown_deletions]).unwrap_err();
        assert!(matches!(&deletion_count_error, Error::Internal { .. }));
        assert!(deletion_count_error.to_string().contains("fragment 2"));
    }

    #[test]
    fn test_fragment_summary_uses_live_rows() {
        let mut deleted = fragment(0, Some(10), 1);
        deleted.deletion_file = Some(DeletionFile {
            read_version: 1,
            id: 1,
            file_type: DeletionFileType::Array,
            num_deleted_rows: Some(4),
            base_id: None,
        });
        let other = fragment(1, Some(20), 2);

        let summary = summarize_fragments(&[deleted, other]).unwrap();
        assert_eq!(summary.min_rows_per_fragment, 6);
        assert_eq!(summary.max_rows_per_fragment, 20);
        assert_eq!(summary.min_data_files_per_fragment, 1);
        assert_eq!(summary.max_data_files_per_fragment, 2);
    }

    #[test]
    fn test_fragment_summary_empty() {
        let summary = summarize_fragments(&[]).unwrap();
        assert_eq!(summary.fragment_count, 0);
        assert_eq!(summary.min_rows_per_fragment, 0);
        assert_eq!(summary.max_rows_per_fragment, 0);
        assert_eq!(summary.min_data_files_per_fragment, 0);
        assert_eq!(summary.max_data_files_per_fragment, 0);
    }
}

/// Statistics about a single field in the dataset
pub struct FieldStatistics {
    /// Id of the field
    pub id: u32,
    /// Amount of data in the field (after compression, if any)
    ///
    /// This will be 0 if the data storage version is less than 2
    pub bytes_on_disk: u64,
}

/// Statistics about the data in the dataset
pub struct DataStatistics {
    /// Statistics about each field in the dataset
    pub fields: Vec<FieldStatistics>,
}

pub trait DatasetStatisticsExt {
    /// Get statistics about the data in the dataset
    fn calculate_data_stats(
        self: &Arc<Self>,
    ) -> impl Future<Output = Result<DataStatistics>> + Send;
}

impl DatasetStatisticsExt for Dataset {
    async fn calculate_data_stats(self: &Arc<Self>) -> Result<DataStatistics> {
        let field_ids = self.schema().field_ids();
        let mut field_stats: HashMap<u32, FieldStatistics> =
            HashMap::from_iter(field_ids.iter().map(|id| {
                (
                    *id as u32,
                    FieldStatistics {
                        id: *id as u32,
                        bytes_on_disk: 0,
                    },
                )
            }));
        versions::collect_data_stats(
            self.manifest().data_storage_format.lance_file_format(),
            self,
            &mut field_stats,
        )
        .await?;
        let field_stats = field_ids
            .into_iter()
            .map(|id| field_stats.remove(&(id as u32)).unwrap())
            .collect();
        Ok(DataStatistics {
            fields: field_stats,
        })
    }
}

pub(super) async fn collect_current_data_stats(
    dataset: &Arc<Dataset>,
    field_stats: &mut HashMap<u32, FieldStatistics>,
) -> Result<()> {
    let scan_scheduler = ScanScheduler::new(
        dataset.object_store.clone(),
        SchedulerConfig::max_bandwidth(dataset.object_store.as_ref()),
    );
    let schema = dataset.schema().clone();
    let fragments = dataset.fragments().as_ref().clone();
    futures::stream::iter(fragments)
        .map(|fragment| {
            let file_fragment = FileFragment::new(dataset.clone(), fragment);
            let schema = schema.clone();
            let scan_scheduler = scan_scheduler.clone();
            async move { file_fragment.storage_stats(&schema, scan_scheduler).await }
        })
        .buffer_unordered(dataset.object_store.io_parallelism())
        .try_for_each(|fragment_stats| {
            for (field_id, bytes) in fragment_stats {
                if let Some(stats) = field_stats.get_mut(&field_id) {
                    stats.bytes_on_disk += bytes;
                }
            }
            futures::future::ready(Ok(()))
        })
        .await
}

/// A read-only handle for cheap, index-derived statistics about a [`Dataset`].
///
/// Obtained via [`Dataset::statistics`]. Groups statistics accessors behind one
/// handle instead of accreting one-off methods on [`Dataset`]. Every accessor is
/// served from index metadata and never scans data.
#[derive(Debug, Clone, Copy)]
pub struct DatasetStatistics<'a> {
    dataset: &'a Dataset,
}

impl<'a> DatasetStatistics<'a> {
    pub(crate) fn new(dataset: &'a Dataset) -> Self {
        Self { dataset }
    }

    /// Global `[min, max]` for `column` from its min/max-capable scalar index
    /// (currently ZoneMap), without a scan.
    ///
    /// `None` unless the column's index segments *jointly* cover every live
    /// fragment and the column can be soundly bounded — fragments appended after
    /// the index was built, a data overlay committed after a segment was built,
    /// or a NaN-bearing column all yield `None`. The disjoint segments of a
    /// multi-segment index are folded together.
    ///
    /// When `Some`, the range is a superset of live values, conservative under
    /// deletion vectors: safe to prune with. See [`ScalarIndex::value_range`].
    ///
    /// [`ScalarIndex::value_range`]: lance_index::scalar::ScalarIndex::value_range
    pub async fn column_value_range(
        &self,
        column: &str,
    ) -> Result<Option<(ScalarValue, ScalarValue)>> {
        let dataset = self.dataset;
        let Some(field) = dataset.schema().field(column) else {
            return Err(Error::invalid_input(format!(
                "column_value_range: column '{column}' not found in dataset schema"
            )));
        };
        let field_id = field.id;
        let field_path = dataset.schema().field_path(field_id)?;

        // A multi-segment ZoneMap is several index entries over the same column,
        // each covering a disjoint fragment subset. Match the field, then the
        // details type (the column may also carry e.g. a BTree).
        let indices = dataset.load_indices().await?;
        let segments: Vec<_> = indices
            .iter()
            .filter(|idx| matches!(idx.fields.as_slice(), [only] if *only == field_id))
            .filter(|idx| {
                idx.index_details
                    .as_ref()
                    .is_some_and(|d| d.type_url.ends_with("ZoneMapIndexDetails"))
            })
            .collect();
        if segments.is_empty() {
            return Ok(None);
        }

        // Soundness: the segments must *jointly* cover every live fragment, else
        // the fold sees only a subset and could prune live rows (e.g. fragments
        // appended after the index was built). Extra dead fragments are harmless.
        let mut covered = RoaringBitmap::new();
        for idx in &segments {
            let Some(bitmap) = idx.fragment_bitmap.as_ref() else {
                return Ok(None);
            };
            covered |= bitmap.clone();
        }
        if !dataset.fragment_bitmap.as_ref().is_subset(&covered) {
            return Ok(None);
        }

        // Soundness: a data overlay committed after a segment was built can move a value
        // outside that segment's summaries without the ZoneMap ever seeing it, so the fold
        // would no longer bound the live values. There is no way to widen the range without
        // reading the overlay, so report "unknown" instead.
        let overlaid = overlaid_fragments(&dataset.manifest.fragments);
        if !overlaid.is_empty() {
            let mut stale = RoaringBitmap::new();
            for idx in &segments {
                collect_overlay_stale_frags(idx, &overlaid, &mut stale, dataset.schema())?;
            }
            if !stale.is_disjoint(dataset.fragment_bitmap.as_ref()) {
                return Ok(None);
            }
        }

        // Keep the opened indices alive so the `ZoneMapIndex` refs we fold over
        // stay borrowed.
        let mut opened = Vec::with_capacity(segments.len());
        for idx in &segments {
            opened.push(
                dataset
                    .open_generic_index(&field_path, &idx.uuid, &NoOpMetricsCollector)
                    .await?,
            );
        }
        let Some(zonemaps) = opened
            .iter()
            .map(|index| index.as_any().downcast_ref::<ZoneMapIndex>())
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        Ok(ZoneMapIndex::value_range_over(zonemaps))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int32Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_core::utils::tempfile::TempStrDir;
    use lance_file::version::LanceFileVersion;

    use crate::dataset::WriteParams;

    use super::*;

    #[tokio::test]
    async fn test_calculate_data_stats_after_dropping_wide_dataset_columns() {
        let num_columns = 64;
        let num_rows = 128;
        let schema = Arc::new(ArrowSchema::new(
            (0..num_columns)
                .map(|idx| ArrowField::new(format!("col_{idx}"), DataType::Int32, true))
                .collect::<Vec<_>>(),
        ));
        let batch = RecordBatch::try_new(
            schema.clone(),
            (0..num_columns)
                .map(|column_idx| {
                    Arc::new(Int32Array::from_iter_values(
                        (0..num_rows).map(|row_idx| row_idx + column_idx),
                    )) as ArrayRef
                })
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let test_dir = TempStrDir::default();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut dataset = Dataset::write(
            reader,
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let columns_to_drop = (1..num_columns)
            .map(|idx| format!("col_{idx}"))
            .collect::<Vec<_>>();
        let column_refs = columns_to_drop
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        dataset.drop_columns(&column_refs).await.unwrap();

        let stats = Arc::new(dataset).calculate_data_stats().await.unwrap();
        assert_eq!(stats.fields.len(), 1);
        assert_eq!(stats.fields[0].id, 0);
        assert!(
            stats.fields[0].bytes_on_disk > 0,
            "bytes_on_disk should include the remaining column after drop_columns"
        );
    }
}
