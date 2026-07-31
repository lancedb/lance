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

use super::{Dataset, fragment::FileFragment};
use crate::index::{DatasetIndexExt, DatasetIndexInternalExt};

/// Statistics about a single field in the dataset
pub struct FieldStatistics {
    /// Id of the field
    pub id: u32,
    /// Amount of data in the field (after compression, if any)
    ///
    /// This will be 0 if the data storage version is less than 2
    ///
    /// For blob v2 fields this includes the payload bytes Lance stores outside
    /// the field's column pages (inline out-of-line buffers plus packed and
    /// dedicated sidecar `.blob` files), as recorded in the manifest at write
    /// time. Bytes of external blobs (foreign URIs) are never included, and
    /// data files written before the tally existed contribute only their
    /// descriptor bytes.
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
        if !self.is_legacy_storage() {
            // Blob v2 payloads live outside column pages (inline out-of-line
            // buffers and sidecar `.blob` files), so file metadata never sees
            // them. Writers record their per-field sizes in the manifest
            // (`DataFile::blob_bytes`); fold those in here. Files without a
            // recorded tally (older writers, externally created data files)
            // contribute only their descriptor bytes.
            for fragment in self.fragments().iter() {
                for file in &fragment.files {
                    for (field_id, blob_bytes) in file.fields.iter().zip(file.blob_bytes.iter()) {
                        if let Some(stats) = field_stats.get_mut(&(*field_id as u32)) {
                            stats.bytes_on_disk += blob_bytes;
                        }
                    }
                }
            }
            let scan_scheduler = ScanScheduler::new(
                self.object_store.clone(),
                SchedulerConfig::max_bandwidth(self.object_store.as_ref()),
            );
            let schema = self.schema().clone();
            let dataset = self.clone();
            let fragments = self.fragments().as_ref().clone();
            futures::stream::iter(fragments)
                .map(|fragment| {
                    let file_fragment = FileFragment::new(dataset.clone(), fragment);
                    let schema = schema.clone();
                    let scan_scheduler = scan_scheduler.clone();
                    async move { file_fragment.storage_stats(&schema, scan_scheduler).await }
                })
                .buffer_unordered(self.object_store.io_parallelism())
                .try_for_each(|fragment_stats| {
                    for (field_id, bytes) in fragment_stats {
                        if let Some(stats) = field_stats.get_mut(&field_id) {
                            stats.bytes_on_disk += bytes;
                        }
                    }
                    futures::future::ready(Ok(()))
                })
                .await?;
        }
        let field_stats = field_ids
            .into_iter()
            .map(|id| field_stats.remove(&(id as u32)).unwrap())
            .collect();
        Ok(DataStatistics {
            fields: field_stats,
        })
    }
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
    /// the index was built, or a NaN-bearing column, yield `None`. The disjoint
    /// segments of a multi-segment index are folded together.
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
    use std::num::NonZeroUsize;
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int32Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_core::utils::tempfile::TempStrDir;
    use lance_file::version::LanceFileVersion;

    use crate::blob::{BlobArrayBuilder, BlobFieldOptions, blob_field_with_options};
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

    #[tokio::test]
    async fn test_calculate_data_stats_includes_blob_v2_payloads() {
        const INLINE_PAYLOAD: usize = 100;
        const PACKED_PAYLOAD: usize = 10 * 1024;
        const DEDICATED_PAYLOAD: usize = 100 * 1024;
        const PAYLOAD_TOTAL: u64 = (INLINE_PAYLOAD + PACKED_PAYLOAD + DEDICATED_PAYLOAD) as u64;

        // Thresholds picked so the three payloads land in inline, packed, and
        // dedicated storage respectively.
        let blobs_field = blob_field_with_options(
            "blobs",
            true,
            BlobFieldOptions::default()
                .with_inline_size_threshold(1024)
                .with_dedicated_size_threshold(NonZeroUsize::new(64 * 1024).unwrap()),
        );
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, false),
            blobs_field,
        ]));
        let mut blobs = BlobArrayBuilder::new(3);
        blobs.push_bytes(vec![1u8; INLINE_PAYLOAD]).unwrap();
        blobs.push_bytes(vec![2u8; PACKED_PAYLOAD]).unwrap();
        blobs.push_bytes(vec![3u8; DEDICATED_PAYLOAD]).unwrap();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                blobs.finish().unwrap(),
            ],
        )
        .unwrap();

        let test_dir = TempStrDir::default();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let dataset = Dataset::write(
            reader,
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // The writer must have recorded the payload bytes in the manifest,
        // attributed to the blob field.
        let blob_field_id = dataset.schema().field("blobs").unwrap().id;
        let fragments = dataset.fragments().clone();
        let data_file = &fragments[0].files[0];
        assert_eq!(data_file.blob_bytes.len(), data_file.fields.len());
        let recorded: u64 = data_file
            .fields
            .iter()
            .zip(data_file.blob_bytes.iter())
            .filter(|(field_id, _)| **field_id == blob_field_id)
            .map(|(_, bytes)| *bytes)
            .sum();
        assert_eq!(recorded, PAYLOAD_TOTAL);

        let stats = Arc::new(dataset).calculate_data_stats().await.unwrap();
        let blob_stats = stats
            .fields
            .iter()
            .find(|f| f.id == blob_field_id as u32)
            .unwrap();
        assert!(
            blob_stats.bytes_on_disk >= PAYLOAD_TOTAL,
            "blob field bytes_on_disk ({}) should include the {} payload bytes",
            blob_stats.bytes_on_disk,
            PAYLOAD_TOTAL
        );
        // The only addition beyond the payloads is the small descriptor
        // column, so a tight upper bound catches double counting.
        assert!(
            blob_stats.bytes_on_disk < PAYLOAD_TOTAL + 4096,
            "blob field bytes_on_disk ({}) suggests payloads were double counted",
            blob_stats.bytes_on_disk
        );
    }

    #[tokio::test]
    async fn test_blob_v2_payload_bytes_survive_compaction() {
        const PAYLOAD: usize = 10 * 1024;
        const NUM_FRAGMENTS: u64 = 2;
        const PAYLOAD_TOTAL: u64 = NUM_FRAGMENTS * PAYLOAD as u64;

        let blobs_field = blob_field_with_options(
            "blobs",
            true,
            BlobFieldOptions::default().with_inline_size_threshold(1024),
        );
        let schema = Arc::new(ArrowSchema::new(vec![blobs_field]));

        let test_dir = TempStrDir::default();
        let mut dataset = None;
        for fragment_idx in 0..NUM_FRAGMENTS {
            let mut blobs = BlobArrayBuilder::new(1);
            blobs.push_bytes(vec![fragment_idx as u8; PAYLOAD]).unwrap();
            let batch =
                RecordBatch::try_new(schema.clone(), vec![blobs.finish().unwrap()]).unwrap();
            let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
            dataset = Some(
                Dataset::write(
                    reader,
                    &test_dir,
                    Some(WriteParams {
                        data_storage_version: Some(LanceFileVersion::V2_2),
                        mode: if fragment_idx == 0 {
                            crate::dataset::WriteMode::Create
                        } else {
                            crate::dataset::WriteMode::Append
                        },
                        ..Default::default()
                    }),
                )
                .await
                .unwrap(),
            );
        }
        let mut dataset = dataset.unwrap();
        assert_eq!(dataset.fragments().len(), NUM_FRAGMENTS as usize);

        // Compaction re-materializes blob payloads into new sidecar files;
        // the rewritten data files must carry a fresh, correct tally.
        crate::dataset::optimize::compact_files(&mut dataset, Default::default(), None)
            .await
            .unwrap();
        assert_eq!(dataset.fragments().len(), 1);

        let blob_field_id = dataset.schema().field("blobs").unwrap().id;
        let stats = Arc::new(dataset).calculate_data_stats().await.unwrap();
        let blob_stats = stats
            .fields
            .iter()
            .find(|f| f.id == blob_field_id as u32)
            .unwrap();
        assert!(
            blob_stats.bytes_on_disk >= PAYLOAD_TOTAL,
            "blob field bytes_on_disk ({}) should include the {} payload bytes after compaction",
            blob_stats.bytes_on_disk,
            PAYLOAD_TOTAL
        );
        assert!(
            blob_stats.bytes_on_disk < PAYLOAD_TOTAL + 4096,
            "blob field bytes_on_disk ({}) suggests payloads were double counted after compaction",
            blob_stats.bytes_on_disk
        );
    }
}
