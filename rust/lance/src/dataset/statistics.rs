// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Module for statistics related to the dataset.

use std::{collections::HashMap, future::Future, sync::Arc};

use arrow_array::{
    Array, ArrayRef, GenericListArray, OffsetSizeTrait, StructArray, UInt8Array, UInt64Array,
};
use datafusion::scalar::ScalarValue;
use futures::{StreamExt, TryStreamExt};
use lance_arrow::list::ListArrayExt;
use lance_core::{
    Error, Result,
    datatypes::{BlobHandling, BlobKind, Field},
};
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::zonemap::ZoneMapIndex;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use roaring::RoaringBitmap;

use super::{Dataset, fragment::FileFragment};
use crate::index::{DatasetIndexExt, DatasetIndexInternalExt};

fn has_blob_v2(field: &Field) -> bool {
    field.is_blob_v2() || field.children.iter().any(has_blob_v2)
}

fn managed_blob_bytes(array: &dyn Array) -> Result<u64> {
    let descriptors = array
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| {
            Error::internal(format!(
                "Blob v2 descriptor must be Struct, got {}",
                array.data_type()
            ))
        })?;
    let fields = descriptors.fields();
    if fields.len() != 5
        || fields[0].name() != "kind"
        || fields[1].name() != "position"
        || fields[2].name() != "size"
        || fields[3].name() != "blob_id"
        || fields[4].name() != "blob_uri"
    {
        return Err(Error::internal(format!(
            "Unexpected blob v2 descriptor fields: {:?}",
            fields.iter().map(|field| field.name()).collect::<Vec<_>>()
        )));
    }
    let kinds = descriptors
        .column_by_name("kind")
        .and_then(|column| column.as_any().downcast_ref::<UInt8Array>())
        .ok_or_else(|| Error::internal("Blob v2 descriptor 'kind' must be UInt8".to_string()))?;
    let sizes = descriptors
        .column_by_name("size")
        .and_then(|column| column.as_any().downcast_ref::<UInt64Array>())
        .ok_or_else(|| Error::internal("Blob v2 descriptor 'size' must be UInt64".to_string()))?;

    let mut total = 0_u64;
    for row in 0..descriptors.len() {
        if descriptors.is_null(row) || kinds.is_null(row) || sizes.is_null(row) {
            continue;
        }
        if matches!(
            BlobKind::try_from(kinds.value(row))?,
            BlobKind::Packed | BlobKind::Dedicated
        ) {
            total = total.checked_add(sizes.value(row)).ok_or_else(|| {
                Error::internal("Blob v2 managed sidecar byte count overflowed u64".to_string())
            })?;
        }
    }
    Ok(total)
}

fn collect_list_blob_stats<O: OffsetSizeTrait>(
    list: &GenericListArray<O>,
    field: &Field,
    stats: &mut HashMap<u32, u64>,
) -> Result<()> {
    let [child] = field.children.as_slice() else {
        return Err(Error::internal(format!(
            "Blob list field '{}' must have exactly one child, got {}",
            field.name,
            field.children.len()
        )));
    };
    collect_blob_v2_sidecar_stats(list.trimmed_values(), child, stats)
}

fn collect_blob_v2_sidecar_stats(
    array: ArrayRef,
    field: &Field,
    stats: &mut HashMap<u32, u64>,
) -> Result<()> {
    if field.is_blob_v2() {
        let bytes = managed_blob_bytes(array.as_ref())?;
        let field_bytes = stats.entry(field.id as u32).or_insert(0_u64);
        *field_bytes = field_bytes.checked_add(bytes).ok_or_else(|| {
            Error::internal(format!(
                "Blob v2 managed sidecar byte count overflowed u64 for field '{}'",
                field.name
            ))
        })?;
        return Ok(());
    }

    if let Some(structure) = array.as_any().downcast_ref::<StructArray>() {
        if structure.num_columns() != field.children.len() {
            return Err(Error::internal(format!(
                "Struct field '{}' has {} arrays but {} schema children",
                field.name,
                structure.num_columns(),
                field.children.len()
            )));
        }
        for (child_array, child_field) in structure.columns().iter().zip(&field.children) {
            if has_blob_v2(child_field) {
                collect_blob_v2_sidecar_stats(child_array.clone(), child_field, stats)?;
            }
        }
        return Ok(());
    }
    if let Some(list) = array.as_any().downcast_ref::<GenericListArray<i32>>() {
        return collect_list_blob_stats(list, field, stats);
    }
    if let Some(list) = array.as_any().downcast_ref::<GenericListArray<i64>>() {
        return collect_list_blob_stats(list, field, stats);
    }
    Err(Error::internal(format!(
        "Blob-containing field '{}' has unsupported type {}",
        field.name,
        array.data_type()
    )))
}

async fn calculate_blob_v2_sidecar_stats(dataset: &Arc<Dataset>) -> Result<HashMap<u32, u64>> {
    let blob_roots = dataset
        .schema()
        .fields
        .iter()
        .filter(|field| has_blob_v2(field))
        .map(|field| {
            Ok((
                field.clone(),
                dataset.schema().field_path_minimal(field.id)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    if blob_roots.is_empty() {
        return Ok(HashMap::new());
    }

    let paths = blob_roots
        .iter()
        .map(|(_, path)| path.as_str())
        .collect::<Vec<_>>();
    let mut scanner = dataset.scan();
    scanner.project(&paths)?;
    scanner.blob_handling(BlobHandling::BlobsDescriptions);
    // Storage statistics cover physical data, including rows hidden by deletion vectors.
    // The scanner requires row ids in order to expose those deleted rows.
    scanner.with_row_id().include_deleted_rows();
    let mut stream = scanner.try_into_stream().await?;
    let mut stats = HashMap::with_capacity(blob_roots.len());
    while let Some(batch) = stream.try_next().await? {
        for (field, path) in &blob_roots {
            let array = batch.column_by_name(path).ok_or_else(|| {
                Error::internal(format!(
                    "Projected blob-containing field '{path}' was missing from the statistics scan"
                ))
            })?;
            collect_blob_v2_sidecar_stats(array.clone(), field, &mut stats)?;
        }
    }
    Ok(stats)
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
        if !self.is_legacy_storage() {
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

            for (field_id, bytes) in calculate_blob_v2_sidecar_stats(self).await? {
                let stats = field_stats.get_mut(&field_id).ok_or_else(|| {
                    Error::internal(format!(
                        "Blob v2 statistics referenced unknown field id {field_id}"
                    ))
                })?;
                stats.bytes_on_disk = stats.bytes_on_disk.checked_add(bytes).ok_or_else(|| {
                    Error::internal(format!(
                        "Data statistics byte count overflowed u64 for field id {field_id}"
                    ))
                })?;
            }
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
    use std::{num::NonZeroUsize, sync::Arc};

    use arrow_array::{ArrayRef, Int32Array, ListArray, RecordBatch, RecordBatchIterator};
    use arrow_buffer::{OffsetBuffer, ScalarBuffer};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_core::utils::tempfile::TempStrDir;
    use lance_file::version::LanceFileVersion;

    use crate::{
        blob::{BlobArrayBuilder, BlobFieldOptions, blob_field_with_options},
        dataset::WriteParams,
    };

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
    async fn test_calculate_data_stats_includes_blob_v2_sidecars() {
        let payloads = [vec![1_u8; 4 * 1024], vec![2_u8; 8 * 1024]];
        let payload_bytes = payloads.iter().map(Vec::len).sum::<usize>() as u64;
        let mut blob_builder = BlobArrayBuilder::new(payloads.len());
        for payload in &payloads {
            blob_builder.push_bytes(payload).unwrap();
        }

        let list_payloads = [vec![3_u8; 2 * 1024], vec![4_u8; 3 * 1024]];
        let list_payload_bytes = list_payloads.iter().map(Vec::len).sum::<usize>() as u64;
        let mut list_blob_builder = BlobArrayBuilder::new(list_payloads.len());
        for payload in &list_payloads {
            list_blob_builder.push_bytes(payload).unwrap();
        }

        let options = BlobFieldOptions::default()
            .with_inline_size_threshold(1)
            .with_dedicated_size_threshold(NonZeroUsize::new(1024).unwrap());
        let blob_field = blob_field_with_options("blob", false, options.clone());
        let list_item = Arc::new(blob_field_with_options("item", false, options));
        let list_array = Arc::new(
            ListArray::try_new(
                list_item.clone(),
                OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 1, 2])),
                list_blob_builder.finish().unwrap(),
                None,
            )
            .unwrap(),
        );
        let schema = Arc::new(ArrowSchema::new(vec![
            blob_field,
            ArrowField::new("blob_list", DataType::List(list_item), false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![blob_builder.finish().unwrap(), list_array],
        )
        .unwrap();
        let test_dir = TempStrDir::default();
        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new(vec![Ok(batch)], schema),
                &test_dir,
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    max_rows_per_file: 1,
                    max_rows_per_group: 1,
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );
        assert_eq!(dataset.get_fragments().len(), 2);

        let blob_field_id = dataset.schema().field("blob").unwrap().id as u32;
        let list_blob_field_id = dataset.schema().field("blob_list").unwrap().children[0].id as u32;
        let stats = dataset.calculate_data_stats().await.unwrap();
        for (field_id, expected_bytes) in [
            (blob_field_id, payload_bytes),
            (list_blob_field_id, list_payload_bytes),
        ] {
            let blob_stats = stats
                .fields
                .iter()
                .find(|stats| stats.id == field_id)
                .unwrap();
            assert!(
                blob_stats.bytes_on_disk >= expected_bytes,
                "blob field {field_id} bytes_on_disk={} should include {expected_bytes} sidecar bytes",
                blob_stats.bytes_on_disk
            );
        }
    }
}
