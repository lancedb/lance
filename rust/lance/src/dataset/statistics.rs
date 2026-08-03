// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Module for statistics related to the dataset.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    future::Future,
    sync::Arc,
};

use arrow_array::{
    Array, ArrayRef, GenericListArray, OffsetSizeTrait, RecordBatch, StructArray, UInt8Array,
    UInt32Array, UInt64Array,
};
use datafusion::scalar::ScalarValue;
use futures::{StreamExt, TryStreamExt};
use lance_core::{
    Error, ROW_ADDR, Result,
    datatypes::{BlobHandling, BlobKind, Field},
    utils::address::RowAddress,
};
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::zonemap::ZoneMapIndex;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use roaring::RoaringBitmap;

use super::{Dataset, fragment::FileFragment};
use crate::index::{DatasetIndexExt, DatasetIndexInternalExt};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct BlobDataFileKey {
    base_id: Option<u32>,
    path: String,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct BlobObjectKey {
    data_file: BlobDataFileKey,
    blob_id: u32,
}

#[derive(Clone, Copy, Debug)]
struct BlobExtent {
    field_id: u32,
    start: u64,
    end: u64,
}

fn field_tree_stored_in_file(field: &Field, file_field_ids: &[i32]) -> bool {
    file_field_ids.contains(&field.id)
        || field
            .children
            .iter()
            .any(|child| field_tree_stored_in_file(child, file_field_ids))
}

fn blob_data_file_key(
    dataset: &Dataset,
    field_id: u32,
    row_addr: u64,
    cache: &mut HashMap<(u32, u32), BlobDataFileKey>,
) -> Result<BlobDataFileKey> {
    let fragment_id = RowAddress::from(row_addr).fragment_id();
    if let Some(key) = cache.get(&(field_id, fragment_id)) {
        return Ok(key.clone());
    }
    let fragment = dataset
        .get_fragment(fragment_id as usize)
        .ok_or_else(|| Error::internal(format!("Fragment {fragment_id} not found")))?;
    let blob_field = dataset
        .schema()
        .field_by_id(field_id as i32)
        .ok_or_else(|| {
            Error::internal(format!(
                "Blob field id {field_id} not found in dataset schema"
            ))
        })?;
    let data_file = fragment
        .metadata()
        .files
        .iter()
        .find(|file| field_tree_stored_in_file(blob_field, &file.fields))
        .ok_or_else(|| {
            Error::internal(format!(
                "Data file not found for blob field id {field_id} in fragment {fragment_id}"
            ))
        })?;
    let key = BlobDataFileKey {
        base_id: data_file.base_id,
        path: data_file.path.clone(),
    };
    cache.insert((field_id, fragment_id), key.clone());
    Ok(key)
}

fn collect_managed_blob_extents(
    descriptors: &StructArray,
    data_files: &[Option<BlobDataFileKey>],
    field_id: u32,
    extents: &mut HashMap<BlobObjectKey, Vec<BlobExtent>>,
) -> Result<()> {
    if descriptors.len() != data_files.len() {
        return Err(Error::internal(format!(
            "Blob v2 descriptor count {} did not match data file count {} for field id {field_id}",
            descriptors.len(),
            data_files.len()
        )));
    }
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
    let positions = descriptors
        .column_by_name("position")
        .and_then(|column| column.as_any().downcast_ref::<UInt64Array>())
        .ok_or_else(|| {
            Error::internal("Blob v2 descriptor 'position' must be UInt64".to_string())
        })?;
    let sizes = descriptors
        .column_by_name("size")
        .and_then(|column| column.as_any().downcast_ref::<UInt64Array>())
        .ok_or_else(|| Error::internal("Blob v2 descriptor 'size' must be UInt64".to_string()))?;
    let blob_ids = descriptors
        .column_by_name("blob_id")
        .and_then(|column| column.as_any().downcast_ref::<UInt32Array>())
        .ok_or_else(|| {
            Error::internal("Blob v2 descriptor 'blob_id' must be UInt32".to_string())
        })?;

    for (row, data_file) in data_files.iter().cloned().enumerate() {
        let Some(data_file) = data_file else {
            continue;
        };
        if descriptors.is_null(row) || kinds.is_null(row) {
            continue;
        }
        let kind = BlobKind::try_from(kinds.value(row))?;
        if !matches!(kind, BlobKind::Packed | BlobKind::Dedicated) {
            continue;
        }
        if sizes.is_null(row) || blob_ids.is_null(row) {
            return Err(Error::internal(format!(
                "Blob v2 {kind:?} descriptor at row {row} must set size and blob_id"
            )));
        }
        let start = if matches!(kind, BlobKind::Packed) {
            if positions.is_null(row) {
                return Err(Error::internal(format!(
                    "Blob v2 packed descriptor at row {row} must set position"
                )));
            }
            positions.value(row)
        } else {
            0
        };
        let size = sizes.value(row);
        let end = start.checked_add(size).ok_or_else(|| {
            Error::internal(format!(
                "Blob v2 {kind:?} range overflowed u64: position={start}, size={size}"
            ))
        })?;
        let object = BlobObjectKey {
            data_file,
            blob_id: blob_ids.value(row),
        };
        extents.entry(object).or_default().push(BlobExtent {
            field_id,
            start,
            end,
        });
    }
    Ok(())
}

fn descend_list<O: OffsetSizeTrait>(
    list: &GenericListArray<O>,
    parent_row_addrs: &[Option<u64>],
) -> Result<(ArrayRef, Vec<Option<u64>>)> {
    if list.len() != parent_row_addrs.len() {
        return Err(Error::internal(format!(
            "Blob list row count {} did not match row address count {}",
            list.len(),
            parent_row_addrs.len()
        )));
    }
    let offsets = list.value_offsets();
    let values_start = offsets[0].as_usize();
    let values_end = offsets[list.len()].as_usize();
    if values_end < values_start {
        return Err(Error::internal(
            "Blob list offsets decreased while collecting statistics".to_string(),
        ));
    }
    let mut child_row_addrs = Vec::with_capacity(values_end - values_start);
    for row in 0..list.len() {
        let start = offsets[row].as_usize();
        let end = offsets[row + 1].as_usize();
        if end < start {
            return Err(Error::internal(format!(
                "Blob list offsets decreased at row {row}: start={start}, end={end}"
            )));
        }
        let row_addr = if list.is_null(row) {
            None
        } else {
            parent_row_addrs[row]
        };
        child_row_addrs.extend(std::iter::repeat_n(row_addr, end - start));
    }
    Ok((
        list.values().slice(values_start, values_end - values_start),
        child_row_addrs,
    ))
}

fn blob_leaf_array(
    batch: &RecordBatch,
    path: &str,
    mut row_addrs: Vec<Option<u64>>,
) -> Result<(ArrayRef, Vec<Option<u64>>)> {
    let mut array = batch.column_by_name(path).cloned().ok_or_else(|| {
        Error::internal(format!(
            "Projected blob field '{path}' was missing from the statistics scan"
        ))
    })?;
    loop {
        if let Some(list) = array.as_any().downcast_ref::<GenericListArray<i32>>() {
            (array, row_addrs) = descend_list(list, &row_addrs)?;
        } else if let Some(list) = array.as_any().downcast_ref::<GenericListArray<i64>>() {
            (array, row_addrs) = descend_list(list, &row_addrs)?;
        } else {
            return Ok((array, row_addrs));
        }
    }
}

fn aggregate_blob_extents(
    extents: HashMap<BlobObjectKey, Vec<BlobExtent>>,
) -> Result<HashMap<u32, u64>> {
    let mut stats = HashMap::new();
    for object_extents in extents.into_values() {
        let mut events = Vec::with_capacity(object_extents.len() * 2);
        for extent in object_extents {
            if extent.start == extent.end {
                continue;
            }
            events.push((extent.start, extent.field_id, true));
            events.push((extent.end, extent.field_id, false));
        }
        events.sort_unstable_by_key(|event| event.0);
        let Some(first_event) = events.first() else {
            continue;
        };
        let mut active_fields = BTreeMap::<u32, usize>::new();
        let mut previous_position = first_event.0;
        let mut event_index = 0;
        while event_index < events.len() {
            let position = events[event_index].0;
            if position > previous_position
                && let Some((&field_id, _)) = active_fields.first_key_value()
            {
                let bytes = position - previous_position;
                let field_bytes = stats.entry(field_id).or_insert(0_u64);
                *field_bytes = field_bytes.checked_add(bytes).ok_or_else(|| {
                    Error::internal(format!(
                        "Blob v2 sidecar byte count overflowed u64 for field id {field_id}"
                    ))
                })?;
            }
            while event_index < events.len() && events[event_index].0 == position {
                let (_, field_id, is_start) = events[event_index];
                if is_start {
                    *active_fields.entry(field_id).or_insert(0) += 1;
                } else {
                    let count = active_fields.get_mut(&field_id).ok_or_else(|| {
                        Error::internal(format!(
                            "Blob v2 extent ended without an active start for field id {field_id}"
                        ))
                    })?;
                    *count -= 1;
                    if *count == 0 {
                        active_fields.remove(&field_id);
                    }
                }
                event_index += 1;
            }
            previous_position = position;
        }
    }
    Ok(stats)
}

async fn calculate_blob_v2_sidecar_stats(
    dataset: &Arc<Dataset>,
    persisted_blob_fields: &HashSet<(u32, u32)>,
) -> Result<HashMap<u32, u64>> {
    let blob_fields = dataset
        .schema()
        .fields_pre_order()
        .filter(|field| field.is_blob_v2())
        .map(|field| {
            Ok((
                field.id as u32,
                dataset.schema().field_path_minimal(field.id)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    if blob_fields.is_empty() {
        return Ok(HashMap::new());
    }

    let fallback_fragments = dataset
        .get_fragments()
        .into_iter()
        .filter(|fragment| {
            blob_fields.iter().any(|(field_id, _)| {
                dataset
                    .schema()
                    .field_by_id(*field_id as i32)
                    .is_some_and(|field| {
                        fragment
                            .metadata()
                            .files
                            .iter()
                            .any(|file| field_tree_stored_in_file(field, &file.fields))
                    })
                    && !persisted_blob_fields.contains(&(fragment.id() as u32, *field_id))
            })
        })
        .map(|fragment| fragment.metadata().clone())
        .collect::<Vec<_>>();
    if fallback_fragments.is_empty() {
        return Ok(HashMap::new());
    }

    let paths = blob_fields
        .iter()
        .map(|(_, path)| path.as_str())
        .collect::<Vec<_>>();
    let mut scanner = dataset.scan();
    scanner.project(&paths)?;
    scanner.with_fragments(fallback_fragments);
    scanner.blob_handling(BlobHandling::BlobsDescriptions);
    // Storage statistics cover physical data, including rows hidden by deletion vectors.
    // Deleted-row scans require row ids; row addresses identify each sidecar's data file.
    scanner
        .with_row_id()
        .with_row_address()
        .include_deleted_rows();
    let mut stream = scanner.try_into_stream().await?;
    let mut extents = HashMap::new();
    let mut data_file_cache = HashMap::new();
    while let Some(batch) = stream.try_next().await? {
        let row_addr_array = batch
            .column_by_name(ROW_ADDR)
            .and_then(|array| array.as_any().downcast_ref::<UInt64Array>())
            .ok_or_else(|| {
                Error::internal(format!(
                    "Blob v2 statistics scan must include UInt64 {ROW_ADDR}"
                ))
            })?;
        let row_addrs = (0..row_addr_array.len())
            .map(|row| (!row_addr_array.is_null(row)).then(|| row_addr_array.value(row)))
            .collect::<Vec<_>>();
        for (field_id, path) in &blob_fields {
            let (array, leaf_row_addrs) = blob_leaf_array(&batch, path, row_addrs.clone())?;
            let descriptors = array
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| {
                    Error::internal(format!(
                        "Blob v2 field '{path}' must be Struct, got {}",
                        array.data_type()
                    ))
                })?;
            let data_files = leaf_row_addrs
                .iter()
                .enumerate()
                .map(|(row, row_addr)| {
                    if descriptors.is_null(row) {
                        return Ok(None);
                    }
                    let Some(row_addr) = row_addr else {
                        return Ok(None);
                    };
                    let fragment_id = RowAddress::from(*row_addr).fragment_id();
                    if persisted_blob_fields.contains(&(fragment_id, *field_id)) {
                        return Ok(None);
                    }
                    blob_data_file_key(dataset, *field_id, *row_addr, &mut data_file_cache)
                        .map(Some)
                })
                .collect::<Result<Vec<_>>>()?;
            collect_managed_blob_extents(descriptors, &data_files, *field_id, &mut extents)?;
        }
    }
    aggregate_blob_extents(extents)
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
            let mut persisted_blob_fields = HashSet::new();
            futures::stream::iter(fragments)
                .map(|fragment| {
                    let file_fragment = FileFragment::new(dataset.clone(), fragment);
                    let schema = schema.clone();
                    let scan_scheduler = scan_scheduler.clone();
                    async move { file_fragment.storage_stats(&schema, scan_scheduler).await }
                })
                .buffer_unordered(self.object_store.io_parallelism())
                .try_for_each(|fragment_stats| {
                    let result = (|| {
                        let fragment_id = fragment_stats.fragment_id;
                        persisted_blob_fields.extend(
                            fragment_stats
                                .persisted_blob_fields
                                .into_iter()
                                .map(|field_id| (fragment_id, field_id)),
                        );
                        for (field_id, bytes) in fragment_stats.fields {
                            if let Some(stats) = field_stats.get_mut(&field_id) {
                                stats.bytes_on_disk = stats
                                    .bytes_on_disk
                                    .checked_add(bytes)
                                    .ok_or_else(|| {
                                    Error::internal(format!(
                                        "Data statistics byte count overflowed u64 for field id {field_id}"
                                    ))
                                })?;
                            }
                        }
                        Ok(())
                    })();
                    futures::future::ready(result)
                })
                .await?;

            for (field_id, bytes) in
                calculate_blob_v2_sidecar_stats(self, &persisted_blob_fields).await?
            {
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

    use arrow_array::{
        ArrayRef, Int32Array, LargeBinaryArray, ListArray, RecordBatch, RecordBatchIterator,
        StringArray,
    };
    use arrow_buffer::{OffsetBuffer, ScalarBuffer};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_core::{datatypes::BLOB_V2_DESC_FIELDS, utils::tempfile::TempStrDir};
    use lance_file::version::LanceFileVersion;

    use crate::{
        blob::{
            BlobArrayBuilder, BlobDescriptorArrayBuilder, BlobFieldOptions, blob_field_with_options,
        },
        dataset::WriteParams,
    };

    use super::*;

    #[test]
    fn test_blob_v2_sidecar_stats_count_unique_physical_extents() {
        let descriptors = StructArray::try_new(
            BLOB_V2_DESC_FIELDS.clone(),
            vec![
                Arc::new(UInt8Array::from(vec![
                    BlobKind::Packed as u8,
                    BlobKind::Packed as u8,
                    BlobKind::Packed as u8,
                    BlobKind::Dedicated as u8,
                    BlobKind::Dedicated as u8,
                ])),
                Arc::new(UInt64Array::from(vec![0, 0, 2, 0, 0])),
                Arc::new(UInt64Array::from(vec![3, 3, 3, 5, 5])),
                Arc::new(UInt32Array::from(vec![7, 7, 7, 8, 8])),
                Arc::new(StringArray::from(vec!["", "", "", "", ""])),
            ],
            None,
        )
        .unwrap();
        let data_file = BlobDataFileKey {
            base_id: None,
            path: "data.lance".to_string(),
        };
        let data_files = vec![Some(data_file); descriptors.len()];
        let mut extents = HashMap::new();
        collect_managed_blob_extents(&descriptors, &data_files, 42, &mut extents).unwrap();

        let stats = aggregate_blob_extents(extents).unwrap();
        assert_eq!(stats.get(&42), Some(&10));
    }

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
        let payloads = [vec![1_u8; 512], vec![2_u8; 8 * 1024]];
        let payload_bytes = payloads.iter().map(Vec::len).sum::<usize>() as u64;
        let mut blob_builder = BlobArrayBuilder::new(payloads.len());
        for payload in &payloads {
            blob_builder.push_bytes(payload).unwrap();
        }

        let list_payloads = [vec![3_u8; 256], vec![4_u8; 3 * 1024]];
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

    #[tokio::test]
    async fn test_blob_v2_sidecar_fallback_projects_only_nested_leaf() {
        let mut blob_descriptors = BlobDescriptorArrayBuilder::new("blob");
        blob_descriptors.push_dedicated(1, 4 * 1024).unwrap();
        let blob_descriptors = blob_descriptors.finish().unwrap();
        let (blob_field, blob_array) = blob_descriptors.into_parts();

        let mut state = 0x9e37_79b9_u32;
        let sibling_bytes = (0..8 * 1024 * 1024)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as u8
            })
            .collect::<Vec<_>>();
        let sibling_field = ArrowField::new("sibling", DataType::LargeBinary, false);
        let info_fields = vec![blob_field, sibling_field];
        let info_array = Arc::new(
            StructArray::try_new(
                info_fields.clone().into(),
                vec![
                    blob_array,
                    Arc::new(LargeBinaryArray::from_iter_values([
                        sibling_bytes.as_slice()
                    ])),
                ],
                None,
            )
            .unwrap(),
        ) as ArrayRef;
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "info",
            DataType::Struct(info_fields.into()),
            false,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![info_array]).unwrap();
        let test_dir = TempStrDir::default();
        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new(vec![Ok(batch)], schema),
                &test_dir,
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );
        let blob_field_id = dataset.schema().field("info").unwrap().children[0].id as u32;

        dataset.object_store.as_ref().io_stats_incremental();
        let stats = dataset.calculate_data_stats().await.unwrap();
        let io_stats = dataset.object_store.as_ref().io_stats_incremental();

        assert!(
            io_stats.read_bytes < 4 * 1024 * 1024,
            "descriptor fallback read {} bytes from an unrelated 8 MiB sibling",
            io_stats.read_bytes
        );
        let blob_stats = stats
            .fields
            .iter()
            .find(|stats| stats.id == blob_field_id)
            .unwrap();
        assert!(
            blob_stats.bytes_on_disk >= 4 * 1024,
            "blob bytes_on_disk={} should include the 4 KiB dedicated sidecar",
            blob_stats.bytes_on_disk
        );
    }
}
