// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Snapshot-bound analysis for reclaiming underutilized Blob v2 pack files.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes::{UInt8Type, UInt32Type, UInt64Type};
use arrow_array::{Array, ArrayRef, GenericListArray, OffsetSizeTrait, StructArray};
use futures::{StreamExt, TryStreamExt};
use lance_arrow::{list::ListArrayExt, r#struct::StructArrayExt};
use lance_core::datatypes::{BlobKind, BlobV2Layout, Field as LanceField};
use lance_core::utils::{address::RowAddress, blob::blob_path};
use lance_core::{Error, ROW_ADDR, Result};
use object_store::path::Path;
use roaring::RoaringBitmap;

use super::field_contains_blob_v2;
use crate::Dataset;
use crate::dataset::blob::{BlobReuseInput, resolve_blob_reuse_input};

#[derive(Debug, Default)]
pub(super) struct BlobRepackPlan {
    repack_sources: HashSet<BlobReuseInput>,
    candidate_fragments: RoaringBitmap,
    reclaimable_bytes: u64,
}

impl BlobRepackPlan {
    pub(super) fn should_repack(&self, source: &BlobReuseInput) -> bool {
        self.repack_sources.contains(source)
    }

    pub(super) fn contains_fragment(&self, fragment_id: u32) -> bool {
        self.candidate_fragments.contains(fragment_id)
    }

    pub(super) fn is_empty(&self) -> bool {
        self.repack_sources.is_empty()
    }

    pub(super) fn reclaimable_bytes(&self) -> u64 {
        self.reclaimable_bytes
    }
}

#[derive(Debug, Default)]
struct LiveRanges {
    ranges: BTreeMap<u64, u64>,
}

impl LiveRanges {
    fn insert(&mut self, mut start: u64, mut end: u64) {
        if start == end {
            return;
        }

        if let Some((&previous_start, &previous_end)) = self.ranges.range(..=start).next_back()
            && previous_end >= start
        {
            start = previous_start;
            end = end.max(previous_end);
            self.ranges.remove(&previous_start);
        }

        while let Some((&next_start, &next_end)) = self.ranges.range(start..).next() {
            if next_start > end {
                break;
            }
            end = end.max(next_end);
            self.ranges.remove(&next_start);
        }
        self.ranges.insert(start, end);
    }

    fn merge(&mut self, other: Self) {
        for (start, end) in other.ranges {
            self.insert(start, end);
        }
    }

    fn live_bytes(&self) -> Result<u64> {
        self.ranges.iter().try_fold(0_u64, |total, (start, end)| {
            total.checked_add(end - start).ok_or_else(|| {
                Error::internal("Packed Blob v2 live byte count overflowed u64".to_string())
            })
        })
    }
}

#[derive(Debug, Default)]
struct SourceUsage {
    ranges: LiveRanges,
    fragments: RoaringBitmap,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PhysicalPackKey {
    store_prefix: String,
    path: String,
}

struct PhysicalPackUsage {
    object_store: Arc<lance_io::object_store::ObjectStore>,
    path: Path,
    sources: HashSet<BlobReuseInput>,
    ranges: LiveRanges,
    fragments: RoaringBitmap,
}

impl PhysicalPackUsage {
    fn merge(&mut self, source: BlobReuseInput, usage: SourceUsage) {
        self.sources.insert(source);
        self.ranges.merge(usage.ranges);
        self.fragments |= usage.fragments;
    }
}

struct DescriptorColumns<'a> {
    kinds: &'a arrow_array::UInt8Array,
    positions: &'a arrow_array::UInt64Array,
    sizes: &'a arrow_array::UInt64Array,
    blob_ids: &'a arrow_array::UInt32Array,
}

impl<'a> DescriptorColumns<'a> {
    fn try_new(descriptions: &'a StructArray, field_name: &str) -> Result<Self> {
        if BlobV2Layout::classify(descriptions.fields()) != Some(BlobV2Layout::Descriptor) {
            return Err(Error::internal(format!(
                "Blob v2 field '{field_name}' did not produce descriptor layout during repack analysis"
            )));
        }
        let column = |name| {
            descriptions.column_by_name(name).ok_or_else(|| {
                Error::internal(format!(
                    "Blob v2 descriptor for field '{field_name}' is missing '{name}'"
                ))
            })
        };
        Ok(Self {
            kinds: column("kind")?.as_primitive::<UInt8Type>(),
            positions: column("position")?.as_primitive::<UInt64Type>(),
            sizes: column("size")?.as_primitive::<UInt64Type>(),
            blob_ids: column("blob_id")?.as_primitive::<UInt32Type>(),
        })
    }
}

fn collect_blob_field(
    dataset: &Dataset,
    field: &LanceField,
    array: ArrayRef,
    row_addrs: &[u64],
    usages: &mut HashMap<BlobReuseInput, SourceUsage>,
) -> Result<()> {
    if !field_contains_blob_v2(field) {
        return Ok(());
    }

    if field.is_blob_v2() {
        let field_id = u32::try_from(field.id).map_err(|_| {
            Error::internal(format!(
                "Blob v2 field id {} for '{}' does not fit in u32",
                field.id, field.name
            ))
        })?;
        let descriptions = array.as_struct();
        if descriptions.len() != row_addrs.len() {
            return Err(Error::internal(format!(
                "Blob v2 field '{}' produced {} descriptors for {} row addresses during repack analysis",
                field.name,
                descriptions.len(),
                row_addrs.len()
            )));
        }
        let columns = DescriptorColumns::try_new(descriptions, &field.name)?;
        for (row_index, row_addr) in row_addrs.iter().copied().enumerate() {
            if descriptions.is_null(row_index) || columns.kinds.is_null(row_index) {
                continue;
            }
            let kind = BlobKind::try_from(columns.kinds.value(row_index)).map_err(|error| {
                Error::internal(format!(
                    "Blob v2 field '{}' has invalid kind at row {row_index}: {error}",
                    field.name
                ))
            })?;
            if kind != BlobKind::Packed {
                continue;
            }
            if columns.positions.is_null(row_index)
                || columns.sizes.is_null(row_index)
                || columns.blob_ids.is_null(row_index)
            {
                return Err(Error::internal(format!(
                    "Packed Blob v2 field '{}' row {row_index} is missing position, size, or blob_id",
                    field.name
                )));
            }
            let start = columns.positions.value(row_index);
            let size = columns.sizes.value(row_index);
            let end = start.checked_add(size).ok_or_else(|| {
                Error::internal(format!(
                    "Packed Blob v2 field '{}' row {row_index} range {start}+{size} overflows u64",
                    field.name
                ))
            })?;
            let source = resolve_blob_reuse_input(
                dataset,
                field_id,
                row_addr,
                columns.blob_ids.value(row_index),
            )?;
            let usage = usages.entry(source).or_default();
            usage.ranges.insert(start, end);
            usage
                .fragments
                .insert(RowAddress::from(row_addr).fragment_id());
        }
        return Ok(());
    }

    match (field.data_type(), array.data_type()) {
        (arrow_schema::DataType::Struct(_), arrow_schema::DataType::Struct(_)) => {
            let struct_array = array.as_struct().normalize_slicing()?.pushdown_nulls()?;
            if field.children.len() != struct_array.num_columns() {
                return Err(Error::internal(format!(
                    "Struct field '{}' has {} schema children but {} array children during Blob v2 repack analysis",
                    field.name,
                    field.children.len(),
                    struct_array.num_columns()
                )));
            }
            for (child, child_array) in field.children.iter().zip(struct_array.columns()) {
                collect_blob_field(dataset, child, child_array.clone(), row_addrs, usages)?;
            }
            Ok(())
        }
        (arrow_schema::DataType::List(_), arrow_schema::DataType::List(_)) => {
            collect_blob_list(dataset, field, array.as_list::<i32>(), row_addrs, usages)
        }
        (arrow_schema::DataType::LargeList(_), arrow_schema::DataType::LargeList(_)) => {
            collect_blob_list(dataset, field, array.as_list::<i64>(), row_addrs, usages)
        }
        (logical_type, physical_type) => Err(Error::internal(format!(
            "Field '{}' contains Blob v2 descendants but logical type {logical_type:?} and descriptor type {physical_type:?} are incompatible",
            field.name
        ))),
    }
}

fn collect_blob_list<O: OffsetSizeTrait>(
    dataset: &Dataset,
    field: &LanceField,
    list_array: &GenericListArray<O>,
    row_addrs: &[u64],
    usages: &mut HashMap<BlobReuseInput, SourceUsage>,
) -> Result<()> {
    let child = field.children.first().ok_or_else(|| {
        Error::internal(format!(
            "List field '{}' is missing its child during Blob v2 repack analysis",
            field.name
        ))
    })?;
    let list_array = if list_array.null_count() > 0 {
        list_array.filter_garbage_nulls()
    } else {
        list_array.clone()
    };
    if list_array.len() != row_addrs.len() {
        return Err(Error::internal(format!(
            "List field '{}' has {} rows but {} row addresses during Blob v2 repack analysis",
            field.name,
            list_array.len(),
            row_addrs.len()
        )));
    }
    let offsets = list_array.value_offsets();
    let values_start = offsets[0].as_usize();
    let values_end = offsets[list_array.len()].as_usize();
    if values_end < values_start {
        return Err(Error::internal(format!(
            "List field '{}' has decreasing offsets during Blob v2 repack analysis",
            field.name
        )));
    }
    let mut child_row_addrs = Vec::with_capacity(values_end - values_start);
    for (row_index, row_addr) in row_addrs.iter().copied().enumerate() {
        let start = offsets[row_index].as_usize();
        let end = offsets[row_index + 1].as_usize();
        if end < start {
            return Err(Error::internal(format!(
                "List field '{}' row {row_index} has decreasing offsets during Blob v2 repack analysis",
                field.name
            )));
        }
        child_row_addrs.extend(std::iter::repeat_n(row_addr, end - start));
    }
    collect_blob_field(
        dataset,
        child,
        list_array
            .values()
            .slice(values_start, values_end - values_start),
        &child_row_addrs,
        usages,
    )
}

async fn collect_source_usages(
    dataset: &Dataset,
    batch_size: Option<usize>,
) -> Result<HashMap<BlobReuseInput, SourceUsage>> {
    let blob_fields = dataset
        .schema()
        .fields
        .iter()
        .filter(|field| field_contains_blob_v2(field))
        .collect::<Vec<_>>();
    if blob_fields.is_empty() {
        return Ok(HashMap::new());
    }

    let projection = blob_fields
        .iter()
        .map(|field| field.name.clone())
        .collect::<Vec<_>>();
    let mut scanner = dataset.scan();
    scanner.project(&projection)?;
    scanner.with_row_address();
    scanner.scan_in_order(true);
    if let Some(batch_size) = batch_size {
        scanner.batch_size(batch_size);
    }
    let mut stream = scanner.try_into_stream().await?;
    let mut usages = HashMap::new();
    while let Some(batch) = stream.try_next().await? {
        let row_addrs = batch
            .column_by_name(ROW_ADDR)
            .ok_or_else(|| {
                Error::internal("Blob v2 repack analysis scan did not return _rowaddr".to_string())
            })?
            .as_primitive::<UInt64Type>();
        for field in &blob_fields {
            let array = batch.column_by_name(&field.name).ok_or_else(|| {
                Error::internal(format!(
                    "Blob v2 repack analysis scan did not return field '{}'; returned fields: {:?}",
                    field.name,
                    batch
                        .schema()
                        .fields()
                        .iter()
                        .map(|field| field.name())
                        .collect::<Vec<_>>()
                ))
            })?;
            collect_blob_field(
                dataset,
                field,
                array.clone(),
                row_addrs.values(),
                &mut usages,
            )?;
        }
    }
    Ok(usages)
}

async fn group_physical_packs(
    dataset: &Dataset,
    source_usages: HashMap<BlobReuseInput, SourceUsage>,
) -> Result<HashMap<PhysicalPackKey, PhysicalPackUsage>> {
    let mut physical_packs: HashMap<PhysicalPackKey, PhysicalPackUsage> = HashMap::new();
    for (source, usage) in source_usages {
        let object_store = dataset.object_store(source.base_id).await?;
        let data_dir = dataset.data_file_dir_for_base(source.base_id)?;
        let path = blob_path(&data_dir, &source.blob_dir, source.physical_id);
        let key = PhysicalPackKey {
            store_prefix: object_store.store_prefix.clone(),
            path: path.to_string(),
        };
        match physical_packs.entry(key) {
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                entry.get_mut().merge(source, usage);
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(PhysicalPackUsage {
                    object_store,
                    path,
                    sources: HashSet::from([source]),
                    ranges: usage.ranges,
                    fragments: usage.fragments,
                });
            }
        }
    }
    Ok(physical_packs)
}

pub(super) async fn analyze(
    dataset: &Dataset,
    utilization_threshold: f32,
    batch_size: Option<usize>,
) -> Result<Arc<BlobRepackPlan>> {
    if utilization_threshold <= 0.0 {
        return Ok(Arc::new(BlobRepackPlan::default()));
    }

    let source_usages = collect_source_usages(dataset, batch_size).await?;
    let physical_packs = group_physical_packs(dataset, source_usages).await?;
    let io_parallelism = dataset.object_store.io_parallelism();
    let mut sized_packs = futures::stream::iter(physical_packs.into_values())
        .map(|usage| async move {
            let physical_size = usage.object_store.size(&usage.path).await?;
            Ok::<_, Error>((usage, physical_size))
        })
        .buffer_unordered(io_parallelism);

    let mut plan = BlobRepackPlan::default();
    while let Some(result) = sized_packs.next().await {
        let (usage, physical_size) = result?;
        let live_bytes = usage.ranges.live_bytes()?;
        if live_bytes > physical_size {
            return Err(Error::corrupt_file(
                usage.path,
                format!(
                    "Packed Blob v2 live range union is {live_bytes} bytes but the physical pack is {physical_size} bytes"
                ),
            ));
        }
        if physical_size == 0 {
            continue;
        }
        let utilization = live_bytes as f64 / physical_size as f64;
        if utilization < f64::from(utilization_threshold) {
            plan.repack_sources.extend(usage.sources);
            plan.candidate_fragments |= usage.fragments;
            plan.reclaimable_bytes = plan
                .reclaimable_bytes
                .checked_add(physical_size - live_bytes)
                .ok_or_else(|| {
                    Error::internal(
                        "Blob v2 repack reclaimable byte count overflowed u64".to_string(),
                    )
                })?;
        }
    }
    Ok(Arc::new(plan))
}

#[cfg(test)]
mod tests {
    use super::LiveRanges;

    #[test]
    fn live_ranges_merge_duplicate_overlap_and_adjacency() {
        let mut ranges = LiveRanges::default();
        for (start, end) in [(10, 20), (10, 20), (15, 30), (40, 50), (30, 40)] {
            ranges.insert(start, end);
        }
        assert_eq!(ranges.live_bytes().unwrap(), 40);
        assert_eq!(
            ranges.ranges.into_iter().collect::<Vec<_>>(),
            vec![(10, 50)]
        );
    }
}
