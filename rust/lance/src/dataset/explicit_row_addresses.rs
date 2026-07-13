// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{BTreeMap, VecDeque};
use std::ops::Range;
use std::sync::Arc;

use arrow_array::{Array, UInt64Array};
use futures::{StreamExt, TryStreamExt, stream};
use lance_core::cache::LanceCache;
use lance_core::utils::address::{LogicalRowAddress, RowAddress};
use lance_core::{Error, ROW_ADDR, ROW_ID, Result};
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_file::reader::FileReader;
use lance_io::ReadBatchParams;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use lance_table::format::{
    ExplicitMapDestination, ExplicitMapRowIdPage, PhysicalRowLocator, PhysicalToLogicalResolution,
    PlacementResolution, RowAddressPlacement, fingerprint_explicit_map_u64_page,
};

use super::Dataset;

pub const EXPLICIT_ROW_ADDRESS_PAGE_ROWS: usize = if cfg!(test) { 4 } else { 4096 };
const EXPLICIT_ROW_ADDRESS_CACHE_PAGES: usize = 128;
const EXPLICIT_ROW_ADDRESS_CACHE_FILES: usize = 128;

fn explicit_object_path(
    base: &object_store::path::Path,
    relative_path: &str,
) -> object_store::path::Path {
    relative_path
        .split('/')
        .filter(|segment| !segment.is_empty())
        .fold(base.clone(), |path, segment| path.join(segment))
}

fn verify_explicit_u64_page(
    relative_path: &str,
    columns: &[&[u64]],
    expected_fingerprint: &[u8],
) -> Result<()> {
    let actual_fingerprint = fingerprint_explicit_map_u64_page(columns)?;
    if actual_fingerprint != expected_fingerprint {
        return Err(Error::corrupt_file(
            object_store::path::Path::from(relative_path),
            format!(
                "ExplicitMap page content fingerprint mismatch: expected={expected_fingerprint:02x?}, actual={actual_fingerprint:02x?}"
            ),
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ExplicitFileCacheKey {
    base_id: Option<u32>,
    relative_path: String,
}

#[derive(Debug, Default)]
struct ExplicitFileCache {
    entries: BTreeMap<ExplicitFileCacheKey, (u64, FileReader)>,
    lru: VecDeque<ExplicitFileCacheKey>,
}

impl ExplicitFileCache {
    fn get(&mut self, key: &ExplicitFileCacheKey) -> Option<(u64, FileReader)> {
        let value = self.entries.get(key)?.clone();
        if let Some(position) = self.lru.iter().position(|candidate| candidate == key) {
            self.lru.remove(position);
        }
        self.lru.push_back(key.clone());
        Some(value)
    }

    fn insert(&mut self, key: ExplicitFileCacheKey, object_size: u64, reader: FileReader) {
        if let Some(position) = self.lru.iter().position(|candidate| candidate == &key) {
            self.lru.remove(position);
        }
        self.entries.insert(key.clone(), (object_size, reader));
        self.lru.push_back(key);
        while self.entries.len() > EXPLICIT_ROW_ADDRESS_CACHE_FILES {
            if let Some(evicted) = self.lru.pop_front() {
                self.entries.remove(&evicted);
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ExplicitPageCacheKey {
    base_id: Option<u32>,
    relative_path: String,
    row_start: u64,
    row_count: u64,
    column_count: usize,
}

#[derive(Debug, Default)]
struct ExplicitPageCache {
    entries: BTreeMap<ExplicitPageCacheKey, Vec<Arc<UInt64Array>>>,
    lru: VecDeque<ExplicitPageCacheKey>,
}

impl ExplicitPageCache {
    fn get(&mut self, key: &ExplicitPageCacheKey) -> Option<Vec<Arc<UInt64Array>>> {
        let value = self.entries.get(key)?.clone();
        if let Some(position) = self.lru.iter().position(|candidate| candidate == key) {
            self.lru.remove(position);
        }
        self.lru.push_back(key.clone());
        Some(value)
    }

    fn insert(&mut self, key: ExplicitPageCacheKey, value: Vec<Arc<UInt64Array>>) {
        if let Some(position) = self.lru.iter().position(|candidate| candidate == &key) {
            self.lru.remove(position);
        }
        self.entries.insert(key.clone(), value);
        self.lru.push_back(key);
        while self.entries.len() > EXPLICIT_ROW_ADDRESS_CACHE_PAGES {
            if let Some(evicted) = self.lru.pop_front() {
                self.entries.remove(&evicted);
            }
        }
    }
}

#[derive(Debug, Default)]
pub struct ExplicitRowAddressCache {
    files: tokio::sync::Mutex<ExplicitFileCache>,
    pages: tokio::sync::Mutex<ExplicitPageCache>,
}

impl Dataset {
    #[cfg(test)]
    pub(crate) async fn explicit_cached_ranges_for_path(
        &self,
        relative_path: &str,
    ) -> Vec<(u64, u64)> {
        self.explicit_row_address_cache
            .pages
            .lock()
            .await
            .entries
            .keys()
            .filter(|key| key.relative_path == relative_path)
            .map(|key| (key.row_start, key.row_count))
            .collect()
    }

    async fn explicit_file_reader(
        &self,
        base_id: Option<u32>,
        relative_path: &str,
        object_size: u64,
    ) -> Result<FileReader> {
        if relative_path.is_empty() || object_size == 0 {
            return Err(Error::invalid_input(
                "ExplicitMap file metadata must include path and object size",
            ));
        }
        let cache_key = ExplicitFileCacheKey {
            base_id,
            relative_path: relative_path.to_owned(),
        };
        if let Some((cached_size, reader)) = self
            .explicit_row_address_cache
            .files
            .lock()
            .await
            .get(&cache_key)
        {
            if cached_size != object_size {
                return Err(Error::invalid_input(
                    "ExplicitMap cache metadata disagrees with the manifest root",
                ));
            }
            return Ok(reader);
        }
        let (object_store, base) = match base_id {
            Some(base_id) => {
                let base = self.manifest.base_paths.get(&base_id).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "ExplicitMap base_id {base_id} is missing from manifest.base_paths"
                    ))
                })?;
                if !base.is_dataset_root {
                    return Err(Error::invalid_input(format!(
                        "ExplicitMap base_id {base_id} must reference a dataset root"
                    )));
                }
                (
                    self.object_store(Some(base_id)).await?,
                    base.extract_path(self.session.store_registry())?,
                )
            }
            None => (self.object_store.clone(), self.base.clone()),
        };
        let path = explicit_object_path(&base, relative_path);
        let scheduler = ScanScheduler::new(
            object_store.clone(),
            SchedulerConfig::max_bandwidth(&object_store),
        );
        let file = scheduler
            .open_file(&path, &CachedFileSize::new(object_size))
            .await?;
        let reader = FileReader::try_open(
            file,
            None,
            Arc::<DecoderPlugins>::default(),
            &LanceCache::no_cache(),
            self.file_reader_options.clone().unwrap_or_default(),
        )
        .await?;
        self.explicit_row_address_cache.files.lock().await.insert(
            cache_key,
            object_size,
            reader.clone(),
        );
        Ok(reader)
    }

    async fn read_explicit_u64_columns(
        &self,
        base_id: Option<u32>,
        relative_path: &str,
        object_size: u64,
        expected_columns: &[&str],
        row_range: Range<u64>,
    ) -> Result<Vec<Arc<UInt64Array>>> {
        let row_count = row_range
            .end
            .checked_sub(row_range.start)
            .ok_or_else(|| Error::invalid_input("ExplicitMap row range ends before it starts"))?;
        if row_count == 0 {
            return Err(Error::invalid_input(
                "ExplicitMap row range must be non-empty",
            ));
        }
        let key = ExplicitPageCacheKey {
            base_id,
            relative_path: relative_path.to_owned(),
            row_start: row_range.start,
            row_count,
            column_count: expected_columns.len(),
        };
        let cacheable = row_count <= EXPLICIT_ROW_ADDRESS_PAGE_ROWS as u64;
        if cacheable
            && let Some(columns) = self.explicit_row_address_cache.pages.lock().await.get(&key)
        {
            return Ok(columns);
        }
        let reader = self
            .explicit_file_reader(base_id, relative_path, object_size)
            .await?;
        if row_range.end > reader.num_rows() {
            return Err(Error::invalid_input(format!(
                "ExplicitMap file {} has {} rows but range {}..{} was requested",
                relative_path,
                reader.num_rows(),
                row_range.start,
                row_range.end
            )));
        }
        let start = usize::try_from(row_range.start)
            .map_err(|_| Error::invalid_input("ExplicitMap row start exceeds usize"))?;
        let end = usize::try_from(row_range.end)
            .map_err(|_| Error::invalid_input("ExplicitMap row end exceeds usize"))?;
        let mut stream = reader
            .read_stream(
                ReadBatchParams::Range(start..end),
                EXPLICIT_ROW_ADDRESS_PAGE_ROWS as u32,
                4,
                FilterExpression::no_filter(),
            )
            .await?;
        let mut chunks = vec![Vec::new(); expected_columns.len()];
        while let Some(batch) = stream.try_next().await? {
            for (column_index, expected_name) in expected_columns.iter().enumerate() {
                let column = batch.column_by_name(expected_name).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "ExplicitMap file {} is missing required column {}",
                        relative_path, expected_name
                    ))
                })?;
                let values = column
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| {
                        Error::invalid_input(format!(
                            "ExplicitMap file {} column {} must be UInt64",
                            relative_path, expected_name
                        ))
                    })?;
                if values.null_count() != 0 {
                    return Err(Error::invalid_input(format!(
                        "ExplicitMap file {} column {} must not contain nulls",
                        relative_path, expected_name
                    )));
                }
                chunks[column_index].push(values.clone());
            }
        }
        let columns = chunks
            .into_iter()
            .map(|arrays| {
                let refs = arrays
                    .iter()
                    .map(|array| array as &dyn Array)
                    .collect::<Vec<_>>();
                arrow_select::concat::concat(&refs)
                    .map_err(Error::from)
                    .and_then(|array| {
                        array
                            .as_any()
                            .downcast_ref::<UInt64Array>()
                            .cloned()
                            .map(Arc::new)
                            .ok_or_else(|| Error::internal("concat changed ExplicitMap type"))
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        if columns
            .iter()
            .any(|column| column.len() as u64 != row_count)
        {
            return Err(Error::invalid_input(format!(
                "ExplicitMap file {} did not return the requested {} rows",
                relative_path, row_count
            )));
        }
        if cacheable {
            self.explicit_row_address_cache
                .pages
                .lock()
                .await
                .insert(key, columns.clone());
        }
        Ok(columns)
    }

    async fn explicit_locator_page(
        &self,
        placement_index: u32,
        page_index: u32,
    ) -> Result<(Arc<UInt64Array>, Arc<UInt64Array>)> {
        let (base_id, object_path, object_size, page) = self
            .manifest
            .row_address_layout
            .as_ref()
            .and_then(|layout| layout.placements.get(placement_index as usize))
            .ok_or_else(|| Error::invalid_input("ExplicitMap placement index is out of bounds"))
            .and_then(|placement| match placement {
                RowAddressPlacement::ExplicitMap(explicit) => explicit
                    .pages
                    .get(page_index as usize)
                    .cloned()
                    .map(|page| {
                        (
                            explicit.base_id,
                            explicit.object_path.clone(),
                            explicit.object_size,
                            page,
                        )
                    })
                    .ok_or_else(|| Error::invalid_input("ExplicitMap page index is out of bounds")),
                _ => Err(Error::invalid_input(
                    "external locator requested for a non-ExplicitMap placement",
                )),
            })?;
        let row_end = page
            .row_start
            .checked_add(page.row_count)
            .ok_or_else(|| Error::invalid_input("ExplicitMap locator page row range overflow"))?;
        let columns = self
            .read_explicit_u64_columns(
                base_id,
                &object_path,
                object_size,
                &[ROW_ID, ROW_ADDR],
                page.row_start..row_end,
            )
            .await?;
        let [logical, physical]: [Arc<UInt64Array>; 2] = columns.try_into().map_err(|_| {
            Error::internal("ExplicitMap locator reader returned the wrong column count")
        })?;
        verify_explicit_u64_page(
            &object_path,
            &[logical.values().as_ref(), physical.values().as_ref()],
            &page.content_fingerprint,
        )?;
        let page_row_count = usize::try_from(page.row_count)
            .map_err(|_| Error::invalid_input("ExplicitMap page row count exceeds usize"))?;
        if logical.len() != page_row_count
            || logical.len() != physical.len()
            || logical.values().first().copied() != Some(page.first_logical_address)
            || logical.values().last().copied() != Some(page.last_logical_address)
            || logical.values().windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(Error::invalid_input(
                "ExplicitMap locator must contain equal-length, strictly logical-sorted columns",
            ));
        }
        Ok((logical, physical))
    }

    pub(crate) async fn resolve_logical_row_ids_async(
        &self,
        row_ids: &[u64],
    ) -> Result<Vec<Option<RowAddress>>> {
        let logical = row_ids
            .iter()
            .copied()
            .map(LogicalRowAddress::try_from)
            .collect::<Result<Vec<_>>>()?;
        let resolutions = self.row_address_router()?.resolve_many(&logical)?;
        let mut result = vec![None; row_ids.len()];
        let mut explicit = BTreeMap::<(u32, u32), Vec<usize>>::new();
        for (index, resolution) in resolutions.into_iter().enumerate() {
            match resolution {
                PlacementResolution::Mapped {
                    locator: PhysicalRowLocator::Physical(address),
                } => result[index] = Some(address),
                PlacementResolution::Mapped {
                    locator:
                        PhysicalRowLocator::ExplicitMap {
                            placement_index,
                            page_index,
                        },
                } => explicit
                    .entry((placement_index, page_index))
                    .or_default()
                    .push(index),
                PlacementResolution::NotLive | PlacementResolution::Unmapped => {}
            }
        }
        let reads = stream::iter(explicit.into_iter().map(
            |((placement_index, page_index), indices)| async move {
                let (logical, physical) = self
                    .explicit_locator_page(placement_index, page_index)
                    .await?;
                Ok::<_, Error>((placement_index, indices, logical, physical))
            },
        ))
        .buffer_unordered(16);
        futures::pin_mut!(reads);
        while let Some((placement_index, indices, logical_column, physical_column)) =
            reads.try_next().await?
        {
            for index in indices {
                if let Ok(position) = logical_column.values().binary_search(&row_ids[index]) {
                    let physical = RowAddress::from(physical_column.value(position));
                    let placement = self
                        .manifest
                        .row_address_layout
                        .as_ref()
                        .and_then(|layout| layout.placements.get(placement_index as usize))
                        .expect("placement was validated before locator read");
                    let owned = placement.destination_ranges().iter().any(
                        |(fragment_id, start, row_count)| {
                            let end = *start as u64 + *row_count;
                            *fragment_id == physical.fragment_id()
                                && *start <= physical.row_offset()
                                && (physical.row_offset() as u64) < end
                        },
                    );
                    if !owned {
                        return Err(Error::invalid_input(format!(
                            "ExplicitMap locator returned physical address {} outside its destination closure",
                            u64::from(physical)
                        )));
                    }
                    result[index] = Some(physical);
                }
            }
        }
        Ok(result)
    }

    async fn explicit_destination_row_ids(
        &self,
        base_id: Option<u32>,
        destination: &ExplicitMapDestination,
        row_range: Range<u64>,
    ) -> Result<Arc<UInt64Array>> {
        let covered_pages = destination
            .row_id_pages
            .iter()
            .filter(|page| {
                page.row_start >= row_range.start
                    && page
                        .row_start
                        .checked_add(page.row_count)
                        .is_some_and(|page_end| page_end <= row_range.end)
            })
            .collect::<Vec<_>>();
        let mut expected_start = row_range.start;
        for page in &covered_pages {
            if page.row_start != expected_start {
                return Err(Error::invalid_input(
                    "ExplicitMap hidden _rowid read must align to persisted page boundaries",
                ));
            }
            expected_start = expected_start.checked_add(page.row_count).ok_or_else(|| {
                Error::invalid_input("ExplicitMap hidden _rowid page row range overflow")
            })?;
        }
        if covered_pages.is_empty() || expected_start != row_range.end {
            return Err(Error::invalid_input(
                "ExplicitMap hidden _rowid read must cover complete persisted pages",
            ));
        }
        let columns = self
            .read_explicit_u64_columns(
                base_id,
                &destination.row_id_file_path,
                destination.row_id_file_size,
                &[ROW_ID],
                row_range.clone(),
            )
            .await?;
        let [row_ids]: [Arc<UInt64Array>; 1] = columns.try_into().map_err(|_| {
            Error::internal("ExplicitMap hidden row-id reader returned the wrong column count")
        })?;
        let expected = row_range.end.checked_sub(row_range.start).ok_or_else(|| {
            Error::invalid_input("ExplicitMap hidden row-id range ends before it starts")
        })?;
        if row_ids.len() as u64 != expected {
            return Err(Error::invalid_input(format!(
                "ExplicitMap hidden row-id page has {} rows but range declares {}",
                row_ids.len(),
                expected
            )));
        }
        for page in covered_pages {
            let local_start = usize::try_from(page.row_start - row_range.start)
                .map_err(|_| Error::invalid_input("ExplicitMap page start exceeds usize"))?;
            let local_end = usize::try_from(page.row_start + page.row_count - row_range.start)
                .map_err(|_| Error::invalid_input("ExplicitMap page end exceeds usize"))?;
            verify_explicit_u64_page(
                &destination.row_id_file_path,
                &[&row_ids.values()[local_start..local_end]],
                &page.content_fingerprint,
            )?;
        }
        Ok(row_ids)
    }

    fn explicit_destination_page(
        destination: &ExplicitMapDestination,
        destination_row_offset: u32,
    ) -> Result<(usize, &ExplicitMapRowIdPage)> {
        let row_offset = destination_row_offset as u64;
        let page_index = destination
            .row_id_pages
            .partition_point(|page| page.row_start <= row_offset)
            .checked_sub(1)
            .ok_or_else(|| {
                Error::invalid_input("ExplicitMap destination row offset precedes its first page")
            })?;
        let page = &destination.row_id_pages[page_index];
        let page_end = page
            .row_start
            .checked_add(page.row_count)
            .ok_or_else(|| Error::invalid_input("ExplicitMap destination page range overflow"))?;
        if row_offset >= page_end {
            return Err(Error::invalid_input(
                "ExplicitMap destination row offset is outside its page directory",
            ));
        }
        Ok((page_index, page))
    }

    pub(crate) async fn explicit_row_ids_for_fragment(
        &self,
        physical_fragment_id: u32,
    ) -> Result<Option<Arc<UInt64Array>>> {
        let Some(layout) = self.manifest.row_address_layout.as_ref() else {
            return Ok(None);
        };
        let mut destination = None;
        for placement in &layout.placements {
            let RowAddressPlacement::ExplicitMap(explicit) = placement else {
                continue;
            };
            for candidate in &explicit.destinations {
                if candidate.physical_fragment_id == physical_fragment_id {
                    if candidate.destination_start != 0 {
                        return Err(Error::not_supported(
                            "partial-fragment ExplicitMap destinations are not supported",
                        ));
                    }
                    if destination.replace((explicit.base_id, candidate)).is_some() {
                        return Err(Error::invalid_input(
                            "physical fragment is owned by multiple ExplicitMap destinations",
                        ));
                    }
                }
            }
        }
        match destination {
            Some((base_id, destination)) => self
                .explicit_destination_row_ids(base_id, destination, 0..destination.row_count as u64)
                .await
                .map(Some),
            None => Ok(None),
        }
    }

    pub(crate) async fn resolve_physical_row_ids_async(
        &self,
        addresses: &[RowAddress],
    ) -> Result<Vec<Option<LogicalRowAddress>>> {
        let resolutions = self
            .row_address_router()?
            .physical_to_logical_many(addresses)?;
        let mut result = vec![None; addresses.len()];
        let mut explicit = BTreeMap::<(u32, u32, usize), Vec<(usize, u32)>>::new();
        for (index, resolution) in resolutions.into_iter().enumerate() {
            match resolution {
                PhysicalToLogicalResolution::Logical(logical) => result[index] = Some(logical),
                PhysicalToLogicalResolution::ExplicitMap {
                    placement_index,
                    destination_index,
                    destination_row_offset,
                } => {
                    let destination = self
                        .manifest
                        .row_address_layout
                        .as_ref()
                        .and_then(|layout| layout.placements.get(placement_index as usize))
                        .and_then(|placement| match placement {
                            RowAddressPlacement::ExplicitMap(explicit) => {
                                explicit.destinations.get(destination_index as usize)
                            }
                            _ => None,
                        })
                        .ok_or_else(|| {
                            Error::invalid_input("ExplicitMap destination index is invalid")
                        })?;
                    let (page_index, _) =
                        Self::explicit_destination_page(destination, destination_row_offset)?;
                    explicit
                        .entry((placement_index, destination_index, page_index))
                        .or_default()
                        .push((index, destination_row_offset));
                }
                PhysicalToLogicalResolution::Unmapped => {}
            }
        }
        let reads = stream::iter(explicit.into_iter().map(
            |((placement_index, destination_index, page_index), indices)| async move {
                let (base_id, destination) = self
                    .manifest
                    .row_address_layout
                    .as_ref()
                    .and_then(|layout| layout.placements.get(placement_index as usize))
                    .and_then(|placement| match placement {
                        RowAddressPlacement::ExplicitMap(explicit) => explicit
                            .destinations
                            .get(destination_index as usize)
                            .map(|destination| (explicit.base_id, destination)),
                        _ => None,
                    })
                    .map(|(base_id, destination)| (base_id, destination.clone()))
                    .ok_or_else(|| {
                        Error::invalid_input("ExplicitMap destination index is invalid")
                    })?;
                let page = destination
                    .row_id_pages
                    .get(page_index)
                    .ok_or_else(|| Error::invalid_input("ExplicitMap page index is invalid"))?;
                let page_start = page.row_start;
                let page_end = page_start.checked_add(page.row_count).ok_or_else(|| {
                    Error::invalid_input("ExplicitMap destination page range overflow")
                })?;
                let row_ids = self
                    .explicit_destination_row_ids(base_id, &destination, page_start..page_end)
                    .await?;
                Ok::<_, Error>((page_start, indices, row_ids))
            },
        ))
        .buffer_unordered(16);
        futures::pin_mut!(reads);
        while let Some((page_start, indices, row_ids)) = reads.try_next().await? {
            let logical = indices
                .iter()
                .map(|(_, destination_row_offset)| {
                    let local = *destination_row_offset as u64 - page_start;
                    LogicalRowAddress::try_from(row_ids.value(local as usize))
                })
                .collect::<Result<Vec<_>>>()?;
            let raw_logical = logical
                .iter()
                .map(|address| address.raw())
                .collect::<Vec<_>>();
            let current_owners = self.resolve_logical_row_ids_async(&raw_logical).await?;
            for (((index, _), logical), current_owner) in
                indices.into_iter().zip(logical).zip(current_owners)
            {
                if current_owner == Some(addresses[index]) {
                    result[index] = Some(logical);
                }
            }
        }
        Ok(result)
    }

    /// Resolve only physical rows that are the current owner of their logical
    /// identity. Historical physical copies can remain behind deletion vectors
    /// after update or maintenance; inverse placement alone must not make those
    /// copies live again.
    pub(crate) async fn resolve_current_physical_row_ids_async(
        &self,
        addresses: &[RowAddress],
    ) -> Result<Vec<Option<LogicalRowAddress>>> {
        let candidates = self.resolve_physical_row_ids_async(addresses).await?;
        let mut candidate_positions = Vec::new();
        let mut logical_row_ids = Vec::new();
        for (position, candidate) in candidates.iter().enumerate() {
            if let Some(candidate) = candidate {
                candidate_positions.push(position);
                logical_row_ids.push(candidate.raw());
            }
        }
        if logical_row_ids.is_empty() {
            return Ok(vec![None; addresses.len()]);
        }

        let current_owners = self.resolve_logical_row_ids_async(&logical_row_ids).await?;
        let mut current = vec![None; addresses.len()];
        for ((position, owner), logical_row_id) in candidate_positions
            .into_iter()
            .zip(current_owners)
            .zip(logical_row_ids)
        {
            if owner == Some(addresses[position]) {
                current[position] = Some(LogicalRowAddress::try_from(logical_row_id)?);
            }
        }
        Ok(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_page_verifier_rejects_same_size_middle_value_corruption() {
        let original = [10, 20, 30, 40];
        let expected = fingerprint_explicit_map_u64_page(&[&original]).unwrap();
        let corrupted = [10, 20, 31, 40];

        let error = verify_explicit_u64_page(
            "data/_row_addresses/row_ids.lance",
            &[&corrupted],
            &expected,
        )
        .unwrap_err();

        assert!(matches!(&error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("content fingerprint mismatch"));
    }
}
