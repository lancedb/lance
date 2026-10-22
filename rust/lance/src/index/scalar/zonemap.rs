// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::DataType;
use itertools::Itertools;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::zonemap::{
    HarvestedZoneSeed, NULL_BITMAP_POLICY_META_KEY, ReusedZones, SeedOutcome,
    ZONEMAP_INDEX_FILENAME, ZoneMapIndex, ZoneMapIndexBuilderParams, ZoneMapSegmentBase,
    build_zonemap_segment,
};
use lance_index::scalar::{
    CreatedIndex, IndexStore, ScalarIndex, TrainingCriteria, index_files_to_table,
};
use lance_select::RowAddrTreeMap;
use lance_table::format::{Fragment, IndexMetadata};
use roaring::RoaringBitmap;
use uuid::Uuid;

use crate::index::append::SeedHarvester;
use crate::index::scalar::load_training_data;
use crate::{Dataset, Error, Result, dataset::index::LanceIndexStoreExt};

/// Merge one caller-defined group of source ZoneMap segments into a single segment.
pub(in crate::index) async fn merge_segments(
    dataset: &Dataset,
    segments: Vec<IndexMetadata>,
    staged: Option<&crate::index::frag_reuse::StagedRemappingPlans>,
) -> Result<IndexMetadata> {
    if segments.is_empty() {
        return Err(Error::index("No segment metadata was provided".to_string()));
    }

    let field_id = *segments[0].fields.first().ok_or_else(|| {
        Error::invalid_input(format!(
            "CreateIndex: segment {} is missing field ids",
            segments[0].uuid
        ))
    })?;
    let field_path = dataset.schema().field_path(field_id)?;

    let mut scalar_indices = Vec::with_capacity(segments.len());
    let mut fragment_bitmap = RoaringBitmap::new();
    let dataset_fragments = dataset.fragment_bitmap.as_ref();
    for segment in &segments {
        let effective = segment
            .effective_fragment_bitmap(dataset_fragments)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "CreateIndex: segment {} is missing fragment coverage",
                    segment.uuid
                ))
            })?;
        fragment_bitmap |= effective;
        let scalar_index = super::open_scalar_index_with_plan(
            dataset,
            &field_path,
            segment,
            staged.and_then(|plans| plans.get(&segment.uuid)),
            crate::index::frag_reuse::OpenPurpose::Maintenance,
            &NoOpMetricsCollector,
        )
        .await?;
        scalar_indices.push((segment.uuid, scalar_index));
    }

    let mut source_indices = Vec::with_capacity(scalar_indices.len());
    for (segment_uuid, scalar_index) in &scalar_indices {
        let zonemap_index = scalar_index
            .as_any()
            .downcast_ref::<ZoneMapIndex>()
            .ok_or_else(|| {
                Error::index(format!(
                    "merge_existing_index_segments: expected zonemap segment {}, got {:?}",
                    segment_uuid,
                    scalar_index.index_type()
                ))
            })?;
        source_indices.push(zonemap_index);
    }

    let new_uuid = Uuid::new_v4();
    let new_store = LanceIndexStore::from_dataset_for_new(dataset, &new_uuid)?;
    let created_index = lance_index::scalar::zonemap::merge_zonemap_indices(
        &source_indices,
        &new_store,
        &fragment_bitmap,
    )
    .await?;

    Ok(IndexMetadata {
        uuid: new_uuid,
        dataset_version: dataset.manifest.version,
        fragment_bitmap: Some(fragment_bitmap),
        index_details: Some(Arc::new(created_index.index_details)),
        index_version: created_index.index_version as i32,
        created_at: Some(chrono::Utc::now()),
        base_id: None,
        files: Some(index_files_to_table(created_index.files)),
        ..segments[0].clone()
    })
}

/// What a seed-aware segment build reused and scanned, recorded for tests.
#[cfg(test)]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SeedReuseReport {
    /// Fragments whose every zone came from whole seed records.
    pub reused_fragments: Vec<u64>,
    /// Fragments with some zones from seeds and scanned intervals.
    pub partial_fragments: Vec<u64>,
    pub reused_records: usize,
    pub zones_from_seeds: usize,
    /// Fragments scanned whole.
    pub scanned_fragments: Vec<u64>,
    /// Physical rows selected for the scan (whole fragments plus intervals).
    pub scanned_rows: u64,
    /// Most raw seed bytes owned by the build at any one moment (live
    /// ownership, decremented when a fragment's buffer is dropped).
    pub max_live_seed_bytes: usize,
    /// Raw seed bytes harvested over the whole build.
    pub total_seed_bytes: usize,
    /// Whether a physical row mask was given to the scan.
    pub masked: bool,
}

#[cfg(test)]
tokio::task_local! {
    /// Collects one [`SeedReuseReport`] per seed-aware segment build.
    pub static SEED_REUSE_REPORTS: std::sync::Mutex<Vec<SeedReuseReport>>;
}

/// The physical row count of `fragment`: the manifest's, else the serving
/// file's footer. Established before any seed decision so every zone's
/// length is known.
async fn physical_rows(dataset: &Dataset, fragment: &Fragment) -> Result<u64> {
    if let Some(rows) = fragment.physical_rows {
        return Ok(rows as u64);
    }
    let file_fragment = dataset.get_fragment(fragment.id as usize).ok_or_else(|| {
        Error::invalid_input(format!("fragment {} is not in the dataset", fragment.id))
    })?;
    Ok(file_fragment.physical_rows().await? as u64)
}

/// Harvest, fold and partition the seeds of `fragments` one fragment at a
/// time, scan the rows the seeds did not cover, and write one segment.
///
/// Partition per fragment: whole seed records inside a target zone are
/// folded; a fragment with crossing records contributes its uncovered
/// intervals to the scan mask, or is scanned whole when partial reuse is
/// not allowed (a deletion file, so physical and live populations are not
/// mixed in one zone; or V1 storage, which cannot take a physical mask).
/// The raw seed buffer of a fragment is dropped before the next is read.
#[allow(clippy::too_many_arguments)]
async fn assemble_segment(
    dataset: &Dataset,
    column: &str,
    base: ZoneMapSegmentBase<'_>,
    data_type: &DataType,
    fragments: &[Fragment],
    harvester: Option<SeedHarvester<'_>>,
    criteria: &TrainingCriteria,
    store: &dyn IndexStore,
) -> Result<CreatedIndex> {
    let rows_per_zone = base.rows_per_zone();
    let legacy_storage = dataset.manifest.data_storage_format.lance_file_format()
        == lance_file::version::ConcreteFileVersion::V1;
    let mut reused = ReusedZones::new(base.keeps_null_positions());
    let mut rows_by_fragment: HashMap<u64, u64> = HashMap::new();
    let mut to_scan: Vec<Fragment> = Vec::new();
    let mut mask = RowAddrTreeMap::new();
    let mut scanned_rows = 0u64;
    let (mut live_seed_bytes, mut max_live_seed_bytes, mut total_seed_bytes) =
        (0usize, 0usize, 0usize);
    let mut harvester = harvester;
    for fragment in fragments {
        let rows = physical_rows(dataset, fragment).await?;
        rows_by_fragment.insert(fragment.id, rows);
        let allow_partial = !legacy_storage && fragment.deletion_file.is_none();
        let outcome = match harvester.as_mut() {
            Some(harvester) => match harvester.harvest(fragment).await? {
                Some(seed) => {
                    let harvested = HarvestedZoneSeed {
                        seed,
                        physical_rows: rows,
                    };
                    let bytes = harvested.seed.bytes.len();
                    live_seed_bytes += bytes;
                    max_live_seed_bytes = max_live_seed_bytes.max(live_seed_bytes);
                    total_seed_bytes += bytes;
                    let outcome =
                        reused.reuse(&harvested, rows_per_zone, data_type, allow_partial)?;
                    // The raw buffer does not outlive its fragment's fold. The
                    // explicit drop keeps the live count honest: moving
                    // `harvested` anywhere else would not compile past it.
                    drop(harvested);
                    live_seed_bytes -= bytes;
                    outcome
                }
                None => SeedOutcome::Rejected,
            },
            None => SeedOutcome::Rejected,
        };
        let base_addr = fragment.id << 32;
        match outcome {
            SeedOutcome::Complete => {}
            SeedOutcome::Partial(intervals) => {
                to_scan.push(fragment.clone());
                for interval in intervals {
                    scanned_rows += interval.end - interval.start;
                    mask.insert_range(base_addr + interval.start..base_addr + interval.end);
                }
            }
            SeedOutcome::Rejected => {
                to_scan.push(fragment.clone());
                scanned_rows += rows;
                mask.insert_range(base_addr..base_addr + rows);
            }
        }
    }
    // A legacy-storage scanner refuses any physical mask; there every
    // fragment to scan is scanned whole (partial reuse was not allowed), so
    // selecting the fragments is the whole selection.
    let masked = !legacy_storage && !to_scan.is_empty();
    log::info!(
        "zone map segment for {column}: {} fragments from seeds, {} partly, {} scanned ({scanned_rows} rows, {} seed records, {total_seed_bytes} seed bytes, at most {max_live_seed_bytes} held at once)",
        reused.reused_fragments.len(),
        reused.partial_fragments.len(),
        to_scan.len(),
        reused.reused_records
    );
    #[cfg(test)]
    let _ = SEED_REUSE_REPORTS.try_with(|reports| {
        reports.lock().unwrap().push(SeedReuseReport {
            reused_fragments: reused.reused_fragments.clone(),
            partial_fragments: reused.partial_fragments.clone(),
            reused_records: reused.reused_records,
            zones_from_seeds: reused.num_zones(),
            scanned_fragments: to_scan.iter().map(|fragment| fragment.id).collect(),
            scanned_rows,
            max_live_seed_bytes,
            total_seed_bytes,
            masked,
        })
    });
    let scan = if to_scan.is_empty() {
        None
    } else {
        Some(
            load_training_data(
                dataset,
                column,
                criteria,
                Some(to_scan),
                true,
                None,
                masked.then_some(mask),
            )
            .await?,
        )
    };
    let refreshed: Vec<u64> = fragments.iter().map(|fragment| fragment.id).collect();
    build_zonemap_segment(
        base,
        data_type,
        reused,
        scan,
        &rows_by_fragment,
        &refreshed,
        store,
    )
    .await
}

/// The fragments with these ids, in id order, erroring on an unknown id.
fn fragments_by_id(dataset: &Dataset, ids: Vec<u32>) -> Result<Vec<Fragment>> {
    let ids: Vec<u32> = ids.into_iter().sorted().dedup().collect();
    ids.iter()
        .zip(dataset.get_frags_from_ordered_ids(&ids))
        .map(|(id, fragment)| {
            fragment.map(|f| f.metadata().clone()).ok_or_else(|| {
                Error::invalid_input_source(format!("No fragment with id {}", id).into())
            })
        })
        .collect()
}

/// Train a new ZoneMap segment over `fragment_ids` (all fragments when
/// `None`), reusing the write seeds those fragments already carry and
/// scanning only the rows they do not cover.
pub(in crate::index) async fn build_segment_with_seeds(
    dataset: &Dataset,
    column: &str,
    params: ZoneMapIndexBuilderParams,
    data_type: &DataType,
    fragment_ids: Option<Vec<u32>>,
    criteria: &TrainingCriteria,
    store: &dyn IndexStore,
) -> Result<CreatedIndex> {
    let fragments = match fragment_ids {
        Some(ids) => fragments_by_id(dataset, ids)?,
        None => dataset.fragments().as_ref().clone(),
    };
    assemble_segment(
        dataset,
        column,
        ZoneMapSegmentBase::New(params),
        data_type,
        &fragments,
        SeedHarvester::for_column(dataset, column),
        criteria,
        store,
    )
    .await
}

/// Append `unindexed` fragments to the ZoneMap segment `reference_index`,
/// reusing their write seeds where usable when `harvest` is set. `None`
/// when the reference is not a ZoneMap segment, in which case the generic
/// update path applies.
#[allow(clippy::too_many_arguments)]
pub(in crate::index) async fn update_segment(
    dataset: &Dataset,
    reference_index: &Arc<dyn ScalarIndex>,
    reference: &IndexMetadata,
    column: &str,
    unindexed: &[Fragment],
    criteria: &TrainingCriteria,
    harvest: bool,
    store: &dyn IndexStore,
) -> Result<Option<CreatedIndex>> {
    let Some(zone_map) = reference_index.as_any().downcast_ref::<ZoneMapIndex>() else {
        return Ok(None);
    };
    let harvester = if harvest {
        reference
            .keyed_field()
            .and_then(|field_id| SeedHarvester::for_field_id(dataset, field_id))
    } else {
        None
    };
    let data_type = zone_map.value_type().clone();
    assemble_segment(
        dataset,
        column,
        ZoneMapSegmentBase::Update(zone_map),
        &data_type,
        unindexed,
        harvester,
        criteria,
        store,
    )
    .await
    .map(Some)
}

/// The persisted null-position policy of a ZoneMap segment, read from its
/// index file's schema metadata (one footer read, served from the file
/// metadata cache afterwards). `true` unless the index was created with
/// `store_null_bitmap: false`; an index without the key (legacy or default)
/// keeps the default.
pub async fn stored_null_bitmap_policy(dataset: &Dataset, index: &IndexMetadata) -> Result<bool> {
    let store = LanceIndexStore::from_dataset_for_existing(dataset, index).await?;
    let file = store.open_index_file(ZONEMAP_INDEX_FILENAME).await?;
    Ok(file
        .schema()
        .metadata
        .get(NULL_BITMAP_POLICY_META_KEY)
        .is_none_or(|value| value != "false"))
}
