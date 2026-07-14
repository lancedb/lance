// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{Array, UInt64Array};
use futures::TryStreamExt;
use lance::Dataset;
use lance::index::{
    build_logical_index_coverage, logical_index_coverage_is_current,
    logical_index_covers_all_current_slots, merge_logical_index_coverage,
    resolve_logical_index_metadata,
};
use lance_core::ROW_ID;
use lance_table::format::{IndexMetadata, LogicalIndexCoverage, LogicalRowAddressSelection};
use roaring::{RoaringBitmap, RoaringTreemap};

type EvidenceError = Box<dyn std::error::Error + Send + Sync>;

fn logical_coverage_selection(
    coverage: &LogicalIndexCoverage,
) -> Result<LogicalRowAddressSelection, EvidenceError> {
    let ranges = coverage
        .shards
        .iter()
        .enumerate()
        .map(|(shard_index, shard)| {
            shard
                .selection
                .as_ref()
                .ok_or_else(|| -> EvidenceError {
                    format!(
                        "logical index coverage shard {shard_index} is missing resolved selection detail"
                    )
                    .into()
                })?
                .to_ranges()
                .map_err(|error| -> EvidenceError { error.into() })
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect();
    Ok(LogicalRowAddressSelection::from_ranges(ranges)?)
}

pub fn covered_live_rows(
    live: &RoaringTreemap,
    ownership: &RoaringTreemap,
    effective: &RoaringTreemap,
) -> Result<u64, String> {
    if !live.is_subset(ownership) {
        return Err(format!(
            "logical index ownership is missing {} live rows",
            live.difference_len(ownership)
        ));
    }
    Ok(live.intersection_len(effective))
}

pub async fn effective_logical_index_covered_rows(
    dataset: &Dataset,
    metadata: &[IndexMetadata],
    live_rows: u64,
) -> Result<u64, EvidenceError> {
    let fields = metadata
        .first()
        .ok_or("benchmark logical index has no catalog segments")?
        .fields
        .clone();
    if metadata.iter().any(|segment| segment.fields != fields) {
        return Err("benchmark logical index has inconsistent field coverage".into());
    }
    let mut current = Vec::with_capacity(metadata.len());
    for segment in metadata {
        if logical_index_coverage_is_current(dataset, segment)? {
            current.push(segment.clone());
        }
    }
    let current_refs = current.iter().collect::<Vec<_>>();
    if logical_index_covers_all_current_slots(dataset, &current_refs)? {
        return Ok(live_rows);
    }

    let physical_fragments = dataset
        .manifest()
        .fragments
        .iter()
        .map(|fragment| {
            u32::try_from(fragment.id).map_err(|_| -> EvidenceError {
                format!(
                    "physical fragment {} exceeds the coverage address space",
                    fragment.id
                )
                .into()
            })
        })
        .collect::<Result<RoaringBitmap, _>>()?;
    let ownership = logical_coverage_selection(
        &build_logical_index_coverage(dataset, &fields, &physical_fragments).await?,
    )?
    .to_roaring_treemap()?;
    let mut live = RoaringTreemap::new();
    let mut scanner = dataset.scan();
    scanner.project(&[ROW_ID])?;
    let mut stream = scanner.try_into_stream().await?;
    while let Some(batch) = stream.try_next().await? {
        let row_ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or("logical coverage evidence did not return UInt64 _rowid values")?;
        if row_ids.null_count() != 0 {
            return Err("logical coverage evidence returned null _rowid values".into());
        }
        live.extend(row_ids.values().iter().copied());
    }
    if live.len() != live_rows {
        return Err(format!(
            "logical liveness evidence contains {} distinct rows for a {live_rows}-row dataset",
            live.len()
        )
        .into());
    }
    let resolved = resolve_logical_index_metadata(dataset, &current).await?;
    let resolved_refs = resolved.iter().collect::<Vec<_>>();
    let effective =
        logical_coverage_selection(&merge_logical_index_coverage(dataset, &resolved_refs)?)?
            .to_roaring_treemap()?;
    covered_live_rows(&live, &ownership, &effective).map_err(Into::into)
}
