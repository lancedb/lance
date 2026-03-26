// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use lance_index::{IndexType, scalar::lance_format::LanceIndexStore};
use lance_table::format::IndexMetadata;

use crate::{
    Dataset, Error, Result,
    dataset::index::LanceIndexStoreExt,
    index::{IndexSegment, IndexSegmentPlan},
};

/// Plan physical segments for staged inverted-index outputs.
///
/// Each staged inverted root remains its own physical segment for now.
pub(crate) fn plan_segments(
    segments: &[IndexMetadata],
    target_segment_bytes: Option<u64>,
) -> Result<Vec<IndexSegmentPlan>> {
    if let Some(0) = target_segment_bytes {
        return Err(Error::invalid_input(
            "target_segment_bytes must be greater than zero".to_string(),
        ));
    }
    if target_segment_bytes.is_some() && segments.len() > 1 {
        // TODO: Support merging multiple staged inverted roots into one segment.
        return Err(Error::invalid_input(
            "Inverted segment builder does not yet support merging multiple source segments"
                .to_string(),
        ));
    }

    segments
        .iter()
        .map(|segment| {
            let fragment_bitmap = segment.fragment_bitmap.as_ref().ok_or_else(|| {
                Error::index(format!(
                    "Segment '{}' is missing fragment coverage",
                    segment.uuid
                ))
            })?;
            let index_details = segment.index_details.as_ref().ok_or_else(|| {
                Error::index(format!(
                    "Segment '{}' is missing index details",
                    segment.uuid
                ))
            })?;
            let built_segment = IndexSegment::new(
                segment.uuid,
                fragment_bitmap.iter(),
                index_details.clone(),
                segment.index_version,
            );
            let estimated_bytes = segment
                .files
                .as_ref()
                .map(|files| files.iter().map(|file| file.size_bytes).sum())
                .unwrap_or(0);
            Ok(IndexSegmentPlan::new(
                built_segment,
                vec![segment.clone()],
                estimated_bytes,
                Some(IndexType::Inverted),
            ))
        })
        .collect()
}

/// Finalize one staged inverted root into a commit-ready physical segment.
pub(crate) async fn build_segment(
    dataset: &Dataset,
    segment_plan: &IndexSegmentPlan,
) -> Result<IndexSegment> {
    let built_segment = segment_plan.segment().clone();
    let source_segments = segment_plan.segments();
    if source_segments.len() != 1 {
        // TODO: Support building one segment from multiple staged inverted roots.
        return Err(Error::invalid_input(
            "Inverted segment builder does not yet support merging multiple source segments"
                .to_string(),
        ));
    }
    let source_segment = &source_segments[0];
    if source_segment.uuid != built_segment.uuid() {
        return Err(Error::invalid_input(
            "Inverted segment builder requires the built segment UUID to match the staged source UUID"
                .to_string(),
        ));
    }

    let index_dir = dataset.indices_dir().child(source_segment.uuid.to_string());
    let metadata_path = index_dir.child(lance_index::scalar::inverted::METADATA_FILE);
    if dataset.object_store().exists(&metadata_path).await? {
        return Ok(built_segment);
    }

    let store = Arc::new(LanceIndexStore::from_dataset_for_new(
        dataset,
        &source_segment.uuid.to_string(),
    )?);
    lance_index::scalar::inverted::builder::merge_index_files(
        dataset.object_store(),
        &index_dir,
        store,
    )
    .await?;
    Ok(built_segment)
}
