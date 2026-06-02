// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_index::IndexType;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::bitmap::BitmapIndex;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_table::format::IndexMetadata;
use roaring::RoaringBitmap;
use std::sync::Arc;
use uuid::Uuid;

use crate::{
    Dataset, Error, Result,
    dataset::index::LanceIndexStoreExt,
    index::api::{IndexSegment, IndexSegmentPlan},
};

/// Plan physical segments for staged bitmap-index outputs.
///
/// Each staged bitmap root is already a complete canonical physical segment,
/// so planning preserves one output segment for each staged source segment.
pub(in crate::index) fn plan_segments(
    segments: &[IndexMetadata],
    target_segment_bytes: Option<u64>,
) -> Result<Vec<IndexSegmentPlan>> {
    if let Some(0) = target_segment_bytes {
        return Err(Error::invalid_input(
            "target_segment_bytes must be greater than zero".to_string(),
        ));
    }
    if target_segment_bytes.is_some() && segments.len() > 1 {
        return Err(Error::invalid_input(
            "Bitmap segment builder does not yet support merging multiple source segments"
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
            if !index_details.type_url.ends_with("BitmapIndexDetails") {
                return Err(Error::index(format!(
                    "Segment '{}' is not a bitmap index segment",
                    segment.uuid
                )));
            }

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
                Some(IndexType::Bitmap),
            ))
        })
        .collect()
}

/// Finalize one staged bitmap root into a commit-ready physical segment.
pub(in crate::index) async fn build_segment(
    segment_plan: &IndexSegmentPlan,
) -> Result<IndexSegment> {
    let built_segment = segment_plan.segment().clone();
    let source_segments = segment_plan.segments();
    if source_segments.len() != 1 {
        return Err(Error::invalid_input(
            "Bitmap segment builder does not yet support merging multiple source segments"
                .to_string(),
        ));
    }
    let source_segment = &source_segments[0];
    if source_segment.uuid != built_segment.uuid() {
        return Err(Error::invalid_input(
            "Bitmap segment builder requires the built segment UUID to match the staged source UUID"
                .to_string(),
        ));
    }

    Ok(built_segment)
}

/// Merge one caller-defined group of source bitmap segments into a single segment.
pub(crate) async fn merge_segments(
    dataset: &Dataset,
    segments: Vec<IndexMetadata>,
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

    let mut source_indices = Vec::with_capacity(segments.len());
    let mut fragment_bitmap = RoaringBitmap::new();
    for segment in &segments {
        fragment_bitmap |= segment.fragment_bitmap.as_ref().cloned().ok_or_else(|| {
            Error::invalid_input(format!(
                "CreateIndex: segment {} is missing fragment coverage",
                segment.uuid
            ))
        })?;
        let scalar_index =
            super::open_scalar_index(dataset, &field_path, segment, &NoOpMetricsCollector).await?;
        let bitmap_index = scalar_index
            .as_any()
            .downcast_ref::<BitmapIndex>()
            .ok_or_else(|| {
                Error::index(format!(
                    "merge_existing_index_segments: expected bitmap segment {}, got {:?}",
                    segment.uuid,
                    scalar_index.index_type()
                ))
            })?;
        source_indices.push(Arc::new(bitmap_index.clone()));
    }

    let new_uuid = Uuid::new_v4();
    let new_store = LanceIndexStore::from_dataset_for_new(dataset, &new_uuid.to_string())?;
    let created_index = lance_index::scalar::bitmap::merge_bitmap_indices(
        &source_indices,
        &new_store,
        lance_index::progress::noop_progress(),
    )
    .await?;

    Ok(IndexMetadata {
        uuid: new_uuid,
        fields: vec![field_id],
        dataset_version: dataset.manifest.version,
        fragment_bitmap: Some(fragment_bitmap),
        index_details: Some(Arc::new(created_index.index_details)),
        index_version: created_index.index_version as i32,
        created_at: Some(chrono::Utc::now()),
        base_id: None,
        files: created_index.files,
        ..segments[0].clone()
    })
}
