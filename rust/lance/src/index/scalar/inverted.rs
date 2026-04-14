// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#![allow(clippy::redundant_pub_crate)]

use std::sync::Arc;

use lance_index::pbold::InvertedIndexDetails;
use lance_index::{IndexType, scalar::lance_format::LanceIndexStore};
use lance_table::format::IndexMetadata;
use prost::Message;

use crate::{
    Dataset, Error, Result,
    dataset::index::LanceIndexStoreExt,
    index::{
        DatasetIndexExt,
        api::{IndexSegment, IndexSegmentPlan},
        scalar::fetch_index_details,
    },
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
        lance_index::progress::noop_progress(),
    )
    .await?;
    Ok(built_segment)
}

/// Load all committed inverted-index segments that belong to the same named index.
pub(crate) async fn load_segments(
    dataset: &Dataset,
    column: &str,
) -> Result<Option<Vec<IndexMetadata>>> {
    let Some(index_meta) = dataset
        .load_scalar_index(
            lance_index::IndexCriteria::default()
                .for_column(column)
                .supports_fts(),
        )
        .await?
    else {
        return Ok(None);
    };

    let indices = dataset.load_indices_by_name(&index_meta.name).await?;
    if indices.is_empty() {
        return Ok(None);
    }

    let expected_fields = indices[0].fields.clone();
    for meta in &indices {
        if meta.fields != expected_fields {
            return Err(Error::invalid_input(format!(
                "FTS index {} has inconsistent fields across segments",
                index_meta.name
            )));
        }
    }

    Ok(Some(indices))
}

/// Load and validate the shared inverted-index details across committed segments.
pub(crate) async fn load_segment_details(
    dataset: &Dataset,
    column: &str,
    segments: &[IndexMetadata],
) -> Result<InvertedIndexDetails> {
    let mut expected_details: Option<InvertedIndexDetails> = None;
    for meta in segments {
        let details_any = fetch_index_details(dataset, column, meta).await?;
        let details =
            InvertedIndexDetails::decode(details_any.value.as_slice()).map_err(|err| {
                Error::io(format!(
                    "failed to decode InvertedIndexDetails payload: {err}"
                ))
            })?;
        match &expected_details {
            Some(expected) if expected != &details => {
                return Err(Error::invalid_input(format!(
                    "FTS index {} has inconsistent inverted index details across segments",
                    meta.name
                )));
            }
            Some(_) => {}
            None => expected_details = Some(details),
        }
    }
    expected_details.ok_or_else(|| {
        Error::invalid_input(format!(
            "FTS index for column {} requires at least one segment",
            column
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_legacy_inverted_details_type_url() {
        let mut details_any = prost_types::Any::from_msg(&InvertedIndexDetails::default()).unwrap();
        details_any.type_url = "/lance.index.pb.InvertedIndexDetails".to_string();

        let decoded = InvertedIndexDetails::decode(details_any.value.as_slice()).unwrap();
        assert_eq!(decoded, InvertedIndexDetails::default());
    }
}
