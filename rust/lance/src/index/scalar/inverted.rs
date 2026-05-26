// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#![allow(clippy::redundant_pub_crate)]

use std::sync::Arc;

use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use lance_core::ROW_ID;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::pbold::InvertedIndexDetails;
use lance_index::scalar::inverted::InvertedIndex;
use lance_index::scalar::registry::VALUE_COLUMN_NAME;
use lance_index::{IndexType, scalar::lance_format::LanceIndexStore};
use lance_table::format::IndexMetadata;
use prost::Message;
use roaring::RoaringBitmap;
use uuid::Uuid;

use crate::{
    Dataset, Error, Result,
    dataset::index::LanceIndexStoreExt,
    index::{
        DatasetIndexExt,
        api::{IndexSegment, IndexSegmentPlan},
        scalar::fetch_index_details,
    },
};

/// Build an empty update stream for the inverted merge API.
///
/// `InvertedIndex::merge_segments` is shaped as "merge old segments plus new
/// rows", so even a pure segment merge needs a stream with the document column
/// and `_rowid` fields. The stream intentionally contains no batches.
fn empty_inverted_update_stream(
    dataset: &Dataset,
    field_id: i32,
) -> Result<SendableRecordBatchStream> {
    let field = dataset.schema().field_by_id(field_id).ok_or_else(|| {
        Error::invalid_input(format!(
            "merge_existing_index_segments: field id {} does not exist",
            field_id
        ))
    })?;
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new(VALUE_COLUMN_NAME, field.data_type(), true),
        ArrowField::new(ROW_ID, arrow_schema::DataType::UInt64, false),
    ]));
    Ok(Box::pin(RecordBatchStreamAdapter::new(
        schema,
        futures::stream::empty(),
    )))
}

async fn finalize_segment_files_if_needed(
    dataset: &Dataset,
    segment: &IndexMetadata,
) -> Result<()> {
    let index_dir = dataset.indices_dir().join(segment.uuid.to_string());
    let metadata_path = index_dir
        .clone()
        .join(lance_index::scalar::inverted::METADATA_FILE);
    if dataset.object_store.as_ref().exists(&metadata_path).await? {
        return Ok(());
    }

    let store = Arc::new(LanceIndexStore::from_dataset_for_new(
        dataset,
        &segment.uuid.to_string(),
    )?);
    lance_index::scalar::inverted::builder::merge_index_files(
        dataset.object_store.as_ref(),
        &index_dir,
        store,
        lance_index::progress::noop_progress(),
    )
    .await
}

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

    finalize_segment_files_if_needed(dataset, source_segment).await?;
    Ok(built_segment)
}

/// Merge one caller-defined group of source FTS segments into a single segment.
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
        finalize_segment_files_if_needed(dataset, segment).await?;
        fragment_bitmap |= segment.fragment_bitmap.as_ref().cloned().ok_or_else(|| {
            Error::invalid_input(format!(
                "CreateIndex: segment {} is missing fragment coverage",
                segment.uuid
            ))
        })?;
        let scalar_index =
            super::open_scalar_index(dataset, &field_path, segment, &NoOpMetricsCollector).await?;
        let inverted_index = scalar_index
            .as_any()
            .downcast_ref::<InvertedIndex>()
            .ok_or_else(|| {
                Error::index(format!(
                    "merge_existing_index_segments: expected inverted segment {}, got {:?}",
                    segment.uuid,
                    scalar_index.index_type()
                ))
            })?;
        source_indices.push(Arc::new(inverted_index.clone()));
    }

    let new_uuid = Uuid::new_v4();
    let new_store = LanceIndexStore::from_dataset_for_new(dataset, &new_uuid.to_string())?;
    let created_index = InvertedIndex::merge_segments(
        &source_indices,
        empty_inverted_update_stream(dataset, field_id)?,
        &new_store,
        None,
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

/// Load all committed inverted-index segments that belong to the same named
/// FTS index on `column`.
///
/// Returns `Ok(None)` if no FTS index exists on the column. When an index
/// exists, the returned vector contains every committed segment's
/// [`IndexMetadata`] (UUID, fragment coverage, index details). All segments
/// must share the same indexed fields; mismatched fields return an error.
pub async fn load_segments(dataset: &Dataset, column: &str) -> Result<Option<Vec<IndexMetadata>>> {
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

/// Load and validate the shared [`InvertedIndexDetails`] across committed
/// segments returned by [`load_segments`].
///
/// All segments are required to agree on their decoded `InvertedIndexDetails`
/// payload (analyzer, tokenizer, position settings, etc.); inconsistent
/// segments return an error. Returns the canonical details that may be used
/// when constructing a tokenizer or running a query against the index.
pub async fn load_segment_details(
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
