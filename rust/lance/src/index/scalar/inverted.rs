// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#![allow(clippy::redundant_pub_crate)]

use std::sync::Arc;

use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use lance_core::{
    ROW_ID,
    datatypes::{
        Field, FieldPathComponent, LogicalType, Schema, format_field_path,
        parse_field_path_components,
    },
};
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::pbold::InvertedIndexDetails;
use lance_index::scalar::inverted::{
    FtsTarget, InvertedIndex, LEGACY_BLOCK_SIZE, default_fts_format_version_for_block_size,
};
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::registry::VALUE_COLUMN_NAME;
use lance_table::format::IndexMetadata;
use prost::Message;
use roaring::RoaringBitmap;
use uuid::Uuid;

use crate::{
    Dataset, Error, Result,
    dataset::index::LanceIndexStoreExt,
    index::{DatasetIndexExt, scalar::fetch_index_details},
};

/// Schema-resolved form of a public FTS target path.
#[derive(Debug, Clone)]
pub(crate) struct ResolvedFtsTarget {
    pub target: FtsTarget,
    pub public_field_id: i32,
    pub scan_column: String,
    pub canonical_path: String,
}

fn text_source_field(field: &Field) -> Result<&Field> {
    if field.logical_type == LogicalType::from("json") {
        return Ok(field);
    }
    match field.data_type() {
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => Ok(field),
        DataType::List(_) | DataType::LargeList(_) => {
            let child = field.children.first().ok_or_else(|| {
                Error::invalid_input(format!(
                    "FTS list field '{}' does not have an item field",
                    field.name
                ))
            })?;
            match child.data_type() {
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => Ok(child),
                data_type => Err(Error::invalid_input(format!(
                    "FTS list field '{}' must contain Utf8 or LargeUtf8 values, got {data_type}",
                    field.name
                ))),
            }
        }
        // JSON fields are represented as extension-annotated string fields, so
        // their physical type is covered above.
        data_type => Err(Error::invalid_input(format!(
            "FTS field '{}' must be Utf8, LargeUtf8, JSON, or a list of strings, got {data_type}",
            field.name
        ))),
    }
}

/// Resolve the public FTS path into a stable schema-id target.
pub(crate) fn resolve_fts_target(schema: &Schema, path: &str) -> Result<ResolvedFtsTarget> {
    let components = parse_field_path_components(path)?;
    let wildcard_count = components
        .iter()
        .filter(|component| matches!(component, FieldPathComponent::ListWildcard))
        .count();

    if wildcard_count == 0 {
        let fields = schema.resolve_case_insensitive(path).ok_or_else(|| {
            Error::index(format!(
                "FTS target '{path}' does not exist in the dataset schema"
            ))
        })?;
        if fields
            .iter()
            .take(fields.len().saturating_sub(1))
            .any(|field| {
                matches!(
                    field.data_type(),
                    DataType::List(_) | DataType::LargeList(_)
                )
            })
        {
            return Err(Error::invalid_input(format!(
                "FTS target '{path}' contains an Arrow-internal list item path; use the public list field for row-document FTS or append [*] for element-document FTS"
            )));
        }
        let field = fields.last().copied().ok_or_else(|| {
            Error::index(format!("FTS target '{path}' does not resolve to a field"))
        })?;
        let source = text_source_field(field)?;
        let names = fields
            .iter()
            .map(|field| field.name.as_str())
            .collect::<Vec<_>>();
        let canonical_path = format_field_path(&names);
        return Ok(ResolvedFtsTarget {
            target: FtsTarget::new(source.id, Vec::new()),
            public_field_id: field.id,
            scan_column: canonical_path.clone(),
            canonical_path,
        });
    }

    if wildcard_count != 1
        || components.len() != 2
        || !matches!(components[0], FieldPathComponent::Field(_))
        || !matches!(components[1], FieldPathComponent::ListWildcard)
    {
        return Err(Error::invalid_input(format!(
            "FTS element target '{path}' is outside the supported Phase 1 shape: expected one top-level List<Utf8> or List<LargeUtf8> field followed by exactly one [*]"
        )));
    }

    let FieldPathComponent::Field(requested_name) = &components[0] else {
        return Err(Error::invalid_input(format!(
            "FTS element target '{path}' must begin with a field name"
        )));
    };
    let field = schema
        .fields
        .iter()
        .find(|field| field.name == *requested_name)
        .or_else(|| {
            schema
                .fields
                .iter()
                .find(|field| field.name.eq_ignore_ascii_case(requested_name))
        })
        .ok_or_else(|| {
            Error::index(format!(
                "FTS target '{path}' does not exist in the dataset schema"
            ))
        })?;
    if !matches!(
        field.data_type(),
        DataType::List(_) | DataType::LargeList(_)
    ) {
        return Err(Error::invalid_input(format!(
            "FTS element target '{path}' requires a List<Utf8> or List<LargeUtf8> field, got {}",
            field.data_type()
        )));
    }
    let source = field.children.first().ok_or_else(|| {
        Error::invalid_input(format!(
            "FTS element target '{path}' does not have an item field"
        ))
    })?;
    if !matches!(source.data_type(), DataType::Utf8 | DataType::LargeUtf8) {
        return Err(Error::invalid_input(format!(
            "FTS element target '{path}' must contain Utf8 or LargeUtf8 values, got {}",
            source.data_type()
        )));
    }
    let scan_column = format_field_path(&[field.name.as_str()]);
    Ok(ResolvedFtsTarget {
        target: FtsTarget::new(source.id, vec![field.id]),
        public_field_id: field.id,
        canonical_path: format!("{scan_column}[*]"),
        scan_column,
    })
}

/// Normalize legacy, target-less inverted metadata to its row-document target.
pub(crate) fn normalize_fts_details(
    schema: &Schema,
    index: &IndexMetadata,
    mut details: InvertedIndexDetails,
) -> Result<InvertedIndexDetails> {
    if details.fts_target.is_none() {
        let field_id = *index.fields.last().ok_or_else(|| {
            Error::invalid_input(format!(
                "FTS index {} does not record a field id",
                index.name
            ))
        })?;
        let field = schema.field_by_id(field_id).ok_or_else(|| {
            Error::invalid_input(format!(
                "FTS index {} refers to missing field id {field_id}",
                index.name
            ))
        })?;
        let source = text_source_field(field)?;
        details.fts_target = Some((&FtsTarget::new(source.id, Vec::new())).into());
    }
    if details.posting_format_version.is_none() {
        let posting_format_version = match index.index_version {
            0 | 1 => 1,
            2 => 2,
            3 => 3,
            4 => default_fts_format_version_for_block_size(
                details.block_size.unwrap_or(LEGACY_BLOCK_SIZE as u32) as usize,
            )?
            .index_version(),
            version => {
                return Err(Error::invalid_input(format!(
                    "FTS index {} has unsupported index version {version}",
                    index.name
                )));
            }
        };
        details.posting_format_version = Some(posting_format_version);
    }
    Ok(details)
}

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

pub(crate) async fn finalize_segment_files_if_needed(
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
        &segment.uuid,
    )?);
    lance_index::scalar::inverted::builder::merge_index_files(
        dataset.object_store.as_ref(),
        &index_dir,
        store,
        lance_index::progress::noop_progress(),
    )
    .await
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
    let new_store = LanceIndexStore::from_dataset_for_new(dataset, &new_uuid)?;
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
        files: Some(created_index.files),
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

    fn fts_test_schema() -> Schema {
        let schema = ArrowSchema::new(vec![
            ArrowField::new("text", DataType::Utf8, true),
            ArrowField::new(
                "tags",
                DataType::List(Arc::new(ArrowField::new("item", DataType::Utf8, true))),
                true,
            ),
        ]);
        Schema::try_from(&schema).unwrap()
    }

    #[test]
    fn decode_legacy_inverted_details_type_url() {
        let mut details_any = prost_types::Any::from_msg(&InvertedIndexDetails::default()).unwrap();
        details_any.type_url = "/lance.index.pb.InvertedIndexDetails".to_string();

        let decoded = InvertedIndexDetails::decode(details_any.value.as_slice()).unwrap();
        assert_eq!(decoded, InvertedIndexDetails::default());
    }

    #[test]
    fn resolve_element_document_target() {
        let schema = fts_test_schema();
        let tags = schema.field("tags").unwrap();
        let target = resolve_fts_target(&schema, "tags[*]").unwrap();
        assert_eq!(target.public_field_id, tags.id);
        assert_eq!(target.scan_column, "tags");
        assert_eq!(target.canonical_path, "tags[*]");
        assert_eq!(target.target.source_field_id(), tags.children[0].id);
        assert_eq!(target.target.boundary_field_ids(), &[tags.id]);

        let err = resolve_fts_target(&schema, "text[*]").unwrap_err();
        assert!(err.to_string().contains("requires a List<Utf8>"), "{err}");
        let err = resolve_fts_target(&schema, "tags[*][*]").unwrap_err();
        assert!(err.to_string().contains("Phase 1 shape"), "{err}");
        let err = resolve_fts_target(&schema, "tags.item").unwrap_err();
        assert!(err.to_string().contains("Arrow-internal"), "{err}");
    }

    #[test]
    fn normalize_legacy_list_metadata_as_row_document() {
        let schema = fts_test_schema();
        let tags = schema.field("tags").unwrap();
        let metadata = IndexMetadata {
            uuid: Uuid::new_v4(),
            fields: vec![tags.id],
            name: "tags_idx".to_string(),
            dataset_version: 1,
            fragment_bitmap: None,
            index_details: None,
            index_version: 3,
            created_at: None,
            base_id: None,
            files: None,
        };
        let details =
            normalize_fts_details(&schema, &metadata, InvertedIndexDetails::default()).unwrap();
        let target = FtsTarget::from(details.fts_target.as_ref().unwrap());
        assert_eq!(target.source_field_id(), tags.children[0].id);
        assert!(target.boundary_field_ids().is_empty());
        assert_eq!(details.posting_format_version, Some(3));
    }
}
