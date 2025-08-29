// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Utilities for integrating scalar indices with datasets
//!

use std::sync::Arc;

use crate::index::DatasetIndexInternalExt;
use crate::{
    dataset::{index::LanceIndexStoreExt, scanner::ColumnOrdering},
    Dataset,
};
use arrow_schema::DataType;
use async_trait::async_trait;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::TryStreamExt;
use itertools::Itertools;
use lance_core::datatypes::Field;
use lance_core::{Error, Result};
use lance_datafusion::{chunker::chunk_concat_stream, exec::LanceExecutionOptions};
use lance_index::metrics::MetricsCollector;
use lance_index::scalar::{
    btree::DEFAULT_BTREE_BATCH_SIZE, inverted::tokenizer::InvertedIndexParams,
};
use lance_index::scalar::{
    inverted::METADATA_FILE,
    ngram::{train_ngram_index, NGramIndex},
    zonemap::{train_zonemap_index, ZoneMapIndex},
};
use lance_index::ScalarIndexCriteria;
use lance_index::{
    scalar::{
        bitmap::{train_bitmap_index, BitmapIndex, BITMAP_LOOKUP_NAME},
        btree::{train_btree_index, BTreeIndex, TrainingSource},
        flat::FlatIndexMetadata,
        inverted::{train_inverted_index, InvertedIndex, INVERT_LIST_FILE},
        label_list::{train_label_list_index, LabelListIndex},
        lance_format::LanceIndexStore,
        ScalarIndex, ScalarIndexParams, ScalarIndexType,
    },
    IndexType,
};
use lance_table::format::Index;
use log::info;
use snafu::location;
use tracing::instrument;

// Log an update every TRAINING_UPDATE_FREQ million rows processed
const TRAINING_UPDATE_FREQ: usize = 1000000;

pub(crate) struct TrainingRequest {
    pub dataset: Arc<Dataset>,
    pub column: String,
    train: bool,
    pub fragment_ids: Option<Vec<u32>>,
}

#[async_trait]
impl TrainingSource for TrainingRequest {
    async fn scan_ordered_chunks(
        self: Box<Self>,
        chunk_size: u32,
    ) -> Result<SendableRecordBatchStream> {
        if !self.train {
            return self.create_empty_stream().await;
        }
        self.scan_chunks(chunk_size, OrderMode::Ordered).await
    }

    async fn scan_unordered_chunks(
        self: Box<Self>,
        chunk_size: u32,
    ) -> Result<SendableRecordBatchStream> {
        if !self.train {
            return self.create_empty_stream().await;
        }
        self.scan_chunks(chunk_size, OrderMode::Unordered).await
    }

    async fn scan_aligned_chunks(
        self: Box<Self>,
        chunk_size: u32,
    ) -> Result<SendableRecordBatchStream> {
        if !self.train {
            return self.create_empty_stream().await;
        }
        self.scan_chunks(chunk_size, OrderMode::Aligned).await
    }
}

enum OrderMode {
    Unordered,
    // ordered by the user specified column
    Ordered,
    // row_address column is added to the scan and it's order by row_address
    Aligned,
}

impl TrainingRequest {
    pub fn new(dataset: Arc<Dataset>, column: String, train: bool) -> Self {
        Self {
            dataset,
            column,
            train,
            fragment_ids: None,
        }
    }

    pub fn with_fragment_ids(
        dataset: Arc<Dataset>,
        column: String,
        fragment_ids: Vec<u32>,
    ) -> Self {
        Self {
            dataset,
            column,
            fragment_ids: Some(fragment_ids),
            train: true, // Default to true for training
        }
    }

    async fn create_empty_stream(&self) -> Result<SendableRecordBatchStream> {
        let column_field =
            self.dataset
                .schema()
                .field(&self.column)
                .ok_or(Error::InvalidInput {
                    source: format!("No column with name {}", self.column).into(),
                    location: location!(),
                })?;

        // Create schema with the column and row_id field (matching scan_chunks behavior)
        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new(&self.column, column_field.data_type(), true),
            arrow_schema::Field::new("_rowid", arrow_schema::DataType::UInt64, false),
        ]));

        // Create empty stream
        let empty_stream = futures::stream::empty();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            empty_stream,
        )))
    }

    async fn scan_chunks(
        self: Box<Self>,
        chunk_size: u32,
        order_mode: OrderMode,
    ) -> Result<SendableRecordBatchStream> {
        let num_rows = self.dataset.count_all_rows().await?;

        let mut scan = self.dataset.scan();

        if let Some(ref fragment_ids) = self.fragment_ids {
            let fragment_ids = fragment_ids.clone().into_iter().dedup().collect_vec();
            let frags = self.dataset.get_frags_from_ordered_ids(&fragment_ids);
            let frags: Result<Vec<_>> = fragment_ids
                .iter()
                .zip(frags)
                .map(|(id, frag)| {
                    let Some(frag) = frag else {
                        return Err(Error::InvalidInput {
                            source: format!("No fragment with id {}", id).into(),
                            location: location!(),
                        });
                    };
                    Ok(frag.metadata().clone())
                })
                .collect();
            scan.with_fragments(frags?);
        }

        let column_field =
            self.dataset
                .schema()
                .field(&self.column)
                .ok_or(Error::InvalidInput {
                    source: format!("No column with name {}", self.column).into(),
                    location: location!(),
                })?;

        // Datafusion currently has bugs with spilling on string columns
        // See https://github.com/apache/datafusion/issues/10073
        //
        // One we upgrade we can remove this
        let use_spilling = !matches!(
            column_field.data_type(),
            DataType::Utf8 | DataType::LargeUtf8
        );

        let scan = match order_mode {
            OrderMode::Aligned => {
                // Since Lance will return data in the order of the row_address, no need to sort.
                scan.with_row_id()
                    .with_row_address()
                    .project(&[&self.column])?
            }
            OrderMode::Ordered => scan
                .with_row_id()
                .order_by(Some(vec![ColumnOrdering::asc_nulls_first(
                    self.column.clone(),
                )]))?
                .project(&[&self.column])?,
            OrderMode::Unordered => scan.with_row_id().project(&[&self.column])?,
        };

        let batches = scan
            .try_into_dfstream(LanceExecutionOptions {
                use_spilling,
                ..Default::default()
            })
            .await?;
        let batches = chunk_concat_stream(batches, chunk_size as usize);

        let schema = batches.schema();
        let mut rows_processed = 0;
        let mut next_update = TRAINING_UPDATE_FREQ;
        let training_uuid = uuid::Uuid::new_v4().to_string();
        info!(
            "Starting index training job with id {} on column {}",
            training_uuid, self.column
        );
        info!("Training index (job_id={}): 0/{}", training_uuid, num_rows);
        let batches = batches.map_ok(move |batch| {
            rows_processed += batch.num_rows();
            if rows_processed >= next_update {
                next_update += TRAINING_UPDATE_FREQ;
                info!(
                    "Training index (job_id={}): {}/{}",
                    training_uuid, rows_processed, num_rows
                );
            }
            batch
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, batches)))
    }
}

/// A trait used by the planner to determine how it can use a scalar index
//
// This may go away at some point but the scanner is a weak spot if we want
// to make index types "generic" and "pluggable".  We will need to create some
// kind of core proto for scalar indices that the scanner can read to determine
// how and when to use a scalar index.
pub trait ScalarIndexDetails {
    fn get_type(&self) -> ScalarIndexType;
}

fn bitmap_index_details() -> prost_types::Any {
    let details = lance_table::format::pb::BitmapIndexDetails {};
    prost_types::Any::from_msg(&details).unwrap()
}

fn btree_index_details() -> prost_types::Any {
    let details = lance_table::format::pb::BTreeIndexDetails {};
    prost_types::Any::from_msg(&details).unwrap()
}

fn label_list_index_details() -> prost_types::Any {
    let details = lance_table::format::pb::LabelListIndexDetails {};
    prost_types::Any::from_msg(&details).unwrap()
}

fn ngram_index_details() -> prost_types::Any {
    let details = lance_table::format::pb::NGramIndexDetails {};
    prost_types::Any::from_msg(&details).unwrap()
}

fn zonemap_index_details() -> prost_types::Any {
    let details = lance_table::format::pb::ZoneMapIndexDetails {};
    prost_types::Any::from_msg(&details).unwrap()
}

pub(super) fn inverted_index_details() -> prost_types::Any {
    let details = lance_table::format::pb::InvertedIndexDetails::default();
    prost_types::Any::from_msg(&details).unwrap()
}

impl ScalarIndexDetails for lance_table::format::pb::BitmapIndexDetails {
    fn get_type(&self) -> ScalarIndexType {
        ScalarIndexType::Bitmap
    }
}

impl ScalarIndexDetails for lance_table::format::pb::BTreeIndexDetails {
    fn get_type(&self) -> ScalarIndexType {
        ScalarIndexType::BTree
    }
}

impl ScalarIndexDetails for lance_table::format::pb::LabelListIndexDetails {
    fn get_type(&self) -> ScalarIndexType {
        ScalarIndexType::LabelList
    }
}

impl ScalarIndexDetails for lance_table::format::pb::InvertedIndexDetails {
    fn get_type(&self) -> ScalarIndexType {
        ScalarIndexType::Inverted
    }
}

impl ScalarIndexDetails for lance_table::format::pb::NGramIndexDetails {
    fn get_type(&self) -> ScalarIndexType {
        ScalarIndexType::NGram
    }
}

impl ScalarIndexDetails for lance_table::format::pb::ZoneMapIndexDetails {
    fn get_type(&self) -> ScalarIndexType {
        ScalarIndexType::ZoneMap
    }
}

fn get_scalar_index_details(
    details: &prost_types::Any,
) -> Result<Option<Box<dyn ScalarIndexDetails>>> {
    if details.type_url.ends_with("BitmapIndexDetails") {
        Ok(Some(Box::new(
            details.to_msg::<lance_table::format::pb::BitmapIndexDetails>()?,
        )))
    } else if details.type_url.ends_with("BTreeIndexDetails") {
        Ok(Some(Box::new(
            details.to_msg::<lance_table::format::pb::BTreeIndexDetails>()?,
        )))
    } else if details.type_url.ends_with("LabelListIndexDetails") {
        Ok(Some(Box::new(
            details.to_msg::<lance_table::format::pb::LabelListIndexDetails>()?,
        )))
    } else if details.type_url.ends_with("InvertedIndexDetails") {
        Ok(Some(Box::new(
            details.to_msg::<lance_table::format::pb::InvertedIndexDetails>()?,
        )))
    } else if details.type_url.ends_with("NGramIndexDetails") {
        Ok(Some(Box::new(
            details.to_msg::<lance_table::format::pb::NGramIndexDetails>()?,
        )))
    } else if details.type_url.ends_with("ZoneMapIndexDetails") {
        Ok(Some(Box::new(
            details.to_msg::<lance_table::format::pb::ZoneMapIndexDetails>()?,
        )))
    } else {
        Ok(None)
    }
}

fn get_vector_index_details(
    details: &prost_types::Any,
) -> Result<Option<lance_table::format::pb::VectorIndexDetails>> {
    if details.type_url.ends_with("VectorIndexDetails") {
        Ok(Some(
            details.to_msg::<lance_table::format::pb::VectorIndexDetails>()?,
        ))
    } else {
        Ok(None)
    }
}

/// Build a Scalar Index (returns details to store in the manifest)
#[instrument(level = "debug", skip_all)]
pub(super) async fn build_scalar_index(
    dataset: &Dataset,
    column: &str,
    uuid: &str,
    params: &ScalarIndexParams,
    train: bool,
    fragment_ids: Option<Vec<u32>>,
) -> Result<prost_types::Any> {
    let training_request = Box::new(TrainingRequest {
        dataset: Arc::new(dataset.clone()),
        column: column.to_string(),
        train,
        fragment_ids,
    });
    let field = dataset.schema().field(column).ok_or(Error::InvalidInput {
        source: format!("No column with name {}", column).into(),
        location: location!(),
    })?;

    // Check if LabelList index is being created on a non-List or non-LargeList type
    if matches!(params.force_index_type, Some(ScalarIndexType::LabelList))
        && !matches!(
            field.data_type(),
            DataType::List(_) | DataType::LargeList(_)
        )
    {
        return Err(Error::InvalidInput {
            source: format!(
                "LabelList index can only be created on List or LargeList type columns. Column '{}' has type {:?}",
                column,
                field.data_type()
            )
            .into(),
            location: location!(),
        });
    }

    // In theory it should be possible to create a btree/bitmap index on a nested field but
    // performance would be poor and I'm not sure we want to allow that unless there is a need.
    if !matches!(params.force_index_type, Some(ScalarIndexType::LabelList))
        && field.data_type().is_nested()
    {
        return Err(Error::InvalidInput {
            source: "A scalar index can only be created on a non-nested field.".into(),
            location: location!(),
        });
    }
    let index_store = LanceIndexStore::from_dataset(dataset, uuid);
    match params.force_index_type {
        Some(ScalarIndexType::Bitmap) => {
            train_bitmap_index(training_request, &index_store).await?;
            Ok(bitmap_index_details())
        }
        Some(ScalarIndexType::LabelList) => {
            train_label_list_index(training_request, &index_store).await?;
            Ok(label_list_index_details())
        }
        Some(ScalarIndexType::Inverted) => {
            train_inverted_index(
                training_request,
                &index_store,
                InvertedIndexParams::default(),
                None,
            )
            .await?;
            Ok(inverted_index_details())
        }
        Some(ScalarIndexType::NGram) => {
            if field.data_type() != DataType::Utf8 && field.data_type() != DataType::LargeUtf8 {
                return Err(Error::InvalidInput {
                    source: "NGram index can only be created on Utf8/LargeUtf8 type columns".into(),
                    location: location!(),
                });
            }
            train_ngram_index(training_request, &index_store).await?;
            Ok(ngram_index_details())
        }
        Some(ScalarIndexType::ZoneMap) => {
            // TODO: Add type check for zone map index
            train_zonemap_index(training_request, &index_store, None).await?;
            Ok(zonemap_index_details())
        }
        _ => {
            let flat_index_trainer = FlatIndexMetadata::new(field.data_type());
            train_btree_index(
                training_request,
                &flat_index_trainer,
                &index_store,
                DEFAULT_BTREE_BATCH_SIZE as u32,
            )
            .await?;
            Ok(btree_index_details())
        }
    }
}

/// Build a Scalar Index
#[instrument(level = "debug", skip_all)]
pub(super) async fn build_inverted_index(
    dataset: &Dataset,
    column: &str,
    uuid: &str,
    params: &InvertedIndexParams,
    train: bool,
    fragment_ids: Option<Vec<u32>>,
) -> Result<()> {
    let training_request = Box::new(match fragment_ids.clone() {
        Some(fragment_ids) => TrainingRequest::with_fragment_ids(
            Arc::new(dataset.clone()),
            column.to_string(),
            fragment_ids,
        ),
        None => TrainingRequest {
            dataset: Arc::new(dataset.clone()),
            column: column.to_string(),
            fragment_ids: None,
            train,
        },
    });
    let index_store = LanceIndexStore::from_dataset(dataset, uuid);
    train_inverted_index(training_request, &index_store, params.clone(), fragment_ids).await
}

pub async fn open_scalar_index(
    dataset: &Dataset,
    column: &str,
    index: &Index,
    metrics: &dyn MetricsCollector,
) -> Result<Arc<dyn ScalarIndex>> {
    let uuid_str = index.uuid.to_string();
    let index_dir = dataset.indice_files_dir(index)?.child(uuid_str.as_str());
    let cache = dataset.metadata_cache.file_metadata_cache(&index_dir);
    let index_store = Arc::new(LanceIndexStore::new(
        dataset.object_store.clone(),
        index_dir,
        Arc::new(cache),
    ));

    let index_type = detect_scalar_index_type(dataset, index, column).await?;
    let frag_reuse_index = dataset.open_frag_reuse_index(metrics).await?;

    let index_cache = dataset
        .index_cache
        .for_index(&uuid_str, frag_reuse_index.as_ref().map(|f| &f.uuid));
    match index_type {
        ScalarIndexType::Bitmap => {
            let bitmap_index =
                BitmapIndex::load(index_store, frag_reuse_index, index_cache).await?;
            Ok(bitmap_index as Arc<dyn ScalarIndex>)
        }
        ScalarIndexType::LabelList => {
            let tag_index =
                LabelListIndex::load(index_store, frag_reuse_index, index_cache).await?;
            Ok(tag_index as Arc<dyn ScalarIndex>)
        }
        ScalarIndexType::Inverted => {
            let inverted_index =
                InvertedIndex::load(index_store, frag_reuse_index, index_cache).await?;
            Ok(inverted_index as Arc<dyn ScalarIndex>)
        }
        ScalarIndexType::NGram => {
            let ngram_index = NGramIndex::load(index_store, frag_reuse_index, index_cache).await?;
            Ok(ngram_index as Arc<dyn ScalarIndex>)
        }
        ScalarIndexType::ZoneMap => {
            let zone_map_index =
                ZoneMapIndex::load(index_store, frag_reuse_index, index_cache).await?;
            Ok(zone_map_index as Arc<dyn ScalarIndex>)
        }
        ScalarIndexType::BTree => {
            let btree_index = BTreeIndex::load(index_store, frag_reuse_index, index_cache).await?;
            Ok(btree_index as Arc<dyn ScalarIndex>)
        }
    }
}

async fn infer_scalar_index_type(
    dataset: &Dataset,
    index: &Index,
    column: &str,
) -> Result<ScalarIndexType> {
    let index_dir = dataset
        .indice_files_dir(index)?
        .child(index.uuid.to_string());
    let col = dataset.schema().field(column).ok_or(Error::Internal {
        message: format!(
            "Index refers to column {} which does not exist in dataset schema",
            column
        ),
        location: location!(),
    })?;

    let bitmap_page_lookup = index_dir.child(BITMAP_LOOKUP_NAME);
    let inverted_list_lookup = index_dir.child(METADATA_FILE);
    let legacy_inverted_list_lookup = index_dir.child(INVERT_LIST_FILE);
    let index_type = if let DataType::List(_) = col.data_type() {
        ScalarIndexType::LabelList
    } else if dataset.object_store.exists(&bitmap_page_lookup).await? {
        ScalarIndexType::Bitmap
    } else if dataset.object_store.exists(&inverted_list_lookup).await?
        || dataset
            .object_store
            .exists(&legacy_inverted_list_lookup)
            .await?
    {
        ScalarIndexType::Inverted
    } else {
        ScalarIndexType::BTree
    };

    Ok(index_type)
}

/// Determines the scalar index type
///
/// If the index was created with Lance newer than 0.19.2 then this simply
/// grabs the type from the index details.  If created with an older version
/// then we may have to perform expensive object_store.exists checks to determine
/// the index type.  To mitigate this we cache the result in the session cache.
#[instrument(level = "debug", skip_all)]
pub async fn detect_scalar_index_type(
    dataset: &Dataset,
    index: &Index,
    column: &str,
) -> Result<ScalarIndexType> {
    if let Some(details) = &index.index_details {
        let details = get_scalar_index_details(details)?;
        if let Some(details) = details {
            return Ok(details.get_type());
        } else {
            return Err(Error::Internal {
                message: format!(
                    "Index details for index {} are not a recognized scalar index type",
                    index.uuid
                ),
                location: location!(),
            });
        }
    } else {
        let uuid = index.uuid.to_string();
        let type_key = crate::session::index_caches::ScalarIndexTypeKey { uuid: &uuid };
        if let Some(index_type) = dataset.index_cache.get_with_key(&type_key).await {
            return Ok(*index_type.as_ref());
        }
        let index_type = infer_scalar_index_type(dataset, index, column).await?;
        dataset
            .index_cache
            .insert_with_key(&type_key, Arc::new(index_type))
            .await;
        Ok(index_type)
    }
}

/// Infers the index type from the index details, if available.
/// Returns None if the index details are not present or the type cannot be determined.
/// This returns IndexType::Vector for all vector index types.
pub fn infer_index_type(index: &Index) -> Option<IndexType> {
    if let Some(details) = &index.index_details {
        // "index details" is a serialized protobuf message (prost_types::Any) stored in the Index struct.
        // It contains type-specific metadata for the index, such as which kind of index it is (scalar, vector, etc).
        // Here, we try to parse it as a scalar index details proto, and if that fails, as a vector index details proto.
        if let Ok(Some(details)) = get_scalar_index_details(details) {
            return Some(details.get_type().into());
        } else if let Ok(Some(_)) = get_vector_index_details(details) {
            return Some(IndexType::Vector);
        } else {
            // If the details are not recognized, we return None
            return None;
        }
    }
    None
}

pub fn index_matches_criteria(
    index: &Index,
    criteria: &ScalarIndexCriteria,
    field: &Field,
    has_multiple_indices: bool,
) -> Result<bool> {
    if let Some(name) = &criteria.has_name {
        if &index.name != name {
            return Ok(false);
        }
    }
    if let Some(expected_type) = criteria.has_type {
        // there are 3 cases:
        // - index_type is a scalar index type
        // - index_type is a vector index type
        // - index_type is not present, in which case that the index was created with an older version of Lance
        match infer_index_type(index) {
            Some(index_type) => {
                if let Ok(index_type) = ScalarIndexType::try_from(index_type) {
                    if index_type != expected_type {
                        return Ok(false);
                    }
                } else {
                    // it's a vector index type,
                    // this method is used to determine if the scalar index matches the criteria
                    // so we should return false
                    return Ok(false);
                }
            }
            _ => {
                // if the index_type is not present, just allow it for backwards compatibility
            }
        }

        // We should not use FTS / NGram indices for exact equality queries
        // (i.e. merge insert with a join on the indexed column)
        if criteria.supports_exact_equality {
            match expected_type {
                ScalarIndexType::Inverted | ScalarIndexType::NGram => {
                    return Ok(false);
                }
                _ => {}
            }
        }

        // we allow FTS / NGram indices to co-exist with each other,
        // but we don't allow for the other scalar index types
        if has_multiple_indices
            && !matches!(
                expected_type,
                ScalarIndexType::Inverted | ScalarIndexType::NGram
            )
        {
            return Err(Error::InvalidInput {
                            source: format!(
                                "An index {} on the field with id {} co-exists with other indices on the same column but was written with an older Lance version, and this is not supported.  Please retrain this index.",
                                index.name,
                                index.fields.first().unwrap_or(&0),
                            ).into(),
                            location: location!(),
                        });
        }
    }
    if let Some(for_column) = criteria.for_column {
        if index.fields.len() != 1 {
            return Ok(false);
        }
        if for_column != field.name {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::Int32Type;
    use arrow_array::RecordBatch;
    use arrow_schema::DataType;
    use futures::TryStreamExt;
    use lance_core::datatypes::Field;
    use lance_datagen::{array, BatchCount, RowCount};
    use lance_table::format::pb::{
        BTreeIndexDetails, InvertedIndexDetails, NGramIndexDetails, VectorIndexDetails,
    };

    fn make_index_metadata(
        name: &str,
        field_id: i32,
        index_type: Option<IndexType>,
    ) -> crate::index::IndexMetadata {
        let index_details = index_type.map(|index_type| match index_type {
            IndexType::BTree => prost_types::Any::from_msg(&BTreeIndexDetails::default()).unwrap(),
            IndexType::Inverted => {
                prost_types::Any::from_msg(&InvertedIndexDetails::default()).unwrap()
            }
            IndexType::NGram => prost_types::Any::from_msg(&NGramIndexDetails::default()).unwrap(),
            IndexType::Vector => {
                prost_types::Any::from_msg(&VectorIndexDetails::default()).unwrap()
            }
            _ => {
                unimplemented!("unsupported index type: {}", index_type)
            }
        });
        crate::index::IndexMetadata {
            uuid: uuid::Uuid::new_v4(),
            name: name.to_string(),
            fields: vec![field_id],
            dataset_version: 1,
            fragment_bitmap: None,
            index_details,
            index_version: 0,
            created_at: None,
            base_id: None,
        }
    }

    #[test]
    fn test_index_matches_criteria_vector_index() {
        let index1 = make_index_metadata("vector_index", 1, Some(IndexType::Vector));

        let criteria = ScalarIndexCriteria {
            has_type: Some(ScalarIndexType::BTree),
            supports_exact_equality: false,
            for_column: None,
            has_name: None,
        };

        let field = Field::new_arrow("mycol", DataType::Int32, true).unwrap();
        let result = index_matches_criteria(&index1, &criteria, &field, true).unwrap();
        assert!(!result);

        let result = index_matches_criteria(&index1, &criteria, &field, false).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_index_matches_criteria_scalar_index() {
        let btree_index = make_index_metadata("btree_index", 1, Some(IndexType::BTree));
        let inverted_index = make_index_metadata("inverted_index", 1, Some(IndexType::Inverted));
        let ngram_index = make_index_metadata("ngram_index", 1, Some(IndexType::NGram));

        let criteria = ScalarIndexCriteria {
            has_type: Some(ScalarIndexType::BTree),
            supports_exact_equality: false,
            for_column: None,
            has_name: None,
        };

        let field = Field::new_arrow("mycol", DataType::Int32, true).unwrap();
        let result = index_matches_criteria(&btree_index, &criteria, &field, true);
        assert!(result.is_err());

        let result = index_matches_criteria(&btree_index, &criteria, &field, false).unwrap();
        assert!(result);

        // test for_column
        let mut criteria = ScalarIndexCriteria {
            has_type: None,
            supports_exact_equality: false,
            for_column: Some("mycol"),
            has_name: None,
        };
        let result = index_matches_criteria(&btree_index, &criteria, &field, false).unwrap();
        assert!(result);

        criteria.for_column = Some("mycol2");
        let result = index_matches_criteria(&btree_index, &criteria, &field, false).unwrap();
        assert!(!result);

        // test has_name
        let mut criteria = ScalarIndexCriteria {
            has_type: None,
            supports_exact_equality: false,
            for_column: None,
            has_name: Some("btree_index"),
        };
        let result = index_matches_criteria(&btree_index, &criteria, &field, true).unwrap();
        assert!(result);
        let result = index_matches_criteria(&btree_index, &criteria, &field, false).unwrap();
        assert!(result);

        criteria.has_name = Some("btree_index2");
        let result = index_matches_criteria(&btree_index, &criteria, &field, true).unwrap();
        assert!(!result);
        let result = index_matches_criteria(&btree_index, &criteria, &field, false).unwrap();
        assert!(!result);

        // test supports_exact_equality
        let mut criteria = ScalarIndexCriteria {
            has_type: Some(ScalarIndexType::BTree),
            supports_exact_equality: true,
            for_column: None,
            has_name: None,
        };
        let result = index_matches_criteria(&btree_index, &criteria, &field, false).unwrap();
        assert!(result);

        criteria.has_type = Some(ScalarIndexType::Inverted);
        let result = index_matches_criteria(&inverted_index, &criteria, &field, false).unwrap();
        assert!(!result);

        criteria.has_type = Some(ScalarIndexType::NGram);
        let result = index_matches_criteria(&ngram_index, &criteria, &field, false).unwrap();
        assert!(!result);

        // test multiple indices
        let mut criteria = ScalarIndexCriteria {
            has_type: Some(ScalarIndexType::BTree),
            supports_exact_equality: false,
            for_column: None,
            has_name: None,
        };
        let result = index_matches_criteria(&btree_index, &criteria, &field, true);
        assert!(result.is_err());

        criteria.has_type = Some(ScalarIndexType::Inverted);
        let result = index_matches_criteria(&inverted_index, &criteria, &field, true).unwrap();
        assert!(result);

        criteria.has_type = Some(ScalarIndexType::NGram);
        let result = index_matches_criteria(&ngram_index, &criteria, &field, true).unwrap();
        assert!(result);
    }

    #[tokio::test]
    async fn test_scan_aligned_chunks() {
        // Create test data using lance_datagen
        let data = lance_datagen::gen_batch()
            .col("values", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));

        // Create a dataset for testing
        let temp_dir = tempfile::tempdir().unwrap();
        let dataset_path = temp_dir.path().join("test_dataset");

        // Create a dataset with multiple fragments by writing them separately
        let mut dataset = Dataset::write(data, dataset_path.to_str().unwrap(), None)
            .await
            .unwrap();

        // Append additional fragments using lance_datagen
        for i in 1..3 {
            let start = i * 10;
            let additional_data = lance_datagen::gen_batch()
                .col("values", array::step_custom::<Int32Type>(start, 1))
                .into_reader_rows(RowCount::from(10), BatchCount::from(1));

            dataset.append(additional_data, None).await.unwrap();
        }

        // Create a TrainingRequest
        let training_request = Box::new(TrainingRequest::new(
            Arc::new(dataset),
            "values".to_string(),
            true,
        ));

        // Test scan_aligned_chunks with different chunk sizes
        log::info!("Testing with chunk_size=10:");
        let stream = training_request.scan_aligned_chunks(10).await.unwrap();

        // Collect all batches
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        // Print information about the chunks and verify _rowaddr format
        log::info!("Total number of chunks: {}", batches.len());

        // Collect all _rowaddr values to analyze fragment distribution
        let mut all_rowaddrs = Vec::new();
        let mut fragment_ids = std::collections::HashSet::new();

        for (i, batch) in batches.iter().enumerate() {
            let rowaddr_array = batch
                .column_by_name("_rowaddr")
                .unwrap()
                .as_any()
                .downcast_ref::<arrow_array::UInt64Array>()
                .unwrap();

            let first_rowaddr = rowaddr_array.value(0);
            let last_rowaddr = rowaddr_array.value(batch.num_rows() - 1);

            // Extract fragment ID (upper 32 bits) and local offset (lower 32 bits)
            let first_fragment_id = (first_rowaddr >> 32) as u32;
            let first_local_offset = (first_rowaddr & 0xFFFFFFFF) as u32;
            let last_fragment_id = (last_rowaddr >> 32) as u32;
            let last_local_offset = (last_rowaddr & 0xFFFFFFFF) as u32;

            log::info!(
                "Chunk {}: {} rows, _rowaddr range: {} to {}",
                i,
                batch.num_rows(),
                first_rowaddr,
                last_rowaddr
            );
            log::info!(
                "  Fragment ID range: {} to {}, Local offset range: {} to {}",
                first_fragment_id,
                last_fragment_id,
                first_local_offset,
                last_local_offset
            );

            // Verify each _rowaddr value in this chunk
            for j in 0..batch.num_rows() {
                let rowaddr = rowaddr_array.value(j);
                let fragment_id = (rowaddr >> 32) as u32;
                let local_offset = (rowaddr & 0xFFFFFFFF) as u32;

                // Verify the expected pattern based on chunk index
                if i < 10 {
                    // Chunks 0-9: Fragment 0, local offset 0-99
                    assert_eq!(
                        fragment_id, 0,
                        "Chunk {}: Expected fragment ID 0, got {}",
                        i, fragment_id
                    );
                    let expected_offset = (i * 10 + j) as u32;
                    assert_eq!(
                        local_offset, expected_offset,
                        "Chunk {} row {}: Expected local offset {}, got {}",
                        i, j, expected_offset, local_offset
                    );
                } else if i == 10 {
                    // Chunk 10: Fragment 1, local offset 0-9
                    assert_eq!(
                        fragment_id, 1,
                        "Chunk {}: Expected fragment ID 1, got {}",
                        i, fragment_id
                    );
                    assert_eq!(
                        local_offset, j as u32,
                        "Chunk {} row {}: Expected local offset {}, got {}",
                        i, j, j, local_offset
                    );
                } else if i == 11 {
                    // Chunk 11: Fragment 2, local offset 0-9
                    assert_eq!(
                        fragment_id, 2,
                        "Chunk {}: Expected fragment ID 2, got {}",
                        i, fragment_id
                    );
                    assert_eq!(
                        local_offset, j as u32,
                        "Chunk {} row {}: Expected local offset {}, got {}",
                        i, j, j, local_offset
                    );
                }

                all_rowaddrs.push(rowaddr);
                fragment_ids.insert(fragment_id);
            }
        }

        // Verify we have multiple fragments
        log::info!("Unique fragment IDs: {:?}", fragment_ids);
        assert_eq!(fragment_ids, std::collections::HashSet::from([0, 1, 2]));

        // Verify _rowaddr values are properly ordered
        assert!(
            all_rowaddrs.windows(2).all(|w| w[0] <= w[1]),
            "_rowaddr values are not properly ordered"
        );

        log::info!(
            "Total _rowaddr values: {}, Unique fragments: {}",
            all_rowaddrs.len(),
            fragment_ids.len()
        );

        // Check that the schema includes the expected columns
        let output_schema = batches[0].schema();
        let field_names: Vec<String> = output_schema
            .fields()
            .iter()
            .map(|f| f.name())
            .cloned()
            .collect();

        // Should have the original column
        assert!(field_names.contains(&"values".to_string()));

        // The _rowaddr column should be present when aligned is true
        // (This is what we're testing - that scan_aligned_chunks adds the _rowaddr column)
        assert!(
            field_names.contains(&"_rowaddr".to_string()),
            "Expected _rowaddr column in aligned scan, got fields: {:?}",
            field_names
        );
    }
}
