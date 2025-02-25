// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Utilities for integrating scalar indices with datasets
//!

use std::sync::Arc;

use arrow_schema::DataType;
use async_trait::async_trait;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::TryStreamExt;
use lance_core::{Error, Result};
use lance_datafusion::{chunker::chunk_concat_stream, exec::LanceExecutionOptions};
use lance_index::scalar::btree::DEFAULT_BTREE_BATCH_SIZE;
use lance_index::scalar::ngram::{train_ngram_index, NGramIndex};
use lance_index::scalar::InvertedIndexParams;
use lance_index::scalar::{
    bitmap::{train_bitmap_index, BitmapIndex, BITMAP_LOOKUP_NAME},
    btree::{train_btree_index, BTreeIndex, TrainingSource},
    flat::FlatIndexMetadata,
    inverted::{train_inverted_index, InvertedIndex, INVERT_LIST_FILE},
    label_list::{train_label_list_index, LabelListIndex},
    lance_format::LanceIndexStore,
    ScalarIndex, ScalarIndexParams, ScalarIndexType,
};
use lance_table::format::Index;
use log::info;
use snafu::location;
use tracing::instrument;

use crate::session::Session;
use crate::{
    dataset::{index::LanceIndexStoreExt, scanner::ColumnOrdering},
    Dataset,
};

// Log an update every TRAINING_UPDATE_FREQ million rows processed
const TRAINING_UPDATE_FREQ: usize = 1000000;

struct TrainingRequest {
    dataset: Arc<Dataset>,
    column: String,
}

#[async_trait]
impl TrainingSource for TrainingRequest {
    async fn scan_ordered_chunks(
        self: Box<Self>,
        chunk_size: u32,
    ) -> Result<SendableRecordBatchStream> {
        self.scan_chunks(chunk_size, true).await
    }

    async fn scan_unordered_chunks(
        self: Box<Self>,
        chunk_size: u32,
    ) -> Result<SendableRecordBatchStream> {
        self.scan_chunks(chunk_size, false).await
    }
}

impl TrainingRequest {
    async fn scan_chunks(
        self: Box<Self>,
        chunk_size: u32,
        sort: bool,
    ) -> Result<SendableRecordBatchStream> {
        let num_rows = self.dataset.count_all_rows().await?;

        let mut scan = self.dataset.scan();

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

        let ordering = match sort {
            true => Some(vec![ColumnOrdering::asc_nulls_first(self.column.clone())]),
            false => None,
        };

        let scan = scan
            .with_row_id()
            .order_by(ordering)?
            .project(&[&self.column])?;

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
                    training_uuid, self.column, rows_processed
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
) -> Result<prost_types::Any> {
    let training_request = Box::new(TrainingRequest {
        dataset: Arc::new(dataset.clone()),
        column: column.to_string(),
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
            )
            .await?;
            Ok(inverted_index_details())
        }
        Some(ScalarIndexType::NGram) => {
            if field.data_type() != DataType::Utf8 {
                return Err(Error::InvalidInput {
                    source: "NGram index can only be created on Utf8 type columns".into(),
                    location: location!(),
                });
            }
            train_ngram_index(training_request, &index_store).await?;
            Ok(ngram_index_details())
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
) -> Result<()> {
    let training_request = Box::new(TrainingRequest {
        dataset: Arc::new(dataset.clone()),
        column: column.to_string(),
    });
    let index_store = LanceIndexStore::from_dataset(dataset, uuid);
    train_inverted_index(training_request, &index_store, params.clone()).await
}

pub async fn open_scalar_index(
    dataset: &Dataset,
    column: &str,
    index: &Index,
) -> Result<Arc<dyn ScalarIndex>> {
    let uuid_str = index.uuid.to_string();
    let index_store = Arc::new(LanceIndexStore::from_dataset(dataset, &uuid_str));
    let index_type = detect_scalar_index_type(dataset, index, column, &dataset.session).await?;
    match index_type {
        ScalarIndexType::Bitmap => {
            let bitmap_index = BitmapIndex::load(index_store).await?;
            Ok(bitmap_index as Arc<dyn ScalarIndex>)
        }
        ScalarIndexType::LabelList => {
            let tag_index = LabelListIndex::load(index_store).await?;
            Ok(tag_index as Arc<dyn ScalarIndex>)
        }
        ScalarIndexType::Inverted => {
            let inverted_index = InvertedIndex::load(index_store).await?;
            Ok(inverted_index as Arc<dyn ScalarIndex>)
        }
        ScalarIndexType::NGram => {
            let ngram_index = NGramIndex::load(index_store).await?;
            Ok(ngram_index as Arc<dyn ScalarIndex>)
        }
        ScalarIndexType::BTree => {
            let btree_index = BTreeIndex::load(index_store).await?;
            Ok(btree_index as Arc<dyn ScalarIndex>)
        }
    }
}

async fn infer_scalar_index_type(
    dataset: &Dataset,
    index_uuid: &str,
    column: &str,
) -> Result<ScalarIndexType> {
    let index_dir = dataset.indices_dir().child(index_uuid.to_string());
    let col = dataset.schema().field(column).ok_or(Error::Internal {
        message: format!(
            "Index refers to column {} which does not exist in dataset schema",
            column
        ),
        location: location!(),
    })?;

    let bitmap_page_lookup = index_dir.child(BITMAP_LOOKUP_NAME);
    let inverted_list_lookup = index_dir.child(INVERT_LIST_FILE);
    let index_type = if let DataType::List(_) = col.data_type() {
        ScalarIndexType::LabelList
    } else if dataset.object_store.exists(&bitmap_page_lookup).await? {
        ScalarIndexType::Bitmap
    } else if dataset.object_store.exists(&inverted_list_lookup).await? {
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
    session: &Session,
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
        if let Some(index_type) = session.index_cache.get_type(&uuid) {
            return Ok(index_type);
        }
        let index_type = infer_scalar_index_type(dataset, &index.uuid.to_string(), column).await?;
        session.index_cache.insert_type(&uuid, index_type);
        Ok(index_type)
    }
}
