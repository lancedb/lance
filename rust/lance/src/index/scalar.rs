// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Utilities for integrating scalar indices with datasets
//!

use std::sync::Arc;

use arrow_schema::DataType;
use async_trait::async_trait;
use datafusion::physical_plan::SendableRecordBatchStream;
use lance_core::{Error, Result};
use lance_datafusion::{chunker::chunk_concat_stream, exec::LanceExecutionOptions};
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
use snafu::{location, Location};
use tracing::instrument;

use crate::{
    dataset::{index::LanceIndexStoreExt, scanner::ColumnOrdering},
    Dataset,
};

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
        let mut scan = self.dataset.scan();

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
                use_spilling: true,
                ..Default::default()
            })
            .await?;
        Ok(chunk_concat_stream(batches, chunk_size as usize))
    }
}

/// Build a Scalar Index
#[instrument(level = "debug", skip_all)]
pub(super) async fn build_scalar_index(
    dataset: &Dataset,
    column: &str,
    uuid: &str,
    params: &ScalarIndexParams,
) -> Result<()> {
    let training_request = Box::new(TrainingRequest {
        dataset: Arc::new(dataset.clone()),
        column: column.to_string(),
    });
    let field = dataset.schema().field(column).ok_or(Error::InvalidInput {
        source: format!("No column with name {}", column).into(),
        location: location!(),
    })?;
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
        Some(ScalarIndexType::Bitmap) => train_bitmap_index(training_request, &index_store).await,
        Some(ScalarIndexType::LabelList) => {
            train_label_list_index(training_request, &index_store).await
        }
        Some(ScalarIndexType::Inverted) => {
            train_inverted_index(
                training_request,
                &index_store,
                InvertedIndexParams::default(),
            )
            .await
        }
        _ => {
            // The BTree index implementation leverages the legacy format's batch offset,
            // which has been removed from new format, so keep using the legacy format for now.
            let index_store = index_store.with_legacy_format(true);
            let flat_index_trainer = FlatIndexMetadata::new(field.data_type());
            train_btree_index(training_request, &flat_index_trainer, &index_store).await
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
    uuid: &str,
) -> Result<Arc<dyn ScalarIndex>> {
    let index_store = Arc::new(LanceIndexStore::from_dataset(dataset, uuid));
    let index_dir = dataset.indices_dir().child(uuid);
    // This works at the moment, since we only have a few index types, may need to introduce better
    // detection method in the future.
    let col = dataset.schema().field(column).ok_or(Error::Internal {
        message: format!(
            "Index refers to column {} which does not exist in dataset schema",
            column
        ),
        location: location!(),
    })?;
    let bitmap_page_lookup = index_dir.child(BITMAP_LOOKUP_NAME);
    let inverted_list_lookup = index_dir.child(INVERT_LIST_FILE);
    if let DataType::List(_) = col.data_type() {
        let tag_index = LabelListIndex::load(index_store).await?;
        Ok(tag_index as Arc<dyn ScalarIndex>)
    } else if dataset.object_store.exists(&bitmap_page_lookup).await? {
        let bitmap_index = BitmapIndex::load(index_store).await?;
        Ok(bitmap_index as Arc<dyn ScalarIndex>)
    } else if dataset.object_store.exists(&inverted_list_lookup).await? {
        let inverted_index = InvertedIndex::load(index_store).await?;
        Ok(inverted_index as Arc<dyn ScalarIndex>)
    } else {
        let btree_index = BTreeIndex::load(index_store).await?;
        Ok(btree_index as Arc<dyn ScalarIndex>)
    }
}
