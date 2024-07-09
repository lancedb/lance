// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Utilities for integrating scalar indices with datasets
//!

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::physical_plan::SendableRecordBatchStream;
use lance_datafusion::{chunker::chunk_concat_stream, exec::LanceExecutionOptions};
use lance_index::{
    scalar::{
        bitmap::{train_bitmap_index, BitmapIndex, BITMAP_LOOKUP_NAME},
        btree::{train_btree_index, BTreeIndex, BtreeTrainingSource},
        flat::FlatIndexMetadata,
        lance_format::LanceIndexStore,
        ScalarIndex,
    },
    IndexType,
};
use snafu::{location, Location};
use tracing::instrument;

use lance_core::{Error, Result};

use crate::{
    dataset::{index::LanceIndexStoreExt, scanner::ColumnOrdering},
    Dataset,
};

use super::IndexParams;

pub const LANCE_SCALAR_INDEX: &str = "__lance_scalar_index";

pub enum ScalarIndexType {
    BTree,
    Bitmap,
}

#[derive(Default)]
pub struct ScalarIndexParams {
    /// If set then always use the given index type and skip auto-detection
    pub force_index_type: Option<ScalarIndexType>,
}

impl IndexParams for ScalarIndexParams {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn index_type(&self) -> IndexType {
        IndexType::Scalar
    }

    fn index_name(&self) -> &str {
        LANCE_SCALAR_INDEX
    }
}

struct TrainingRequest {
    dataset: Arc<Dataset>,
    column: String,
}

#[async_trait]
impl BtreeTrainingSource for TrainingRequest {
    async fn scan_ordered_chunks(
        self: Box<Self>,
        chunk_size: u32,
    ) -> Result<SendableRecordBatchStream> {
        let mut scan = self.dataset.scan();
        let scan = scan
            .with_row_id()
            .order_by(Some(vec![ColumnOrdering::asc_nulls_first(
                self.column.clone(),
            )]))?
            .project(&[&self.column])?;

        let ordered_batches = scan
            .try_into_dfstream(LanceExecutionOptions {
                use_spilling: true,
                ..Default::default()
            })
            .await?;
        Ok(chunk_concat_stream(ordered_batches, chunk_size as usize))
    }
}

/// Build a Scalar Index
#[instrument(level = "debug", skip_all)]
pub async fn build_scalar_index(
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
    // In theory it should be possible to create a scalar index (e.g. btree) on a nested field but
    // performance would be poor and I'm not sure we want to allow that unless there is a need.
    if field.data_type().is_nested() {
        return Err(Error::InvalidInput {
            source: "A scalar index can only be created on a non-nested field.".into(),
            location: location!(),
        });
    }
    let index_store = LanceIndexStore::from_dataset(dataset, uuid);
    match params.force_index_type {
        Some(ScalarIndexType::Bitmap) => train_bitmap_index(training_request, &index_store).await,
        _ => {
            let flat_index_trainer = FlatIndexMetadata::new(field.data_type());
            train_btree_index(training_request, &flat_index_trainer, &index_store).await
        }
    }
}

pub async fn open_scalar_index(dataset: &Dataset, uuid: &str) -> Result<Arc<dyn ScalarIndex>> {
    let index_store = Arc::new(LanceIndexStore::from_dataset(dataset, uuid));
    let index_dir = dataset.indices_dir().child(uuid);
    // This works at the moment, since we only have two index types, may need to introduce better
    // detection method in the future.
    let bitmap_page_lookup = index_dir.child(BITMAP_LOOKUP_NAME);
    if dataset.object_store.exists(&bitmap_page_lookup).await? {
        let bitmap_index = BitmapIndex::load(index_store).await?;
        Ok(bitmap_index as Arc<dyn ScalarIndex>)
    } else {
        let btree_index = BTreeIndex::load(index_store).await?;
        Ok(btree_index as Arc<dyn ScalarIndex>)
    }
}
