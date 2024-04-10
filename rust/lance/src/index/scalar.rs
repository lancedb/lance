// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Utilities for integrating scalar indices with datasets
//!

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::physical_plan::SendableRecordBatchStream;
use lance_datafusion::{chunker::chunk_concat_stream, exec::LanceExecutionOptions};
use lance_index::scalar::{
    btree::{train_btree_index, BTreeIndex, BtreeTrainingSource},
    flat::FlatIndexMetadata,
    lance_format::LanceIndexStore,
    ScalarIndex,
};
use snafu::{location, Location};
use tracing::instrument;

use lance_core::{Error, Result};

use crate::{dataset::scanner::ColumnOrdering, Dataset};

use super::IndexParams;

#[derive(Default)]
pub struct ScalarIndexParams {}

impl IndexParams for ScalarIndexParams {
    fn as_any(&self) -> &dyn std::any::Any {
        self
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

/// Build a Vector Index
#[instrument(level = "debug", skip(dataset))]
pub async fn build_scalar_index(dataset: &Dataset, column: &str, uuid: &str) -> Result<()> {
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
    let flat_index_trainer = FlatIndexMetadata::new(field.data_type());
    let index_dir = dataset.indices_dir().child(uuid);
    let index_store = LanceIndexStore::new((*dataset.object_store).clone(), index_dir);
    train_btree_index(training_request, &flat_index_trainer, &index_store).await
}

pub async fn open_scalar_index(dataset: &Dataset, uuid: &str) -> Result<Arc<dyn ScalarIndex>> {
    let index_dir = dataset.indices_dir().child(uuid);
    let index_store = Arc::new(LanceIndexStore::new(
        (*dataset.object_store).clone(),
        index_dir,
    ));
    // Currently we assume all scalar indices are btree indices.  In the future, if this is not the
    // case, we may need to store a metadata file in the index directory with scalar index metadata
    let btree_index = BTreeIndex::load(index_store).await?;
    Ok(btree_index as Arc<dyn ScalarIndex>)
}
