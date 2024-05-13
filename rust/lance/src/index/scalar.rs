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
        btree::{train_btree_index, BTreeIndex, BtreeTrainingSource},
        flat::FlatIndexMetadata,
        lance_format::LanceIndexStore,
        ScalarIndex,
    },
    IndexType,
};
use lance_table::format::{Fragment, Index as IndexMetadata};
use snafu::{location, Location};
use tracing::instrument;

use lance_core::{datatypes::Field, Error, Result};
use uuid::Uuid;

use crate::{
    dataset::scanner::{ColumnOrdering, DatasetRecordBatchStream},
    Dataset,
};

use super::{DatasetIndexInternalExt, IndexParams};

pub const LANCE_SCALAR_INDEX: &str = "__lance_scalar_index";

#[derive(Default)]
pub struct ScalarIndexParams {}

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

/// Optimize a scalar index.
///
/// Index the new data in `frags_to_index` and merge with all the indices in
/// `indices_to_merge`. If `indices_to_merge` is empty, create a new index.
pub async fn optimize_scalar_index(
    dataset: &Dataset,
    column: &Field,
    indices_to_merge: &[&IndexMetadata],
    mut frags_to_index: Vec<Fragment>,
) -> Result<Uuid> {
    let existing_index = if let Some(idx) = indices_to_merge.first() {
        // We will extend the first index, and scan the data in all the other ones.
        let other_covered_frags = indices_to_merge[1..]
            .iter()
            .flat_map(|idx| idx.fragment_bitmap.iter().flat_map(|bitmap| bitmap.iter()))
            .collect::<Vec<u32>>();
        let other_covered_frags = dataset
            .get_fragments()
            .into_iter()
            .map(|f| f.metadata().clone())
            .filter(|frag| other_covered_frags.contains(&(frag.id as u32)));
        frags_to_index.extend(other_covered_frags);
        Some(idx)
    } else {
        None
    };

    let new_data_stream = if frags_to_index.is_empty() {
        None
    } else {
        let mut scanner = dataset.scan();
        scanner
            .with_fragments(frags_to_index)
            .with_row_id()
            .order_by(Some(vec![ColumnOrdering::asc_nulls_first(
                column.name.clone(),
            )]))?
            .project(&[&column.name])?;
        Some(scanner.try_into_stream().await?)
    };

    let new_uuid = Uuid::new_v4();
    let index_dir = dataset.indices_dir().child(new_uuid.to_string());
    let new_store = LanceIndexStore::new((*dataset.object_store).clone(), index_dir);

    match (existing_index, new_data_stream) {
        // Note: since we put other delta indexes into data stream, this
        // is also the case where we merge indices.
        (Some(existing_index), Some(new_data_stream)) => {
            // TODO: how can I downcast `existing_index` and use that?
            let index = dataset
                .open_scalar_index(&column.name, &existing_index.uuid.to_string())
                .await?;
            index.update(new_data_stream.into(), &new_store).await?;
            Ok(new_uuid)
        }
        (None, Some(new_data_stream)) => {
            let training_source = StreamTrainingSource::new(new_data_stream);
            let flat_index_trainer = FlatIndexMetadata::new(column.data_type());

            train_btree_index(Box::new(training_source), &flat_index_trainer, &new_store).await?;
            Ok(new_uuid)
        }
        _ => unreachable!(),
    }
}

struct StreamTrainingSource {
    stream: DatasetRecordBatchStream,
}

impl StreamTrainingSource {
    pub fn new(stream: DatasetRecordBatchStream) -> Self {
        Self { stream }
    }
}

#[async_trait]
impl BtreeTrainingSource for StreamTrainingSource {
    async fn scan_ordered_chunks(
        self: Box<Self>,
        chunk_size: u32,
    ) -> Result<SendableRecordBatchStream> {
        Ok(chunk_concat_stream(self.stream.into(), chunk_size as usize))
    }
}
