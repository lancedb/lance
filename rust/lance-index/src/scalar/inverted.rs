// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

pub mod builder;
mod encoding;
mod index;
mod iter;
pub mod json;
mod merger;
pub mod parser;
pub mod query;
mod scorer;
pub mod tokenizer;
mod wand;

use std::sync::Arc;

use arrow_schema::{DataType, Field};
use async_trait::async_trait;
pub use builder::InvertedIndexBuilder;
use datafusion::execution::SendableRecordBatchStream;
pub use index::*;
use lance_core::{cache::LanceCache, Result};
use tantivy::tokenizer::Language;
pub use tokenizer::*;

use lance_core::Error;
use snafu::location;

use crate::pbold;
use crate::{
    frag_reuse::FragReuseIndex,
    scalar::{
        expression::{FtsQueryParser, ScalarQueryParser},
        registry::{ScalarIndexPlugin, TrainingCriteria, TrainingOrdering, TrainingRequest},
        CreatedIndex, ScalarIndex,
    },
};

use super::IndexStore;

#[derive(Debug, Default)]
pub struct InvertedIndexPlugin;

impl InvertedIndexPlugin {
    pub async fn train_inverted_index(
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        params: InvertedIndexParams,
        fragment_ids: Option<Vec<u32>>,
    ) -> Result<CreatedIndex> {
        let fragment_mask = fragment_ids.as_ref().and_then(|frag_ids| {
            if !frag_ids.is_empty() {
                // Create a mask with fragment_id in high 32 bits for distributed indexing
                // This mask is used to filter partitions belonging to specific fragments
                // If multiple fragments processed, use first fragment_id <<32 as mask
                Some((frag_ids[0] as u64) << 32)
            } else {
                None
            }
        });

        let details = pbold::InvertedIndexDetails::try_from(&params)?;
        let mut inverted_index =
            InvertedIndexBuilder::new_with_fragment_mask(params, fragment_mask);
        inverted_index.update(data, index_store).await?;
        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&details).unwrap(),
            index_version: INVERTED_INDEX_VERSION,
        })
    }

    /// Return true if the query can be used to speed up contains_tokens queries
    fn can_accelerate_queries(details: &pbold::InvertedIndexDetails) -> bool {
        details.base_tokenizer == Some("simple".to_string())
            && details.max_token_length.is_none()
            && details.language == serde_json::to_string(&Language::English).unwrap()
            && !details.stem
    }
}

struct InvertedIndexTrainingRequest {
    parameters: InvertedIndexParams,
    criteria: TrainingCriteria,
}

impl InvertedIndexTrainingRequest {
    pub fn new(parameters: InvertedIndexParams) -> Self {
        Self {
            parameters,
            criteria: TrainingCriteria::new(TrainingOrdering::None).with_row_id(),
        }
    }
}

impl TrainingRequest for InvertedIndexTrainingRequest {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn criteria(&self) -> &TrainingCriteria {
        &self.criteria
    }
}

#[async_trait]
impl ScalarIndexPlugin for InvertedIndexPlugin {
    fn new_training_request(
        &self,
        params: &str,
        field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        match field.data_type() {
            DataType::Utf8 | DataType::LargeUtf8 | DataType::LargeBinary => (),
            DataType::List(f) if matches!(f.data_type(), DataType::Utf8 | DataType::LargeUtf8) => (),
            DataType::LargeList(f) if matches!(f.data_type(), DataType::Utf8 | DataType::LargeUtf8) => (),

            _ => return Err(Error::InvalidInput {
                source: format!(
                    "A inverted index can only be created on a Utf8 or LargeUtf8 field/list or LargeBinary field. Column has type {:?}",
                    field.data_type()
                )
                    .into(),
                location: location!(),
            })
        }

        let params = serde_json::from_str::<InvertedIndexParams>(params)?;
        Ok(Box::new(InvertedIndexTrainingRequest::new(params)))
    }

    fn provides_exact_answer(&self) -> bool {
        false
    }

    fn version(&self) -> u32 {
        INVERTED_INDEX_VERSION
    }

    fn new_query_parser(
        &self,
        index_name: String,
        _index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        let Ok(index_details) = _index_details.to_msg::<pbold::InvertedIndexDetails>() else {
            return None;
        };

        if Self::can_accelerate_queries(&index_details) {
            Some(Box::new(FtsQueryParser::new(index_name)))
        } else {
            None
        }
    }

    /// Train a new index
    ///
    /// The provided data must fulfill all the criteria returned by `training_criteria`.
    /// It is the caller's responsibility to ensure this.
    ///
    /// Returns index details that describe the index.  These details can potentially be
    /// useful for planning (although this will currently require inside information on
    /// the index type) and they will need to be provided when loading the index.
    ///
    /// It is the caller's responsibility to store these details somewhere.
    async fn train_index(
        &self,
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        request: Box<dyn TrainingRequest>,
        fragment_ids: Option<Vec<u32>>,
    ) -> Result<CreatedIndex> {
        let request = (request as Box<dyn std::any::Any>)
            .downcast::<InvertedIndexTrainingRequest>()
            .map_err(|_| Error::InvalidInput {
                source: "must provide training request created by new_training_request".into(),
                location: location!(),
            })?;
        Self::train_inverted_index(data, index_store, request.parameters.clone(), fragment_ids)
            .await
    }

    /// Load an index from storage
    ///
    /// The index details should match the details that were returned when the index was
    /// originally trained.
    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        _index_details: &prost_types::Any,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        Ok(
            InvertedIndex::load(index_store, frag_reuse_index, cache).await?
                as Arc<dyn ScalarIndex>,
        )
    }
}
