// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

pub mod builder;
mod encoding;
mod index;
mod iter;
mod merger;
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

use crate::{
    frag_reuse::FragReuseIndex,
    pb,
    scalar::{
        expression::{FtsQueryParser, ScalarQueryParser},
        registry::{ScalarIndexPlugin, TrainingCriteria, TrainingOrdering},
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
    ) -> Result<CreatedIndex> {
        let details = pb::InvertedIndexDetails::try_from(&params)?;
        let mut inverted_index = InvertedIndexBuilder::new(params);
        inverted_index.update(data, index_store).await?;
        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&details).unwrap(),
            index_version: INVERTED_INDEX_VERSION,
        })
    }

    /// Return true if the query can be used to speed up contains_tokens queries
    fn can_accelerate_queries(details: &pb::InvertedIndexDetails) -> bool {
        details.base_tokenizer == Some("simple".to_string())
            && details.max_token_length.is_none()
            && details.language == serde_json::to_string(&Language::English).unwrap()
            && !details.stem
    }
}

#[async_trait]
impl ScalarIndexPlugin for InvertedIndexPlugin {
    fn training_criteria(&self, _params: &prost_types::Any) -> Result<TrainingCriteria> {
        Ok(TrainingCriteria::new(TrainingOrdering::None).with_row_id())
    }

    fn check_can_train(&self, field: &Field) -> Result<()> {
        if !matches!(field.data_type(), DataType::Utf8 | DataType::LargeUtf8) {
            return Err(Error::InvalidInput {
                source: format!(
                    "A ngram index can only be created on a Utf8 or LargeUtf8 field.  Column has type {:?}",
                    field.data_type()
                )
                .into(),
                location: location!(),
            });
        }
        Ok(())
    }

    fn provides_exact_answer(&self) -> bool {
        false
    }

    fn version(&self) -> u32 {
        0
    }

    fn new_query_parser(
        &self,
        index_name: String,
        _index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        let Ok(index_details) = _index_details.to_msg::<pb::InvertedIndexDetails>() else {
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
        params: &prost_types::Any,
    ) -> Result<CreatedIndex> {
        let params = params.to_msg::<pb::InvertedIndexParams>()?;
        Self::train_inverted_index(data, index_store, (&params).try_into()?).await
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
        cache: LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        Ok(
            InvertedIndex::load(index_store, frag_reuse_index, cache).await?
                as Arc<dyn ScalarIndex>,
        )
    }
}
