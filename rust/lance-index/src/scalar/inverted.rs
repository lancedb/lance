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

pub use builder::InvertedIndexBuilder;
pub use index::*;
use lance_core::Result;
pub use tokenizer::*;

use super::btree::TrainingSource;
use super::IndexStore;

pub async fn train_inverted_index(
    data_source: Box<dyn TrainingSource>,
    index_store: &dyn IndexStore,
    params: InvertedIndexParams,
) -> Result<()> {
    let batch_stream = data_source.scan_unordered_chunks(4096).await?;
    // mapping from item to list of the row ids where it is present
    let mut inverted_index = InvertedIndexBuilder::new(params);
    inverted_index.update(batch_stream, index_store).await
}
