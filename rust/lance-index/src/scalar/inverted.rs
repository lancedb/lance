// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

mod builder;
mod index;
mod wand;

pub use builder::InvertedIndexBuilder;
pub use index::*;
use lance_core::Result;

use super::btree::TrainingSource;
use super::{IndexStore, InvertedIndexParams};

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
