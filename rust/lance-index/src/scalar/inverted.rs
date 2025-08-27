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
    fragment_ids: Option<Vec<u32>>,
) -> Result<()> {
    let batch_stream = data_source.scan_unordered_chunks(4096).await?;

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

    let mut inverted_index = InvertedIndexBuilder::new_with_mask(params, fragment_mask);
    inverted_index.update(batch_stream, index_store).await
}
