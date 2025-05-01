// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::{BinaryHeap, HashMap},
    sync::Arc,
};

use lance_core::Result;

use crate::{
    metrics::NoOpMetricsCollector,
    scalar::{inverted::PostingList, IndexStore},
};

use super::{
    builder::{doc_file_path, posting_file_path, token_file_path, InnerBuilder, PositionRecorder},
    InvertedPartition, PostingListBuilder,
};

pub(crate) trait Merger {
    // Merge the partitions and write new partitions,
    // the new partitions are returned.
    // This method would read all the input partitions at the same time,
    // so it's not recommended to pass too many partitions.
    async fn merge(&mut self) -> Result<Vec<u64>>;
}

// A merger that merges partitions based on their size,
// it would read the posting lists for each token from
// the partitions and write them to a new partition,
// until the size of the new partition reaches the target size.
pub(crate) struct SizeBasedMerger<'a> {
    src_store: Arc<dyn IndexStore>,
    dest_store: &'a dyn IndexStore,
    input: Vec<&'a InvertedPartition>,
    target_size: u64,
    builder: InnerBuilder,
    partitions: Vec<u64>,
}

impl<'a> SizeBasedMerger<'a> {
    // Create a new SizeBasedMerger with the target size,
    // the size is compressed size in byte.
    // Typically, just set the size to the memory limit,
    // because less partitions means faster query.
    pub fn new(
        src_store: Arc<dyn IndexStore>,
        dest_store: &'a dyn IndexStore,
        input: impl IntoIterator<Item = &'a InvertedPartition>,
        target_size: u64,
    ) -> Self {
        let input = input.into_iter().collect::<Vec<_>>();
        let max_id = input.iter().map(|p| p.id()).max().unwrap_or(0);
        Self {
            src_store,
            dest_store,
            input,
            target_size,
            builder: InnerBuilder::new(max_id + 1),
            partitions: Vec::new(),
        }
    }

    async fn flush(&mut self) -> Result<()> {
        if self.builder.tokens.len() > 0 {
            log::info!("flushing partition {}", self.builder.id());
            let start = std::time::Instant::now();
            self.builder.write(self.dest_store).await?;
            log::info!(
                "flushed partition {} in {:?}",
                self.builder.id(),
                start.elapsed()
            );
            self.partitions.push(self.builder.id());
            self.builder = InnerBuilder::new(self.builder.id() + 1);
        }
        Ok(())
    }
}

impl<'a> Merger for SizeBasedMerger<'a> {
    async fn merge(&mut self) -> Result<Vec<u64>> {
        if self.input.len() <= 1 {
            let part_ids = self.input.iter().map(|p| p.id()).collect::<Vec<_>>();
            for &part_id in part_ids.iter() {
                self.src_store
                    .copy_index_file(&token_file_path(part_id), self.dest_store)
                    .await?;
                self.src_store
                    .copy_index_file(&posting_file_path(part_id), self.dest_store)
                    .await?;
                self.src_store
                    .copy_index_file(&doc_file_path(part_id), self.dest_store)
                    .await?;
            }

            return Ok(part_ids);
        }

        // for token set, union the tokens,
        // for doc set, concatenate the row ids, assign the doc id to offset + doc_id
        // for posting list, concatenate the posting lists
        log::info!("merging {} partitions", self.input.len());
        let start = std::time::Instant::now();
        for idx in 0..self.input.len() {
            let part = self.input[idx];
            let doc_id_offset = self.builder.docs.len() as u32;
            // single partition can index up to u32::MAX documents
            if self.builder.docs.len() + part.docs.len() > u32::MAX as usize {
                self.flush().await?;
            }

            let mut inv_token = HashMap::with_capacity(part.tokens.len());
            // merge token set
            for (token, token_id) in part.tokens.iter() {
                self.builder.tokens.add(token.clone());
                inv_token.insert(token_id, token);
            }
            // merge doc set
            for (row_id, num_tokens) in part.docs.iter() {
                self.builder.docs.add(*row_id, *num_tokens);
            }
            // merge posting lists
            self.builder
                .posting_lists
                .resize_with(self.builder.tokens.len(), || {
                    PostingListBuilder::new(part.inverted_list.has_positions())
                });

            let postings = part
                .inverted_list
                .read_batch(part.inverted_list.has_positions())
                .await?;
            for token_id in 0..part.tokens.len() as u32 {
                let posting_list = part
                    .inverted_list
                    .posting_list_from_batch(&postings.slice(token_id as usize, 1), token_id)?;

                let new_token_id = self.builder.tokens.get(&inv_token[&token_id]).unwrap();
                let builder = &mut self.builder.posting_lists[new_token_id as usize];
                for (doc_id, freq, positions) in posting_list.iter() {
                    let new_doc_id = doc_id_offset + doc_id as u32;
                    let positions = match positions {
                        Some(positions) => PositionRecorder::Position(positions.collect()),
                        None => PositionRecorder::Count(freq),
                    };
                    builder.add(new_doc_id, positions);
                }
            }
            log::info!(
                "merged {}/{} partitions in {:?}",
                idx + 1,
                self.input.len(),
                start.elapsed()
            );
        }

        self.flush().await?;
        Ok(self.partitions.clone())
    }
}
