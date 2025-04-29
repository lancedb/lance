// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::{BinaryHeap, HashMap},
    sync::Arc,
};

use lance_core::Result;

use crate::{metrics::NoOpMetricsCollector, scalar::IndexStore};

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
    input: HashMap<u64, &'a InvertedPartition>,
    target_size: u64,
    builder: InnerBuilder,
    partitions: Vec<u64>,
}

impl<'a> SizeBasedMerger<'a> {
    // Create a new SizeBasedMerger with the target size,
    // the size is uncompressed size in byte.
    // Typically, just set the size to the memory limit,
    // because less partitions means faster query.
    pub fn new(
        src_store: Arc<dyn IndexStore>,
        dest_store: &'a dyn IndexStore,
        input: impl IntoIterator<Item = &'a InvertedPartition>,
        target_size: u64,
    ) -> Self {
        let input = input
            .into_iter()
            .map(|p| (p.id(), p))
            .collect::<HashMap<_, _>>();
        let max_id = input.keys().copied().max().unwrap_or(0);
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
            self.builder.write(self.dest_store).await?;
            self.partitions.push(self.builder.id());
            self.builder = InnerBuilder::new(self.builder.id() + 1);
        }
        Ok(())
    }
}

impl<'a> Merger for SizeBasedMerger<'a> {
    async fn merge(&mut self) -> Result<Vec<u64>> {
        // if self.input.len() == 1 {
        //     let part_id = self.input.keys().next().unwrap().clone();
        //     self.src_store
        //         .copy_index_file(&token_file_path(part_id), self.dest_store)
        //         .await?;
        //     self.src_store
        //         .copy_index_file(&posting_file_path(part_id), self.dest_store)
        //         .await?;
        //     self.src_store
        //         .copy_index_file(&doc_file_path(part_id), self.dest_store)
        //         .await?;
        //     return Ok(vec![part_id]);
        // }

        // for token set, union the tokens,
        // for doc set, concatenate the row ids, assign the doc id to offset + doc_id
        // for posting list, concatenate the posting lists
        for (part_id, part) in self.input.iter() {
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
                    PostingListBuilder::empty(part.inverted_list.has_positions())
                });
            for token_id in 0..part.tokens.len() as u32 {
                let posting_list = part
                    .inverted_list
                    .posting_list(
                        token_id,
                        part.inverted_list.has_positions(),
                        &NoOpMetricsCollector,
                    )
                    .await?;

                let new_token_id = self.builder.tokens.get(&inv_token[&token_id]).unwrap();
                let builder = &mut self.builder.posting_lists[new_token_id as usize];
                posting_list.positions(row_id)
                for (doc_id, freq) in posting_list.iter() {
                    let new_doc_id = doc_id_offset + doc_id as u32;
                    builder.add(new_doc_id, term_positions);
                }
            }
        }
        while let Some(location) = heap.pop() {
            let (token, locations) = (location.token, location.locations);
            self.builder.tokens.add(token);
            let mut posting_builder = PostingListBuilder::empty(with_position);

            let posting_lists = locations.iter().map(|(id, token_id)| {
                let part = self.input[id];
                part.inverted_list.posting_list(
                    *token_id,
                    part.inverted_list.has_positions(),
                    &NoOpMetricsCollector,
                )
            });
            let posting_lists = futures::future::try_join_all(posting_lists).await?;
            for (posting_list, (part_id, _)) in std::iter::zip(posting_lists, locations) {
                let part = self.input[&part_id];
                for (doc_id, freq) in posting_list.iter() {
                    let row_id = part.docs.row_id(doc_id as u32);
                    let positions = PositionRecorder::Count(freq);
                    let new_doc_id = self.builder.docs.add(row_id, positions.len());
                    // the new doc id is not guaranteed to be correct if the row id has been added,
                    // so must use `doc_ids` to check if the row id is already added.
                    let new_doc_id = doc_ids.entry(row_id).or_insert(new_doc_id).clone();

                    posting_builder.add(new_doc_id, positions);
                }
            }

            // add the posting list to the builder
            total_size += posting_builder.size();
            self.builder.posting_lists.push(posting_builder);
            if total_size > self.target_size {
                self.builder.write(self.dest_store).await?;
                new_partitions.push(self.builder.id());
                self.builder = InnerBuilder::new(self.builder.id() + 1);
                doc_ids.clear();
                total_size = 0;
            }
            assert_eq!(self.builder.posting_lists.len(), self.builder.tokens.len());
        }

        if self.builder.tokens.len() > 0 {
            self.builder.write(self.dest_store).await?;
            new_partitions.push(self.builder.id());
        }

        Ok(new_partitions)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TokenLocation {
    token: String,
    // the partition id and the token id in the partition
    locations: Vec<(u64, u32)>,
}

impl PartialOrd for TokenLocation {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TokenLocation {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.locations
            .len()
            .cmp(&other.locations.len())
            .then(self.token.cmp(&other.token))
    }
}
