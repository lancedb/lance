// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use fst::Streamer;
use lance_core::Result;

use crate::scalar::IndexStore;

use super::{
    builder::{doc_file_path, posting_file_path, token_file_path, InnerBuilder, PositionRecorder},
    InvertedPartition, PostingListBuilder, TokenMap, TokenSetFormat,
};

pub trait Merger {
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
pub struct SizeBasedMerger<'a> {
    dest_store: &'a dyn IndexStore,
    input: Vec<InvertedPartition>,
    with_position: bool,
    target_size: u64,
    token_set_format: TokenSetFormat,
    builder: InnerBuilder,
    partitions: Vec<u64>,
}

impl<'a> SizeBasedMerger<'a> {
    // Create a new SizeBasedMerger with the target size,
    // the size is compressed size in byte.
    // Typically, just set the size to the memory limit,
    // because less partitions means faster query.
    pub fn new(
        dest_store: &'a dyn IndexStore,
        input: Vec<InvertedPartition>,
        target_size: u64,
        token_set_format: TokenSetFormat,
    ) -> Self {
        let max_id = input.iter().map(|p| p.id()).max().unwrap_or(0);
        let with_position = input
            .first()
            .map(|p| p.inverted_list.has_positions())
            .unwrap_or(false);

        Self {
            dest_store,
            input,
            with_position,
            target_size,
            token_set_format,
            builder: InnerBuilder::new(max_id + 1, with_position, token_set_format),
            partitions: Vec::new(),
        }
    }

    async fn flush(&mut self) -> Result<()> {
        if !self.builder.tokens.is_empty() {
            log::info!("flushing partition {}", self.builder.id());
            let start = std::time::Instant::now();
            self.builder.write(self.dest_store).await?;
            log::info!(
                "flushed partition {} in {:?}",
                self.builder.id(),
                start.elapsed()
            );
            self.partitions.push(self.builder.id());
            self.builder = InnerBuilder::new(
                self.builder.id() + 1,
                self.with_position,
                self.token_set_format,
            );
        }
        Ok(())
    }
}

impl Merger for SizeBasedMerger<'_> {
    async fn merge(&mut self) -> Result<Vec<u64>> {
        if self.input.len() <= 1 {
            for part in self.input.iter() {
                part.store()
                    .copy_index_file(&token_file_path(part.id()), self.dest_store)
                    .await?;
                part.store()
                    .copy_index_file(&posting_file_path(part.id()), self.dest_store)
                    .await?;
                part.store()
                    .copy_index_file(&doc_file_path(part.id()), self.dest_store)
                    .await?;
            }

            return Ok(self.input.iter().map(|p| p.id()).collect());
        }

        // for token set, union the tokens,
        // for doc set, concatenate the row ids, assign the doc id to offset + doc_id
        // for posting list, concatenate the posting lists
        log::info!(
            "merging {} partitions with target size {} MiB",
            self.input.len(),
            self.target_size / 1024 / 1024
        );
        let mut estimated_size = 0;
        let start = std::time::Instant::now();
        let parts = std::mem::take(&mut self.input);
        let num_parts = parts.len();
        for (idx, part) in parts.into_iter().enumerate() {
            // single partition can index up to u32::MAX documents,
            // or target size is reached
            if self.builder.docs.len() + part.docs.len() > u32::MAX as usize
                || estimated_size >= self.target_size
            {
                self.flush().await?;
                estimated_size = 0;
            }

            // merge token set
            let mut token_id_map = vec![u32::MAX; part.tokens.len()];
            match &part.tokens.tokens {
                TokenMap::HashMap(map) => {
                    for (token, token_id) in map.iter() {
                        let new_token_id = self.builder.tokens.get_or_add(token.as_str());
                        let index = *token_id as usize;
                        debug_assert!(index < token_id_map.len());
                        token_id_map[index] = new_token_id;
                    }
                }
                TokenMap::Fst(map) => {
                    let mut stream = map.stream();
                    while let Some((token, token_id)) = stream.next() {
                        let token_id = token_id as u32;
                        let token = String::from_utf8_lossy(token);
                        let new_token_id = self.builder.tokens.get_or_add(token.as_ref());
                        let index = token_id as usize;
                        debug_assert!(index < token_id_map.len());
                        token_id_map[index] = new_token_id;
                    }
                }
            }
            // merge doc set
            let doc_id_offset = self.builder.docs.len() as u32;
            for (row_id, num_tokens) in part.docs.iter() {
                self.builder.docs.append(*row_id, *num_tokens);
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
                let new_token_id = token_id_map[token_id as usize];
                debug_assert_ne!(new_token_id, u32::MAX);
                let builder = &mut self.builder.posting_lists[new_token_id as usize];
                let old_size = builder.size();
                for (doc_id, freq, positions) in posting_list.iter() {
                    let new_doc_id = doc_id_offset + doc_id as u32;
                    let positions = match positions {
                        Some(positions) => PositionRecorder::Position(positions.collect()),
                        None => PositionRecorder::Count(freq),
                    };
                    builder.add(new_doc_id, positions);
                }
                let new_size = builder.size();
                estimated_size += new_size - old_size;
            }
            log::info!(
                "merged {}/{} partitions in {:?}",
                idx + 1,
                num_parts,
                start.elapsed()
            );
        }

        self.flush().await?;
        Ok(self.partitions.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::NoOpMetricsCollector;
    use crate::scalar::lance_format::LanceIndexStore;
    use lance_core::cache::LanceCache;
    use lance_core::utils::tempfile::TempObjDir;
    use lance_io::object_store::ObjectStore;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_merge_reuses_token_ids_for_shared_tokens() -> Result<()> {
        let src_dir = TempObjDir::default();
        let dest_dir = TempObjDir::default();
        let src_store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            src_dir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));
        let dest_store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            dest_dir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let token_set_format = TokenSetFormat::default();

        let mut builder0 = InnerBuilder::new(0, false, token_set_format);
        let apple_id = builder0.tokens.add("apple".to_owned());
        let banana_id = builder0.tokens.add("banana".to_owned());
        builder0
            .posting_lists
            .resize_with(builder0.tokens.len(), || PostingListBuilder::new(false));
        let doc_id = builder0.docs.append(10, 2);
        builder0.posting_lists[apple_id as usize].add(doc_id, PositionRecorder::Count(1));
        builder0.posting_lists[banana_id as usize].add(doc_id, PositionRecorder::Count(1));
        builder0.write(src_store.as_ref()).await?;

        let mut builder1 = InnerBuilder::new(1, false, token_set_format);
        let banana_id = builder1.tokens.add("banana".to_owned());
        let carrot_id = builder1.tokens.add("carrot".to_owned());
        builder1
            .posting_lists
            .resize_with(builder1.tokens.len(), || PostingListBuilder::new(false));
        let doc_id = builder1.docs.append(20, 2);
        builder1.posting_lists[banana_id as usize].add(doc_id, PositionRecorder::Count(1));
        builder1.posting_lists[carrot_id as usize].add(doc_id, PositionRecorder::Count(1));
        builder1.write(src_store.as_ref()).await?;

        let partition0 = InvertedPartition::load(
            src_store.clone(),
            0,
            None,
            &LanceCache::no_cache(),
            token_set_format,
        )
        .await?;
        let partition1 = InvertedPartition::load(
            src_store.clone(),
            1,
            None,
            &LanceCache::no_cache(),
            token_set_format,
        )
        .await?;

        let mut merger = SizeBasedMerger::new(
            dest_store.as_ref(),
            vec![partition0, partition1],
            u64::MAX,
            token_set_format,
        );
        let merged_partitions = merger.merge().await?;
        assert_eq!(merged_partitions, vec![2]);

        let merged = InvertedPartition::load(
            dest_store.clone(),
            merged_partitions[0],
            None,
            &LanceCache::no_cache(),
            token_set_format,
        )
        .await?;

        assert_eq!(merged.tokens.len(), 3);
        assert_eq!(merged.docs.len(), 2);
        assert_eq!(merged.docs.row_id(0), 10);
        assert_eq!(merged.docs.row_id(1), 20);

        let banana_token_id = merged.tokens.get("banana").unwrap();
        let posting = merged
            .inverted_list
            .posting_list(banana_token_id, false, &NoOpMetricsCollector)
            .await?;
        let doc_ids: Vec<u64> = posting.iter().map(|(doc_id, _, _)| doc_id).collect();
        assert_eq!(doc_ids, vec![0, 1]);

        Ok(())
    }
}
