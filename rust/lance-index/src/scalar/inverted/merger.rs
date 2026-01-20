// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use fst::Streamer;
use futures::{stream, StreamExt, TryStreamExt};
use lance_core::{cache::LanceCache, utils::tokio::get_num_compute_intensive_cpus, Error, Result};
use snafu::location;

use crate::scalar::IndexStore;

use super::{
    builder::{doc_file_path, posting_file_path, token_file_path, InnerBuilder, PositionRecorder},
    InvertedPartition, PostingListBuilder, TokenMap, TokenSetFormat,
};

pub trait Merger {
    // Merge the partitions and write new partitions,
    // the new partitions are returned.
    // This method streams partitions with bounded buffering to avoid
    // loading all partitions into memory at once.
    async fn merge(&mut self) -> Result<Vec<u64>>;
}

#[derive(Debug, Clone)]
pub(super) struct PartitionSource {
    store: std::sync::Arc<dyn IndexStore>,
    id: u64,
}

impl PartitionSource {
    pub(super) fn new(store: std::sync::Arc<dyn IndexStore>, id: u64) -> Self {
        Self { store, id }
    }

    async fn load(
        &self,
        cache: &LanceCache,
        token_set_format: TokenSetFormat,
    ) -> Result<InvertedPartition> {
        InvertedPartition::load(self.store.clone(), self.id, None, cache, token_set_format).await
    }
}

// A merger that merges partitions based on their size,
// it would read the posting lists for each token from
// the partitions and write them to a new partition,
// until the size of the new partition reaches the target size.
pub struct SizeBasedMerger<'a> {
    dest_store: &'a dyn IndexStore,
    input: Vec<PartitionSource>,
    with_position: Option<bool>,
    target_size: u64,
    token_set_format: TokenSetFormat,
    builder: Option<InnerBuilder>,
    next_id: u64,
    partitions: Vec<u64>,
}

impl<'a> SizeBasedMerger<'a> {
    // Create a new SizeBasedMerger with the target size,
    // the size is compressed size in byte.
    // Typically, just set the size to the memory limit,
    // because less partitions means faster query.
    pub fn new(
        dest_store: &'a dyn IndexStore,
        input: Vec<PartitionSource>,
        target_size: u64,
        token_set_format: TokenSetFormat,
    ) -> Self {
        let max_id = input.iter().map(|p| p.id).max().unwrap_or(0);

        Self {
            dest_store,
            input,
            with_position: None,
            target_size,
            token_set_format,
            builder: None,
            next_id: max_id + 1,
            partitions: Vec::new(),
        }
    }

    async fn flush(&mut self) -> Result<()> {
        let Some(builder) = self.builder.as_mut() else {
            return Ok(());
        };

        if !builder.tokens.is_empty() {
            log::info!("flushing partition {}", builder.id());
            let start = std::time::Instant::now();
            builder.write(self.dest_store).await?;
            log::info!(
                "flushed partition {} in {:?}",
                builder.id(),
                start.elapsed()
            );
            self.partitions.push(builder.id());
            let with_position = self.with_position.expect("with_position must be set");
            let next_id = self.next_id;
            self.builder = Some(InnerBuilder::new(
                next_id,
                with_position,
                self.token_set_format,
            ));
            self.next_id += 1;
        }
        Ok(())
    }

    fn ensure_builder(&mut self, part: &InvertedPartition) -> Result<()> {
        let with_position = part.inverted_list.has_positions();
        match self.with_position {
            Some(existing) => {
                if existing != with_position {
                    return Err(Error::Index {
                        message: "partition position settings do not match".to_string(),
                        location: location!(),
                    });
                }
            }
            None => {
                self.with_position = Some(with_position);
            }
        }

        if self.builder.is_none() {
            let with_position = self.with_position.expect("with_position must be set");
            self.builder = Some(InnerBuilder::new(
                self.next_id,
                with_position,
                self.token_set_format,
            ));
            self.next_id += 1;
        }
        Ok(())
    }

    async fn merge_partition(
        &mut self,
        part: InvertedPartition,
        estimated_size: &mut u64,
    ) -> Result<()> {
        self.ensure_builder(&part)?;

        {
            let builder = self.builder.as_ref().expect("builder must exist");
            if builder.docs.len() + part.docs.len() > u32::MAX as usize
                || *estimated_size >= self.target_size
            {
                self.flush().await?;
                *estimated_size = 0;
                self.ensure_builder(&part)?;
            }
        }

        let builder = self.builder.as_mut().expect("builder must exist");
        let mut token_id_map = vec![u32::MAX; part.tokens.len()];
        match &part.tokens.tokens {
            TokenMap::HashMap(map) => {
                for (token, token_id) in map.iter() {
                    let new_token_id = builder.tokens.get_or_add(token.as_str());
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
                    let new_token_id = builder.tokens.get_or_add(token.as_ref());
                    let index = token_id as usize;
                    debug_assert!(index < token_id_map.len());
                    token_id_map[index] = new_token_id;
                }
            }
        }
        let doc_id_offset = builder.docs.len() as u32;
        for (row_id, num_tokens) in part.docs.iter() {
            builder.docs.append(*row_id, *num_tokens);
        }
        builder.posting_lists.resize_with(builder.tokens.len(), || {
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
            let builder = &mut builder.posting_lists[new_token_id as usize];
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
            *estimated_size += new_size - old_size;
        }
        Ok(())
    }
}

impl Merger for SizeBasedMerger<'_> {
    async fn merge(&mut self) -> Result<Vec<u64>> {
        if self.input.len() <= 1 {
            for part in self.input.iter() {
                part.store
                    .copy_index_file(&token_file_path(part.id), self.dest_store)
                    .await?;
                part.store
                    .copy_index_file(&posting_file_path(part.id), self.dest_store)
                    .await?;
                part.store
                    .copy_index_file(&doc_file_path(part.id), self.dest_store)
                    .await?;
            }

            return Ok(self.input.iter().map(|p| p.id).collect());
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
        let buffer_size = std::cmp::max(
            1,
            std::cmp::min(get_num_compute_intensive_cpus(), num_parts),
        );
        let cache = LanceCache::no_cache();
        let token_set_format = self.token_set_format;
        let mut stream = stream::iter(parts.into_iter().map(|part| {
            let cache = cache.clone();
            async move { part.load(&cache, token_set_format).await }
        }))
        .buffered(buffer_size);

        let mut idx = 0;
        while let Some(part) = stream.try_next().await? {
            idx += 1;
            self.merge_partition(part, &mut estimated_size).await?;
            log::info!(
                "merged {}/{} partitions in {:?}",
                idx,
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

        let mut merger = SizeBasedMerger::new(
            dest_store.as_ref(),
            vec![
                PartitionSource::new(src_store.clone(), 0),
                PartitionSource::new(src_store.clone(), 1),
            ],
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

    #[tokio::test]
    async fn test_merge_streams_partitions_in_batches() -> Result<()> {
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
        let num_parts = get_num_compute_intensive_cpus().saturating_add(2);
        let mut sources = Vec::with_capacity(num_parts);
        for id in 0..num_parts as u64 {
            let mut builder = InnerBuilder::new(id, false, token_set_format);
            let token_id = builder.tokens.add(format!("token_{}", id));
            builder
                .posting_lists
                .resize_with(builder.tokens.len(), || PostingListBuilder::new(false));
            let doc_id = builder.docs.append(id * 10, 1);
            builder.posting_lists[token_id as usize].add(doc_id, PositionRecorder::Count(1));
            builder.write(src_store.as_ref()).await?;
            sources.push(PartitionSource::new(src_store.clone(), id));
        }

        let mut merger =
            SizeBasedMerger::new(dest_store.as_ref(), sources, u64::MAX, token_set_format);
        let merged_partitions = merger.merge().await?;
        assert_eq!(merged_partitions, vec![num_parts as u64]);

        let merged = InvertedPartition::load(
            dest_store.clone(),
            merged_partitions[0],
            None,
            &LanceCache::no_cache(),
            token_set_format,
        )
        .await?;
        assert_eq!(merged.tokens.len(), num_parts);
        assert_eq!(merged.docs.len(), num_parts);

        Ok(())
    }
}
