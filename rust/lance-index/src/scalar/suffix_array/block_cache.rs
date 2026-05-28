// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Block cache for byte-range reads on raw segment files.
//!
//! Divides a file into fixed-size blocks and caches them on demand,
//! avoiding the need to load entire multi-hundred-megabyte files into
//! memory for random-access binary search.

use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;
use lance_core::Result;
use tokio::sync::RwLock;

use crate::scalar::IndexStore;

/// Default block size: 64 KB.
///
/// Binary search does ~60 random reads of `query.len()` bytes each.
/// With 64 KB blocks, at most ~60 distinct blocks = ~3.8 MB cached,
/// compared to ~500 MB for a full segment load.
const DEFAULT_BLOCK_SIZE: usize = 64 * 1024;

/// Block cache for byte-range reads on a raw file stored in an [`IndexStore`].
///
/// Instead of loading the entire file (which can be hundreds of megabytes),
/// this cache lazily fetches fixed-size blocks on demand via
/// [`IndexStore::read_raw_range`]. Blocks are cached in memory for the
/// lifetime of this struct.
///
/// # Memory usage
///
/// For a 500 MB file with 64 KB blocks:
/// - Binary search: ~60 blocks × 64 KB = **3.8 MB**
/// - Position iteration (200K matches): ~1000 blocks × 64 KB = **64 MB**
/// - Full load baseline: **500 MB**
pub struct BlockCache {
    store: Arc<dyn IndexStore>,
    filename: String,
    file_size: usize,
    block_size: usize,
    blocks: RwLock<HashMap<usize, Bytes>>,
}

impl std::fmt::Debug for BlockCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockCache")
            .field("filename", &self.filename)
            .field("file_size", &self.file_size)
            .field("block_size", &self.block_size)
            .finish()
    }
}

impl BlockCache {
    /// Create a new block cache for a raw file.
    pub fn new(store: Arc<dyn IndexStore>, filename: String, file_size: usize) -> Self {
        Self {
            store,
            filename,
            file_size,
            block_size: DEFAULT_BLOCK_SIZE,
            blocks: RwLock::new(HashMap::new()),
        }
    }

    /// Create a block cache from in-memory bytes.
    ///
    /// This is used when data has been fully loaded (e.g. from a `.bin` file)
    /// but the query path expects a `BlockCache`. The data is pre-populated
    /// into the cache so `read()` never hits the store.
    pub fn from_bytes(store: Arc<dyn IndexStore>, filename: String, data: Bytes) -> Self {
        let file_size = data.len();
        let block_size = DEFAULT_BLOCK_SIZE;
        let mut blocks = HashMap::new();

        // Slice the data into blocks and pre-populate the cache
        let mut offset = 0;
        let mut block_idx = 0;
        while offset < file_size {
            let end = (offset + block_size).min(file_size);
            blocks.insert(block_idx, data.slice(offset..end));
            offset = end;
            block_idx += 1;
        }

        Self {
            store,
            filename,
            file_size,
            block_size,
            blocks: RwLock::new(blocks),
        }
    }

    /// Read `len` bytes starting at `offset`.
    ///
    /// Fetches blocks on demand and caches them. Reads spanning multiple
    /// blocks are assembled from cached blocks.
    pub async fn read(&self, offset: usize, len: usize) -> Result<Bytes> {
        if len == 0 {
            return Ok(Bytes::new());
        }

        let block_idx = offset / self.block_size;
        let block_offset = offset % self.block_size;

        // Fast path: data fits in one block
        if block_offset + len <= self.block_size {
            let block = self.get_block(block_idx).await?;
            let end = (block_offset + len).min(block.len());
            return Ok(block.slice(block_offset..end));
        }

        // Slow path: spans multiple blocks — concatenate
        let mut result = Vec::with_capacity(len);
        let mut remaining = len;
        let mut cur_block = block_idx;
        let mut cur_offset = block_offset;
        while remaining > 0 {
            let block = self.get_block(cur_block).await?;
            if cur_offset >= block.len() {
                break; // past end of file
            }
            let available = block.len() - cur_offset;
            let take = remaining.min(available);
            result.extend_from_slice(&block[cur_offset..cur_offset + take]);
            remaining -= take;
            cur_block += 1;
            cur_offset = 0;
        }
        Ok(Bytes::from(result))
    }

    /// Total number of bytes in the underlying file.
    pub fn len(&self) -> usize {
        self.file_size
    }

    /// Number of blocks currently cached.
    pub async fn cached_blocks(&self) -> usize {
        self.blocks.read().await.len()
    }

    /// Approximate memory usage of cached blocks.
    pub async fn cached_bytes(&self) -> usize {
        self.blocks
            .read()
            .await
            .values()
            .map(|b| b.len())
            .sum()
    }

    async fn get_block(&self, block_idx: usize) -> Result<Bytes> {
        // Check cache (read lock)
        {
            let cache = self.blocks.read().await;
            if let Some(block) = cache.get(&block_idx) {
                return Ok(block.clone());
            }
        }

        // Cache miss — fetch from store (write lock)
        let start = block_idx * self.block_size;
        if start >= self.file_size {
            return Ok(Bytes::new());
        }
        let end = (start + self.block_size).min(self.file_size);
        let data = self
            .store
            .read_raw_range(&self.filename, start..end)
            .await?;

        let mut cache = self.blocks.write().await;
        cache.insert(block_idx, data.clone());
        Ok(data)
    }
}

impl deepsize::DeepSizeOf for BlockCache {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        // Approximate: we can't await inside DeepSizeOf, so estimate
        // based on file_size as upper bound. Actual cached size is lower.
        // The LRU cache in SuffixArrayIndex tracks segment size via
        // SuffixArraySegment::deep_size_of, which includes this estimate.
        self.file_size / 10 // conservative estimate: ~10% loaded
    }
}
