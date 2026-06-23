// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! FM-Index for exact substring search (following the Infini-gram Mini paper)
//!
//! The FM-Index is a compressed full-text index based on the Burrows-Wheeler Transform (BWT).
//! It supports exact substring matching via backward search and returns exact row ids.
//!
//! Architecture (matching the paper):
//!   - Huffman-shaped Wavelet Tree over BWT for entropy-compressed rank queries (~0.26N)
//!   - Sampled Suffix Array every D-th position for locate (~N/D × 8 bytes)
//!   - doc_start_positions for mapping text positions to documents (tiny)
//!   - No doc_array — documents are resolved via SA sampling + LF-mapping + binary search
//!
//! Total index size: ~0.44N (matching paper's claim)
//!
//! Storage layout (v10 - blocked, partitioned):
//!   - BWT wavelet tree bitvectors in blocks of BLOCK_WORDS (32KB each)
//!   - SA samples stored as packed binary blocks after wavelet blocks
//!   - Row IDs and doc_start_positions in metadata
//!   - File metadata: c_table, huffman_codes, tree topology

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::sync::{Arc, OnceLock};

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field};
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use futures::StreamExt;
use lance_core::cache::LanceCache;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, ROW_ADDR, Result};
use roaring::RoaringBitmap;

use crate::frag_reuse::FragReuseIndex;
use crate::metrics::MetricsCollector;
use crate::pb;
use crate::scalar::expression::{ScalarQueryParser, TextQueryParser};
use crate::scalar::registry::{
    DefaultTrainingRequest, ScalarIndexPlugin, TrainingCriteria, TrainingOrdering, TrainingRequest,
    VALUE_COLUMN_NAME,
};
use crate::scalar::{
    AnyQuery, BuiltinIndexType, CreatedIndex, IndexFile, IndexStore, OldIndexDataFilter,
    ScalarIndex, ScalarIndexParams, SearchResult, TextQuery, UpdateCriteria,
};
use crate::{Index, IndexType};

const FMINDEX_INDEX_VERSION: u32 = 10;
const BLOCK_WORDS: usize = 4096;
const PARTITION_SIZE: usize = 10_000;
const SENTINEL_BYTE: u8 = 0xFF;

/// SA sampling rate. Store every D-th SA entry. Locate walks at most D LF steps.
const SA_SAMPLE_RATE: usize = 32;

fn fmindex_partition_path(partition_id: u64) -> String {
    format!("part_{partition_id}_fm.lance")
}

// ── Bitvector with O(1) rank ─────────────────────────────────────────────────

const SUPERBLOCK_BITS: usize = 512;
const WORDS_PER_SUPERBLOCK: usize = SUPERBLOCK_BITS / 64;

#[derive(Debug, Clone)]
struct RankBitVec {
    words: Vec<u64>,
    superblocks: Vec<u32>,
    len: usize,
}

#[allow(dead_code)]
impl RankBitVec {
    fn new(len: usize) -> Self {
        Self {
            words: vec![0u64; len.div_ceil(64)],
            superblocks: Vec::new(),
            len,
        }
    }

    #[inline]
    fn set(&mut self, pos: usize) {
        self.words[pos / 64] |= 1u64 << (pos % 64);
    }

    #[inline]
    fn get(&self, pos: usize) -> bool {
        (self.words[pos / 64] >> (pos % 64)) & 1 != 0
    }

    fn build_rank_index(&mut self) {
        let num_sb = self.words.len().div_ceil(WORDS_PER_SUPERBLOCK) + 1;
        self.superblocks = Vec::with_capacity(num_sb);
        let mut cum = 0u32;
        for (i, chunk) in self.words.chunks(WORDS_PER_SUPERBLOCK).enumerate() {
            self.superblocks.push(if i == 0 { 0 } else { cum });
            for &w in chunk {
                cum += w.count_ones();
            }
        }
        self.superblocks.push(cum);
    }

    #[inline]
    fn rank1(&self, pos: usize) -> usize {
        if pos == 0 {
            return 0;
        }
        let word_idx = pos / 64;
        let bit_idx = pos % 64;
        let sb_idx = word_idx / WORDS_PER_SUPERBLOCK;
        let mut count = self.superblocks[sb_idx] as usize;
        for i in (sb_idx * WORDS_PER_SUPERBLOCK)..word_idx {
            count += self.words[i].count_ones() as usize;
        }
        if bit_idx > 0 {
            count += (self.words[word_idx] & ((1u64 << bit_idx) - 1)).count_ones() as usize;
        }
        count
    }

    #[inline]
    fn rank0(&self, pos: usize) -> usize {
        pos - self.rank1(pos)
    }

    fn deep_size(&self) -> usize {
        self.words.len() * 8 + self.superblocks.len() * 4
    }
}

// ── Huffman-shaped Wavelet Tree ──────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
struct HuffmanCode {
    bits: u32,
    length: u8,
    node_path: Vec<usize>,
}

#[derive(Debug, Clone)]
enum WaveletChild {
    Node(usize),
    Leaf(u8),
}

#[derive(Debug, Clone)]
struct HuffmanWaveletTree {
    nodes: Vec<RankBitVec>,
    codes: [HuffmanCode; 256],
    children: Vec<(WaveletChild, WaveletChild)>,
    len: usize,
}

#[derive(Debug)]
enum HuffNode {
    Leaf(u8),
    Internal { left: Box<Self>, right: Box<Self> },
}

impl PartialEq for HuffNode {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}
impl Eq for HuffNode {}
impl PartialOrd for HuffNode {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for HuffNode {
    fn cmp(&self, _: &Self) -> std::cmp::Ordering {
        std::cmp::Ordering::Equal
    }
}

#[allow(dead_code)]
impl HuffmanWaveletTree {
    fn build(data: &[u8]) -> Self {
        let n = data.len();
        if n == 0 {
            return Self {
                nodes: Vec::new(),
                codes: std::array::from_fn(|_| HuffmanCode::default()),
                children: Vec::new(),
                len: 0,
            };
        }

        let mut freq = [0u64; 256];
        for &b in data {
            freq[b as usize] += 1;
        }

        let mut heap: BinaryHeap<(Reverse<u64>, Reverse<usize>, Box<HuffNode>)> = BinaryHeap::new();
        let mut tie = 0;
        for (v, &f) in freq.iter().enumerate() {
            if f > 0 {
                heap.push((Reverse(f), Reverse(tie), Box::new(HuffNode::Leaf(v as u8))));
                tie += 1;
            }
        }
        if heap.len() == 1 {
            let (f, _, node) = heap.pop().unwrap();
            heap.push((Reverse(0), Reverse(tie), Box::new(HuffNode::Leaf(255))));
            tie += 1;
            heap.push((f, Reverse(tie), node));
            tie += 1;
        }
        while heap.len() > 1 {
            let (Reverse(f1), _, l) = heap.pop().unwrap();
            let (Reverse(f2), _, r) = heap.pop().unwrap();
            heap.push((
                Reverse(f1 + f2),
                Reverse(tie),
                Box::new(HuffNode::Internal { left: l, right: r }),
            ));
            tie += 1;
        }
        let root = heap.pop().unwrap().2;

        let mut codes: [HuffmanCode; 256] = std::array::from_fn(|_| HuffmanCode::default());
        let mut node_count = 0;
        let mut children_map: Vec<(WaveletChild, WaveletChild)> = Vec::new();

        fn assign(
            node: &HuffNode,
            bits: u32,
            len: u8,
            path: &mut Vec<usize>,
            nid: &mut usize,
            codes: &mut [HuffmanCode; 256],
            cm: &mut Vec<(WaveletChild, WaveletChild)>,
        ) -> WaveletChild {
            match node {
                HuffNode::Leaf(b) => {
                    codes[*b as usize] = HuffmanCode {
                        bits,
                        length: len,
                        node_path: path.clone(),
                    };
                    WaveletChild::Leaf(*b)
                }
                HuffNode::Internal { left, right } => {
                    let my = *nid;
                    *nid += 1;
                    path.push(my);
                    cm.push((WaveletChild::Leaf(0), WaveletChild::Leaf(0)));
                    let lc = assign(left, bits << 1, len + 1, path, nid, codes, cm);
                    let rc = assign(right, (bits << 1) | 1, len + 1, path, nid, codes, cm);
                    cm[my] = (lc, rc);
                    path.pop();
                    WaveletChild::Node(my)
                }
            }
        }
        assign(
            &root,
            0,
            0,
            &mut Vec::new(),
            &mut node_count,
            &mut codes,
            &mut children_map,
        );

        let mut node_sizes = vec![0usize; node_count];
        for &b in data {
            for &nid in &codes[b as usize].node_path {
                node_sizes[nid] += 1;
            }
        }
        let mut nodes: Vec<RankBitVec> = node_sizes.iter().map(|&sz| RankBitVec::new(sz)).collect();
        let mut cursors = vec![0usize; node_count];
        for &b in data {
            let code = &codes[b as usize];
            for (level, &nid) in code.node_path.iter().enumerate() {
                if (code.bits >> (code.length - 1 - level as u8)) & 1 == 1 {
                    nodes[nid].set(cursors[nid]);
                }
                cursors[nid] += 1;
            }
        }
        for n in &mut nodes {
            n.build_rank_index();
        }
        Self {
            nodes,
            codes,
            children: children_map,
            len: n,
        }
    }

    /// Retrieve the byte at position `pos` in the original BWT.
    #[inline]
    fn access(&self, mut pos: usize) -> u8 {
        if self.nodes.is_empty() {
            return 0;
        }
        let mut node_idx = 0;
        loop {
            let bit = self.nodes[node_idx].get(pos);
            let (ref left, ref right) = self.children[node_idx];
            if bit {
                pos = self.nodes[node_idx].rank1(pos);
                match right {
                    WaveletChild::Leaf(b) => return *b,
                    WaveletChild::Node(next) => node_idx = *next,
                }
            } else {
                pos = self.nodes[node_idx].rank0(pos);
                match left {
                    WaveletChild::Leaf(b) => return *b,
                    WaveletChild::Node(next) => node_idx = *next,
                }
            }
        }
    }

    /// Count occurrences of byte `c` in positions `[0, pos)`.
    #[inline]
    fn rank(&self, c: u8, pos: usize) -> usize {
        let code = &self.codes[c as usize];
        if code.length == 0 {
            return 0;
        }
        let (mut lo, mut hi) = (0, pos);
        for (level, &nid) in code.node_path.iter().enumerate() {
            if (code.bits >> (code.length - 1 - level as u8)) & 1 == 0 {
                lo = self.nodes[nid].rank0(lo);
                hi = self.nodes[nid].rank0(hi);
            } else {
                lo = self.nodes[nid].rank1(lo);
                hi = self.nodes[nid].rank1(hi);
            }
        }
        hi - lo
    }

    #[inline]
    fn rank_pair(&self, c: u8, lo: usize, hi: usize) -> (usize, usize) {
        let code = &self.codes[c as usize];
        if code.length == 0 {
            return (0, 0);
        }
        let (mut s, mut l, mut h) = (0, lo, hi);
        for (level, &nid) in code.node_path.iter().enumerate() {
            if (code.bits >> (code.length - 1 - level as u8)) & 1 == 0 {
                s = self.nodes[nid].rank0(s);
                l = self.nodes[nid].rank0(l);
                h = self.nodes[nid].rank0(h);
            } else {
                s = self.nodes[nid].rank1(s);
                l = self.nodes[nid].rank1(l);
                h = self.nodes[nid].rank1(h);
            }
        }
        (l - s, h - s)
    }

    fn deep_size(&self) -> usize {
        self.nodes.iter().map(|n| n.deep_size()).sum::<usize>()
            + self
                .codes
                .iter()
                .map(|c| c.node_path.len() * 8)
                .sum::<usize>()
            + self.children.len() * 24
    }
}

// ── Suffix Array ─────────────────────────────────────────────────────────────

fn build_suffix_array(text: &[u8]) -> Vec<usize> {
    let n = text.len();
    if n == 0 {
        return Vec::new();
    }
    if n > i32::MAX as usize {
        let mut sa = vec![0i64; n];
        assert_eq!(libsais_rs::libsais64(text, &mut sa, 0, None), 0);
        sa.iter().map(|&x| x as usize).collect()
    } else {
        let mut sa = vec![0i32; n];
        assert_eq!(libsais_rs::libsais(text, &mut sa, 0, None), 0);
        sa.iter().map(|&x| x as usize).collect()
    }
}

// ── Lazy Block Loading ───────────────────────────────────────────────────────

const BLOCK_BITS: usize = BLOCK_WORDS * 64;

struct LazyRankBitVec {
    prefix_ranks: Vec<u64>,
    blocks: Vec<OnceLock<Vec<u64>>>,
    reader: Arc<dyn crate::scalar::IndexReader>,
    block_row_offset: usize,
    len: usize,
}

impl std::fmt::Debug for LazyRankBitVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyRankBitVec")
            .field("len", &self.len)
            .finish()
    }
}

impl LazyRankBitVec {
    fn new(
        prefix_ranks: Vec<u64>,
        num_blocks: usize,
        reader: Arc<dyn crate::scalar::IndexReader>,
        offset: usize,
        len: usize,
    ) -> Self {
        Self {
            prefix_ranks,
            blocks: (0..num_blocks).map(|_| OnceLock::new()).collect(),
            reader,
            block_row_offset: offset,
            len,
        }
    }

    /// Pre-load all blocks into memory. Call this before sync rank/access operations
    /// to avoid the need for `block_in_place` during queries.
    async fn load_all_blocks(&self) -> Result<()> {
        for (idx, lock) in self.blocks.iter().enumerate() {
            if lock.get().is_none() {
                let words = self.load_block(idx).await?;
                let _ = lock.set(words);
            }
        }
        Ok(())
    }

    #[inline]
    fn ensure_block(&self, idx: usize) -> &[u64] {
        self.blocks[idx].get_or_init(|| {
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(self.load_block(idx))
            })
            .unwrap_or_else(|e| panic!("FM-Index block load failed: {e}"))
        })
    }

    async fn load_block(&self, idx: usize) -> Result<Vec<u64>> {
        let row = self.block_row_offset + idx;
        let batch = self
            .reader
            .read_range(row..row + 1, Some(&["words"]))
            .await?;
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::LargeBinaryArray>()
            .ok_or_else(|| Error::invalid_input("expected LargeBinary words column"))?;
        Ok(col
            .value(0)
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    #[inline]
    fn rank1(&self, pos: usize) -> usize {
        if pos == 0 {
            return 0;
        }
        let bi = pos / BLOCK_BITS;
        let local = pos % BLOCK_BITS;
        if local == 0 {
            return self.prefix_ranks[bi] as usize;
        }
        let mut count = self.prefix_ranks[bi] as usize;
        let block = self.ensure_block(bi);
        let wi = local / 64;
        let bit = local % 64;
        for w in &block[..wi] {
            count += w.count_ones() as usize;
        }
        if bit > 0 {
            count += (block[wi] & ((1u64 << bit) - 1)).count_ones() as usize;
        }
        count
    }

    #[inline]
    fn rank0(&self, pos: usize) -> usize {
        pos - self.rank1(pos)
    }

    #[inline]
    fn get(&self, pos: usize) -> bool {
        let bi = pos / BLOCK_BITS;
        let local = pos % BLOCK_BITS;
        let block = self.ensure_block(bi);
        (block[local / 64] >> (local % 64)) & 1 != 0
    }

    fn deep_size(&self) -> usize {
        let loaded: usize = self
            .blocks
            .iter()
            .filter_map(|b| b.get())
            .map(|w| w.len() * 8)
            .sum();
        self.prefix_ranks.len() * 8 + loaded
    }
}

struct LazyHuffmanWaveletTree {
    nodes: Vec<LazyRankBitVec>,
    codes: [HuffmanCode; 256],
    children: Vec<(WaveletChild, WaveletChild)>,
    len: usize,
}

impl std::fmt::Debug for LazyHuffmanWaveletTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyHuffmanWaveletTree")
            .field("len", &self.len)
            .finish()
    }
}

impl LazyHuffmanWaveletTree {
    /// Pre-load all wavelet tree blocks into memory.
    async fn load_all(&self) -> Result<()> {
        for node in &self.nodes {
            node.load_all_blocks().await?;
        }
        Ok(())
    }

    #[inline]
    fn access(&self, mut pos: usize) -> u8 {
        if self.nodes.is_empty() {
            return 0;
        }
        let mut node_idx = 0;
        loop {
            let bit = self.nodes[node_idx].get(pos);
            let (ref left, ref right) = self.children[node_idx];
            if bit {
                pos = self.nodes[node_idx].rank1(pos);
                match right {
                    WaveletChild::Leaf(b) => return *b,
                    WaveletChild::Node(next) => node_idx = *next,
                }
            } else {
                pos = self.nodes[node_idx].rank0(pos);
                match left {
                    WaveletChild::Leaf(b) => return *b,
                    WaveletChild::Node(next) => node_idx = *next,
                }
            }
        }
    }

    #[inline]
    fn rank(&self, c: u8, pos: usize) -> usize {
        let code = &self.codes[c as usize];
        if code.length == 0 {
            return 0;
        }
        let (mut lo, mut hi) = (0, pos);
        for (level, &nid) in code.node_path.iter().enumerate() {
            if (code.bits >> (code.length - 1 - level as u8)) & 1 == 0 {
                lo = self.nodes[nid].rank0(lo);
                hi = self.nodes[nid].rank0(hi);
            } else {
                lo = self.nodes[nid].rank1(lo);
                hi = self.nodes[nid].rank1(hi);
            }
        }
        hi - lo
    }

    #[inline]
    fn rank_pair(&self, c: u8, lo: usize, hi: usize) -> (usize, usize) {
        let code = &self.codes[c as usize];
        if code.length == 0 {
            return (0, 0);
        }
        let (mut s, mut l, mut h) = (0, lo, hi);
        for (level, &nid) in code.node_path.iter().enumerate() {
            if (code.bits >> (code.length - 1 - level as u8)) & 1 == 0 {
                s = self.nodes[nid].rank0(s);
                l = self.nodes[nid].rank0(l);
                h = self.nodes[nid].rank0(h);
            } else {
                s = self.nodes[nid].rank1(s);
                l = self.nodes[nid].rank1(l);
                h = self.nodes[nid].rank1(h);
            }
        }
        (l - s, h - s)
    }

    fn deep_size(&self) -> usize {
        self.nodes.iter().map(|n| n.deep_size()).sum::<usize>()
            + self
                .codes
                .iter()
                .map(|c| c.node_path.len() * 8)
                .sum::<usize>()
    }
}

// ── FM-Index (in-memory, build-time) ─────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct FMIndex {
    wavelet: HuffmanWaveletTree,
    row_ids: Vec<u64>,
    /// Sampled SA: sa_samples[i] = SA[i * SA_SAMPLE_RATE]. Size: N/D × 8 bytes.
    sa_samples: Vec<u64>,
    /// Starting byte offset of each document in the concatenated text.
    doc_start_positions: Vec<u64>,
    c_table: Vec<usize>,
    alphabet_size: usize,
}

impl DeepSizeOf for FMIndex {
    fn deep_size_of_children(&self, _context: &mut lance_core::deepsize::Context) -> usize {
        self.wavelet.deep_size()
            + self.row_ids.len() * 8
            + self.sa_samples.len() * 8
            + self.doc_start_positions.len() * 8
            + self.c_table.len() * std::mem::size_of::<usize>()
    }
}

#[allow(dead_code)]
impl FMIndex {
    fn build(texts: &[(u64, &[u8])]) -> Result<Self> {
        if texts.is_empty() {
            return Ok(Self {
                wavelet: HuffmanWaveletTree {
                    nodes: Vec::new(),
                    codes: std::array::from_fn(|_| HuffmanCode::default()),
                    children: Vec::new(),
                    len: 0,
                },
                row_ids: Vec::new(),
                sa_samples: Vec::new(),
                doc_start_positions: Vec::new(),
                c_table: vec![0; 257],
                alphabet_size: 256,
            });
        }

        let mut concat = Vec::new();
        let mut doc_row_ids = Vec::new();
        let mut doc_starts: Vec<u64> = Vec::new();
        for (row_id, text) in texts {
            doc_starts.push(concat.len() as u64);
            doc_row_ids.push(*row_id);
            concat.extend_from_slice(text);
            concat.push(SENTINEL_BYTE); // \xFF separator between documents
        }
        // Append unique terminator \x00 so SA-IS produces a proper suffix array
        // with a single-cycle LF-mapping permutation.
        concat.push(0x00);
        let n = concat.len();
        let sa = build_suffix_array(&concat);

        let bwt: Vec<u8> = sa
            .iter()
            .map(|&pos| {
                if pos == 0 {
                    concat[n - 1]
                } else {
                    concat[pos - 1]
                }
            })
            .collect();

        let mut counts = vec![0usize; 257];
        for &b in &concat {
            counts[b as usize + 1] += 1;
        }
        for i in 1..257 {
            counts[i] += counts[i - 1];
        }

        // Sampled SA: store every D-th entry
        let sa_samples: Vec<u64> = sa
            .iter()
            .step_by(SA_SAMPLE_RATE)
            .map(|&pos| pos as u64)
            .collect();

        let wavelet = HuffmanWaveletTree::build(&bwt);

        Ok(Self {
            wavelet,
            row_ids: doc_row_ids,
            sa_samples,
            doc_start_positions: doc_starts,
            c_table: counts,
            alphabet_size: 256,
        })
    }

    /// Locate: resolve SA[pos] by walking LF-mapping until hitting a sampled position.
    /// For large data (N >> SA_SAMPLE_RATE), converges within SA_SAMPLE_RATE steps.
    /// For small data with short LF cycles, may need up to N steps.
    #[inline]
    fn locate(&self, mut pos: usize) -> usize {
        let mut steps = 0;
        let n = self.wavelet.len;
        loop {
            if pos.is_multiple_of(SA_SAMPLE_RATE) && (pos / SA_SAMPLE_RATE) < self.sa_samples.len()
            {
                return (self.sa_samples[pos / SA_SAMPLE_RATE] as usize + steps) % n;
            }
            let c = self.wavelet.access(pos);
            pos = self.c_table[c as usize] + self.wavelet.rank(c, pos);
            steps += 1;
            if steps >= n {
                log::warn!("FM-Index SA locate exceeded {n} steps, possible index corruption");
                return 0;
            }
        }
    }

    /// Map a text position to document index via binary search on doc_start_positions.
    #[inline]
    fn doc_for_position(&self, text_pos: usize) -> usize {
        let tp = text_pos as u64;
        match self.doc_start_positions.binary_search(&tp) {
            Ok(idx) => idx,
            Err(idx) => idx - 1,
        }
    }

    fn backward_search(&self, pattern: &[u8]) -> (usize, usize) {
        if pattern.is_empty() || self.wavelet.len == 0 {
            return (0, 0);
        }
        let (mut lo, mut hi) = (0, self.wavelet.len);
        for &b in pattern.iter().rev() {
            let c = self.c_table[b as usize];
            let (occ_lo, occ_hi) = self.wavelet.rank_pair(b, lo, hi);
            lo = c + occ_lo;
            hi = c + occ_hi;
            if lo >= hi {
                return (0, 0);
            }
        }
        (lo, hi)
    }

    #[cfg(test)]
    fn search(&self, pattern: &[u8]) -> RoaringBitmap {
        let (lo, hi) = self.backward_search(pattern);
        if lo >= hi {
            return RoaringBitmap::new();
        }
        let mut result = RoaringBitmap::new();
        for i in lo..hi {
            let text_pos = self.locate(i);
            let doc_idx = self.doc_for_position(text_pos);
            result.insert(self.row_ids[doc_idx] as u32);
        }
        result
    }

    /// Search returning full u64 row addresses (preserving fragment ID in upper bits).
    fn search_row_addrs(&self, pattern: &[u8]) -> Vec<u64> {
        let (lo, hi) = self.backward_search(pattern);
        if lo >= hi {
            return Vec::new();
        }
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        for i in lo..hi {
            let text_pos = self.locate(i);
            let doc_idx = self.doc_for_position(text_pos);
            let row_addr = self.row_ids[doc_idx];
            if seen.insert(row_addr) {
                result.push(row_addr);
            }
        }
        result
    }

    fn serialize_huffman_codes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        for code in &self.wavelet.codes {
            buf.extend_from_slice(&code.bits.to_le_bytes());
            buf.push(code.length);
            buf.extend_from_slice(&(code.node_path.len() as u16).to_le_bytes());
            for &nid in &code.node_path {
                buf.extend_from_slice(&(nid as u32).to_le_bytes());
            }
        }
        buf
    }

    fn deserialize_huffman_codes(data: &[u8]) -> [HuffmanCode; 256] {
        let mut codes: [HuffmanCode; 256] = std::array::from_fn(|_| HuffmanCode::default());
        let mut cur = 0;
        for code in &mut codes {
            let bits = u32::from_le_bytes(data[cur..cur + 4].try_into().unwrap());
            cur += 4;
            let length = data[cur];
            cur += 1;
            let plen = u16::from_le_bytes(data[cur..cur + 2].try_into().unwrap()) as usize;
            cur += 2;
            let mut node_path = Vec::with_capacity(plen);
            for _ in 0..plen {
                node_path.push(u32::from_le_bytes(data[cur..cur + 4].try_into().unwrap()) as usize);
                cur += 4;
            }
            *code = HuffmanCode {
                bits,
                length,
                node_path,
            };
        }
        codes
    }

    fn serialize_tree_topology(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(self.wavelet.children.len() as u32).to_le_bytes());
        for (left, right) in &self.wavelet.children {
            for child in [left, right] {
                match child {
                    WaveletChild::Node(id) => {
                        buf.push(0);
                        buf.extend_from_slice(&(*id as u32).to_le_bytes());
                    }
                    WaveletChild::Leaf(b) => {
                        buf.push(1);
                        buf.extend_from_slice(&(*b as u32).to_le_bytes());
                    }
                }
            }
        }
        buf
    }

    fn deserialize_tree_topology(data: &[u8]) -> Vec<(WaveletChild, WaveletChild)> {
        let mut cur = 0;
        let count = u32::from_le_bytes(data[cur..cur + 4].try_into().unwrap()) as usize;
        cur += 4;
        let mut children = Vec::with_capacity(count);
        for _ in 0..count {
            let mut read_child = || {
                let t = data[cur];
                cur += 1;
                let v = u32::from_le_bytes(data[cur..cur + 4].try_into().unwrap());
                cur += 4;
                if t == 0 {
                    WaveletChild::Node(v as usize)
                } else {
                    WaveletChild::Leaf(v as u8)
                }
            };
            let l = read_child();
            let r = read_child();
            children.push((l, r));
        }
        children
    }

    fn serialize_c_table(&self) -> Vec<u8> {
        self.c_table
            .iter()
            .flat_map(|&v| (v as u64).to_le_bytes())
            .collect()
    }

    fn deserialize_c_table(data: &[u8]) -> Vec<usize> {
        data.chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()) as usize)
            .collect()
    }

    fn u64_to_bytes(data: &[u64]) -> Vec<u8> {
        data.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn build_wavelet_batch(&self) -> Result<RecordBatch> {
        use arrow_array::{LargeBinaryArray, UInt32Array, UInt64Array};
        let mut nid_b = Vec::new();
        let mut bid_b = Vec::new();
        let mut words_b: Vec<Vec<u8>> = Vec::new();
        let mut pr_b = Vec::new();
        let mut bl_b = Vec::new();

        for (i, node) in self.wavelet.nodes.iter().enumerate() {
            let mut pr: u64 = 0;
            if node.words.is_empty() {
                nid_b.push(i as u32);
                bid_b.push(0u32);
                words_b.push(Vec::new());
                pr_b.push(0u64);
                bl_b.push(node.len as u64);
            } else {
                for (bi, chunk) in node.words.chunks(BLOCK_WORDS).enumerate() {
                    nid_b.push(i as u32);
                    bid_b.push(bi as u32);
                    words_b.push(Self::u64_to_bytes(chunk));
                    pr_b.push(pr);
                    bl_b.push(node.len as u64);
                    pr += chunk.iter().map(|w| w.count_ones() as u64).sum::<u64>();
                }
            }
        }
        let refs: Vec<&[u8]> = words_b.iter().map(|v| v.as_slice()).collect();
        let schema = Arc::new(Self::block_schema());
        Ok(RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt32Array::from(nid_b)),
                Arc::new(UInt32Array::from(bid_b)),
                Arc::new(LargeBinaryArray::from(refs)),
                Arc::new(UInt64Array::from(pr_b)),
                Arc::new(UInt64Array::from(bl_b)),
            ],
        )?)
    }

    fn block_schema() -> arrow_schema::Schema {
        arrow_schema::Schema::new(vec![
            Field::new("node_id", DataType::UInt32, false),
            Field::new("block_id", DataType::UInt32, false),
            Field::new("words", DataType::LargeBinary, false),
            Field::new("prefix_rank", DataType::UInt64, false),
            Field::new("bit_len", DataType::UInt64, false),
        ])
    }
}

// ── Lazy FM-Index ────────────────────────────────────────────────────────────

#[derive(Debug)]
struct LazyFMIndex {
    wavelet: LazyHuffmanWaveletTree,
    row_ids: Vec<u64>,
    sa_samples: Vec<u64>,
    doc_start_positions: Vec<u64>,
    c_table: Vec<usize>,
}

impl LazyFMIndex {
    /// Pre-load all wavelet tree blocks before sync search operations.
    async fn prewarm(&self) -> Result<()> {
        self.wavelet.load_all().await
    }

    fn backward_search(&self, pattern: &[u8]) -> (usize, usize) {
        if pattern.is_empty() || self.wavelet.len == 0 {
            return (0, 0);
        }
        let (mut lo, mut hi) = (0, self.wavelet.len);
        for &b in pattern.iter().rev() {
            let c = self.c_table[b as usize];
            let (occ_lo, occ_hi) = self.wavelet.rank_pair(b, lo, hi);
            lo = c + occ_lo;
            hi = c + occ_hi;
            if lo >= hi {
                return (0, 0);
            }
        }
        (lo, hi)
    }

    #[inline]
    fn locate(&self, mut pos: usize) -> usize {
        let mut steps = 0;
        let n = self.wavelet.len;
        loop {
            if pos.is_multiple_of(SA_SAMPLE_RATE) && (pos / SA_SAMPLE_RATE) < self.sa_samples.len()
            {
                return (self.sa_samples[pos / SA_SAMPLE_RATE] as usize + steps) % n;
            }
            let c = self.wavelet.access(pos);
            pos = self.c_table[c as usize] + self.wavelet.rank(c, pos);
            steps += 1;
            if steps >= n {
                log::warn!("FM-Index SA locate exceeded {n} steps, possible index corruption");
                return 0;
            }
        }
    }

    #[inline]
    fn doc_for_position(&self, text_pos: usize) -> usize {
        let tp = text_pos as u64;
        match self.doc_start_positions.binary_search(&tp) {
            Ok(idx) => idx,
            Err(idx) => idx - 1,
        }
    }

    #[cfg(test)]
    fn search(&self, pattern: &[u8]) -> RoaringBitmap {
        let (lo, hi) = self.backward_search(pattern);
        if lo >= hi {
            return RoaringBitmap::new();
        }
        let mut result = RoaringBitmap::new();
        for i in lo..hi {
            let text_pos = self.locate(i);
            let doc_idx = self.doc_for_position(text_pos);
            result.insert(self.row_ids[doc_idx] as u32);
        }
        result
    }

    /// Search returning full u64 row addresses (preserving fragment ID in upper bits).
    fn search_row_addrs(&self, pattern: &[u8]) -> Vec<u64> {
        let (lo, hi) = self.backward_search(pattern);
        if lo >= hi {
            return Vec::new();
        }
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        for i in lo..hi {
            let text_pos = self.locate(i);
            let doc_idx = self.doc_for_position(text_pos);
            let row_addr = self.row_ids[doc_idx];
            if seen.insert(row_addr) {
                result.push(row_addr);
            }
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    async fn from_reader(
        reader: Arc<dyn crate::scalar::IndexReader>,
        num_bwt_nodes: usize,
        huffman_codes: [HuffmanCode; 256],
        children: Vec<(WaveletChild, WaveletChild)>,
        c_table: Vec<usize>,
        bwt_len: usize,
        total_wavelet_rows: usize,
        num_sa_blocks: usize,
        sa_samples_len: usize,
        row_ids: Vec<u64>,
        doc_start_positions: Vec<u64>,
    ) -> Result<Self> {
        use arrow_array::UInt64Array;

        let meta = reader
            .read_range(
                0..total_wavelet_rows,
                Some(&["node_id", "prefix_rank", "bit_len"]),
            )
            .await?;
        let nid_col = meta
            .column_by_name("node_id")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::UInt32Array>()
            .unwrap();
        let pr_col = meta
            .column_by_name("prefix_rank")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let bl_col = meta
            .column_by_name("bit_len")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();

        struct NM {
            prs: Vec<u64>,
            offset: usize,
            blen: usize,
        }
        let mut nms: Vec<NM> = (0..num_bwt_nodes)
            .map(|_| NM {
                prs: Vec::new(),
                offset: 0,
                blen: 0,
            })
            .collect();
        for row in 0..meta.num_rows() {
            let nid = nid_col.value(row) as usize;
            if nid >= num_bwt_nodes {
                continue;
            }
            let nm = &mut nms[nid];
            if nm.prs.is_empty() {
                nm.offset = row;
            }
            nm.prs.push(pr_col.value(row));
            nm.blen = bl_col.value(row) as usize;
        }

        let mut bwt_nodes = Vec::with_capacity(num_bwt_nodes);
        for nm in &nms {
            bwt_nodes.push(LazyRankBitVec::new(
                nm.prs.clone(),
                nm.prs.len(),
                reader.clone(),
                nm.offset,
                nm.blen,
            ));
        }
        let wavelet = LazyHuffmanWaveletTree {
            nodes: bwt_nodes,
            codes: huffman_codes,
            children,
            len: bwt_len,
        };

        // Read SA samples from packed binary blocks
        let mut sa_samples = Vec::with_capacity(sa_samples_len);
        let sa_batch = reader
            .read_range(
                total_wavelet_rows..total_wavelet_rows + num_sa_blocks,
                Some(&["words"]),
            )
            .await?;
        let words_col = sa_batch
            .column_by_name("words")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::LargeBinaryArray>()
            .unwrap();
        for i in 0..sa_batch.num_rows() {
            let raw = words_col.value(i);
            for chunk in raw.chunks_exact(8) {
                sa_samples.push(u64::from_le_bytes(chunk.try_into().unwrap()));
            }
        }
        sa_samples.truncate(sa_samples_len);

        Ok(Self {
            wavelet,
            row_ids,
            sa_samples,
            doc_start_positions,
            c_table,
        })
    }

    fn deep_size(&self) -> usize {
        self.wavelet.deep_size()
            + self.row_ids.len() * 8
            + self.sa_samples.len() * 8
            + self.doc_start_positions.len() * 8
            + self.c_table.len() * std::mem::size_of::<usize>()
    }
}

// ── FMIndexScalarIndex ───────────────────────────────────────────────────────

#[derive(Debug)]
struct FMIndexPartition {
    #[allow(dead_code)]
    id: u64,
    fm: LazyFMIndex,
}

#[derive(Debug)]
pub struct FMIndexScalarIndex {
    partitions: Vec<Arc<FMIndexPartition>>,
}

impl DeepSizeOf for FMIndexScalarIndex {
    fn deep_size_of_children(&self, _ctx: &mut lance_core::deepsize::Context) -> usize {
        self.partitions.iter().map(|p| p.fm.deep_size()).sum()
    }
}

impl FMIndexScalarIndex {
    async fn load_partition(
        store: &dyn IndexStore,
        filename: &str,
        pid: u64,
    ) -> Result<FMIndexPartition> {
        let reader = store.open_index_file(filename).await?;
        let md = &reader.schema().metadata;

        let parse = |key: &str| -> Result<usize> {
            md.get(key)
                .ok_or_else(|| Error::invalid_input(format!("missing {key}")))?
                .parse()
                .map_err(|e| Error::invalid_input(format!("invalid {key}: {e}")))
        };

        let num_bwt_nodes = parse("num_bwt_nodes")?;
        let bwt_len = parse("bwt_len")?;
        let num_sa_blocks = parse("num_sa_blocks")?;
        let sa_samples_len = parse("sa_samples_len")?;
        let total_wavelet_rows = parse("total_wavelet_rows")?;

        let c_table = FMIndex::deserialize_c_table(&hex_decode(
            md.get("c_table")
                .ok_or_else(|| Error::invalid_input("missing c_table"))?,
        )?);
        let huffman_codes = FMIndex::deserialize_huffman_codes(&hex_decode(
            md.get("huffman_codes")
                .ok_or_else(|| Error::invalid_input("missing huffman_codes"))?,
        )?);
        let children = FMIndex::deserialize_tree_topology(&hex_decode(
            md.get("tree_topology")
                .ok_or_else(|| Error::invalid_input("missing tree_topology"))?,
        )?);

        // row_ids and doc_start_positions stored in metadata (small)
        let row_ids_hex = md
            .get("row_ids")
            .ok_or_else(|| Error::invalid_input("missing row_ids"))?;
        let row_ids_bytes = hex_decode(row_ids_hex)?;
        let row_ids: Vec<u64> = row_ids_bytes
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect();

        let doc_starts_hex = md
            .get("doc_start_positions")
            .ok_or_else(|| Error::invalid_input("missing doc_start_positions"))?;
        let doc_starts_bytes = hex_decode(doc_starts_hex)?;
        let doc_start_positions: Vec<u64> = doc_starts_bytes
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect();

        let fm = Box::pin(LazyFMIndex::from_reader(
            reader,
            num_bwt_nodes,
            huffman_codes,
            children,
            c_table,
            bwt_len,
            total_wavelet_rows,
            num_sa_blocks,
            sa_samples_len,
            row_ids,
            doc_start_positions,
        ))
        .await?;
        Ok(FMIndexPartition { id: pid, fm })
    }

    async fn load(
        store: Arc<dyn IndexStore>,
        _fri: Option<Arc<FragReuseIndex>>,
        _cache: &LanceCache,
    ) -> Result<Arc<Self>> {
        let files = store.list_files_with_sizes().await?;
        let mut pfiles: Vec<(u64, String)> = Vec::new();
        for f in &files {
            if let Some(id) = f
                .path
                .strip_prefix("part_")
                .and_then(|r| r.strip_suffix("_fm.lance"))
                .and_then(|s| s.parse::<u64>().ok())
            {
                pfiles.push((id, f.path.clone()));
            }
        }
        if pfiles.is_empty() {
            return Err(Error::invalid_input("no FM-Index partition files found"));
        }
        pfiles.sort_by_key(|(id, _)| *id);
        let mut parts = Vec::with_capacity(pfiles.len());
        for (id, name) in &pfiles {
            parts.push(Arc::new(
                Self::load_partition(store.as_ref(), name, *id).await?,
            ));
        }
        Ok(Arc::new(Self { partitions: parts }))
    }
}

#[async_trait]
impl Index for FMIndexScalarIndex {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }
    async fn prewarm(&self) -> Result<()> {
        Ok(())
    }
    fn statistics(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "type": "Fm",
            "num_partitions": self.partitions.len(),
            "total_bwt_len": self.partitions.iter().map(|p| p.fm.wavelet.len).sum::<usize>(),
            "total_docs": self.partitions.iter().map(|p| p.fm.row_ids.len()).sum::<usize>(),
        }))
    }
    fn index_type(&self) -> IndexType {
        IndexType::Fm
    }
    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        let mut frags = RoaringBitmap::new();
        for p in &self.partitions {
            for &rid in &p.fm.row_ids {
                frags.insert((rid >> 32) as u32);
            }
        }
        Ok(frags)
    }
}

#[async_trait]
impl ScalarIndex for FMIndexScalarIndex {
    async fn search(
        &self,
        query: &dyn AnyQuery,
        _metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let tq = query
            .as_any()
            .downcast_ref::<TextQuery>()
            .ok_or_else(|| Error::invalid_input("Fm only supports TextQuery"))?;
        match tq {
            TextQuery::StringContains(pattern) => {
                let pb = pattern.as_bytes();
                use lance_select::RowAddrTreeMap;
                let mut tree = RowAddrTreeMap::new();
                for p in &self.partitions {
                    p.fm.prewarm().await?;
                    for row_addr in p.fm.search_row_addrs(pb) {
                        tree.insert(row_addr);
                    }
                }
                Ok(SearchResult::Exact(lance_select::NullableRowAddrSet::new(
                    tree,
                    Default::default(),
                )))
            }
            // Regex queries are routed only to the ngram index (the FM-index's
            // query parser advertises `supports_regex = false`), so this is
            // unreachable in practice; reject it explicitly rather than silently.
            TextQuery::Regex(_) => Err(Error::invalid_input(
                "FMIndex does not support regular expression queries",
            )),
        }
    }
    fn can_remap(&self) -> bool {
        false
    }
    async fn remap(
        &self,
        _: &HashMap<u64, Option<u64>>,
        _: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        Err(Error::not_supported("Fm does not support remap"))
    }
    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest: &dyn IndexStore,
        _old_data_filter: Option<OldIndexDataFilter>,
    ) -> Result<CreatedIndex> {
        let files = write_partitioned_fmindex_stream(new_data, dest).await?;
        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pb::FmIndexIndexDetails {}).unwrap(),
            index_version: FMINDEX_INDEX_VERSION,
            files,
        })
    }
    fn update_criteria(&self) -> UpdateCriteria {
        UpdateCriteria::requires_old_data(
            TrainingCriteria::new(TrainingOrdering::None).with_row_addr(),
        )
    }
    fn derive_index_params(&self) -> Result<ScalarIndexParams> {
        Ok(ScalarIndexParams::for_builtin(BuiltinIndexType::Fm))
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

async fn write_partitioned_fmindex_stream(
    mut stream: SendableRecordBatchStream,
    store: &dyn IndexStore,
) -> Result<Vec<IndexFile>> {
    let mut files = Vec::new();
    let mut partition = Vec::with_capacity(PARTITION_SIZE);
    let mut partition_id = 0;

    while let Some(batch) = stream.next().await {
        let batch = batch?;
        // Prefer _rowaddr (global row address) over _rowid to ensure stable,
        // globally unique identifiers across segments.
        let row_addrs: &arrow_array::UInt64Array = batch
            .column_by_name(ROW_ADDR)
            .or_else(|| batch.column_by_name("_rowid"))
            .and_then(|c| c.as_any().downcast_ref())
            .ok_or_else(|| {
                Error::invalid_input("Fm training data must include _rowaddr or _rowid column")
            })?;
        // Use the named value column; fall back to column(0) for legacy streams
        let value_col = batch
            .column_by_name(VALUE_COLUMN_NAME)
            .unwrap_or_else(|| batch.column(0));
        for i in 0..batch.num_rows() {
            let rid = row_addrs.value(i);
            if let Some(bytes) = extract_sanitized_text_bytes(value_col.as_ref(), i)? {
                partition.push((rid, bytes));
                if partition.len() == PARTITION_SIZE {
                    files.push(write_fmindex_partition(&partition, store, partition_id).await?);
                    partition.clear();
                    partition_id += 1;
                }
            }
        }
    }

    if !partition.is_empty() {
        files.push(write_fmindex_partition(&partition, store, partition_id).await?);
    } else if files.is_empty() {
        files.push(write_empty_fmindex_partition(store).await?);
    }

    Ok(files)
}

fn sanitize_text_bytes(bytes: &[u8]) -> Vec<u8> {
    bytes
        .iter()
        .map(|&b| {
            if b == SENTINEL_BYTE || b == 0x00 {
                b' '
            } else {
                b
            }
        })
        .collect()
}

fn extract_sanitized_text_bytes(
    array: &dyn arrow_array::Array,
    index: usize,
) -> Result<Option<Vec<u8>>> {
    if array.is_null(index) {
        return Ok(None);
    }
    match array.data_type() {
        DataType::Utf8 => Ok(Some(sanitize_text_bytes(
            array
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .unwrap()
                .value(index)
                .as_bytes(),
        ))),
        DataType::LargeUtf8 => Ok(Some(sanitize_text_bytes(
            array
                .as_any()
                .downcast_ref::<arrow_array::LargeStringArray>()
                .unwrap()
                .value(index)
                .as_bytes(),
        ))),
        DataType::Binary => Ok(Some(sanitize_text_bytes(
            array
                .as_any()
                .downcast_ref::<arrow_array::BinaryArray>()
                .unwrap()
                .value(index),
        ))),
        DataType::LargeBinary => Ok(Some(sanitize_text_bytes(
            array
                .as_any()
                .downcast_ref::<arrow_array::LargeBinaryArray>()
                .unwrap()
                .value(index),
        ))),
        _ => Err(Error::invalid_input(format!(
            "Fm does not support data type: {:?}",
            array.data_type()
        ))),
    }
}

#[cfg(test)]
fn extract_text_bytes(array: &dyn arrow_array::Array, index: usize) -> Result<Option<Vec<u8>>> {
    if array.is_null(index) {
        return Ok(None);
    }
    match array.data_type() {
        DataType::Utf8 => Ok(Some(
            array
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .unwrap()
                .value(index)
                .as_bytes()
                .to_vec(),
        )),
        DataType::LargeUtf8 => Ok(Some(
            array
                .as_any()
                .downcast_ref::<arrow_array::LargeStringArray>()
                .unwrap()
                .value(index)
                .as_bytes()
                .to_vec(),
        )),
        DataType::Binary => Ok(Some(
            array
                .as_any()
                .downcast_ref::<arrow_array::BinaryArray>()
                .unwrap()
                .value(index)
                .to_vec(),
        )),
        DataType::LargeBinary => Ok(Some(
            array
                .as_any()
                .downcast_ref::<arrow_array::LargeBinaryArray>()
                .unwrap()
                .value(index)
                .to_vec(),
        )),
        _ => Err(Error::invalid_input(format!(
            "Fm does not support data type: {:?}",
            array.data_type()
        ))),
    }
}

fn hex_encode(data: &[u8]) -> String {
    data.iter().map(|b| format!("{b:02x}")).collect()
}
fn hex_decode(s: &str) -> Result<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return Err(Error::invalid_input("invalid hex length"));
    }
    (0..s.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&s[i..i + 2], 16)
                .map_err(|e| Error::invalid_input(format!("invalid hex: {e}")))
        })
        .collect()
}

/// Write an FM-Index partition to storage.
///
/// Layout:
///   - Wavelet block rows (BWT nodes)
///   - SA sample blocks (packed u64 in LargeBinary)
///   - Metadata: c_table, huffman_codes, tree_topology, row_ids, doc_start_positions
async fn write_fmindex(fm: &FMIndex, store: &dyn IndexStore, filename: &str) -> Result<IndexFile> {
    let schema = Arc::new(FMIndex::block_schema());

    let mut writer = store.new_index_file(filename, schema.clone()).await?;

    // 1. Wavelet blocks
    let wb = fm.build_wavelet_batch()?;
    let nw = wb.num_rows();
    writer.write_record_batch(wb).await?;

    // 2. SA samples packed as binary blocks
    let u64s_per_block = BLOCK_WORDS; // 4096 u64s per block = 32KB
    let mut sa_nid = Vec::new();
    let mut sa_bid = Vec::new();
    let mut sa_words: Vec<Vec<u8>> = Vec::new();
    let mut sa_pr = Vec::new();
    let mut sa_bl = Vec::new();
    for (bi, chunk) in fm.sa_samples.chunks(u64s_per_block).enumerate() {
        sa_nid.push(u32::MAX);
        sa_bid.push(bi as u32);
        sa_words.push(FMIndex::u64_to_bytes(chunk));
        sa_pr.push(0u64);
        sa_bl.push(fm.sa_samples.len() as u64);
    }
    let num_sa_blocks = sa_nid.len();
    if num_sa_blocks > 0 {
        let refs: Vec<&[u8]> = sa_words.iter().map(|v| v.as_slice()).collect();
        writer
            .write_record_batch(RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(arrow_array::UInt32Array::from(sa_nid)),
                    Arc::new(arrow_array::UInt32Array::from(sa_bid)),
                    Arc::new(arrow_array::LargeBinaryArray::from(refs)),
                    Arc::new(arrow_array::UInt64Array::from(sa_pr)),
                    Arc::new(arrow_array::UInt64Array::from(sa_bl)),
                ],
            )?)
            .await?;
    }

    // Metadata
    let mut metadata = HashMap::new();
    metadata.insert("num_bwt_nodes".into(), fm.wavelet.nodes.len().to_string());
    metadata.insert("bwt_len".into(), fm.wavelet.len.to_string());
    metadata.insert("num_sa_blocks".into(), num_sa_blocks.to_string());
    metadata.insert("sa_samples_len".into(), fm.sa_samples.len().to_string());
    metadata.insert("total_wavelet_rows".into(), nw.to_string());
    metadata.insert("sa_sample_rate".into(), SA_SAMPLE_RATE.to_string());
    metadata.insert("alphabet_size".into(), fm.alphabet_size.to_string());
    metadata.insert("c_table".into(), hex_encode(&fm.serialize_c_table()));
    metadata.insert(
        "huffman_codes".into(),
        hex_encode(&fm.serialize_huffman_codes()),
    );
    metadata.insert(
        "tree_topology".into(),
        hex_encode(&fm.serialize_tree_topology()),
    );
    // row_ids in metadata (10K × 8 = 80KB per partition — small)
    let row_ids_bytes: Vec<u8> = fm.row_ids.iter().flat_map(|&v| v.to_le_bytes()).collect();
    metadata.insert("row_ids".into(), hex_encode(&row_ids_bytes));
    // doc_start_positions in metadata (10K × 8 = 80KB per partition — small)
    let doc_starts_bytes: Vec<u8> = fm
        .doc_start_positions
        .iter()
        .flat_map(|&v| v.to_le_bytes())
        .collect();
    metadata.insert("doc_start_positions".into(), hex_encode(&doc_starts_bytes));

    writer.finish_with_metadata(metadata).await
}

#[cfg(test)]
async fn write_partitioned_fmindex(
    texts: &[(u64, Vec<u8>)],
    store: &dyn IndexStore,
) -> Result<Vec<IndexFile>> {
    if texts.is_empty() {
        return Ok(vec![write_empty_fmindex_partition(store).await?]);
    }
    let mut files = Vec::new();
    for (pid, chunk) in texts.chunks(PARTITION_SIZE).enumerate() {
        files.push(write_fmindex_partition(chunk, store, pid as u64).await?);
    }
    Ok(files)
}

async fn write_fmindex_partition(
    texts: &[(u64, Vec<u8>)],
    store: &dyn IndexStore,
    partition_id: u64,
) -> Result<IndexFile> {
    let refs: Vec<(u64, &[u8])> = texts.iter().map(|(id, t)| (*id, t.as_slice())).collect();
    let fm = FMIndex::build(&refs)?;
    write_fmindex(&fm, store, &fmindex_partition_path(partition_id)).await
}

async fn write_empty_fmindex_partition(store: &dyn IndexStore) -> Result<IndexFile> {
    let fm = FMIndex::build(&[])?;
    write_fmindex(&fm, store, &fmindex_partition_path(0)).await
}

// ── Plugin ───────────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct FMIndexPlugin;

#[async_trait]
impl ScalarIndexPlugin for FMIndexPlugin {
    fn name(&self) -> &str {
        "Fm"
    }
    fn new_training_request(
        &self,
        _params: &str,
        field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        match field.data_type() {
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary => {}
            _ => {
                return Err(Error::invalid_input(format!(
                    "FM-Index does not support {:?}",
                    field.data_type()
                )));
            }
        }
        Ok(Box::new(DefaultTrainingRequest::new(
            TrainingCriteria::new(TrainingOrdering::None).with_row_addr(),
        )))
    }
    async fn train_index(
        &self,
        data: SendableRecordBatchStream,
        store: &dyn IndexStore,
        _req: Box<dyn TrainingRequest>,
        _fids: Option<Vec<u32>>,
        _progress: Arc<dyn crate::progress::IndexBuildProgress>,
    ) -> Result<CreatedIndex> {
        let files = write_partitioned_fmindex_stream(data, store).await?;
        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pb::FmIndexIndexDetails {}).unwrap(),
            index_version: FMINDEX_INDEX_VERSION,
            files,
        })
    }
    fn provides_exact_answer(&self) -> bool {
        true
    }
    fn version(&self) -> u32 {
        FMINDEX_INDEX_VERSION
    }
    fn new_query_parser(
        &self,
        index_name: String,
        _details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        Some(Box::new(TextQueryParser::new(
            index_name,
            self.name().to_string(),
            // needs_recheck: the FM-index returns exact substring matches.
            false,
            // supports_regex: regex acceleration is only implemented for ngram.
            false,
        )))
    }
    async fn load_index(
        &self,
        store: Arc<dyn IndexStore>,
        details: &prost_types::Any,
        fri: Option<Arc<FragReuseIndex>>,
        cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        let _ = details
            .to_msg::<pb::FmIndexIndexDetails>()
            .unwrap_or_default();
        Ok(FMIndexScalarIndex::load(store, fri, cache).await? as Arc<dyn ScalarIndex>)
    }
    async fn load_statistics(
        &self,
        _: Arc<dyn IndexStore>,
        _: &prost_types::Any,
    ) -> Result<Option<serde_json::Value>> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{BinaryArray, LargeBinaryArray, LargeStringArray, StringArray, UInt64Array};
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use futures::stream;
    use lance_core::{ROW_ADDR, cache::LanceCache};
    use lance_io::object_store::ObjectStore;
    use object_store::path::Path;
    use std::sync::Arc;

    use crate::scalar::lance_format::LanceIndexStore;

    #[test]
    fn test_fmindex_build_and_search() {
        let texts: Vec<(u64, &[u8])> = vec![
            (0, b"hello world"),
            (1, b"hello rust"),
            (2, b"goodbye world"),
        ];
        let fm = FMIndex::build(&texts).unwrap();

        let r = fm.search(b"hello");
        assert!(r.contains(0));
        assert!(r.contains(1));
        assert!(!r.contains(2));

        let r = fm.search(b"world");
        assert!(r.contains(0));
        assert!(!r.contains(1));
        assert!(r.contains(2));

        let r = fm.search(b"goodbye");
        assert!(!r.contains(0));
        assert!(!r.contains(1));
        assert!(r.contains(2));

        assert!(fm.search(b"xyz").is_empty());
    }

    #[test]
    fn test_fmindex_empty() {
        let fm = FMIndex::build(&[]).unwrap();
        assert!(fm.search(b"anything").is_empty());
    }

    #[test]
    fn test_fmindex_single_char_search() {
        let texts: Vec<(u64, &[u8])> = vec![(0, b"abc"), (1, b"def")];
        let fm = FMIndex::build(&texts).unwrap();
        assert!(fm.search(b"a").contains(0));
        assert!(!fm.search(b"a").contains(1));
        assert!(!fm.search(b"d").contains(0));
        assert!(fm.search(b"d").contains(1));
    }

    #[test]
    fn test_fmindex_repeated_pattern() {
        let texts: Vec<(u64, &[u8])> = vec![(0, b"ababab"), (1, b"cdcd")];
        let fm = FMIndex::build(&texts).unwrap();
        assert!(fm.search(b"ab").contains(0));
        assert!(!fm.search(b"ab").contains(1));
        assert!(!fm.search(b"cd").contains(0));
        assert!(fm.search(b"cd").contains(1));
    }

    #[test]
    fn test_early_exit_all_docs_match() {
        let texts: Vec<(u64, &[u8])> = vec![(0, b"the cat"), (1, b"the dog"), (2, b"the bird")];
        let fm = FMIndex::build(&texts).unwrap();
        assert_eq!(fm.search(b"the").len(), 3);
    }

    #[test]
    fn test_locate_correctness() {
        let texts: Vec<(u64, &[u8])> = vec![
            (0, b"the quick brown fox jumps over the lazy dog"),
            (1, b"pack my box with five dozen liquor jugs"),
            (2, b"how vexingly quick daft zebras jump"),
        ];
        let fm = FMIndex::build(&texts).unwrap();

        let r = fm.search(b"quick");
        assert!(r.contains(0));
        assert!(!r.contains(1));
        assert!(r.contains(2));

        let r = fm.search(b"the");
        assert!(r.contains(0));
        assert!(!r.contains(1));
        assert!(!r.contains(2));

        let r = fm.search(b"jump");
        assert!(r.contains(0));
        assert!(r.contains(2));
    }

    #[test]
    fn test_many_documents() {
        let docs: Vec<Vec<u8>> = (0..100)
            .map(|i| format!("document number {} with hello world data xyz", i).into_bytes())
            .collect();
        let texts: Vec<(u64, &[u8])> = docs
            .iter()
            .enumerate()
            .map(|(i, d)| (i as u64, d.as_slice()))
            .collect();
        let fm = FMIndex::build(&texts).unwrap();

        assert_eq!(fm.search(b"hello world").len(), 100);
        assert_eq!(fm.search(b"document number 42").len(), 1);
        assert_eq!(fm.search(b"nonexistent").len(), 0);
    }

    #[test]
    fn test_index_size_ratio() {
        let docs: Vec<Vec<u8>> = (0..200)
            .map(|i| {
                format!(
                    "document {} with enough text to test size ratio properly end",
                    i
                )
                .into_bytes()
            })
            .collect();
        let texts: Vec<(u64, &[u8])> = docs
            .iter()
            .enumerate()
            .map(|(i, d)| (i as u64, d.as_slice()))
            .collect();
        let fm = FMIndex::build(&texts).unwrap();

        let text_size: usize = docs.iter().map(|d| d.len()).sum();
        let wavelet_size = fm.wavelet.deep_size();
        let sa_size = fm.sa_samples.len() * 8;
        let total = wavelet_size + sa_size;

        let ratio = total as f64 / text_size as f64;
        assert!(
            ratio < 1.5,
            "index should be much smaller than text, got ratio={ratio:.2}"
        );
    }

    #[test]
    fn test_wavelet_access_consistency() {
        let docs: Vec<Vec<u8>> = (0..50)
            .map(|i| format!("document {i} hello world test").into_bytes())
            .collect();
        let texts: Vec<(u64, &[u8])> = docs
            .iter()
            .enumerate()
            .map(|(i, d)| (i as u64, d.as_slice()))
            .collect();

        let mut concat = Vec::new();
        for (_, text) in &texts {
            concat.extend_from_slice(text);
            concat.push(SENTINEL_BYTE);
        }
        concat.push(0x00);
        let sa = build_suffix_array(&concat);
        let n = concat.len();
        let bwt: Vec<u8> = sa
            .iter()
            .map(|&pos| {
                if pos == 0 {
                    concat[n - 1]
                } else {
                    concat[pos - 1]
                }
            })
            .collect();
        let wavelet = HuffmanWaveletTree::build(&bwt);

        for (i, &expected) in bwt.iter().enumerate().take(n.min(500)) {
            assert_eq!(wavelet.access(i), expected, "access mismatch at {i}");
        }
    }

    #[test]
    fn test_serialization_roundtrip() {
        let texts: Vec<(u64, &[u8])> = vec![
            (10, b"alpha beta gamma"),
            (20, b"beta gamma delta"),
            (30, b"gamma delta epsilon"),
        ];
        let fm = FMIndex::build(&texts).unwrap();

        // Test huffman codes roundtrip
        let hc_bytes = fm.serialize_huffman_codes();
        let hc = FMIndex::deserialize_huffman_codes(&hc_bytes);
        for (i, (loaded, original)) in hc.iter().zip(fm.wavelet.codes.iter()).enumerate() {
            assert_eq!(loaded.bits, original.bits, "bits mismatch at {i}");
            assert_eq!(loaded.length, original.length, "length mismatch at {i}");
            assert_eq!(loaded.node_path, original.node_path, "path mismatch at {i}");
        }

        // Test tree topology roundtrip
        let topo_bytes = fm.serialize_tree_topology();
        let topo = FMIndex::deserialize_tree_topology(&topo_bytes);
        assert_eq!(topo.len(), fm.wavelet.children.len());

        // Test c_table roundtrip
        let ct_bytes = fm.serialize_c_table();
        let ct = FMIndex::deserialize_c_table(&ct_bytes);
        assert_eq!(ct, fm.c_table);
    }

    #[test]
    fn test_hex_roundtrip() {
        let data = vec![0u8, 1, 127, 255, 42];
        let encoded = hex_encode(&data);
        let decoded = hex_decode(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn test_sentinel_sanitization() {
        // Text containing \xFF should be sanitized to space during training.
        let texts: Vec<(u64, &[u8])> = vec![(0, b"hello\xFFworld")];
        let fm = FMIndex::build(&texts).unwrap();
        // Build itself does not sanitize, but search should still work.
        let r = fm.search(b"hello");
        assert!(r.contains(0));
    }

    #[test]
    fn test_wavelet_rank_pair_consistency() {
        let docs: Vec<Vec<u8>> = (0..30)
            .map(|i| format!("doc {i} with repeated words hello world test data").into_bytes())
            .collect();
        let texts: Vec<(u64, &[u8])> = docs
            .iter()
            .enumerate()
            .map(|(i, d)| (i as u64, d.as_slice()))
            .collect();
        let fm = FMIndex::build(&texts).unwrap();

        let n = fm.wavelet.len;
        for b in [b'a', b'e', b' ', SENTINEL_BYTE] {
            for &(lo, hi) in &[(0usize, 1usize), (0, n), (n / 4, n / 2)] {
                if lo >= n || hi > n || lo >= hi {
                    continue;
                }
                let (pl, ph) = fm.wavelet.rank_pair(b, lo, hi);
                let rl = fm.wavelet.rank(b, lo);
                let rh = fm.wavelet.rank(b, hi);
                assert_eq!(pl, rl, "rank_pair lo mismatch for b={b} [{lo},{hi})");
                assert_eq!(ph, rh, "rank_pair hi mismatch for b={b} [{lo},{hi})");
            }
        }
    }

    #[test]
    fn test_large_sa_sampling() {
        // Test with enough documents to have multiple SA sample points
        let docs: Vec<Vec<u8>> = (0..50)
            .map(|i| {
                format!(
                    "document number {} with lots of text to ensure we have enough bytes for multiple SA samples across the suffix array positions",
                    i
                )
                .into_bytes()
            })
            .collect();
        let texts: Vec<(u64, &[u8])> = docs
            .iter()
            .enumerate()
            .map(|(i, d)| (i as u64, d.as_slice()))
            .collect();
        let fm = FMIndex::build(&texts).unwrap();

        assert!(fm.sa_samples.len() > 1, "should have multiple SA samples");
        assert_eq!(fm.search(b"document number 25").len(), 1);
        assert_eq!(fm.search(b"document number").len(), 50);
        assert_eq!(fm.search(b"nonexistent pattern").len(), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_write_and_load_roundtrip() {
        let texts: Vec<(u64, &[u8])> = vec![
            (0, b"hello world foo bar"),
            (1, b"hello rust baz qux"),
            (2, b"goodbye world quux"),
        ];
        let fm = FMIndex::build(&texts).unwrap();

        let tempdir = tempfile::tempdir().unwrap();
        let index_dir = Path::from_filesystem_path(tempdir.path()).unwrap();
        let store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            index_dir,
            Arc::new(LanceCache::no_cache()),
        ));

        // Write
        write_fmindex(&fm, store.as_ref(), &fmindex_partition_path(0))
            .await
            .unwrap();

        // Load
        let part =
            FMIndexScalarIndex::load_partition(store.as_ref(), &fmindex_partition_path(0), 0)
                .await
                .unwrap();

        // Verify search results match
        let r = part.fm.search(b"hello");
        assert!(r.contains(0));
        assert!(r.contains(1));
        assert!(!r.contains(2));

        let r = part.fm.search(b"world");
        assert!(r.contains(0));
        assert!(!r.contains(1));
        assert!(r.contains(2));

        assert!(part.fm.search(b"xyz").is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_partitioned_write_and_load() {
        let docs: Vec<Vec<u8>> = (0..30)
            .map(|i| format!("document {i} hello world test data").into_bytes())
            .collect();
        let texts: Vec<(u64, Vec<u8>)> = docs
            .into_iter()
            .enumerate()
            .map(|(i, d)| (i as u64, d))
            .collect();

        let tempdir = tempfile::tempdir().unwrap();
        let index_dir = Path::from_filesystem_path(tempdir.path()).unwrap();
        let store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            index_dir,
            Arc::new(LanceCache::no_cache()),
        ));

        write_partitioned_fmindex(&texts, store.as_ref())
            .await
            .unwrap();

        let index = FMIndexScalarIndex::load(store, None, &LanceCache::no_cache())
            .await
            .unwrap();

        // Search across partitions
        let r = index
            .search(
                &TextQuery::StringContains("hello world".to_string()),
                &crate::metrics::NoOpMetricsCollector,
            )
            .await
            .unwrap();
        match r {
            SearchResult::Exact(set) => {
                assert_eq!(set.len(), Some(30));
            }
            _ => panic!("expected exact result"),
        }

        let r = index
            .search(
                &TextQuery::StringContains("document 15".to_string()),
                &crate::metrics::NoOpMetricsCollector,
            )
            .await
            .unwrap();
        match r {
            SearchResult::Exact(set) => {
                assert_eq!(set.len(), Some(1));
            }
            _ => panic!("expected exact result"),
        }

        let r = index
            .search(
                &TextQuery::StringContains("nonexistent".to_string()),
                &crate::metrics::NoOpMetricsCollector,
            )
            .await
            .unwrap();
        match r {
            SearchResult::Exact(set) => {
                assert_eq!(set.len(), Some(0));
            }
            _ => panic!("expected exact result"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_plugin_train_and_load() {
        let docs = vec!["hello world", "hello rust", "goodbye world"];
        let row_addrs: Vec<u64> = vec![0, 1, 2];
        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new(
                crate::scalar::registry::VALUE_COLUMN_NAME,
                DataType::Utf8,
                false,
            ),
            arrow_schema::Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(docs)),
                Arc::new(UInt64Array::from(row_addrs)),
            ],
        )
        .unwrap();

        let tempdir = tempfile::tempdir().unwrap();
        let index_dir = Path::from_filesystem_path(tempdir.path()).unwrap();
        let store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            index_dir,
            Arc::new(LanceCache::no_cache()),
        ));

        let stream = RecordBatchStreamAdapter::new(batch.schema(), stream::iter(vec![Ok(batch)]));
        let req = FMIndexPlugin
            .new_training_request("", &arrow_schema::Field::new("val", DataType::Utf8, false))
            .unwrap();
        let created = FMIndexPlugin
            .train_index(
                Box::pin(stream),
                store.as_ref(),
                req,
                None,
                Arc::new(crate::progress::NoopIndexBuildProgress),
            )
            .await
            .unwrap();

        let index = FMIndexPlugin
            .load_index(store, &created.index_details, None, &LanceCache::no_cache())
            .await
            .unwrap();

        let r = index
            .search(
                &TextQuery::StringContains("hello".to_string()),
                &crate::metrics::NoOpMetricsCollector,
            )
            .await
            .unwrap();
        match r {
            SearchResult::Exact(set) => {
                assert_eq!(set.len(), Some(2));
            }
            _ => panic!("expected exact result"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_plugin_train_streams_multiple_partitions() {
        fn training_batch(
            schema: Arc<arrow_schema::Schema>,
            start: usize,
            len: usize,
        ) -> RecordBatch {
            let docs = vec!["x"; len];
            let row_addrs: Vec<u64> = (start..start + len).map(|i| i as u64).collect();
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(docs)),
                    Arc::new(UInt64Array::from(row_addrs)),
                ],
            )
            .unwrap()
        }

        let total_rows = PARTITION_SIZE + 5;
        let first_batch_rows = PARTITION_SIZE - 3;
        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new(
                crate::scalar::registry::VALUE_COLUMN_NAME,
                DataType::Utf8,
                false,
            ),
            arrow_schema::Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));
        let batches = vec![
            Ok(training_batch(schema.clone(), 0, first_batch_rows)),
            Ok(training_batch(
                schema.clone(),
                first_batch_rows,
                total_rows - first_batch_rows,
            )),
        ];

        let tempdir = tempfile::tempdir().unwrap();
        let index_dir = Path::from_filesystem_path(tempdir.path()).unwrap();
        let store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            index_dir,
            Arc::new(LanceCache::no_cache()),
        ));

        let stream = RecordBatchStreamAdapter::new(schema, stream::iter(batches));
        let req = FMIndexPlugin
            .new_training_request("", &arrow_schema::Field::new("val", DataType::Utf8, false))
            .unwrap();
        let created = FMIndexPlugin
            .train_index(
                Box::pin(stream),
                store.as_ref(),
                req,
                None,
                Arc::new(crate::progress::NoopIndexBuildProgress),
            )
            .await
            .unwrap();

        assert_eq!(created.files.len(), 2);

        let index = FMIndexPlugin
            .load_index(store, &created.index_details, None, &LanceCache::no_cache())
            .await
            .unwrap();
        let r = index
            .search(
                &TextQuery::StringContains("x".to_string()),
                &crate::metrics::NoOpMetricsCollector,
            )
            .await
            .unwrap();
        match r {
            SearchResult::Exact(set) => {
                assert_eq!(set.len(), Some(total_rows as u64));
            }
            _ => panic!("expected exact result"),
        }
    }

    #[test]
    fn test_build_wavelet_batch() {
        let texts: Vec<(u64, &[u8])> = vec![(0, b"hello world"), (1, b"test data")];
        let fm = FMIndex::build(&texts).unwrap();
        let batch = fm.build_wavelet_batch().unwrap();
        assert!(batch.num_rows() > 0);
        assert_eq!(batch.num_columns(), 5);
    }

    #[test]
    fn test_extract_text_bytes_types() {
        let utf8 = StringArray::from(vec!["hello"]);
        assert_eq!(
            extract_text_bytes(&utf8, 0).unwrap(),
            Some(b"hello".to_vec())
        );

        let large_utf8 = LargeStringArray::from(vec!["world"]);
        assert_eq!(
            extract_text_bytes(&large_utf8, 0).unwrap(),
            Some(b"world".to_vec())
        );

        let binary = BinaryArray::from(vec![b"bytes" as &[u8]]);
        assert_eq!(
            extract_text_bytes(&binary, 0).unwrap(),
            Some(b"bytes".to_vec())
        );
        let binary_with_sentinels = BinaryArray::from(vec![b"a\xFFb\0c" as &[u8]]);
        assert_eq!(
            extract_sanitized_text_bytes(&binary_with_sentinels, 0).unwrap(),
            Some(b"a b c".to_vec())
        );

        let large_binary = LargeBinaryArray::from(vec![b"large" as &[u8]]);
        assert_eq!(
            extract_text_bytes(&large_binary, 0).unwrap(),
            Some(b"large".to_vec())
        );

        // Null handling
        let nullable = StringArray::from(vec![None::<&str>]);
        assert_eq!(extract_text_bytes(&nullable, 0).unwrap(), None);
    }

    #[test]
    fn test_fmindex_statistics() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        rt.block_on(async {
            let docs: Vec<Vec<u8>> = (0..10).map(|i| format!("doc {i}").into_bytes()).collect();
            let texts: Vec<(u64, Vec<u8>)> = docs
                .into_iter()
                .enumerate()
                .map(|(i, d)| (i as u64, d))
                .collect();

            let tempdir = tempfile::tempdir().unwrap();
            let index_dir = Path::from_filesystem_path(tempdir.path()).unwrap();
            let store = Arc::new(LanceIndexStore::new(
                Arc::new(ObjectStore::local()),
                index_dir,
                Arc::new(LanceCache::no_cache()),
            ));

            write_partitioned_fmindex(&texts, store.as_ref())
                .await
                .unwrap();
            let index = FMIndexScalarIndex::load(store, None, &LanceCache::no_cache())
                .await
                .unwrap();

            let stats = index.statistics().unwrap();
            assert_eq!(stats["type"], "Fm");
            assert_eq!(stats["total_docs"], 10);
            assert!(stats["total_bwt_len"].as_u64().unwrap() > 0);
        });
    }
}
