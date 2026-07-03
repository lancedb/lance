// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::mem::{size_of, size_of_val};
use std::sync::{Arc, Mutex, MutexGuard};

use arrow_array::builder::LargeBinaryBuilder;
use arrow_array::{Array, LargeBinaryArray};
use lance_core::{Error, Result};

use super::scorer::Scorer;

pub const IMPACT_LEVEL1_BLOCKS: usize = 32;
const SMALL_FRONTIER_FREQ_LIMIT: usize = 256;

/// On-disk encoding of one impact entry.
///
/// `FixedU32` (128-doc blocks): [doc_up_to u32][pair_count u32][(freq u32,
/// doc_len u32)...]. `Varint` (256-doc blocks, format V3): [doc_up_to varint]
/// [pair_count varint][(freq delta varint, doc_len varint)...] with the
/// frontier's freqs delta-encoded in ascending order — pairs shrink from 8
/// bytes to ~2-3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImpactFormat {
    FixedU32,
    Varint,
}

impl ImpactFormat {
    pub fn for_block_size(block_size: usize) -> Self {
        if block_size == 256 {
            Self::Varint
        } else {
            Self::FixedU32
        }
    }
}

#[derive(Debug, Clone)]
pub struct ImpactSkipData {
    entries: LargeBinaryArray,
    level0_len: usize,
    format: ImpactFormat,
    // Last doc id covered by each entry (level0 entries then level1 entries),
    // decoded once at construction. Level1 markers are fully validated because
    // WAND may use them to skip a group; u32::MAX marks malformed entries.
    entry_doc_up_tos: Arc<[u32]>,
    // The most recently baked bounds with a stable scorer key. Each query holds
    // its own Arc in ImpactScoreCache, so replacing this slot for another scorer
    // cannot change bounds already in use. Scorers without a key never enter the
    // shared slot. Malformed entries bake to INFINITY so pruning stays safe.
    last_keyed_bounds: Arc<Mutex<LastKeyedImpactBounds>>,
}

impl PartialEq for ImpactSkipData {
    fn eq(&self, other: &Self) -> bool {
        self.entries == other.entries
            && self.level0_len == other.level0_len
            && self.format == other.format
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ImpactScore {
    pub score: f32,
    pub entries_scanned: usize,
}

#[derive(Debug)]
struct ImpactBounds {
    per_entry: Box<[f32]>,
    global: f32,
}

type LastKeyedImpactBounds = Option<(u64, Arc<ImpactBounds>)>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ImpactBoundsCacheKey {
    Keyed(u64),
    QueryLocal,
}

#[derive(Debug, Default, Clone)]
pub struct ImpactScoreCache {
    key: Option<ImpactBoundsCacheKey>,
    bounds: Option<Arc<ImpactBounds>>,
}

impl ImpactScoreCache {
    fn bounds<'a, S: Scorer + ?Sized>(
        &'a mut self,
        impacts: &ImpactSkipData,
        scorer: &S,
    ) -> &'a ImpactBounds {
        let scorer_key = scorer.doc_weight_cache_key();
        let cache_key = scorer_key
            .map(ImpactBoundsCacheKey::Keyed)
            .unwrap_or(ImpactBoundsCacheKey::QueryLocal);
        if self.key != Some(cache_key) {
            self.key = Some(cache_key);
            self.bounds = None;
        }

        self.bounds
            .get_or_insert_with(|| impacts.bounds_for_scorer(scorer, scorer_key))
    }

    fn entry_score<S: Scorer + ?Sized>(
        &mut self,
        impacts: &ImpactSkipData,
        entry_idx: usize,
        query_weight: f32,
        scorer: &S,
    ) -> f32 {
        if query_weight <= 0.0 {
            return 0.0;
        }
        query_weight * self.bounds(impacts, scorer).per_entry[entry_idx]
    }
}

#[derive(Debug, Clone, Copy)]
struct ImpactEntryHeader {
    doc_up_to: u32,
    pair_count: usize,
}

impl ImpactSkipData {
    pub fn new(entries: LargeBinaryArray, level0_len: usize, format: ImpactFormat) -> Result<Self> {
        let expected_len = level0_len + level1_len(level0_len);
        if entries.len() != expected_len {
            return Err(Error::index(format!(
                "impact entry count mismatch: got {}, expected {} for {} level0 blocks",
                entries.len(),
                expected_len,
                level0_len
            )));
        }
        let entry_doc_up_tos = (0..entries.len())
            .map(|entry_idx| {
                if entries.is_null(entry_idx) {
                    return u32::MAX;
                }
                let bytes = entries.value(entry_idx);
                let doc_up_to = if entry_idx < level0_len {
                    decode_level0_entry_doc_up_to(bytes, format)
                } else {
                    decode_entry_doc_up_to(bytes, format)
                };
                doc_up_to.unwrap_or(u32::MAX)
            })
            .collect::<Arc<[u32]>>();
        Ok(Self {
            entries,
            level0_len,
            format,
            entry_doc_up_tos,
            last_keyed_bounds: Arc::new(Mutex::new(None)),
        })
    }

    fn keyed_bounds_guard(&self) -> MutexGuard<'_, LastKeyedImpactBounds> {
        match self.last_keyed_bounds.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn bounds_for_scorer<S: Scorer + ?Sized>(
        &self,
        scorer: &S,
        scorer_key: Option<u64>,
    ) -> Arc<ImpactBounds> {
        let Some(scorer_key) = scorer_key else {
            return Arc::new(self.compute_bounds(scorer));
        };

        {
            let cached = self.keyed_bounds_guard();
            if let Some((cached_key, bounds)) = cached.as_ref()
                && *cached_key == scorer_key
            {
                return bounds.clone();
            }
        }

        // Compute outside the mutex. Concurrent misses may duplicate this work,
        // but only the short publication/check below holds the shared lock.
        let computed = Arc::new(self.compute_bounds(scorer));
        let mut cached = self.keyed_bounds_guard();
        if let Some((cached_key, bounds)) = cached.as_ref()
            && *cached_key == scorer_key
        {
            return bounds.clone();
        }
        *cached = Some((scorer_key, computed.clone()));
        computed
    }

    fn compute_bounds<S: Scorer + ?Sized>(&self, scorer: &S) -> ImpactBounds {
        // V3 (Varint impacts <=> 256-doc blocks) scores with quantized doc
        // lengths, so bake the bounds against the same quantized lengths.
        // Quantization is monotone, so pareto dominance among the stored
        // (freq, doc_len) frontier pairs is preserved and the frontier max
        // still bounds every doc in the range.
        let quantized = self.format == ImpactFormat::Varint;
        let per_entry = (0..self.entries.len())
            .map(|entry_idx| {
                if self.entries.is_null(entry_idx) {
                    return f32::INFINITY;
                }
                let bytes = self.entries.value(entry_idx);
                let mut max_doc_weight = 0.0_f32;
                match for_each_entry_pair(bytes, self.format, |freq, doc_len| {
                    let doc_len = if quantized {
                        super::index::dequantize_doc_length(super::index::quantize_doc_length(
                            doc_len,
                        ))
                    } else {
                        doc_len
                    };
                    max_doc_weight = max_doc_weight.max(scorer.doc_weight(freq, doc_len));
                }) {
                    Ok(()) => max_doc_weight,
                    Err(_) => f32::INFINITY,
                }
            })
            .collect::<Box<[f32]>>();
        // The level1 entries cover every block, so their max is the list-wide
        // max doc weight; zero-entry lists fall back to the empty level0 slab.
        let global = if per_entry.len() > self.level0_len {
            per_entry[self.level0_len..]
                .iter()
                .copied()
                .fold(0.0_f32, f32::max)
        } else {
            per_entry.iter().copied().fold(0.0_f32, f32::max)
        };
        ImpactBounds { per_entry, global }
    }

    /// List-wide max doc weight, from the scorer-specific cached bounds. The
    /// tightest valid global score bound is `query_weight * this`, matching what
    /// the non-impact format stores as `max_score` at build time.
    pub fn global_max_doc_weight_cached<S: Scorer + ?Sized>(
        &self,
        scorer: &S,
        cache: &mut ImpactScoreCache,
    ) -> f32 {
        cache.bounds(self, scorer).global
    }

    pub fn entries(&self) -> &LargeBinaryArray {
        &self.entries
    }

    /// Conservative heap charge for query-independent derived state and one
    /// shared keyed-bound slab, whether or not that slab has been initialized
    /// yet. The Arrow impact entries are owned by the enclosing batch and are
    /// deliberately excluded so packed-group cache accounting counts them once.
    pub(crate) fn derived_cache_bytes(&self) -> usize {
        size_of_val(&*self.entry_doc_up_tos)
            + size_of::<Mutex<LastKeyedImpactBounds>>()
            + size_of::<ImpactBounds>()
            + self.entries.len() * size_of::<f32>()
    }

    #[cfg(test)]
    pub(crate) fn shares_derived_state_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.entry_doc_up_tos, &other.entry_doc_up_tos)
            && Arc::ptr_eq(&self.last_keyed_bounds, &other.last_keyed_bounds)
    }

    #[cfg(test)]
    pub fn level0_len(&self) -> usize {
        self.level0_len
    }

    #[cfg(test)]
    pub fn level1_len(&self) -> usize {
        level1_len(self.level0_len)
    }

    pub(crate) fn level1_doc_up_to(&self, group_idx: usize) -> Option<u32> {
        if group_idx >= level1_len(self.level0_len) {
            return None;
        }
        match self.entry_doc_up_tos[self.level0_len + group_idx] {
            u32::MAX => None,
            doc_up_to => Some(doc_up_to),
        }
    }

    /// Max score of the docs covered by the level0 entry of `block_idx`,
    /// answered from the baked bounds slab.
    // Only tests exercise the uncached form until the maxscore rework
    // (stacked follow-up) anchors its block-max caches on it.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn level0_score<S: Scorer + ?Sized>(
        &self,
        block_idx: usize,
        query_weight: f32,
        scorer: &S,
    ) -> f32 {
        if block_idx >= self.level0_len || query_weight <= 0.0 {
            return 0.0;
        }
        let mut cache = ImpactScoreCache::default();
        cache.entry_score(self, block_idx, query_weight, scorer)
    }

    pub fn level0_score_cached<S: Scorer + ?Sized>(
        &self,
        block_idx: usize,
        query_weight: f32,
        scorer: &S,
        cache: &mut ImpactScoreCache,
    ) -> f32 {
        if block_idx >= self.level0_len {
            return 0.0;
        }
        cache.entry_score(self, block_idx, query_weight, scorer)
    }

    #[cfg(test)]
    pub fn max_score_up_to<S, F>(
        &self,
        start_block_idx: usize,
        up_to: u64,
        _block_least_doc_id: F,
        query_weight: f32,
        scorer: &S,
    ) -> ImpactScore
    where
        S: Scorer + ?Sized,
        F: FnMut(usize) -> u32,
    {
        self.max_score_up_to_with(start_block_idx, up_to, |impacts, entry_idx| {
            impacts.entry_score(entry_idx, query_weight, scorer)
        })
    }

    pub fn max_score_up_to_cached<S>(
        &self,
        start_block_idx: usize,
        up_to: u64,
        query_weight: f32,
        scorer: &S,
        cache: &mut ImpactScoreCache,
    ) -> ImpactScore
    where
        S: Scorer + ?Sized,
    {
        self.max_score_up_to_with(start_block_idx, up_to, |impacts, entry_idx| {
            cache.entry_score(impacts, entry_idx, query_weight, scorer)
        })
    }

    fn max_score_up_to_with<E>(
        &self,
        start_block_idx: usize,
        up_to: u64,
        mut entry_score: E,
    ) -> ImpactScore
    where
        E: FnMut(&Self, usize) -> f32,
    {
        let mut block_idx = start_block_idx;
        let mut max_score = 0.0_f32;
        let mut entries_scanned = 0usize;

        while block_idx < self.level0_len {
            let group_idx = block_idx / IMPACT_LEVEL1_BLOCKS;
            let group_start = group_idx * IMPACT_LEVEL1_BLOCKS;
            let group_end = ((group_idx + 1) * IMPACT_LEVEL1_BLOCKS).min(self.level0_len);
            if block_idx == group_start {
                let level1_entry_idx = self.level0_len + group_idx;
                match self.entry_doc_up_tos[level1_entry_idx] {
                    u32::MAX => {
                        return ImpactScore {
                            score: f32::INFINITY,
                            entries_scanned: entries_scanned + 1,
                        };
                    }
                    doc_up_to if u64::from(doc_up_to) <= up_to => {
                        max_score = max_score.max(entry_score(self, level1_entry_idx));
                        entries_scanned += 1;
                        block_idx = group_end;
                        continue;
                    }
                    _ => {}
                }
            }

            max_score = max_score.max(entry_score(self, block_idx));
            entries_scanned += 1;
            match self.entry_doc_up_tos[block_idx] {
                u32::MAX => {
                    return ImpactScore {
                        score: f32::INFINITY,
                        entries_scanned,
                    };
                }
                doc_up_to if u64::from(doc_up_to) >= up_to => break,
                _ => {}
            }
            block_idx += 1;
        }

        ImpactScore {
            score: max_score,
            entries_scanned,
        }
    }

    #[cfg(test)]
    fn entry_score<S: Scorer + ?Sized>(
        &self,
        entry_idx: usize,
        query_weight: f32,
        scorer: &S,
    ) -> f32 {
        if query_weight <= 0.0 {
            return 0.0;
        }
        let bytes = self.entries.value(entry_idx);
        let mut max_doc_weight = 0.0_f32;
        if for_each_entry_pair(bytes, self.format, |freq, doc_len| {
            max_doc_weight = max_doc_weight.max(scorer.doc_weight(freq, doc_len));
        })
        .is_err()
        {
            return f32::INFINITY;
        }
        query_weight * max_doc_weight
    }
}

pub struct ImpactSkipDataBuilder {
    entries: LargeBinaryBuilder,
    level0_len: usize,
    level1_entries: Vec<Vec<u8>>,
    level1_docs: Vec<(u32, u32, u32)>,
    format: ImpactFormat,
}

impl ImpactSkipDataBuilder {
    pub fn with_capacity(level0_blocks: usize, block_size: usize) -> Self {
        Self {
            entries: LargeBinaryBuilder::with_capacity(
                level0_blocks + level1_len(level0_blocks),
                0,
            ),
            level0_len: 0,
            level1_entries: Vec::with_capacity(level1_len(level0_blocks)),
            level1_docs: Vec::with_capacity(IMPACT_LEVEL1_BLOCKS * block_size),
            format: ImpactFormat::for_block_size(block_size),
        }
    }

    pub fn append_block(&mut self, docs: &[(u32, u32, u32)]) -> Result<()> {
        let bytes = encode_impact_entry(docs, self.format)?;
        self.entries.append_value(bytes.as_slice());
        self.level0_len += 1;
        self.level1_docs.extend_from_slice(docs);
        if self.level0_len.is_multiple_of(IMPACT_LEVEL1_BLOCKS) {
            self.flush_level1()?;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<ImpactSkipData> {
        if !self.level1_docs.is_empty() {
            self.flush_level1()?;
        }
        for entry in self.level1_entries {
            self.entries.append_value(entry.as_slice());
        }
        ImpactSkipData::new(self.entries.finish(), self.level0_len, self.format)
    }

    fn flush_level1(&mut self) -> Result<()> {
        let bytes = encode_impact_entry(self.level1_docs.as_slice(), self.format)?;
        self.level1_entries.push(bytes);
        self.level1_docs.clear();
        Ok(())
    }
}

#[cfg(test)]
pub fn build_impact_skip_data(blocks: &[Vec<(u32, u32, u32)>]) -> Result<ImpactSkipData> {
    let block_size = blocks.iter().map(Vec::len).max().unwrap_or(0).max(1);
    let mut builder = ImpactSkipDataBuilder::with_capacity(blocks.len(), block_size);
    for block in blocks {
        builder.append_block(block)?;
    }
    builder.finish()
}

fn encode_impact_entry(docs: &[(u32, u32, u32)], format: ImpactFormat) -> Result<Vec<u8>> {
    if docs.is_empty() {
        return Err(Error::index(
            "cannot encode an empty impact entry".to_owned(),
        ));
    }
    let doc_up_to = docs
        .last()
        .map(|(doc_id, _, _)| *doc_id)
        .expect("non-empty impact entry was validated above");
    let frontier = impact_frontier(docs);
    let pair_count = u32::try_from(frontier.len()).map_err(|_| {
        Error::index("impact frontier too large to encode as u32 pair count".to_string())
    })?;
    let mut bytes = Vec::with_capacity(8 + frontier.len() * 8);
    match format {
        ImpactFormat::FixedU32 => {
            bytes.extend_from_slice(&doc_up_to.to_le_bytes());
            bytes.extend_from_slice(&pair_count.to_le_bytes());
            for (freq, doc_len) in frontier {
                bytes.extend_from_slice(&freq.to_le_bytes());
                bytes.extend_from_slice(&doc_len.to_le_bytes());
            }
        }
        ImpactFormat::Varint => {
            super::encoding::encode_varint_u32(&mut bytes, doc_up_to);
            super::encoding::encode_varint_u32(&mut bytes, pair_count);
            let mut previous_freq = 0u32;
            for (freq, doc_len) in frontier {
                super::encoding::encode_varint_u32(&mut bytes, freq - previous_freq);
                super::encoding::encode_varint_u32(&mut bytes, doc_len);
                previous_freq = freq;
            }
        }
    }
    Ok(bytes)
}

fn decode_entry_doc_up_to(bytes: &[u8], format: ImpactFormat) -> Result<u32> {
    match format {
        ImpactFormat::FixedU32 => {
            let header = decode_header(bytes)?;
            if header.pair_count == 0 {
                return Err(Error::index(
                    "impact entry must contain at least one frontier pair".to_owned(),
                ));
            }
            Ok(header.doc_up_to)
        }
        ImpactFormat::Varint => {
            let mut offset = 0usize;
            let doc_up_to = super::encoding::decode_varint_u32(bytes, &mut offset)?;
            // Level-1 doc ids drive whole-group skips, so only publish a doc
            // id after validating the complete entry. A truncated entry may
            // still have a valid first varint.
            for_each_entry_pair(bytes, format, |_, _| {})?;
            Ok(doc_up_to)
        }
    }
}

fn decode_level0_entry_doc_up_to(bytes: &[u8], format: ImpactFormat) -> Result<u32> {
    match format {
        // Fixed entries validate their complete layout from the header in O(1).
        ImpactFormat::FixedU32 => decode_entry_doc_up_to(bytes, format),
        // A malformed level0 frontier bakes an INFINITY score before this
        // marker can terminate a range scan, so avoid parsing every frontier
        // twice on the query-load path. Level1 entries are fully validated.
        ImpactFormat::Varint => {
            let mut offset = 0usize;
            super::encoding::decode_varint_u32(bytes, &mut offset)
        }
    }
}

/// Walk an entry's (freq, doc_len) frontier pairs, validating the layout.
fn for_each_entry_pair(
    bytes: &[u8],
    format: ImpactFormat,
    mut visit: impl FnMut(u32, u32),
) -> Result<()> {
    match format {
        ImpactFormat::FixedU32 => {
            let header = decode_header(bytes)?;
            if header.pair_count == 0 {
                return Err(Error::index(
                    "impact entry must contain at least one frontier pair".to_owned(),
                ));
            }
            let mut offset = 8usize;
            for _ in 0..header.pair_count {
                visit(read_u32_le(bytes, offset), read_u32_le(bytes, offset + 4));
                offset += 8;
            }
        }
        ImpactFormat::Varint => {
            let mut offset = 0usize;
            let _doc_up_to = super::encoding::decode_varint_u32(bytes, &mut offset)?;
            let pair_count = super::encoding::decode_varint_u32(bytes, &mut offset)?;
            if pair_count == 0 {
                return Err(Error::index(
                    "impact entry must contain at least one frontier pair".to_owned(),
                ));
            }
            let mut freq = 0u32;
            for _ in 0..pair_count {
                freq = freq
                    .checked_add(super::encoding::decode_varint_u32(bytes, &mut offset)?)
                    .ok_or_else(|| Error::index("impact freq delta overflow".to_owned()))?;
                let doc_len = super::encoding::decode_varint_u32(bytes, &mut offset)?;
                visit(freq, doc_len);
            }
            if offset != bytes.len() {
                return Err(Error::index(format!(
                    "impact varint entry has {} trailing bytes",
                    bytes.len() - offset
                )));
            }
        }
    }
    Ok(())
}

fn impact_frontier(docs: &[(u32, u32, u32)]) -> Vec<(u32, u32)> {
    let max_freq = docs.iter().map(|(_, freq, _)| *freq).max().unwrap_or(0) as usize;
    if max_freq <= SMALL_FRONTIER_FREQ_LIMIT {
        return impact_frontier_small_freq(docs, max_freq);
    }

    impact_frontier_sparse_freq(docs)
}

fn impact_frontier_small_freq(docs: &[(u32, u32, u32)], max_freq: usize) -> Vec<(u32, u32)> {
    let mut min_doc_len_by_freq = [u32::MAX; SMALL_FRONTIER_FREQ_LIMIT + 1];
    for (_, freq, doc_len) in docs {
        min_doc_len_by_freq[*freq as usize] = min_doc_len_by_freq[*freq as usize].min(*doc_len);
    }

    let min_doc_lens = min_doc_len_by_freq[..=max_freq]
        .iter()
        .enumerate()
        .filter_map(|(freq, doc_len)| (*doc_len != u32::MAX).then_some((freq as u32, *doc_len)))
        .collect::<Vec<_>>();
    frontier_from_min_doc_lens(min_doc_lens)
}

fn impact_frontier_sparse_freq(docs: &[(u32, u32, u32)]) -> Vec<(u32, u32)> {
    let mut pairs = docs
        .iter()
        .map(|(_, freq, doc_len)| (*freq, *doc_len))
        .collect::<Vec<_>>();
    pairs.sort_unstable_by_key(|(freq, _)| *freq);

    let mut min_doc_lens: Vec<(u32, u32)> = Vec::with_capacity(pairs.len());
    for (freq, doc_len) in pairs {
        match min_doc_lens.last_mut() {
            Some((last_freq, last_doc_len)) if *last_freq == freq => {
                *last_doc_len = (*last_doc_len).min(doc_len);
            }
            _ => min_doc_lens.push((freq, doc_len)),
        }
    }

    frontier_from_min_doc_lens(min_doc_lens)
}

fn frontier_from_min_doc_lens(min_doc_lens: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    let mut best_doc_len = u32::MAX;
    let mut frontier = Vec::with_capacity(min_doc_lens.len());
    for (freq, doc_len) in min_doc_lens.into_iter().rev() {
        if doc_len < best_doc_len {
            frontier.push((freq, doc_len));
            best_doc_len = doc_len;
        }
    }
    frontier.reverse();
    frontier
}

fn decode_header(bytes: &[u8]) -> Result<ImpactEntryHeader> {
    if bytes.len() < 8 {
        return Err(Error::index(format!(
            "impact entry too short: {} bytes",
            bytes.len()
        )));
    }
    let pair_count = read_u32_le(bytes, 4) as usize;
    let expected_len = pair_count
        .checked_mul(8)
        .and_then(|pairs_len| 8usize.checked_add(pairs_len))
        .ok_or_else(|| Error::index("impact entry length overflow".to_owned()))?;
    if bytes.len() != expected_len {
        return Err(Error::index(format!(
            "impact entry length mismatch: got {} bytes, expected {} for {} pairs",
            bytes.len(),
            expected_len,
            pair_count
        )));
    }
    Ok(ImpactEntryHeader {
        doc_up_to: read_u32_le(bytes, 0),
        pair_count,
    })
}

#[inline]
fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
    let mut value = [0u8; 4];
    value.copy_from_slice(&bytes[offset..offset + 4]);
    u32::from_le_bytes(value)
}

fn level1_len(level0_len: usize) -> usize {
    level0_len.div_ceil(IMPACT_LEVEL1_BLOCKS)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow::buffer::{Buffer, NullBuffer, OffsetBuffer, ScalarBuffer};

    use super::*;
    use crate::scalar::inverted::scorer::{MemBM25Scorer, Scorer};

    struct KeyedCountingScorer {
        key: u64,
        calls: Arc<AtomicUsize>,
    }

    impl Scorer for KeyedCountingScorer {
        fn query_weight(&self, _token: &str) -> f32 {
            1.0
        }

        fn doc_weight(&self, freq: u32, doc_tokens: u32) -> f32 {
            self.calls.fetch_add(1, Ordering::Relaxed);
            freq as f32 / doc_tokens as f32
        }

        fn doc_weight_cache_key(&self) -> Option<u64> {
            Some(self.key)
        }
    }

    #[test]
    fn impact_entry_frontier_drops_dominated_pairs() {
        let docs = vec![(0, 1, 10), (1, 1, 8), (2, 2, 9), (3, 3, 20)];
        assert_eq!(impact_frontier(&docs), vec![(1, 8), (2, 9), (3, 20)]);
    }

    #[test]
    fn impact_entry_frontier_handles_sparse_large_frequencies() {
        let docs = vec![
            (0, 1, 100),
            (1, 1, 80),
            (2, 512, 90),
            (3, 1_000, 120),
            (4, 1_000, 110),
        ];
        assert_eq!(
            impact_frontier(&docs),
            vec![(1, 80), (512, 90), (1_000, 110)]
        );
    }

    #[test]
    fn impact_max_score_can_use_level1_entry() {
        let blocks = (0..40)
            .map(|block| vec![(block as u32, 1 + block as u32 % 3, 10)])
            .collect::<Vec<_>>();
        let impacts = build_impact_skip_data(&blocks).unwrap();
        assert_eq!(impacts.level0_len(), 40);
        assert_eq!(impacts.level1_len(), 2);
        let scorer = MemBM25Scorer::new(400, 40, HashMap::from([(String::from("token"), 40usize)]));
        let score = impacts.max_score_up_to(0, 31, |idx| idx as u32, 1.0, &scorer);
        assert!(score.entries_scanned < IMPACT_LEVEL1_BLOCKS);
        assert!(score.score > 0.0);
    }

    #[test]
    fn impact_level1_doc_up_to_reports_full_and_partial_groups() {
        let blocks = (0..40)
            .map(|block| vec![(block as u32, 1, 10)])
            .collect::<Vec<_>>();
        let impacts = build_impact_skip_data(&blocks).unwrap();

        assert_eq!(
            impacts.level1_doc_up_to(0),
            Some((IMPACT_LEVEL1_BLOCKS - 1) as u32)
        );
        assert_eq!(impacts.level1_doc_up_to(1), Some(39));
        assert_eq!(impacts.level1_doc_up_to(2), None);
    }

    #[test]
    fn impact_level1_doc_up_to_returns_none_for_malformed_entry() {
        let level0 = encode_impact_entry(&[(0, 1, 10)], ImpactFormat::FixedU32).unwrap();
        let malformed_level1 = vec![1, 2, 3];
        let entries = LargeBinaryArray::from_opt_vec(vec![
            Some(level0.as_slice()),
            Some(malformed_level1.as_slice()),
        ]);
        let impacts = ImpactSkipData::new(entries, 1, ImpactFormat::FixedU32).unwrap();

        assert_eq!(impacts.level1_doc_up_to(0), None);
    }

    #[test]
    fn impact_level1_doc_up_to_validates_complete_varint_entry() {
        let level0 = encode_impact_entry(&[(0, 1, 10)], ImpactFormat::Varint).unwrap();
        // A complete first varint is not enough: the pair count and frontier
        // are required before this doc id can safely drive a group skip.
        let truncated_level1 = [31_u8];
        let entries = LargeBinaryArray::from_opt_vec(vec![
            Some(level0.as_slice()),
            Some(truncated_level1.as_slice()),
        ]);
        let impacts = ImpactSkipData::new(entries, 1, ImpactFormat::Varint).unwrap();

        assert_eq!(impacts.level1_doc_up_to(0), None);
    }

    #[test]
    fn empty_impact_frontiers_are_malformed_bounds() {
        let scorer = MemBM25Scorer::new(10, 1, HashMap::new());
        for format in [ImpactFormat::FixedU32, ImpactFormat::Varint] {
            let level0 = encode_impact_entry(&[(0, 1, 10)], format).unwrap();
            let empty_level1 = match format {
                ImpactFormat::FixedU32 => {
                    let mut bytes = 0_u32.to_le_bytes().to_vec();
                    bytes.extend_from_slice(&0_u32.to_le_bytes());
                    bytes
                }
                ImpactFormat::Varint => vec![0, 0],
            };
            let entries = LargeBinaryArray::from_opt_vec(vec![
                Some(level0.as_slice()),
                Some(empty_level1.as_slice()),
            ]);
            let impacts = ImpactSkipData::new(entries, 1, format).unwrap();
            let mut cache = ImpactScoreCache::default();

            assert_eq!(impacts.level1_doc_up_to(0), None);
            assert!(
                impacts
                    .global_max_doc_weight_cached(&scorer, &mut cache)
                    .is_infinite()
            );
        }
        assert!(
            ImpactSkipDataBuilder::with_capacity(1, 128)
                .append_block(&[])
                .is_err()
        );
    }

    #[test]
    fn null_impact_entry_is_an_infinite_bound_even_with_hidden_bytes() {
        let level0 = encode_impact_entry(&[(0, 1, 10)], ImpactFormat::FixedU32).unwrap();
        let level1 = encode_impact_entry(&[(0, 2, 8)], ImpactFormat::FixedU32).unwrap();
        let mut values = level0.clone();
        values.extend_from_slice(&level1);
        let entries = LargeBinaryArray::new(
            OffsetBuffer::new(ScalarBuffer::from(vec![
                0_i64,
                level0.len() as i64,
                values.len() as i64,
            ])),
            Buffer::from_vec(values),
            Some(NullBuffer::from(vec![true, false])),
        );
        let impacts = ImpactSkipData::new(entries, 1, ImpactFormat::FixedU32).unwrap();
        let scorer = MemBM25Scorer::new(10, 1, HashMap::new());
        let mut cache = ImpactScoreCache::default();

        assert_eq!(impacts.level1_doc_up_to(0), None);
        assert!(
            impacts
                .global_max_doc_weight_cached(&scorer, &mut cache)
                .is_infinite()
        );
    }

    #[test]
    fn impact_score_cache_matches_uncached_scores() {
        let blocks = (0..40)
            .map(|block| vec![(block as u32, 1 + block as u32 % 3, 10)])
            .collect::<Vec<_>>();
        let impacts = build_impact_skip_data(&blocks).unwrap();
        let scorer = MemBM25Scorer::new(400, 40, HashMap::from([(String::from("token"), 40usize)]));
        let mut cache = ImpactScoreCache::default();

        let uncached_level0 = impacts.level0_score(3, 1.0, &scorer);
        let cached_level0 = impacts.level0_score_cached(3, 1.0, &scorer, &mut cache);
        assert_eq!(cached_level0, uncached_level0);

        let uncached = impacts.max_score_up_to(0, 31, |idx| idx as u32, 1.0, &scorer);
        let cached = impacts.max_score_up_to_cached(0, 31, 1.0, &scorer, &mut cache);
        assert_eq!(cached.score, uncached.score);
        assert_eq!(cached.entries_scanned, uncached.entries_scanned);
    }

    #[test]
    fn impact_bounds_follow_changed_bm25_average_doc_length() {
        let impacts = build_impact_skip_data(&[vec![(0, 1, 100)]]).unwrap();
        let low_avgdl = MemBM25Scorer::new(1, 1, HashMap::new());
        let high_avgdl = MemBM25Scorer::new(100, 1, HashMap::new());
        let mut low_cache = ImpactScoreCache::default();
        let mut high_cache = ImpactScoreCache::default();

        let low_bound = impacts.global_max_doc_weight_cached(&low_avgdl, &mut low_cache);
        let high_bound = impacts.global_max_doc_weight_cached(&high_avgdl, &mut high_cache);

        assert!((low_bound - low_avgdl.doc_weight(1, 100)).abs() < 1e-6);
        assert!((high_bound - high_avgdl.doc_weight(1, 100)).abs() < 1e-6);
        assert!(
            high_bound > low_bound,
            "larger avgdl must recompute a larger bound: low={low_bound}, high={high_bound}"
        );
    }

    #[test]
    fn impact_bounds_reuse_same_scorer_key_across_queries() {
        let impacts = build_impact_skip_data(&[vec![(0, 2, 10)]]).unwrap();
        let cloned = impacts.clone();
        assert!(impacts.shares_derived_state_with(&cloned));
        let calls = Arc::new(AtomicUsize::new(0));
        let scorer = KeyedCountingScorer {
            key: 7,
            calls: calls.clone(),
        };
        let mut first_query = ImpactScoreCache::default();
        let mut second_query = ImpactScoreCache::default();

        let first = impacts.global_max_doc_weight_cached(&scorer, &mut first_query);
        let baked_calls = calls.load(Ordering::Relaxed);
        assert!(baked_calls > 0);
        let second = cloned.global_max_doc_weight_cached(&scorer, &mut second_query);

        assert_eq!(second, first);
        assert_eq!(
            calls.load(Ordering::Relaxed),
            baked_calls,
            "the same keyed bounds should be shared across query caches"
        );
    }

    #[test]
    fn impact_entries_are_decoded_lazily() {
        let level0_0 = encode_impact_entry(&[(0, 1, 10)], ImpactFormat::FixedU32).unwrap();
        let malformed_level0_1 = vec![1, 2, 3];
        let level1 =
            encode_impact_entry(&[(0, 1, 10), (1, 1, 10)], ImpactFormat::FixedU32).unwrap();
        let entries = LargeBinaryArray::from_opt_vec(vec![
            Some(level0_0.as_slice()),
            Some(malformed_level0_1.as_slice()),
            Some(level1.as_slice()),
        ]);
        let impacts = ImpactSkipData::new(entries, 2, ImpactFormat::FixedU32).unwrap();
        let scorer = MemBM25Scorer::new(10, 10, HashMap::from([(String::from("token"), 2usize)]));

        let score = impacts.max_score_up_to(0, 0, |_| 0, 1.0, &scorer);
        assert!(score.score.is_finite());
        assert_eq!(score.entries_scanned, 1);

        let mut cache = ImpactScoreCache::default();
        assert_eq!(
            impacts.level0_score_cached(1, 1.0, &scorer, &mut cache),
            f32::INFINITY
        );
    }

    #[test]
    fn impact_varint_entries_roundtrip_and_match_fixed() {
        let docs = vec![(3, 1, 100), (9, 2, 40), (200, 7, 80), (4095, 130, 900)];
        let fixed = encode_impact_entry(&docs, ImpactFormat::FixedU32).unwrap();
        let varint = encode_impact_entry(&docs, ImpactFormat::Varint).unwrap();
        assert!(varint.len() < fixed.len());
        assert_eq!(
            decode_entry_doc_up_to(&fixed, ImpactFormat::FixedU32).unwrap(),
            decode_entry_doc_up_to(&varint, ImpactFormat::Varint).unwrap()
        );
        let mut fixed_pairs = Vec::new();
        for_each_entry_pair(&fixed, ImpactFormat::FixedU32, |f, l| {
            fixed_pairs.push((f, l))
        })
        .unwrap();
        let mut varint_pairs = Vec::new();
        for_each_entry_pair(&varint, ImpactFormat::Varint, |f, l| {
            varint_pairs.push((f, l))
        })
        .unwrap();
        assert_eq!(fixed_pairs, varint_pairs);
        assert!(!fixed_pairs.is_empty());

        // a 256-doc-block skip data goes through the varint path end to end
        let blocks: Vec<Vec<(u32, u32, u32)>> = (0..3)
            .map(|b| (0..256).map(|i| (b * 256 + i, 1 + i % 5, 10)).collect())
            .collect();
        let impacts = build_impact_skip_data(&blocks).unwrap();
        assert_eq!(impacts.level1_doc_up_to(0), Some(767));
        let scorer = MemBM25Scorer::new(400, 768, HashMap::from([(String::from("t"), 768usize)]));
        assert!(impacts.level0_score(0, 1.0, &scorer).is_finite());
        let level1 = impacts.max_score_up_to(0, 767, |idx| (idx * 256) as u32, 1.0, &scorer);
        assert!(level1.score.is_finite() && level1.score > 0.0);
    }

    #[test]
    fn impact_upper_bound_covers_real_scores() {
        let blocks = vec![
            vec![(0, 1, 100), (3, 2, 40), (7, 4, 80)],
            vec![(9, 3, 15), (10, 1, 5), (12, 5, 30)],
            vec![(16, 2, 10), (18, 6, 70), (21, 3, 12)],
            vec![(24, 1, 4), (28, 7, 100), (30, 2, 8)],
        ];
        let impacts = build_impact_skip_data(&blocks).unwrap();
        let scorer = MemBM25Scorer::new(474, 31, HashMap::from([(String::from("token"), 4usize)]));
        let query_weight = scorer.query_weight("token");

        for start_block_idx in 0..blocks.len() {
            let up_to = blocks
                .iter()
                .skip(start_block_idx)
                .take(2)
                .flatten()
                .map(|(doc_id, _, _)| *doc_id)
                .max()
                .unwrap();
            let upper_bound = impacts.max_score_up_to(
                start_block_idx,
                u64::from(up_to),
                |idx| blocks[idx][0].0,
                query_weight,
                &scorer,
            );
            let exact_max = blocks
                .iter()
                .skip(start_block_idx)
                .flatten()
                .take_while(|(doc_id, _, _)| *doc_id <= up_to)
                .map(|(_, freq, doc_len)| query_weight * scorer.doc_weight(*freq, *doc_len))
                .fold(0.0_f32, f32::max);
            assert!(
                upper_bound.score + 1e-6 >= exact_max,
                "upper bound {} should cover exact max {} from block {} up to doc {}",
                upper_bound.score,
                exact_max,
                start_block_idx,
                up_to
            );
        }
    }
}
