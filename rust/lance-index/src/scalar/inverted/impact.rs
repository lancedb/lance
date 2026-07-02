// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::OnceLock;

use arrow_array::builder::LargeBinaryBuilder;
use arrow_array::{Array, LargeBinaryArray};
use lance_core::{Error, Result};

use super::scorer::Scorer;

pub const IMPACT_LEVEL1_BLOCKS: usize = 32;
const SMALL_FRONTIER_FREQ_LIMIT: usize = 256;

#[derive(Debug, Clone)]
pub struct ImpactSkipData {
    entries: LargeBinaryArray,
    level0_len: usize,
    // Per-entry max doc weight (level0 entries then level1 entries), baked on
    // first use. Doc weights depend only on (freq, doc_len) and the scorer's
    // index-wide stats (e.g. BM25 avgdl), not on the query, so one pass over
    // the frontiers serves every query for the lifetime of the cached list.
    // Malformed entries bake to INFINITY so pruning stays safe.
    doc_weight_bounds: OnceLock<Box<[f32]>>,
}

impl PartialEq for ImpactSkipData {
    fn eq(&self, other: &Self) -> bool {
        self.entries == other.entries && self.level0_len == other.level0_len
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub struct ImpactScore {
    pub score: f32,
    pub entries_scanned: usize,
}

#[derive(Debug, Clone, Copy)]
struct ImpactEntryHeader {
    doc_up_to: u32,
    pair_count: usize,
}

impl ImpactSkipData {
    pub fn new(entries: LargeBinaryArray, level0_len: usize) -> Result<Self> {
        let expected_len = level0_len + level1_len(level0_len);
        if entries.len() != expected_len {
            return Err(Error::index(format!(
                "impact entry count mismatch: got {}, expected {} for {} level0 blocks",
                entries.len(),
                expected_len,
                level0_len
            )));
        }
        Ok(Self {
            entries,
            level0_len,
            doc_weight_bounds: OnceLock::new(),
        })
    }

    fn doc_weight_bounds<S: Scorer + ?Sized>(&self, scorer: &S) -> &[f32] {
        self.doc_weight_bounds.get_or_init(|| {
            (0..self.entries.len())
                .map(|entry_idx| self.entry_max_doc_weight(entry_idx, scorer))
                .collect()
        })
    }

    fn entry_max_doc_weight<S: Scorer + ?Sized>(&self, entry_idx: usize, scorer: &S) -> f32 {
        let bytes = self.entries.value(entry_idx);
        let Ok(header) = decode_header(bytes) else {
            return f32::INFINITY;
        };
        let mut max_doc_weight = 0.0_f32;
        let mut offset = 8usize;
        for _ in 0..header.pair_count {
            let freq = read_u32_le(bytes, offset);
            let doc_len = read_u32_le(bytes, offset + 4);
            max_doc_weight = max_doc_weight.max(scorer.doc_weight(freq, doc_len));
            offset += 8;
        }
        max_doc_weight
    }

    pub fn entries(&self) -> &LargeBinaryArray {
        &self.entries
    }

    #[cfg(test)]
    pub fn level0_len(&self) -> usize {
        self.level0_len
    }

    #[cfg(test)]
    pub fn level1_len(&self) -> usize {
        level1_len(self.level0_len)
    }

    /// Last doc id covered by the level0 entry of `block_idx`, or `None` when
    /// the entry is missing or malformed (callers must fall back to a coarser
    /// bound in that case).
    pub(crate) fn level0_doc_up_to(&self, block_idx: usize) -> Option<u32> {
        if block_idx >= self.level0_len || self.entries.is_null(block_idx) {
            return None;
        }
        decode_header(self.entries.value(block_idx))
            .ok()
            .map(|header| header.doc_up_to)
    }

    pub(crate) fn level1_doc_up_to(&self, group_idx: usize) -> Option<u32> {
        if group_idx >= level1_len(self.level0_len) {
            return None;
        }
        let entry_idx = self.level0_len.checked_add(group_idx)?;
        if entry_idx >= self.entries.len() || self.entries.is_null(entry_idx) {
            return None;
        }
        decode_header(self.entries.value(entry_idx))
            .ok()
            .map(|header| header.doc_up_to)
    }

    /// Max score of the docs covered by the level0 entry of `block_idx`.
    /// Malformed entries yield `f32::INFINITY` so pruning stays safe.
    pub fn level0_score<S: Scorer + ?Sized>(
        &self,
        block_idx: usize,
        query_weight: f32,
        scorer: &S,
    ) -> f32 {
        if block_idx >= self.level0_len || query_weight <= 0.0 {
            return 0.0;
        }
        query_weight * self.doc_weight_bounds(scorer)[block_idx]
    }

    /// Max score of the docs covered by the level1 entry of `group_idx`
    /// (a span of [`IMPACT_LEVEL1_BLOCKS`] level0 blocks). Malformed entries
    /// yield `f32::INFINITY` so pruning stays safe.
    pub fn level1_score<S: Scorer + ?Sized>(
        &self,
        group_idx: usize,
        query_weight: f32,
        scorer: &S,
    ) -> f32 {
        if group_idx >= level1_len(self.level0_len) || query_weight <= 0.0 {
            return 0.0;
        }
        query_weight * self.doc_weight_bounds(scorer)[self.level0_len + group_idx]
    }

    #[cfg(test)]
    pub fn max_score_up_to<S, F>(
        &self,
        start_block_idx: usize,
        up_to: u64,
        block_least_doc_id: F,
        query_weight: f32,
        scorer: &S,
    ) -> ImpactScore
    where
        S: Scorer + ?Sized,
        F: FnMut(usize) -> u32,
    {
        self.max_score_up_to_with(
            start_block_idx,
            up_to,
            block_least_doc_id,
            |impacts, entry_idx| impacts.entry_score(entry_idx, query_weight, scorer),
        )
    }

    #[cfg(test)]
    fn max_score_up_to_with<E, F>(
        &self,
        start_block_idx: usize,
        up_to: u64,
        mut block_least_doc_id: F,
        mut entry_score: E,
    ) -> ImpactScore
    where
        E: FnMut(&Self, usize) -> f32,
        F: FnMut(usize) -> u32,
    {
        let mut block_idx = start_block_idx;
        let mut max_score = 0.0_f32;
        let mut entries_scanned = 0usize;

        while block_idx < self.level0_len {
            if u64::from(block_least_doc_id(block_idx)) > up_to {
                break;
            }
            let group_idx = block_idx / IMPACT_LEVEL1_BLOCKS;
            let group_start = group_idx * IMPACT_LEVEL1_BLOCKS;
            let group_end = ((group_idx + 1) * IMPACT_LEVEL1_BLOCKS).min(self.level0_len);
            if block_idx == group_start {
                let level1_entry_idx = self.level0_len + group_idx;
                match decode_header(self.entries.value(level1_entry_idx)) {
                    Ok(header) if u64::from(header.doc_up_to) <= up_to => {
                        max_score = max_score.max(entry_score(self, level1_entry_idx));
                        entries_scanned += 1;
                        block_idx = group_end;
                        continue;
                    }
                    Ok(_) => {}
                    Err(_) => {
                        return ImpactScore {
                            score: f32::INFINITY,
                            entries_scanned: entries_scanned + 1,
                        };
                    }
                }
            }

            max_score = max_score.max(entry_score(self, block_idx));
            entries_scanned += 1;
            match decode_header(self.entries.value(block_idx)) {
                Ok(header) if u64::from(header.doc_up_to) >= up_to => break,
                Ok(_) => {}
                Err(_) => {
                    return ImpactScore {
                        score: f32::INFINITY,
                        entries_scanned,
                    };
                }
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
        let Ok(header) = decode_header(bytes) else {
            return f32::INFINITY;
        };
        let mut max_doc_weight = 0.0_f32;
        let mut offset = 8usize;
        for _ in 0..header.pair_count {
            let freq = read_u32_le(bytes, offset);
            let doc_len = read_u32_le(bytes, offset + 4);
            max_doc_weight = max_doc_weight.max(scorer.doc_weight(freq, doc_len));
            offset += 8;
        }
        query_weight * max_doc_weight
    }
}

pub struct ImpactSkipDataBuilder {
    entries: LargeBinaryBuilder,
    level0_len: usize,
    level1_entries: Vec<Vec<u8>>,
    level1_docs: Vec<(u32, u32, u32)>,
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
        }
    }

    pub fn append_block(&mut self, docs: &[(u32, u32, u32)]) -> Result<()> {
        let bytes = encode_impact_entry(docs)?;
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
        ImpactSkipData::new(self.entries.finish(), self.level0_len)
    }

    fn flush_level1(&mut self) -> Result<()> {
        let bytes = encode_impact_entry(self.level1_docs.as_slice())?;
        self.level1_entries.push(bytes);
        self.level1_docs.clear();
        Ok(())
    }
}

#[cfg(test)]
pub fn build_impact_skip_data(blocks: &[Vec<(u32, u32, u32)>]) -> Result<ImpactSkipData> {
    let block_size = blocks.iter().map(Vec::len).max().unwrap_or(0);
    let mut builder = ImpactSkipDataBuilder::with_capacity(blocks.len(), block_size);
    for block in blocks {
        builder.append_block(block)?;
    }
    builder.finish()
}

fn encode_impact_entry(docs: &[(u32, u32, u32)]) -> Result<Vec<u8>> {
    let doc_up_to = docs
        .last()
        .map(|(doc_id, _, _)| *doc_id)
        .unwrap_or_default();
    let frontier = impact_frontier(docs);
    let mut bytes = Vec::with_capacity(8 + frontier.len() * 8);
    bytes.extend_from_slice(&doc_up_to.to_le_bytes());
    let pair_count = u32::try_from(frontier.len()).map_err(|_| {
        Error::index("impact frontier too large to encode as u32 pair count".to_string())
    })?;
    bytes.extend_from_slice(&pair_count.to_le_bytes());
    for (freq, doc_len) in frontier {
        bytes.extend_from_slice(&freq.to_le_bytes());
        bytes.extend_from_slice(&doc_len.to_le_bytes());
    }
    Ok(bytes)
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
    let expected_len = 8 + pair_count * 8;
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

    use super::*;
    use crate::scalar::inverted::scorer::{MemBM25Scorer, Scorer};

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
        let level0 = encode_impact_entry(&[(0, 1, 10)]).unwrap();
        let malformed_level1 = vec![1, 2, 3];
        let entries = LargeBinaryArray::from_opt_vec(vec![
            Some(level0.as_slice()),
            Some(malformed_level1.as_slice()),
        ]);
        let impacts = ImpactSkipData::new(entries, 1).unwrap();

        assert_eq!(impacts.level1_doc_up_to(0), None);
    }

    #[test]
    fn impact_level1_score_covers_level0_scores_in_group() {
        let blocks = (0..40)
            .map(|block| vec![(block as u32, 1 + block as u32 % 3, 10)])
            .collect::<Vec<_>>();
        let impacts = build_impact_skip_data(&blocks).unwrap();
        let scorer = MemBM25Scorer::new(400, 40, HashMap::from([(String::from("token"), 40usize)]));

        for group_idx in 0..impacts.level1_len() {
            let group_score = impacts.level1_score(group_idx, 1.0, &scorer);
            let group_start = group_idx * IMPACT_LEVEL1_BLOCKS;
            let group_end = ((group_idx + 1) * IMPACT_LEVEL1_BLOCKS).min(impacts.level0_len());
            for block_idx in group_start..group_end {
                let block_score = impacts.level0_score(block_idx, 1.0, &scorer);
                assert!(
                    group_score + 1e-6 >= block_score,
                    "level1 score {} must cover level0 score {} of block {}",
                    group_score,
                    block_score,
                    block_idx
                );
            }
        }
    }

    #[test]
    fn impact_level0_doc_up_to_reports_block_bounds() {
        let blocks = vec![
            vec![(0, 1, 10), (5, 2, 20)],
            vec![(7, 1, 10), (12, 1, 10)],
            vec![(20, 3, 30)],
        ];
        let impacts = build_impact_skip_data(&blocks).unwrap();
        assert_eq!(impacts.level0_doc_up_to(0), Some(5));
        assert_eq!(impacts.level0_doc_up_to(1), Some(12));
        assert_eq!(impacts.level0_doc_up_to(2), Some(20));
        assert_eq!(impacts.level0_doc_up_to(3), None);
    }

    #[test]
    fn impact_entries_are_decoded_lazily() {
        let level0_0 = encode_impact_entry(&[(0, 1, 10)]).unwrap();
        let malformed_level0_1 = vec![1, 2, 3];
        let level1 = encode_impact_entry(&[(0, 1, 10), (1, 1, 10)]).unwrap();
        let entries = LargeBinaryArray::from_opt_vec(vec![
            Some(level0_0.as_slice()),
            Some(malformed_level0_1.as_slice()),
            Some(level1.as_slice()),
        ]);
        let impacts = ImpactSkipData::new(entries, 2).unwrap();
        let scorer = MemBM25Scorer::new(10, 10, HashMap::from([(String::from("token"), 2usize)]));

        let score = impacts.max_score_up_to(0, 0, |_| 0, 1.0, &scorer);
        assert!(score.score.is_finite());
        assert_eq!(score.entries_scanned, 1);

        assert_eq!(impacts.level0_score(1, 1.0, &scorer), f32::INFINITY);
        assert_eq!(impacts.level0_doc_up_to(1), None);
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
