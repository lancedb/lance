// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

mod should_maxscore;

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};

use futures::{StreamExt, TryStreamExt, stream};
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, spawn_cpu};
use lance_core::{Error, Result};
use lance_select::RowAddrMask;
use lance_tokenizer::{SimpleTokenizer, TextAnalyzer};

use super::{
    InvertedIndex, build_global_bm25_scorer,
    document_tokenizer::{DocType, JsonTokenizer, LanceTokenizer},
    documents::{DocId, DocLengths, DocVisibility, PartitionDocuments, ResidentAddressProjection},
    index::{DocSet, InvertedPartition},
    query::{
        FtsQuery, FtsSearchParams, MatchQuery, Operator, PhraseQuery, Tokens, collect_query_tokens,
    },
    scorer::MemBM25Scorer,
    tokenizer::document_tokenizer::TextTokenizer,
    wand::{
        FLAT_SEARCH_PERCENT_THRESHOLD, LegacyWandDocuments, ModernWandDocuments, PostingIterator,
        WandCursor, WandDocuments, score_sum_upper_bound_factor,
    },
};
use crate::{metrics::MetricsCollector, prefilter::PreFilter};

use self::should_maxscore::ShouldMaxScoreScorer;

const DEFAULT_BLOCK_SIZE: usize = 128;
const SCORE_FLOOR_RESOLUTION_BATCH_SIZE: usize = DEFAULT_BLOCK_SIZE;
/// Bound deferred exclusion I/O concurrency while letting each completed
/// batch raise the global score floor before more partitions are admitted.
const DEFERRED_MUST_NOT_LOAD_BATCH_SIZE: usize = 8;

/// One exact FTS result in a compound collector's candidate domain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ScoredRow<K = u64> {
    pub row_id: K,
    pub score: f32,
}

#[cfg(test)]
impl ScoredRow<u64> {
    pub(super) fn new(row_id: u64, score: f32) -> Result<Self> {
        if !score.is_finite() {
            return Err(Error::invalid_input(format!(
                "FTS score for row_id={row_id} must be finite, got {score}"
            )));
        }
        Ok(Self { row_id, score })
    }
}

/// Conservative score bounds for a document range.
///
/// The lower bound is needed by signed compositions such as [`BoostScorer`].
/// Arithmetic widens both sides by one representable `f32` so nested
/// operations cannot round an upper bound below an exact score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ScoreBounds {
    lower: f32,
    upper: f32,
}

impl ScoreBounds {
    const ZERO: Self = Self {
        lower: 0.0,
        upper: 0.0,
    };
    const UNBOUNDED: Self = Self {
        lower: f32::NEG_INFINITY,
        upper: f32::INFINITY,
    };

    #[cfg(test)]
    fn point(score: f32) -> Result<Self> {
        if !score.is_finite() {
            return Err(Error::invalid_input(format!(
                "FTS score bounds require a finite score, got {score}"
            )));
        }
        Ok(Self {
            lower: score,
            upper: score,
        })
    }

    fn scale_non_negative(self, factor: f32) -> Self {
        debug_assert!(factor.is_finite() && factor >= 0.0);
        if factor == 0.0 {
            return Self::ZERO;
        }
        if !self.lower.is_finite() || !self.upper.is_finite() {
            return Self::UNBOUNDED;
        }
        Self {
            lower: next_down(self.lower * factor),
            upper: next_up(self.upper * factor),
        }
    }

    fn include_zero(self) -> Self {
        Self {
            lower: self.lower.min(0.0),
            upper: self.upper.max(0.0),
        }
    }

    fn add(self, other: Self) -> Self {
        if !self.lower.is_finite()
            || !self.upper.is_finite()
            || !other.lower.is_finite()
            || !other.upper.is_finite()
        {
            return Self::UNBOUNDED;
        }
        Self {
            lower: next_down(self.lower + other.lower),
            upper: next_up(self.upper + other.upper),
        }
    }

    fn subtract_scaled(self, other: Self, factor: f32) -> Self {
        let penalty = other.scale_non_negative(factor);
        if !self.lower.is_finite()
            || !self.upper.is_finite()
            || !penalty.lower.is_finite()
            || !penalty.upper.is_finite()
        {
            return Self::UNBOUNDED;
        }
        Self {
            lower: next_down(self.lower - penalty.upper),
            upper: next_up(self.upper - penalty.lower),
        }
    }
}

fn next_up(value: f32) -> f32 {
    if !value.is_finite() {
        return value;
    }
    if value == 0.0 {
        return f32::from_bits(1);
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f32::from_bits(bits + 1)
    } else {
        f32::from_bits(bits - 1)
    }
}

fn next_down(value: f32) -> f32 {
    if !value.is_finite() {
        return value;
    }
    if value == 0.0 {
        return f32::from_bits((1_u32 << 31) | 1);
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f32::from_bits(bits - 1)
    } else {
        f32::from_bits(bits + 1)
    }
}

fn checked_score(score: f32, context: &str) -> Result<f32> {
    if score.is_finite() {
        Ok(score)
    } else {
        Err(Error::invalid_input(format!(
            "{context} produced a non-finite FTS score: {score}"
        )))
    }
}

/// Internal document-at-a-time scorer protocol for compound FTS.
///
/// Implementations iterate matching partition-local document ids in ascending
/// order and expose the corresponding candidate key separately. A collector
/// may shallow-advance independently of the exact iterator, inspect a
/// conservative range bound, and monotonically raise the competitive score.
/// `matches` is the optional two-phase confirmation hook: cheap approximations
/// return a candidate from `next` / `advance` and defer expensive checks such
/// as phrase positions until confirmation.
pub(super) trait ComposableScorer: Send {
    fn doc(&self) -> Option<u64>;
    fn document_key(&self) -> Option<u64> {
        self.doc()
    }
    fn next(&mut self) -> Result<Option<u64>>;
    fn advance(&mut self, target: u64) -> Result<Option<u64>>;
    fn cost(&self) -> usize;
    fn score(&mut self) -> Result<f32>;
    fn advance_shallow(&mut self, target: u64) -> Result<u64>;
    fn score_bounds(&mut self, up_to: u64) -> Result<ScoreBounds>;
    /// Conservative list-wide score upper bound, independent of iterator
    /// position. `None` keeps the scorer on exact eager composition paths.
    fn global_score_upper_bound(&self) -> Option<f32> {
        None
    }
    fn set_min_competitive_score(&mut self, min_score: f32) -> Result<()>;

    fn matches(&mut self) -> Result<bool> {
        Ok(true)
    }

    /// Estimated relative cost of [`Self::matches`], stable for this scorer's
    /// lifetime. `None` means no ordering hint, not that confirmation may be skipped.
    fn match_cost(&self) -> Option<f32> {
        None
    }

    fn scores_non_negative(&self) -> bool {
        false
    }
}

type BoxScorer<'a> = Box<dyn ComposableScorer + 'a>;

fn sum_global_score_upper_bounds(children: &[BoxScorer<'_>]) -> Option<f32> {
    children.iter().try_fold(0.0, |upper, child| {
        let child_upper = child.global_score_upper_bound()?;
        if !child_upper.is_finite() || child_upper < 0.0 {
            return None;
        }
        let combined = ScoreBounds { lower: 0.0, upper }
            .add(ScoreBounds {
                lower: 0.0,
                upper: child_upper,
            })
            .upper;
        combined.is_finite().then_some(combined)
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompoundScoreMode {
    Scoring,
    CompleteNoScores,
}

#[derive(Debug, Clone)]
enum CompoundScorerPlan {
    Leaf {
        index: usize,
        boost: f32,
    },
    Boost {
        positive: Box<Self>,
        negative: Box<Self>,
        negative_boost: f32,
    },
    MultiMatch(Vec<Self>),
    Boolean {
        should: Vec<Self>,
        must: Vec<Self>,
        must_not: Vec<Self>,
        score_mode: CompoundScoreMode,
    },
}

impl CompoundScorerPlan {
    fn from_query(
        query: &FtsQuery,
        num_leaves: &mut usize,
        score_mode: CompoundScoreMode,
    ) -> Result<Self> {
        match query {
            FtsQuery::Match(query) => {
                let index = *num_leaves;
                *num_leaves += 1;
                Ok(Self::Leaf {
                    index,
                    boost: query.boost,
                })
            }
            FtsQuery::Phrase(_) => {
                let index = *num_leaves;
                *num_leaves += 1;
                Ok(Self::Leaf { index, boost: 1.0 })
            }
            FtsQuery::Boost(query) => {
                if !query.negative_boost.is_finite() || query.negative_boost < 0.0 {
                    return Err(Error::invalid_input(format!(
                        "BoostQuery negative_boost must be finite and non-negative, got {}",
                        query.negative_boost
                    )));
                }
                let positive = Self::from_query(&query.positive, num_leaves, score_mode)?;
                if score_mode == CompoundScoreMode::CompleteNoScores {
                    return Ok(positive);
                }
                Ok(Self::Boost {
                    positive: Box::new(positive),
                    negative: Box::new(Self::from_query(&query.negative, num_leaves, score_mode)?),
                    negative_boost: query.negative_boost,
                })
            }
            FtsQuery::MultiMatch(query) => Ok(Self::MultiMatch(
                query
                    .match_queries
                    .iter()
                    .map(|query| {
                        Self::from_query(&FtsQuery::Match(query.clone()), num_leaves, score_mode)
                    })
                    .collect::<Result<Vec<_>>>()?,
            )),
            FtsQuery::Boolean(query) => {
                let should = if score_mode == CompoundScoreMode::CompleteNoScores
                    && !query.must.is_empty()
                {
                    Vec::new()
                } else {
                    query
                        .should
                        .iter()
                        .map(|query| Self::from_query(query, num_leaves, score_mode))
                        .collect::<Result<Vec<_>>>()?
                };
                Ok(Self::Boolean {
                    should,
                    must: query
                        .must
                        .iter()
                        .map(|query| Self::from_query(query, num_leaves, score_mode))
                        .collect::<Result<Vec<_>>>()?,
                    must_not: query
                        .must_not
                        .iter()
                        .map(|query| {
                            Self::from_query(query, num_leaves, CompoundScoreMode::CompleteNoScores)
                        })
                        .collect::<Result<Vec<_>>>()?,
                    score_mode,
                })
            }
        }
    }

    fn build<'a>(
        &self,
        leaves: &mut [Option<BoxScorer<'a>>],
        metrics: &'a dyn MetricsCollector,
    ) -> Result<BoxScorer<'a>> {
        match self {
            Self::Leaf { index, boost } => {
                let leaf = leaves
                    .get_mut(*index)
                    .and_then(Option::take)
                    .ok_or_else(|| {
                        Error::internal(format!(
                            "compound FTS scorer references missing leaf index {index}"
                        ))
                    })?;
                Ok(Box::new(ScaleScorer::try_new(leaf, *boost)?))
            }
            Self::Boost {
                positive,
                negative,
                negative_boost,
            } => Ok(Box::new(BoostScorer::try_new(
                positive.build(leaves, metrics)?,
                negative.build(leaves, metrics)?,
                *negative_boost,
            )?)),
            Self::MultiMatch(children) => Ok(Box::new(DisjunctionScorer::try_new(
                children
                    .iter()
                    .map(|child| child.build(leaves, metrics))
                    .collect::<Result<Vec<_>>>()?,
                DisjunctionScore::Max,
            )?)),
            Self::Boolean {
                should,
                must,
                must_not,
                score_mode,
            } => Ok(Box::new(
                BooleanScorer::try_new_with_metrics_and_score_mode(
                    should
                        .iter()
                        .map(|child| child.build(leaves, metrics))
                        .collect::<Result<Vec<_>>>()?,
                    must.iter()
                        .map(|child| child.build(leaves, metrics))
                        .collect::<Result<Vec<_>>>()?,
                    must_not
                        .iter()
                        .map(|child| child.build(leaves, metrics))
                        .collect::<Result<Vec<_>>>()?,
                    Some(metrics),
                    *score_mode,
                )?,
            )),
        }
    }

    /// Build a scorer whose score is an upper envelope of the complete query
    /// score while ignoring every prohibited subtree.
    ///
    /// The gate runs before prohibited postings are loaded. Boolean
    /// exclusions can only remove matches. A BoostQuery's negative side can be
    /// dropped when [`Self::positive_gate_is_safe`] proves it cannot itself
    /// score below zero. Under that precondition, false positives only load
    /// deferred postings earlier than necessary.
    fn build_gate<'a>(
        &self,
        leaves: &mut [Option<BoxScorer<'a>>],
        metrics: &'a dyn MetricsCollector,
    ) -> Result<BoxScorer<'a>> {
        match self {
            Self::Leaf { index, boost } => {
                let leaf = leaves
                    .get_mut(*index)
                    .and_then(Option::take)
                    .ok_or_else(|| {
                        Error::internal(format!(
                            "compound FTS positive gate references missing leaf index {index}"
                        ))
                    })?;
                Ok(Box::new(ScaleScorer::try_new(leaf, *boost)?))
            }
            Self::Boost { positive, .. } => positive.build_gate(leaves, metrics),
            Self::MultiMatch(children) => Ok(Box::new(DisjunctionScorer::try_new(
                children
                    .iter()
                    .map(|child| child.build_gate(leaves, metrics))
                    .collect::<Result<Vec<_>>>()?,
                DisjunctionScore::Max,
            )?)),
            Self::Boolean { should, must, .. } => {
                Ok(Box::new(BooleanScorer::try_new_with_metrics(
                    should
                        .iter()
                        .map(|child| child.build_gate(leaves, metrics))
                        .collect::<Result<Vec<_>>>()?,
                    must.iter()
                        .map(|child| child.build_gate(leaves, metrics))
                        .collect::<Result<Vec<_>>>()?,
                    Vec::new(),
                    Some(metrics),
                )?))
            }
        }
    }

    /// Whether [`Self::build_gate`] is guaranteed not to underestimate any
    /// final score. A BoostQuery may only drop its negative side when that
    /// subtree cannot itself produce a negative score.
    fn positive_gate_is_safe(&self) -> bool {
        match self {
            Self::Leaf { boost, .. } => boost.is_finite() && *boost >= 0.0,
            Self::Boost {
                positive,
                negative,
                negative_boost,
            } => {
                positive.positive_gate_is_safe()
                    && (*negative_boost == 0.0 || negative.scores_provably_non_negative())
            }
            Self::MultiMatch(children) => children.iter().all(Self::positive_gate_is_safe),
            Self::Boolean { should, must, .. } => {
                should.iter().chain(must).all(Self::positive_gate_is_safe)
            }
        }
    }

    fn scores_provably_non_negative(&self) -> bool {
        match self {
            Self::Leaf { boost, .. } => boost.is_finite() && *boost >= 0.0,
            Self::Boost {
                positive,
                negative_boost,
                ..
            } => *negative_boost == 0.0 && positive.scores_provably_non_negative(),
            Self::MultiMatch(children) => children.iter().all(Self::scores_provably_non_negative),
            Self::Boolean { should, must, .. } => should
                .iter()
                .chain(must)
                .all(Self::scores_provably_non_negative),
        }
    }
}

impl<D: WandDocuments + Sync> ComposableScorer for WandCursor<'_, D> {
    fn doc(&self) -> Option<u64> {
        self.doc()
    }

    fn document_key(&self) -> Option<u64> {
        self.document_key()
    }

    fn next(&mut self) -> Result<Option<u64>> {
        self.next()
    }

    fn advance(&mut self, target: u64) -> Result<Option<u64>> {
        self.advance(target)
    }

    fn cost(&self) -> usize {
        self.cost()
    }

    fn score(&mut self) -> Result<f32> {
        self.current_score()
    }

    fn advance_shallow(&mut self, target: u64) -> Result<u64> {
        self.advance_shallow(target)
    }

    fn score_bounds(&mut self, up_to: u64) -> Result<ScoreBounds> {
        Ok(ScoreBounds {
            lower: 0.0,
            upper: self.score_upper_bound(up_to)?,
        })
    }

    fn global_score_upper_bound(&self) -> Option<f32> {
        WandCursor::global_score_upper_bound(self)
    }

    fn set_min_competitive_score(&mut self, min_score: f32) -> Result<()> {
        self.set_min_competitive_score(min_score)
    }

    fn matches(&mut self) -> Result<bool> {
        self.matches()
    }

    fn match_cost(&self) -> Option<f32> {
        self.match_cost()
    }

    fn scores_non_negative(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy)]
#[cfg(test)]
struct ShallowRange {
    target: u64,
    up_to: u64,
    start: usize,
    end: usize,
}

/// Exact in-memory scorer used to unit-test compound nodes and the collector.
#[cfg(test)]
struct MaterializedScorer {
    rows: Vec<ScoredRow>,
    block_size: usize,
    index: Option<usize>,
    shallow: Option<ShallowRange>,
    min_competitive_score: f32,
    scores_non_negative: bool,
}

#[cfg(test)]
impl MaterializedScorer {
    fn try_new(mut rows: Vec<ScoredRow>) -> Result<Self> {
        rows.sort_unstable_by_key(|row| row.row_id);
        for pair in rows.windows(2) {
            if pair[0].row_id == pair[1].row_id {
                return Err(Error::internal(format!(
                    "FTS leaf scorer produced duplicate row_id={}",
                    pair[0].row_id
                )));
            }
        }
        let scores_non_negative = rows.iter().all(|row| row.score >= 0.0);
        Ok(Self {
            rows,
            block_size: DEFAULT_BLOCK_SIZE,
            index: None,
            shallow: None,
            min_competitive_score: f32::NEG_INFINITY,
            scores_non_negative,
        })
    }

    #[cfg(test)]
    fn with_block_size(mut self, block_size: usize) -> Self {
        assert!(block_size > 0);
        self.block_size = block_size;
        self
    }

    fn block_bounds(&self, start: usize, end: usize) -> Result<ScoreBounds> {
        let Some(first) = self.rows.get(start) else {
            return Ok(ScoreBounds::ZERO);
        };
        let mut bounds = ScoreBounds::point(first.score)?;
        for row in &self.rows[start + 1..end] {
            bounds.lower = bounds.lower.min(row.score);
            bounds.upper = bounds.upper.max(row.score);
        }
        Ok(bounds)
    }

    fn position_at(&mut self, mut index: usize) -> Result<Option<u64>> {
        while index < self.rows.len() {
            let block_start = (index / self.block_size) * self.block_size;
            let block_end = (block_start + self.block_size).min(self.rows.len());
            if self.block_bounds(block_start, block_end)?.upper < self.min_competitive_score {
                index = block_end;
                continue;
            }
            self.index = Some(index);
            self.shallow = None;
            return Ok(Some(self.rows[index].row_id));
        }
        self.index = None;
        self.shallow = None;
        Ok(None)
    }
}

#[cfg(test)]
impl ComposableScorer for MaterializedScorer {
    fn doc(&self) -> Option<u64> {
        self.index.map(|index| self.rows[index].row_id)
    }

    fn next(&mut self) -> Result<Option<u64>> {
        self.position_at(self.index.map_or(0, |index| index + 1))
    }

    fn advance(&mut self, target: u64) -> Result<Option<u64>> {
        if self.doc().is_some_and(|doc| doc >= target) {
            return Ok(self.doc());
        }
        let start = self.index.map_or(0, |index| index + 1);
        let offset = self.rows[start..].partition_point(|row| row.row_id < target);
        self.position_at(start + offset)
    }

    fn cost(&self) -> usize {
        self.rows.len()
    }

    fn score(&mut self) -> Result<f32> {
        self.index
            .map(|index| self.rows[index].score)
            .ok_or_else(|| Error::internal("FTS scorer is not positioned on a document"))
    }

    fn advance_shallow(&mut self, target: u64) -> Result<u64> {
        let start = self.rows.partition_point(|row| row.row_id < target);
        if start == self.rows.len() {
            self.shallow = Some(ShallowRange {
                target,
                up_to: u64::MAX,
                start,
                end: start,
            });
            return Ok(u64::MAX);
        }
        let block_start = (start / self.block_size) * self.block_size;
        let end = (block_start + self.block_size).min(self.rows.len());
        let up_to = self
            .rows
            .get(end)
            .map(|next| next.row_id.saturating_sub(1))
            .unwrap_or(u64::MAX);
        self.shallow = Some(ShallowRange {
            target,
            up_to,
            start,
            end,
        });
        Ok(up_to)
    }

    fn score_bounds(&mut self, up_to: u64) -> Result<ScoreBounds> {
        let shallow = self.shallow.ok_or_else(|| {
            Error::internal("score_bounds requires advance_shallow on the FTS scorer")
        })?;
        if up_to < shallow.target || up_to > shallow.up_to {
            return Err(Error::internal(format!(
                "FTS score bound up_to={up_to} is outside shallow range [{}, {}]",
                shallow.target, shallow.up_to
            )));
        }
        let end = shallow.start
            + self.rows[shallow.start..shallow.end].partition_point(|row| row.row_id <= up_to);
        self.block_bounds(shallow.start, end)
    }

    fn global_score_upper_bound(&self) -> Option<f32> {
        self.rows
            .iter()
            .map(|row| row.score)
            .max_by(f32::total_cmp)
            .or(Some(0.0))
    }

    fn set_min_competitive_score(&mut self, min_score: f32) -> Result<()> {
        if min_score.is_nan() {
            return Err(Error::invalid_input(
                "minimum competitive FTS score cannot be NaN",
            ));
        }
        if min_score > self.min_competitive_score {
            self.min_competitive_score = min_score;
        }
        Ok(())
    }

    fn scores_non_negative(&self) -> bool {
        self.scores_non_negative
    }
}

/// Monotonic score-only floor shared by partition-local top-k collectors.
///
/// Equal-score candidates are never pruned because final ordering also uses
/// row id. The score-only floor is therefore a safe lower bound even when
/// partitions encounter ties in different orders.
#[derive(Debug)]
pub(super) struct CompetitiveScore {
    bits: AtomicU32,
}

impl Default for CompetitiveScore {
    fn default() -> Self {
        Self {
            bits: AtomicU32::new(f32::NEG_INFINITY.to_bits()),
        }
    }
}

impl CompetitiveScore {
    fn get(&self) -> f32 {
        f32::from_bits(self.bits.load(AtomicOrdering::Relaxed))
    }

    fn raise(&self, score: f32) {
        debug_assert!(!score.is_nan());
        let mut current = self.bits.load(AtomicOrdering::Relaxed);
        while score > f32::from_bits(current) {
            match self.bits.compare_exchange_weak(
                current,
                score.to_bits(),
                AtomicOrdering::Relaxed,
                AtomicOrdering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct HeapRow<K>(ScoredRow<K>);

impl<K: PartialEq> PartialEq for HeapRow<K> {
    fn eq(&self, other: &Self) -> bool {
        self.0.row_id == other.0.row_id && self.0.score.to_bits() == other.0.score.to_bits()
    }
}

impl<K: Eq> Eq for HeapRow<K> {}

impl<K: Ord> PartialOrd for HeapRow<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: Ord> Ord for HeapRow<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        // The worst result is the heap maximum: lower score, then higher row id.
        other
            .0
            .score
            .total_cmp(&self.0.score)
            .then_with(|| self.0.row_id.cmp(&other.0.row_id))
    }
}

fn compare_scored_rows<K: Ord>(left: &ScoredRow<K>, right: &ScoredRow<K>) -> Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| left.row_id.cmp(&right.row_id))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CollectionStatus {
    Complete,
    ScoreFloorOverflow,
}

#[derive(Debug, Clone, Copy)]
enum TieHandling {
    ResolveByKey,
    RetainScoreFloor { max_buffered: usize },
}

/// The sole owner of top-k state for a compound scorer tree.
///
/// Resolved document keys use the normal `(score DESC, row_id ASC)` ordering
/// and keep exactly `limit` rows. An unresolved partition temporarily retains
/// its kth-score floor, but stops once the bounded resolution buffer fills.
pub(super) struct TopKCollector<K = u64> {
    limit: usize,
    heap: BinaryHeap<HeapRow<K>>,
    competitive_score: Arc<CompetitiveScore>,
    tie_handling: TieHandling,
    peak_buffered: usize,
}

impl<K: Copy + Ord> TopKCollector<K> {
    pub(super) fn new(limit: usize) -> Self {
        Self::with_competitive_score(limit, Arc::new(CompetitiveScore::default()))
    }

    pub(super) fn with_competitive_score(
        limit: usize,
        competitive_score: Arc<CompetitiveScore>,
    ) -> Self {
        Self::with_tie_handling(limit, competitive_score, TieHandling::ResolveByKey)
    }

    fn retaining_score_floor(
        limit: usize,
        competitive_score: Arc<CompetitiveScore>,
        max_buffered: usize,
    ) -> Self {
        debug_assert!(max_buffered >= limit);
        Self::with_tie_handling(
            limit,
            competitive_score,
            TieHandling::RetainScoreFloor { max_buffered },
        )
    }

    fn with_tie_handling(
        limit: usize,
        competitive_score: Arc<CompetitiveScore>,
        tie_handling: TieHandling,
    ) -> Self {
        Self {
            limit,
            heap: BinaryHeap::with_capacity(limit.min(DEFAULT_BLOCK_SIZE)),
            competitive_score,
            tie_handling,
            peak_buffered: 0,
        }
    }

    fn insert(&mut self, row: ScoredRow<K>) -> CollectionStatus {
        if self.limit == 0 {
            return CollectionStatus::Complete;
        }
        if self.heap.len() < self.limit {
            self.heap.push(HeapRow(row));
        } else {
            let worst = self.heap.peek().expect("a full top-k heap is non-empty").0;
            match row.score.total_cmp(&worst.score) {
                Ordering::Less => {}
                Ordering::Equal => match self.tie_handling {
                    TieHandling::ResolveByKey => {
                        if row.row_id < worst.row_id {
                            self.heap.pop();
                            self.heap.push(HeapRow(row));
                        }
                    }
                    TieHandling::RetainScoreFloor { max_buffered } => {
                        if self.heap.len() >= max_buffered {
                            self.raise_competitive_score();
                            return CollectionStatus::ScoreFloorOverflow;
                        }
                        self.heap.push(HeapRow(row));
                    }
                },
                Ordering::Greater => {
                    match self.tie_handling {
                        TieHandling::ResolveByKey => {
                            self.heap.pop();
                            self.heap.push(HeapRow(row));
                        }
                        TieHandling::RetainScoreFloor { max_buffered } => {
                            self.heap.push(HeapRow(row));
                            self.prune_obsolete_score_floors();
                            if self.heap.len() > max_buffered {
                                // This collector is discarded and the partition is
                                // retried with resolved keys. Keep its observable
                                // working set within the advertised bound meanwhile.
                                self.heap.pop();
                                self.raise_competitive_score();
                                return CollectionStatus::ScoreFloorOverflow;
                            }
                        }
                    }
                }
            }
        }
        self.peak_buffered = self.peak_buffered.max(self.heap.len());
        self.raise_competitive_score();
        CollectionStatus::Complete
    }

    fn raise_competitive_score(&self) {
        if self.heap.len() >= self.limit
            && let Some(worst) = self.heap.peek()
        {
            self.competitive_score.raise(worst.0.score);
        }
    }

    fn prune_obsolete_score_floors(&mut self) {
        while self.heap.len() > self.limit {
            let floor = self
                .heap
                .peek()
                .expect("a non-empty top-k heap has a score floor")
                .0
                .score;
            let floor_count = self
                .heap
                .iter()
                .filter(|row| row.0.score.total_cmp(&floor) == Ordering::Equal)
                .count();
            if self.heap.len() - floor_count < self.limit {
                break;
            }
            self.heap
                .retain(|row| row.0.score.total_cmp(&floor) != Ordering::Equal);
        }
    }

    #[cfg(test)]
    pub(super) fn collect_mapped(
        &mut self,
        scorer: &mut dyn ComposableScorer,
        map_document: impl FnMut(u64) -> Result<K>,
    ) -> Result<CollectionStatus> {
        self.collect_mapped_from(scorer, None, map_document)
    }

    fn collect_mapped_from(
        &mut self,
        scorer: &mut dyn ComposableScorer,
        start_doc: Option<u64>,
        mut map_document: impl FnMut(u64) -> Result<K>,
    ) -> Result<CollectionStatus> {
        if self.limit == 0 {
            return Ok(CollectionStatus::Complete);
        }
        let capacity_limit = match self.tie_handling {
            TieHandling::ResolveByKey => self.limit,
            TieHandling::RetainScoreFloor { max_buffered } => max_buffered,
        };
        let expected = scorer.cost().min(capacity_limit);
        self.heap
            .reserve(expected.saturating_sub(self.heap.capacity()));

        scorer.set_min_competitive_score(self.competitive_score.get())?;
        let mut doc = match start_doc {
            Some(start_doc) => scorer.advance(start_doc)?,
            None => scorer.next()?,
        };
        while let Some(doc_id) = doc {
            let min_score = self.competitive_score.get();
            scorer.set_min_competitive_score(min_score)?;
            let up_to = scorer.advance_shallow(doc_id)?;
            let bounds = scorer.score_bounds(up_to)?;
            if bounds.upper < min_score {
                doc = if up_to == u64::MAX {
                    None
                } else {
                    scorer.advance(up_to + 1)?
                };
                continue;
            }

            if let Some(match_cost) = scorer.match_cost()
                && (!match_cost.is_finite() || match_cost < 0.0)
            {
                return Err(Error::internal(format!(
                    "FTS scorer reported invalid two-phase match cost: {match_cost}"
                )));
            }
            if scorer.matches()? {
                let score = checked_score(scorer.score()?, "compound scorer")?;
                // A shared partition floor is already known to be globally
                // competitive. Scores strictly below it cannot enter final top-k.
                if score >= self.competitive_score.get() {
                    let document_key = scorer.document_key().ok_or_else(|| {
                        Error::internal(
                            "compound FTS scorer did not expose its current document key",
                        )
                    })?;
                    let status = self.insert(ScoredRow {
                        row_id: map_document(document_key)?,
                        score,
                    });
                    if status == CollectionStatus::ScoreFloorOverflow {
                        return Ok(status);
                    }
                }
            }
            let next_min_score = self.competitive_score.get();
            if next_min_score > min_score {
                // A scorer may have an unbounded fast path until the heap first
                // establishes a score floor. Publish that floor before asking
                // it for another document so it can switch modes immediately.
                scorer.set_min_competitive_score(next_min_score)?;
            }
            doc = scorer.next()?;
        }

        Ok(CollectionStatus::Complete)
    }

    fn into_candidates(self) -> Vec<ScoredRow<K>> {
        let mut rows = self.heap.into_iter().map(|row| row.0).collect::<Vec<_>>();
        rows.sort_unstable_by(compare_scored_rows);
        rows
    }

    fn into_rows(self) -> Vec<ScoredRow<K>> {
        let limit = self.limit;
        let mut rows = self.into_candidates();
        rows.truncate(limit);
        rows
    }
}

impl TopKCollector<u64> {
    #[cfg(test)]
    fn collect(mut self, scorer: &mut dyn ComposableScorer) -> Result<Vec<ScoredRow>> {
        self.collect_mapped(scorer, Ok)?;
        Ok(self.into_rows())
    }
}

/// Return the first exact match at or above a fixed score floor without
/// mutating the authoritative top-k collector.
fn first_competitive_match(
    scorer: &mut dyn ComposableScorer,
    min_score: f32,
) -> Result<Option<u64>> {
    if min_score.is_nan() {
        return Err(Error::invalid_input(
            "minimum competitive FTS score cannot be NaN",
        ));
    }
    scorer.set_min_competitive_score(min_score)?;
    let mut doc = scorer.next()?;
    while let Some(doc_id) = doc {
        if min_score != f32::NEG_INFINITY {
            let up_to = scorer.advance_shallow(doc_id)?;
            if scorer.score_bounds(up_to)?.upper < min_score {
                doc = if up_to == u64::MAX {
                    None
                } else {
                    scorer.advance(up_to + 1)?
                };
                continue;
            }
        }
        if let Some(match_cost) = scorer.match_cost()
            && (!match_cost.is_finite() || match_cost < 0.0)
        {
            return Err(Error::internal(format!(
                "FTS scorer reported invalid two-phase match cost: {match_cost}"
            )));
        }
        if scorer.matches()?
            && (min_score == f32::NEG_INFINITY
                || checked_score(scorer.score()?, "compound positive gate")? >= min_score)
        {
            return Ok(Some(doc_id));
        }
        doc = scorer.next()?;
    }
    Ok(None)
}

#[derive(Debug, Clone, Copy)]
pub(super) enum DisjunctionScore {
    Sum,
    Max,
}

struct EmptyScorer;

impl ComposableScorer for EmptyScorer {
    fn doc(&self) -> Option<u64> {
        None
    }

    fn next(&mut self) -> Result<Option<u64>> {
        Ok(None)
    }

    fn advance(&mut self, _target: u64) -> Result<Option<u64>> {
        Ok(None)
    }

    fn cost(&self) -> usize {
        0
    }

    fn score(&mut self) -> Result<f32> {
        Err(Error::internal(
            "score requested from an empty compound FTS scorer",
        ))
    }

    fn advance_shallow(&mut self, _target: u64) -> Result<u64> {
        Ok(u64::MAX)
    }

    fn score_bounds(&mut self, _up_to: u64) -> Result<ScoreBounds> {
        Ok(ScoreBounds::ZERO)
    }

    fn global_score_upper_bound(&self) -> Option<f32> {
        Some(0.0)
    }

    fn set_min_competitive_score(&mut self, _min_score: f32) -> Result<()> {
        Ok(())
    }

    fn scores_non_negative(&self) -> bool {
        true
    }
}

struct ScaleScorer<'a> {
    child: BoxScorer<'a>,
    factor: f32,
}

impl<'a> ScaleScorer<'a> {
    fn try_new(child: BoxScorer<'a>, factor: f32) -> Result<Self> {
        if !factor.is_finite() || factor < 0.0 {
            return Err(Error::invalid_input(format!(
                "MatchQuery boost must be finite and non-negative, got {factor}"
            )));
        }
        Ok(Self { child, factor })
    }
}

impl ComposableScorer for ScaleScorer<'_> {
    fn doc(&self) -> Option<u64> {
        self.child.doc()
    }

    fn document_key(&self) -> Option<u64> {
        self.child.document_key()
    }

    fn next(&mut self) -> Result<Option<u64>> {
        self.child.next()
    }

    fn advance(&mut self, target: u64) -> Result<Option<u64>> {
        self.child.advance(target)
    }

    fn cost(&self) -> usize {
        self.child.cost()
    }

    fn score(&mut self) -> Result<f32> {
        checked_score(self.child.score()? * self.factor, "MatchQuery boost")
    }

    fn advance_shallow(&mut self, target: u64) -> Result<u64> {
        self.child.advance_shallow(target)
    }

    fn score_bounds(&mut self, up_to: u64) -> Result<ScoreBounds> {
        Ok(self
            .child
            .score_bounds(up_to)?
            .scale_non_negative(self.factor))
    }

    fn global_score_upper_bound(&self) -> Option<f32> {
        self.child
            .global_score_upper_bound()
            .map(|upper| {
                ScoreBounds { lower: 0.0, upper }
                    .scale_non_negative(self.factor)
                    .upper
            })
            .filter(|upper| upper.is_finite() && *upper >= 0.0)
    }

    fn set_min_competitive_score(&mut self, min_score: f32) -> Result<()> {
        if self.factor > 0.0 {
            self.child
                .set_min_competitive_score(next_down(min_score / self.factor))?;
        }
        Ok(())
    }

    fn matches(&mut self) -> Result<bool> {
        self.child.matches()
    }

    fn match_cost(&self) -> Option<f32> {
        self.child.match_cost()
    }

    fn scores_non_negative(&self) -> bool {
        self.child.scores_non_negative()
    }
}

/// Union scorer used for Boolean SHOULD sums and MultiMatch DisMax.
pub(super) struct DisjunctionScorer<'a> {
    children: Vec<BoxScorer<'a>>,
    mode: DisjunctionScore,
    current: Option<u64>,
    confirmed_doc: Option<u64>,
    confirmed: Vec<bool>,
    min_competitive_score: f32,
}

impl<'a> DisjunctionScorer<'a> {
    pub(super) fn try_new(children: Vec<BoxScorer<'a>>, mode: DisjunctionScore) -> Result<Self> {
        if children.is_empty() {
            return Err(Error::internal(
                "FTS disjunction scorer requires at least one child",
            ));
        }
        let confirmed = vec![false; children.len()];
        Ok(Self {
            children,
            mode,
            current: None,
            confirmed_doc: None,
            confirmed,
            min_competitive_score: f32::NEG_INFINITY,
        })
    }

    fn set_current_from_children(&mut self) -> Option<u64> {
        self.current = self.children.iter().filter_map(|child| child.doc()).min();
        self.confirmed_doc = None;
        self.confirmed.fill(false);
        self.current
    }

    fn ensure_confirmed(&mut self) -> Result<bool> {
        let Some(current) = self.current else {
            return Ok(false);
        };
        if self.confirmed_doc == Some(current) {
            return Ok(self.confirmed.iter().any(|matched| *matched));
        }
        self.confirmed.fill(false);
        for (matched, child) in self.confirmed.iter_mut().zip(&mut self.children) {
            if child.doc() == Some(current) {
                *matched = child.matches()?;
            }
        }
        self.confirmed_doc = Some(current);
        Ok(self.confirmed.iter().any(|matched| *matched))
    }
}

impl ComposableScorer for DisjunctionScorer<'_> {
    fn doc(&self) -> Option<u64> {
        self.current
    }

    fn document_key(&self) -> Option<u64> {
        let current = self.current?;
        self.children
            .iter()
            .find(|child| child.doc() == Some(current))
            .and_then(|child| child.document_key())
    }

    fn next(&mut self) -> Result<Option<u64>> {
        match self.current {
            None => {
                for child in &mut self.children {
                    child.next()?;
                }
            }
            Some(current) => {
                for child in &mut self.children {
                    if child.doc() == Some(current) {
                        child.next()?;
                    }
                }
            }
        }
        Ok(self.set_current_from_children())
    }

    fn advance(&mut self, target: u64) -> Result<Option<u64>> {
        if self.current.is_some_and(|current| current >= target) {
            return Ok(self.current);
        }
        for child in &mut self.children {
            if child.doc().is_none_or(|doc| doc < target) {
                child.advance(target)?;
            }
        }
        Ok(self.set_current_from_children())
    }

    fn cost(&self) -> usize {
        self.children
            .iter()
            .map(|child| child.cost())
            .fold(0, usize::saturating_add)
    }

    fn score(&mut self) -> Result<f32> {
        if !self.ensure_confirmed()? {
            return Err(Error::internal(
                "FTS disjunction score requested for an unconfirmed document",
            ));
        }
        let mut score = match self.mode {
            DisjunctionScore::Sum => 0.0_f32,
            DisjunctionScore::Max => f32::NEG_INFINITY,
        };
        for (matched, child) in self.confirmed.iter().zip(&mut self.children) {
            if !matched {
                continue;
            }
            let child_score = child.score()?;
            score = match self.mode {
                DisjunctionScore::Sum => score + child_score,
                DisjunctionScore::Max => score.max(child_score),
            };
        }
        checked_score(score, "FTS disjunction")
    }

    fn advance_shallow(&mut self, target: u64) -> Result<u64> {
        let mut up_to = u64::MAX;
        for child in &mut self.children {
            if let Some(doc) = child.doc() {
                up_to = up_to.min(child.advance_shallow(target.max(doc))?);
            }
        }
        Ok(up_to)
    }

    fn score_bounds(&mut self, up_to: u64) -> Result<ScoreBounds> {
        let mut bounds = match self.mode {
            DisjunctionScore::Sum => ScoreBounds::ZERO,
            DisjunctionScore::Max => ScoreBounds {
                lower: f32::INFINITY,
                upper: f32::NEG_INFINITY,
            },
        };
        for child in &mut self.children {
            let child_bounds = if child.doc().is_some_and(|doc| doc <= up_to) {
                child.score_bounds(up_to)?
            } else {
                ScoreBounds::ZERO
            };
            bounds = match self.mode {
                DisjunctionScore::Sum => bounds.add(child_bounds.include_zero()),
                DisjunctionScore::Max => ScoreBounds {
                    lower: bounds.lower.min(child_bounds.lower),
                    upper: bounds.upper.max(child_bounds.upper),
                },
            };
        }
        if bounds.lower == f32::INFINITY {
            Ok(ScoreBounds::ZERO)
        } else {
            Ok(bounds)
        }
    }

    fn global_score_upper_bound(&self) -> Option<f32> {
        match self.mode {
            DisjunctionScore::Sum => sum_global_score_upper_bounds(&self.children),
            DisjunctionScore::Max => self.children.iter().try_fold(0.0_f32, |upper, child| {
                let child_upper = child.global_score_upper_bound()?;
                child_upper.is_finite().then_some(upper.max(child_upper))
            }),
        }
    }

    fn set_min_competitive_score(&mut self, min_score: f32) -> Result<()> {
        if min_score.is_nan() {
            return Err(Error::invalid_input(
                "minimum competitive FTS score cannot be NaN",
            ));
        }
        if min_score <= self.min_competitive_score {
            return Ok(());
        }
        self.min_competitive_score = min_score;
        // A child below a DisMax threshold cannot affect a competitive max.
        // Sum scorers need sibling-global bounds before translating the floor,
        // so they keep it at this node and prune from their combined block bound.
        if matches!(self.mode, DisjunctionScore::Max) {
            for child in &mut self.children {
                child.set_min_competitive_score(min_score)?;
            }
        }
        Ok(())
    }

    fn matches(&mut self) -> Result<bool> {
        self.ensure_confirmed()
    }

    fn match_cost(&self) -> Option<f32> {
        self.children
            .iter()
            .filter_map(|child| child.match_cost())
            .reduce(|left, right| left + right)
    }

    fn scores_non_negative(&self) -> bool {
        self.children
            .iter()
            .all(|child| child.scores_non_negative())
    }
}

/// Intersection scorer that requires and scores every Boolean MUST child.
pub(super) struct RequiredConjunctionScorer<'a> {
    children: Vec<BoxScorer<'a>>,
    /// Child indices sorted by approximation cost, omitted when query order is
    /// already cheapest-first. `children` remains in query order so scoring and
    /// score-bound arithmetic stay bit-for-bit stable.
    approximation_order: Option<Vec<usize>>,
    /// Child indices sorted by two-phase confirmation cost. Children without a
    /// cost hint remain in query order after costed confirmations.
    confirmation_order: Option<Vec<usize>>,
    current: Option<u64>,
    confirmed_doc: Option<u64>,
    confirmed: bool,
}

fn align_conjunction_children(
    children: &mut [BoxScorer<'_>],
    mut target: u64,
    child_index: impl Fn(usize) -> usize,
) -> Result<Option<u64>> {
    loop {
        for position in 0..children.len() {
            let child = &mut children[child_index(position)];
            if child.doc().is_none_or(|doc| doc < target) {
                let Some(doc) = child.advance(target)? else {
                    return Ok(None);
                };
                target = target.max(doc);
            }
        }
        let min_doc = children.iter().filter_map(|child| child.doc()).min();
        let max_doc = children.iter().filter_map(|child| child.doc()).max();
        if min_doc == max_doc {
            return Ok(min_doc);
        }
        target = max_doc.ok_or_else(|| {
            Error::internal("FTS conjunction lost a child while aligning scorers")
        })?;
    }
}

fn compare_confirmation_cost(
    left: &dyn ComposableScorer,
    right: &dyn ComposableScorer,
) -> Ordering {
    match (left.match_cost(), right.match_cost()) {
        (Some(left), Some(right)) => left.total_cmp(&right),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

fn confirm_conjunction_children(
    children: &mut [BoxScorer<'_>],
    child_index: impl Fn(usize) -> usize,
) -> Result<bool> {
    for position in 0..children.len() {
        if !children[child_index(position)].matches()? {
            return Ok(false);
        }
    }
    Ok(true)
}

impl<'a> RequiredConjunctionScorer<'a> {
    pub(super) fn try_new(children: Vec<BoxScorer<'a>>) -> Result<Self> {
        if children.is_empty() {
            return Err(Error::internal(
                "FTS conjunction scorer requires at least one child",
            ));
        }
        let approximation_order = if children
            .windows(2)
            .all(|pair| pair[0].cost() <= pair[1].cost())
        {
            None
        } else {
            let mut order = (0..children.len()).collect::<Vec<_>>();
            order.sort_by_key(|&index| (children[index].cost(), index));
            Some(order)
        };
        for (index, child) in children.iter().enumerate() {
            if let Some(match_cost) = child.match_cost()
                && (!match_cost.is_finite() || match_cost < 0.0)
            {
                return Err(Error::internal(format!(
                    "FTS conjunction child {index} reported invalid two-phase match cost: {match_cost}"
                )));
            }
        }
        let confirmation_order = if children.windows(2).all(|pair| {
            compare_confirmation_cost(pair[0].as_ref(), pair[1].as_ref()) != Ordering::Greater
        }) {
            None
        } else {
            let mut order = (0..children.len()).collect::<Vec<_>>();
            order.sort_by(|&left, &right| {
                compare_confirmation_cost(children[left].as_ref(), children[right].as_ref())
                    .then_with(|| left.cmp(&right))
            });
            Some(order)
        };
        Ok(Self {
            children,
            approximation_order,
            confirmation_order,
            current: None,
            confirmed_doc: None,
            confirmed: false,
        })
    }

    fn align(&mut self, target: u64) -> Result<Option<u64>> {
        self.current = if let Some(order) = &self.approximation_order {
            align_conjunction_children(&mut self.children, target, |position| order[position])?
        } else {
            align_conjunction_children(&mut self.children, target, |position| position)?
        };
        if self.current.is_some() {
            self.confirmed_doc = None;
            self.confirmed = false;
        }
        Ok(self.current)
    }

    fn ensure_confirmed(&mut self) -> Result<bool> {
        let Some(current) = self.current else {
            return Ok(false);
        };
        if self.confirmed_doc == Some(current) {
            return Ok(self.confirmed);
        }
        self.confirmed = if let Some(order) = &self.confirmation_order {
            confirm_conjunction_children(&mut self.children, |position| order[position])?
        } else {
            confirm_conjunction_children(&mut self.children, |position| position)?
        };
        self.confirmed_doc = Some(current);
        Ok(self.confirmed)
    }
}

impl ComposableScorer for RequiredConjunctionScorer<'_> {
    fn doc(&self) -> Option<u64> {
        self.current
    }

    fn document_key(&self) -> Option<u64> {
        self.children.first().and_then(|child| child.document_key())
    }

    fn next(&mut self) -> Result<Option<u64>> {
        let target = match self.current {
            None => 0,
            Some(u64::MAX) => return Ok(None),
            Some(current) => current + 1,
        };
        self.align(target)
    }

    fn advance(&mut self, target: u64) -> Result<Option<u64>> {
        if self.current.is_some_and(|current| current >= target) {
            return Ok(self.current);
        }
        self.align(target)
    }

    fn cost(&self) -> usize {
        self.children
            .iter()
            .map(|child| child.cost())
            .min()
            .unwrap_or(0)
    }

    fn score(&mut self) -> Result<f32> {
        if !self.ensure_confirmed()? {
            return Err(Error::internal(
                "FTS conjunction score requested for an unconfirmed document",
            ));
        }
        let mut score = 0.0_f32;
        for child in &mut self.children {
            score += child.score()?;
        }
        checked_score(score, "FTS conjunction")
    }

    fn advance_shallow(&mut self, target: u64) -> Result<u64> {
        let mut up_to = u64::MAX;
        for child in &mut self.children {
            let child_target = child.doc().map_or(target, |doc| target.max(doc));
            up_to = up_to.min(child.advance_shallow(child_target)?);
        }
        Ok(up_to)
    }

    fn score_bounds(&mut self, up_to: u64) -> Result<ScoreBounds> {
        let mut bounds = ScoreBounds::ZERO;
        for child in &mut self.children {
            bounds = bounds.add(child.score_bounds(up_to)?);
        }
        Ok(bounds)
    }

    fn global_score_upper_bound(&self) -> Option<f32> {
        self.scores_non_negative()
            .then(|| sum_global_score_upper_bounds(&self.children))
            .flatten()
    }

    fn set_min_competitive_score(&mut self, min_score: f32) -> Result<()> {
        if min_score.is_nan() {
            return Err(Error::invalid_input(
                "minimum competitive FTS score cannot be NaN",
            ));
        }
        // Propagating the full conjunction floor to one child is unsafe because
        // individually sub-threshold MUST scores may sum to a competitive hit.
        if self.children.len() == 1 {
            self.children[0].set_min_competitive_score(min_score)?;
        }
        Ok(())
    }

    fn matches(&mut self) -> Result<bool> {
        self.ensure_confirmed()
    }

    fn match_cost(&self) -> Option<f32> {
        self.children
            .iter()
            .filter_map(|child| child.match_cost())
            .reduce(|left, right| left + right)
    }

    fn scores_non_negative(&self) -> bool {
        self.children
            .iter()
            .all(|child| child.scores_non_negative())
    }
}

/// Positive-driven Boost scorer with signed conservative bounds.
pub(super) struct BoostScorer<'a> {
    positive: BoxScorer<'a>,
    negative: BoxScorer<'a>,
    negative_boost: f32,
    negative_matches_doc: Option<u64>,
    negative_matches: bool,
}

impl<'a> BoostScorer<'a> {
    pub(super) fn try_new(
        positive: BoxScorer<'a>,
        negative: BoxScorer<'a>,
        negative_boost: f32,
    ) -> Result<Self> {
        if !negative_boost.is_finite() || negative_boost < 0.0 {
            return Err(Error::invalid_input(format!(
                "BoostQuery negative_boost must be finite and non-negative, got {negative_boost}"
            )));
        }
        Ok(Self {
            positive,
            negative,
            negative_boost,
            negative_matches_doc: None,
            negative_matches: false,
        })
    }

    fn reset_confirmation(&mut self) {
        self.negative_matches_doc = None;
        self.negative_matches = false;
    }

    fn confirm_negative(&mut self) -> Result<bool> {
        let Some(current) = self.positive.doc() else {
            return Ok(false);
        };
        if self.negative_matches_doc == Some(current) {
            return Ok(self.negative_matches);
        }
        self.negative_matches =
            self.negative.advance(current)? == Some(current) && self.negative.matches()?;
        self.negative_matches_doc = Some(current);
        Ok(self.negative_matches)
    }
}

impl ComposableScorer for BoostScorer<'_> {
    fn doc(&self) -> Option<u64> {
        self.positive.doc()
    }

    fn document_key(&self) -> Option<u64> {
        self.positive.document_key()
    }

    fn next(&mut self) -> Result<Option<u64>> {
        self.reset_confirmation();
        self.positive.next()
    }

    fn advance(&mut self, target: u64) -> Result<Option<u64>> {
        self.reset_confirmation();
        self.positive.advance(target)
    }

    fn cost(&self) -> usize {
        self.positive.cost()
    }

    fn score(&mut self) -> Result<f32> {
        let positive = self.positive.score()?;
        let score = if self.confirm_negative()? {
            positive - self.negative_boost * self.negative.score()?
        } else {
            positive
        };
        checked_score(score, "BoostQuery scorer")
    }

    fn advance_shallow(&mut self, target: u64) -> Result<u64> {
        let mut up_to = self.positive.advance_shallow(target)?;
        if self.negative.doc().is_none_or(|doc| doc < target) {
            self.negative.advance(target)?;
        }
        if let Some(doc) = self.negative.doc() {
            up_to = up_to.min(self.negative.advance_shallow(target.max(doc))?);
        }
        Ok(up_to)
    }

    fn score_bounds(&mut self, up_to: u64) -> Result<ScoreBounds> {
        let positive = self.positive.score_bounds(up_to)?;
        let negative = if self.negative.doc().is_some_and(|doc| doc <= up_to) {
            self.negative.score_bounds(up_to)?.include_zero()
        } else {
            ScoreBounds::ZERO
        };
        Ok(positive.subtract_scaled(negative, self.negative_boost))
    }

    fn set_min_competitive_score(&mut self, min_score: f32) -> Result<()> {
        // With a non-negative negative scorer, Boost can only demote the
        // positive score, so the parent's floor is safe for the positive side.
        if self.negative.scores_non_negative() {
            self.positive.set_min_competitive_score(min_score)?;
        }
        Ok(())
    }

    fn matches(&mut self) -> Result<bool> {
        self.positive.matches()
    }

    fn match_cost(&self) -> Option<f32> {
        self.positive.match_cost()
    }
}

/// Required-plus-optional scorer that only touches the optional side when it
/// can change competitiveness or an exact score is requested.
///
/// The required scorer drives iteration. Once a score floor is available,
/// block bounds may either skip the whole range or temporarily turn the
/// optional approximation into a required iterator when the required score
/// cannot reach the floor on its own.
#[derive(Clone, Copy)]
struct ReqOptBounds {
    up_to: u64,
    required: ScoreBounds,
    combined: ScoreBounds,
}

struct ReqOptScorer<'a> {
    required: BoxScorer<'a>,
    optional: BoxScorer<'a>,
    current: Option<u64>,
    exhausted: bool,
    optional_initialized: bool,
    optional_is_required: bool,
    optional_checked_doc: Option<u64>,
    optional_matches: bool,
    confirmed_doc: Option<u64>,
    confirmed: bool,
    min_competitive_score: f32,
    shallow_bounds: Option<ReqOptBounds>,
}

impl<'a> ReqOptScorer<'a> {
    fn new(required: BoxScorer<'a>, optional: BoxScorer<'a>) -> Self {
        debug_assert!(required.scores_non_negative());
        debug_assert!(optional.scores_non_negative());
        Self {
            required,
            optional,
            current: None,
            exhausted: false,
            optional_initialized: false,
            optional_is_required: false,
            optional_checked_doc: None,
            optional_matches: false,
            confirmed_doc: None,
            confirmed: false,
            min_competitive_score: f32::NEG_INFINITY,
            shallow_bounds: None,
        }
    }

    fn set_current(&mut self, current: Option<u64>) {
        if self.current != current {
            self.optional_checked_doc = None;
            self.optional_matches = false;
            self.confirmed_doc = None;
            self.confirmed = false;
        }
        self.current = current;
        self.optional_is_required = false;
    }

    fn set_optional_required(&mut self, required: bool) {
        if self.optional_is_required != required {
            self.confirmed_doc = None;
            self.confirmed = false;
        }
        self.optional_is_required = required;
    }

    fn exhaust(&mut self) -> Option<u64> {
        self.exhausted = true;
        self.set_current(None);
        None
    }

    fn ensure_optional_at_or_after(&mut self, target: u64) -> Result<Option<u64>> {
        if !self.optional_initialized || self.optional.doc().is_some_and(|doc| doc < target) {
            self.optional.advance(target)?;
            self.optional_initialized = true;
        }
        Ok(self.optional.doc())
    }

    fn optional_matches_current(&mut self) -> Result<bool> {
        let Some(current) = self.current else {
            return Ok(false);
        };
        if self.optional_checked_doc == Some(current) {
            return Ok(self.optional_matches);
        }
        self.optional_matches = self.ensure_optional_at_or_after(current)? == Some(current)
            && self.optional.matches()?;
        self.optional_checked_doc = Some(current);
        Ok(self.optional_matches)
    }

    fn usable_bounds(bounds: ScoreBounds) -> bool {
        bounds.lower.is_finite() && bounds.upper.is_finite() && bounds.lower <= bounds.upper
    }

    fn bounds(&mut self, up_to: u64) -> Result<ReqOptBounds> {
        if let Some(bounds) = self.shallow_bounds
            && bounds.up_to == up_to
        {
            return Ok(bounds);
        }

        let required = self.required.score_bounds(up_to)?;
        let optional = if self.optional.doc().is_some_and(|doc| doc <= up_to) {
            let bounds = self.optional.score_bounds(up_to)?;
            if Self::usable_bounds(bounds) {
                bounds.include_zero()
            } else {
                ScoreBounds::UNBOUNDED
            }
        } else {
            ScoreBounds::ZERO
        };
        let combined = required.add(optional);
        let bounds = ReqOptBounds {
            up_to,
            required,
            combined,
        };
        self.shallow_bounds = Some(bounds);
        Ok(bounds)
    }

    fn position(&mut self, mut target: u64) -> Result<Option<u64>> {
        if self.exhausted {
            return Ok(None);
        }

        'search: loop {
            let bounds = self.shallow_bounds.filter(|bounds| target <= bounds.up_to);

            if self.min_competitive_score.is_finite()
                && self.min_competitive_score > 0.0
                && let Some(bounds) = bounds
                && Self::usable_bounds(bounds.required)
                && Self::usable_bounds(bounds.combined)
            {
                if bounds.combined.upper < self.min_competitive_score {
                    if bounds.up_to == u64::MAX {
                        return Ok(self.exhaust());
                    }
                    target = bounds.up_to + 1;
                    self.shallow_bounds = None;
                    continue;
                }

                if bounds.required.upper < self.min_competitive_score {
                    // The optional contribution is necessary throughout this
                    // cached shallow range. Intersect approximations until
                    // both sides agree or the range is exhausted.
                    let Some(mut required_doc) = self.required.advance(target)? else {
                        return Ok(self.exhaust());
                    };
                    if required_doc > bounds.up_to {
                        target = required_doc;
                        self.shallow_bounds = None;
                        continue;
                    }
                    self.set_current(Some(required_doc));
                    self.set_optional_required(true);
                    loop {
                        self.set_current(Some(required_doc));
                        self.set_optional_required(true);
                        let Some(optional_doc) = self.ensure_optional_at_or_after(required_doc)?
                        else {
                            if bounds.up_to == u64::MAX {
                                return Ok(self.exhaust());
                            }
                            target = bounds.up_to + 1;
                            self.shallow_bounds = None;
                            continue 'search;
                        };
                        if optional_doc > bounds.up_to {
                            if bounds.up_to == u64::MAX {
                                return Ok(self.exhaust());
                            }
                            target = bounds.up_to + 1;
                            self.shallow_bounds = None;
                            continue 'search;
                        }
                        if optional_doc == required_doc {
                            return Ok(self.current);
                        }
                        let Some(next_required) = self.required.advance(optional_doc)? else {
                            return Ok(self.exhaust());
                        };
                        if next_required > bounds.up_to {
                            target = next_required;
                            self.shallow_bounds = None;
                            continue 'search;
                        }
                        required_doc = next_required;
                    }
                }
            }

            let Some(required_doc) = self.required.advance(target)? else {
                return Ok(self.exhaust());
            };
            self.set_current(Some(required_doc));
            self.set_optional_required(false);
            return Ok(self.current);
        }
    }

    fn ensure_confirmed(&mut self) -> Result<bool> {
        let Some(current) = self.current else {
            return Ok(false);
        };
        if self.confirmed_doc == Some(current) {
            return Ok(self.confirmed);
        }
        self.confirmed = self.required.matches()?
            && (!self.optional_is_required || self.optional_matches_current()?);
        self.confirmed_doc = Some(current);
        Ok(self.confirmed)
    }
}

impl ComposableScorer for ReqOptScorer<'_> {
    fn doc(&self) -> Option<u64> {
        self.current
    }

    fn document_key(&self) -> Option<u64> {
        self.required.document_key()
    }

    fn next(&mut self) -> Result<Option<u64>> {
        let target = match self.current {
            None => 0,
            Some(u64::MAX) => return Ok(self.exhaust()),
            Some(current) => current + 1,
        };
        self.position(target)
    }

    fn advance(&mut self, target: u64) -> Result<Option<u64>> {
        if self.current.is_some_and(|current| current >= target) {
            return Ok(self.current);
        }
        self.position(target)
    }

    fn cost(&self) -> usize {
        self.required.cost()
    }

    fn score(&mut self) -> Result<f32> {
        if !self.ensure_confirmed()? {
            return Err(Error::internal(
                "required-plus-optional FTS score requested for an unconfirmed document",
            ));
        }
        let mut score = self.required.score()?;
        if self.optional_matches_current()? {
            score += self.optional.score()?;
        }
        checked_score(score, "required-plus-optional FTS scorer")
    }

    fn advance_shallow(&mut self, target: u64) -> Result<u64> {
        let target = self.current.map_or(target, |current| target.max(current));
        if let Some(bounds) = self.shallow_bounds
            && target <= bounds.up_to
        {
            return Ok(bounds.up_to);
        }
        self.shallow_bounds = None;
        let mut up_to = self.required.advance_shallow(target)?;
        match self.ensure_optional_at_or_after(target)? {
            Some(optional_doc) if optional_doc <= target => {
                up_to = up_to.min(self.optional.advance_shallow(target)?);
            }
            Some(optional_doc) => {
                up_to = up_to.min(optional_doc.saturating_sub(1));
            }
            None => {}
        }
        Ok(up_to)
    }

    fn score_bounds(&mut self, up_to: u64) -> Result<ScoreBounds> {
        Ok(self.bounds(up_to)?.combined)
    }

    fn global_score_upper_bound(&self) -> Option<f32> {
        let required = self.required.global_score_upper_bound()?;
        let optional = self.optional.global_score_upper_bound()?;
        let combined = ScoreBounds {
            lower: 0.0,
            upper: required,
        }
        .add(ScoreBounds {
            lower: 0.0,
            upper: optional,
        })
        .upper;
        combined.is_finite().then_some(combined)
    }

    fn set_min_competitive_score(&mut self, min_score: f32) -> Result<()> {
        if min_score.is_nan() {
            return Err(Error::invalid_input(
                "minimum competitive FTS score cannot be NaN",
            ));
        }
        if min_score > self.min_competitive_score {
            self.min_competitive_score = min_score;
        }
        Ok(())
    }

    fn matches(&mut self) -> Result<bool> {
        self.ensure_confirmed()
    }

    fn match_cost(&self) -> Option<f32> {
        self.required
            .match_cost()
            .into_iter()
            .chain(self.optional.match_cost())
            .reduce(|left, right| left + right)
    }

    fn scores_non_negative(&self) -> bool {
        true
    }
}

#[derive(Default)]
struct BooleanWork {
    positive_survivors: usize,
    must_not_probes: usize,
}

/// Boolean scorer preserving the current membership and score semantics.
pub(super) struct BooleanScorer<'a> {
    driver: BoxScorer<'a>,
    optional: Option<BoxScorer<'a>>,
    prohibited: Option<BoxScorer<'a>>,
    score_mode: CompoundScoreMode,
    current: Option<u64>,
    optional_matches: bool,
    min_competitive_score: f32,
    positive_checked_doc: Option<u64>,
    positive_score: Option<f32>,
    positive_survivor_doc: Option<u64>,
    prohibited_checked_doc: Option<u64>,
    prohibited_matches: bool,
    metrics: Option<&'a dyn MetricsCollector>,
    work: BooleanWork,
}

impl<'a> BooleanScorer<'a> {
    #[cfg(test)]
    pub(super) fn try_new(
        should: Vec<BoxScorer<'a>>,
        must: Vec<BoxScorer<'a>>,
        must_not: Vec<BoxScorer<'a>>,
    ) -> Result<Self> {
        Self::try_new_with_metrics(should, must, must_not, None)
    }

    fn try_new_with_metrics(
        should: Vec<BoxScorer<'a>>,
        must: Vec<BoxScorer<'a>>,
        must_not: Vec<BoxScorer<'a>>,
        metrics: Option<&'a dyn MetricsCollector>,
    ) -> Result<Self> {
        Self::try_new_with_metrics_and_score_mode(
            should,
            must,
            must_not,
            metrics,
            CompoundScoreMode::Scoring,
        )
    }

    fn try_new_with_metrics_and_score_mode(
        should: Vec<BoxScorer<'a>>,
        must: Vec<BoxScorer<'a>>,
        must_not: Vec<BoxScorer<'a>>,
        metrics: Option<&'a dyn MetricsCollector>,
        score_mode: CompoundScoreMode,
    ) -> Result<Self> {
        let (driver, optional) = if must.is_empty() {
            if should.is_empty() {
                return Err(Error::invalid_input(
                    "boolean query must have at least one should/must query",
                ));
            }
            let driver = if score_mode == CompoundScoreMode::Scoring
                && let Some(global_bounds) = ShouldMaxScoreScorer::global_bounds(&should)
            {
                Box::new(ShouldMaxScoreScorer::new(should, global_bounds, metrics)) as BoxScorer<'a>
            } else {
                Box::new(DisjunctionScorer::try_new(should, DisjunctionScore::Sum)?)
                    as BoxScorer<'a>
            };
            (driver, None)
        } else {
            let mut optional =
                if should.is_empty() || score_mode == CompoundScoreMode::CompleteNoScores {
                    None
                } else {
                    Some(
                        Box::new(DisjunctionScorer::try_new(should, DisjunctionScore::Sum)?)
                            as BoxScorer<'a>,
                    )
                };
            let required = Box::new(RequiredConjunctionScorer::try_new(must)?) as BoxScorer<'a>;
            let driver = if score_mode == CompoundScoreMode::Scoring
                && required.scores_non_negative()
                && optional
                    .as_ref()
                    .is_some_and(|optional| optional.scores_non_negative())
            {
                Box::new(ReqOptScorer::new(
                    required,
                    optional
                        .take()
                        .expect("checked that the optional scorer is present"),
                )) as BoxScorer<'a>
            } else {
                required
            };
            (driver, optional)
        };
        let prohibited = if must_not.is_empty() {
            None
        } else {
            Some(
                Box::new(DisjunctionScorer::try_new(must_not, DisjunctionScore::Max)?)
                    as BoxScorer<'a>,
            )
        };
        let metrics = if prohibited.is_some() { metrics } else { None };
        Ok(Self {
            driver,
            optional,
            prohibited,
            score_mode,
            current: None,
            optional_matches: false,
            min_competitive_score: f32::NEG_INFINITY,
            positive_checked_doc: None,
            positive_score: None,
            positive_survivor_doc: None,
            prohibited_checked_doc: None,
            prohibited_matches: false,
            metrics,
            work: BooleanWork::default(),
        })
    }

    fn accept_driver_doc_without_prohibited(&mut self) -> Result<bool> {
        debug_assert!(self.prohibited.is_none());
        let Some(current) = self.driver.doc() else {
            return Ok(false);
        };
        if !self.driver.matches()? {
            return Ok(false);
        }
        self.optional_matches = if let Some(optional) = &mut self.optional {
            optional.advance(current)? == Some(current) && optional.matches()?
        } else {
            false
        };
        self.current = Some(current);
        Ok(true)
    }

    fn next_accepted_without_prohibited(&mut self, target: Option<u64>) -> Result<Option<u64>> {
        debug_assert!(self.prohibited.is_none());
        let mut doc = match target {
            Some(target) => self.driver.advance(target)?,
            None => self.driver.next()?,
        };
        while doc.is_some() {
            if self.accept_driver_doc_without_prohibited()? {
                return Ok(self.current);
            }
            doc = self.driver.next()?;
        }
        self.current = None;
        self.optional_matches = false;
        Ok(None)
    }

    fn accept_driver_doc_without_score_floor(&mut self) -> Result<bool> {
        debug_assert!(self.prohibited.is_some());
        debug_assert_eq!(self.min_competitive_score, f32::NEG_INFINITY);
        let Some(current) = self.driver.doc() else {
            return Ok(false);
        };
        if !self.driver.matches()? {
            return Ok(false);
        }

        // Probe after positive confirmation. Scoring mode scores only accepted
        // documents, while COMPLETE_NO_SCORES stops after exact membership.
        self.work.positive_survivors += 1;
        self.work.must_not_probes += 1;
        let prohibited_matches = {
            let prohibited = self
                .prohibited
                .as_mut()
                .expect("unbounded MUST_NOT path requires a prohibited scorer");
            prohibited.advance(current)? == Some(current) && prohibited.matches()?
        };
        if prohibited_matches {
            return Ok(false);
        }

        // Only returned documents need repeat-call caches. Rejected documents
        // remain internal to this loop and are never observable by a caller.
        self.set_current(Some(current));
        self.positive_checked_doc = Some(current);
        self.positive_survivor_doc = Some(current);
        self.prohibited_checked_doc = Some(current);
        if self.score_mode == CompoundScoreMode::CompleteNoScores {
            return Ok(true);
        }
        self.optional_matches = if let Some(optional) = &mut self.optional {
            optional.advance(current)? == Some(current) && optional.matches()?
        } else {
            false
        };
        let mut score = self.driver.score()?;
        if self.optional_matches
            && let Some(optional) = &mut self.optional
        {
            score += optional.score()?;
        }
        self.positive_score = Some(checked_score(score, "BooleanQuery scorer")?);
        Ok(true)
    }

    fn next_accepted_without_score_floor(&mut self, target: Option<u64>) -> Result<Option<u64>> {
        debug_assert!(self.prohibited.is_some());
        debug_assert_eq!(self.min_competitive_score, f32::NEG_INFINITY);
        let mut doc = match target {
            Some(target) => self.driver.advance(target)?,
            None => self.driver.next()?,
        };
        while doc.is_some() {
            if self.accept_driver_doc_without_score_floor()? {
                return Ok(self.current);
            }
            doc = self.driver.next()?;
        }
        self.set_current(None);
        Ok(None)
    }

    fn set_current(&mut self, current: Option<u64>) {
        if self.current != current {
            self.optional_matches = false;
            self.positive_checked_doc = None;
            self.positive_score = None;
            self.positive_survivor_doc = None;
            self.prohibited_checked_doc = None;
            self.prohibited_matches = false;
        }
        self.current = current;
    }

    fn position(&mut self, target: Option<u64>) -> Result<Option<u64>> {
        let current = match target {
            Some(target) => self.driver.advance(target)?,
            None => self.driver.next()?,
        };
        // The separate optional scorer is the exact fallback for shapes that
        // cannot use ReqOpt. Keep its approximation positioned so parent
        // shallow bounds include a possible optional contribution.
        if let Some(current) = current
            && let Some(optional) = &mut self.optional
        {
            optional.advance(current)?;
        }
        self.set_current(current);
        Ok(current)
    }

    fn ensure_positive_score(&mut self) -> Result<Option<f32>> {
        let Some(current) = self.current else {
            return Ok(None);
        };
        if self.positive_checked_doc == Some(current) {
            return Ok(self.positive_score);
        }
        if !self.driver.matches()? {
            self.positive_checked_doc = Some(current);
            self.positive_score = None;
            return Ok(None);
        }
        self.optional_matches = if let Some(optional) = &mut self.optional {
            debug_assert!(optional.doc().is_none_or(|doc| doc >= current));
            optional.doc() == Some(current) && optional.matches()?
        } else {
            false
        };
        let mut score = self.driver.score()?;
        if self.optional_matches
            && let Some(optional) = &mut self.optional
        {
            score += optional.score()?;
        }
        let score = checked_score(score, "BooleanQuery scorer")?;
        self.positive_checked_doc = Some(current);
        self.positive_score = Some(score);
        Ok(Some(score))
    }

    fn ensure_not_prohibited(&mut self) -> Result<bool> {
        let Some(current) = self.current else {
            return Ok(false);
        };
        if self.prohibited_checked_doc == Some(current) {
            return Ok(!self.prohibited_matches);
        }
        self.prohibited_matches = if let Some(prohibited) = &mut self.prohibited {
            self.work.must_not_probes += 1;
            prohibited.advance(current)? == Some(current) && prohibited.matches()?
        } else {
            false
        };
        self.prohibited_checked_doc = Some(current);
        Ok(!self.prohibited_matches)
    }
}

impl ComposableScorer for BooleanScorer<'_> {
    fn doc(&self) -> Option<u64> {
        self.current
    }

    fn document_key(&self) -> Option<u64> {
        self.driver.document_key()
    }

    fn next(&mut self) -> Result<Option<u64>> {
        if self.prohibited.is_none() {
            self.next_accepted_without_prohibited(None)
        } else if self.score_mode == CompoundScoreMode::CompleteNoScores
            || self.min_competitive_score == f32::NEG_INFINITY
        {
            self.next_accepted_without_score_floor(None)
        } else {
            self.position(None)
        }
    }

    fn advance(&mut self, target: u64) -> Result<Option<u64>> {
        if self.current.is_some_and(|current| current >= target) {
            return Ok(self.current);
        }
        if self.prohibited.is_none() {
            self.next_accepted_without_prohibited(Some(target))
        } else if self.score_mode == CompoundScoreMode::CompleteNoScores
            || self.min_competitive_score == f32::NEG_INFINITY
        {
            self.next_accepted_without_score_floor(Some(target))
        } else {
            self.position(Some(target))
        }
    }

    fn cost(&self) -> usize {
        self.driver.cost()
    }

    fn score(&mut self) -> Result<f32> {
        if self.score_mode == CompoundScoreMode::CompleteNoScores {
            return Err(Error::internal(
                "score requested from a COMPLETE_NO_SCORES Boolean FTS scorer",
            ));
        }
        if self.prohibited.is_none() {
            if self.current.is_none() {
                return Err(Error::internal(
                    "Boolean FTS scorer is not positioned on a document",
                ));
            }
            let mut score = self.driver.score()?;
            if self.optional_matches
                && let Some(optional) = &mut self.optional
            {
                score += optional.score()?;
            }
            return checked_score(score, "BooleanQuery scorer");
        }
        let current = self
            .current
            .ok_or_else(|| Error::internal("Boolean FTS scorer is not positioned on a document"))?;
        if self.positive_checked_doc != Some(current) {
            return Err(Error::internal(
                "Boolean FTS score requested before confirming the positive scorer",
            ));
        }
        self.positive_score.ok_or_else(|| {
            Error::internal("Boolean FTS score requested for a rejected positive candidate")
        })
    }

    fn advance_shallow(&mut self, target: u64) -> Result<u64> {
        if self.score_mode == CompoundScoreMode::CompleteNoScores {
            return Ok(u64::MAX);
        }
        let mut up_to = self.driver.advance_shallow(target)?;
        if let Some(optional) = &mut self.optional
            && let Some(doc) = optional.doc()
        {
            up_to = up_to.min(optional.advance_shallow(target.max(doc))?);
        }
        Ok(up_to)
    }

    fn score_bounds(&mut self, up_to: u64) -> Result<ScoreBounds> {
        if self.score_mode == CompoundScoreMode::CompleteNoScores {
            return Ok(ScoreBounds::UNBOUNDED);
        }
        let mut bounds = self.driver.score_bounds(up_to)?;
        if let Some(optional) = &mut self.optional
            && optional.doc().is_some_and(|doc| doc <= up_to)
        {
            bounds = bounds.add(optional.score_bounds(up_to)?.include_zero());
        }
        Ok(bounds)
    }

    fn global_score_upper_bound(&self) -> Option<f32> {
        if self.score_mode == CompoundScoreMode::CompleteNoScores {
            return None;
        }
        if !self.scores_non_negative() {
            return None;
        }
        let driver = self.driver.global_score_upper_bound()?;
        let combined = if let Some(optional) = &self.optional {
            let optional = optional.global_score_upper_bound()?;
            ScoreBounds {
                lower: 0.0,
                upper: driver,
            }
            .add(ScoreBounds {
                lower: 0.0,
                upper: optional,
            })
            .upper
        } else {
            driver
        };
        (combined.is_finite() && combined >= 0.0).then_some(combined)
    }

    fn set_min_competitive_score(&mut self, min_score: f32) -> Result<()> {
        if self.score_mode == CompoundScoreMode::CompleteNoScores {
            if min_score.is_nan() {
                return Err(Error::invalid_input(
                    "minimum competitive FTS score cannot be NaN",
                ));
            }
            return Ok(());
        }
        if self.prohibited.is_some() {
            if min_score.is_nan() {
                return Err(Error::invalid_input(
                    "minimum competitive FTS score cannot be NaN",
                ));
            }
            if min_score > self.min_competitive_score {
                self.min_competitive_score = min_score;
            }
        }
        // When SHOULD is also present, a global sibling bound is required to
        // translate the parent threshold safely. The combined block bound still
        // prunes at this node. Without SHOULD, driver score is the full score.
        if self.optional.is_none() {
            self.driver.set_min_competitive_score(min_score)?;
        }
        Ok(())
    }

    fn matches(&mut self) -> Result<bool> {
        if self.score_mode == CompoundScoreMode::CompleteNoScores {
            return Ok(self.current.is_some());
        }
        if self.prohibited.is_none() {
            return Ok(self.current.is_some());
        }
        let Some(score) = self.ensure_positive_score()? else {
            return Ok(false);
        };
        if score < self.min_competitive_score {
            return Ok(false);
        }
        if self.prohibited.is_some() && self.positive_survivor_doc != self.current {
            self.positive_survivor_doc = self.current;
            self.work.positive_survivors += 1;
        }
        self.ensure_not_prohibited()
    }

    fn scores_non_negative(&self) -> bool {
        if self.score_mode == CompoundScoreMode::CompleteNoScores {
            return false;
        }
        self.driver.scores_non_negative()
            && self
                .optional
                .as_ref()
                .is_none_or(|optional| optional.scores_non_negative())
    }
}

impl Drop for BooleanScorer<'_> {
    fn drop(&mut self) {
        let Some(metrics) = self.metrics else {
            return;
        };
        if self.work.positive_survivors > 0 {
            metrics.record_compound_positive_survivors(self.work.positive_survivors);
        }
        if self.work.must_not_probes > 0 {
            metrics.record_compound_must_not_probes(self.work.must_not_probes);
        }
    }
}

#[derive(Clone)]
enum LeafQuery {
    Match(MatchQuery),
    Phrase(PhraseQuery),
}

impl LeafQuery {
    fn terms(&self) -> &str {
        match self {
            Self::Match(query) => &query.terms,
            Self::Phrase(query) => &query.terms,
        }
    }

    fn operator(&self) -> Operator {
        match self {
            Self::Match(query) => query.operator,
            Self::Phrase(_) => Operator::And,
        }
    }

    fn effective_params(&self, params: &FtsSearchParams) -> FtsSearchParams {
        match self {
            Self::Match(query) => params
                .clone()
                .with_limit(None)
                .with_phrase_slop(None)
                .with_fuzziness(query.fuzziness)
                .with_max_expansions(query.max_expansions)
                .with_prefix_length(query.prefix_length),
            Self::Phrase(query) => params
                .clone()
                .with_limit(None)
                .with_phrase_slop(Some(query.slop)),
        }
    }
}

fn collect_leaf_queries(
    query: &FtsQuery,
    score_mode: CompoundScoreMode,
    leaves: &mut Vec<(LeafQuery, CompoundScoreMode)>,
) -> Result<()> {
    match query {
        FtsQuery::Match(query) => leaves.push((LeafQuery::Match(query.clone()), score_mode)),
        FtsQuery::Phrase(query) => leaves.push((LeafQuery::Phrase(query.clone()), score_mode)),
        FtsQuery::Boost(query) => {
            collect_leaf_queries(&query.positive, score_mode, leaves)?;
            if score_mode == CompoundScoreMode::Scoring {
                collect_leaf_queries(&query.negative, score_mode, leaves)?;
            }
        }
        FtsQuery::MultiMatch(query) => {
            leaves.extend(
                query
                    .match_queries
                    .iter()
                    .cloned()
                    .map(|query| (LeafQuery::Match(query), score_mode)),
            );
        }
        FtsQuery::Boolean(query) => {
            if score_mode == CompoundScoreMode::Scoring || query.must.is_empty() {
                for child in &query.should {
                    collect_leaf_queries(child, score_mode, leaves)?;
                }
            }
            for child in &query.must {
                collect_leaf_queries(child, score_mode, leaves)?;
            }
            for child in &query.must_not {
                collect_leaf_queries(child, CompoundScoreMode::CompleteNoScores, leaves)?;
            }
        }
    }
    Ok(())
}

struct PreparedLeaf {
    tokens_by_segment: Vec<Arc<Tokens>>,
    params: Arc<FtsSearchParams>,
    operator: Operator,
    scorer: Arc<MemBM25Scorer>,
    score_mode: CompoundScoreMode,
}

fn tokenize_leaf(index: &InvertedIndex, leaf: &LeafQuery, params: &FtsSearchParams) -> Tokens {
    let is_fuzzy_match = matches!(leaf, LeafQuery::Match(_))
        && matches!(params.fuzziness, Some(distance) if distance != 0);
    let mut tokenizer = if is_fuzzy_match {
        let analyzer = TextAnalyzer::from(SimpleTokenizer::default());
        match index.tokenizer().doc_type() {
            DocType::Text => Box::new(TextTokenizer::new(analyzer)) as Box<dyn LanceTokenizer>,
            DocType::Json => Box::new(JsonTokenizer::new(analyzer)) as Box<dyn LanceTokenizer>,
        }
    } else {
        index.tokenizer()
    };
    collect_query_tokens(leaf.terms(), &mut tokenizer)
}

fn expanded_leaf_tokens(
    index: &InvertedIndex,
    tokens: &Tokens,
    params: &FtsSearchParams,
    operator: Operator,
) -> Result<Tokens> {
    if !matches!(params.fuzziness, Some(distance) if distance != 0) {
        return Ok(tokens.clone());
    }
    let expanded = index.expand_fuzzy_tokens(tokens, params)?;
    if operator == Operator::And || params.phrase_slop.is_some() {
        let surviving = (0..expanded.len())
            .map(|index| expanded.position(index))
            .collect::<HashSet<_>>();
        if (0..tokens.len()).any(|index| !surviving.contains(&tokens.position(index))) {
            return Ok(Tokens::with_positions(
                Vec::new(),
                Vec::new(),
                tokens.token_type().clone(),
            ));
        }
    }
    Ok(expanded)
}

fn validate_injected_scorer_tokens(scorer: &MemBM25Scorer, tokens: &Tokens) -> Result<()> {
    for token in tokens {
        if !scorer.token_docs.contains_key(token) {
            return Err(Error::invalid_input(format!(
                "injected BM25 scorer is missing compound FTS token '{token}'"
            )));
        }
    }
    Ok(())
}

fn validate_compound_query(query: &FtsQuery) -> Result<()> {
    fn validate_multiplier(name: &str, value: f32) -> Result<()> {
        if value.is_finite() && value >= 0.0 {
            Ok(())
        } else {
            Err(Error::invalid_input(format!(
                "{name} must be finite and non-negative, got {value}"
            )))
        }
    }

    match query {
        FtsQuery::Match(query) => validate_multiplier("MatchQuery boost", query.boost),
        FtsQuery::Phrase(_) => Ok(()),
        FtsQuery::Boost(query) => {
            validate_multiplier("BoostQuery negative_boost", query.negative_boost)?;
            validate_compound_query(&query.positive)?;
            validate_compound_query(&query.negative)
        }
        FtsQuery::MultiMatch(query) => {
            if query.match_queries.is_empty() {
                return Err(Error::invalid_input(
                    "MultiMatchQuery must have at least one match query",
                ));
            }
            for match_query in &query.match_queries {
                validate_multiplier("MultiMatchQuery boost", match_query.boost)?;
            }
            Ok(())
        }
        FtsQuery::Boolean(query) => {
            if query.should.is_empty() && query.must.is_empty() {
                return Err(Error::invalid_input(
                    "boolean query must have at least one should/must query",
                ));
            }
            for child in query
                .should
                .iter()
                .chain(&query.must)
                .chain(&query.must_not)
            {
                validate_compound_query(child)?;
            }
            Ok(())
        }
    }
}

async fn prepare_compound_query(
    indices: &[Arc<InvertedIndex>],
    query: &FtsQuery,
    params: &FtsSearchParams,
    metrics: &dyn MetricsCollector,
    base_scorer: Option<Arc<MemBM25Scorer>>,
) -> Result<(CompoundScorerPlan, Vec<PreparedLeaf>)> {
    let first_index = indices
        .first()
        .ok_or_else(|| Error::invalid_input("compound FTS requires at least one index segment"))?;
    // CompleteNoScores prunes score-only branches. Validate the original AST
    // first so pruning cannot make malformed nested queries silently valid.
    validate_compound_query(query)?;
    let mut leaf_queries = Vec::new();
    collect_leaf_queries(query, CompoundScoreMode::Scoring, &mut leaf_queries)?;
    let mut num_plan_leaves = 0;
    let plan =
        CompoundScorerPlan::from_query(query, &mut num_plan_leaves, CompoundScoreMode::Scoring)?;
    if num_plan_leaves != leaf_queries.len() {
        return Err(Error::internal(format!(
            "compound FTS planned {num_plan_leaves} leaves but prepared {}",
            leaf_queries.len()
        )));
    }

    let mut leaves = Vec::with_capacity(leaf_queries.len());
    let mut membership_scorer = None;
    for (leaf, score_mode) in leaf_queries {
        let effective_params = leaf.effective_params(params);
        let tokens = tokenize_leaf(first_index, &leaf, &effective_params);
        let scorer = match score_mode {
            CompoundScoreMode::CompleteNoScores => membership_scorer
                .get_or_insert_with(|| {
                    // Posting loading still carries a scorer, but membership
                    // cursors never observe its weights or score bounds.
                    Arc::new(MemBM25Scorer::new(1, 1, Default::default()))
                })
                .clone(),
            CompoundScoreMode::Scoring => match &base_scorer {
                Some(scorer) => scorer.clone(),
                None => Arc::new(
                    build_global_bm25_scorer(indices, &tokens, &effective_params, Some(metrics))
                        .await?,
                ),
            },
        };
        let mut tokens_by_segment = Vec::with_capacity(indices.len());
        for index in indices {
            let expanded_tokens =
                expanded_leaf_tokens(index, &tokens, &effective_params, leaf.operator())?;
            if score_mode == CompoundScoreMode::Scoring && base_scorer.is_some() {
                validate_injected_scorer_tokens(&scorer, &expanded_tokens)?;
            }
            tokens_by_segment.push(Arc::new(expanded_tokens));
        }
        leaves.push(PreparedLeaf {
            tokens_by_segment,
            params: Arc::new(effective_params),
            operator: leaf.operator(),
            scorer,
            score_mode,
        });
    }
    Ok((plan, leaves))
}

struct LoadedLeaf {
    postings: Option<Vec<PostingIterator>>,
    params: Arc<FtsSearchParams>,
    operator: Operator,
    scorer: Arc<MemBM25Scorer>,
    score_mode: CompoundScoreMode,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PartitionLoadMode {
    ScoringOnly,
    All,
}

enum LoadedDocuments {
    Legacy(Arc<DocSet>),
    Modern {
        documents: Arc<PartitionDocuments>,
        lengths: Arc<DocLengths>,
        visibility: DocVisibility,
        projection: Option<ResidentAddressProjection>,
    },
}

struct LoadedPartition {
    segment_ordinal: usize,
    partition_ordinal: usize,
    partition: Arc<InvertedPartition>,
    documents: LoadedDocuments,
    leaves: Vec<LoadedLeaf>,
    /// First positive document that can still meet the score floor used by
    /// the preflight gate. Documents before it cannot enter the final top-k.
    collection_start_doc: Option<u64>,
}

async fn load_compound_partition(
    segment_ordinal: usize,
    partition_ordinal: usize,
    partition: Arc<InvertedPartition>,
    leaves: &[PreparedLeaf],
    mask: Arc<RowAddrMask>,
    metrics: Arc<dyn MetricsCollector>,
    load_mode: PartitionLoadMode,
) -> Result<Option<LoadedPartition>> {
    let leaf_loads = leaves.iter().map(|leaf| {
        let partition = partition.clone();
        let tokens = leaf.tokens_by_segment[segment_ordinal].clone();
        let params = leaf.params.clone();
        let scorer = leaf.scorer.clone();
        let score_mode = leaf.score_mode;
        let metrics = metrics.clone();
        let operator = leaf.operator;
        async move {
            let postings = if tokens.is_empty() {
                Some(Vec::new())
            } else if load_mode == PartitionLoadMode::ScoringOnly
                && score_mode == CompoundScoreMode::CompleteNoScores
            {
                None
            } else if score_mode == CompoundScoreMode::CompleteNoScores {
                Some(
                    partition
                        .load_membership_posting_lists(
                            tokens.as_ref(),
                            params.as_ref(),
                            operator,
                            metrics.as_ref(),
                        )
                        .await?,
                )
            } else {
                Some(
                    partition
                        .load_posting_lists(
                            tokens.as_ref(),
                            params.as_ref(),
                            operator,
                            scorer.as_ref(),
                            metrics.as_ref(),
                            true,
                        )
                        .await?
                        .postings,
                )
            };
            Result::Ok(LoadedLeaf {
                postings,
                params,
                operator,
                scorer,
                score_mode,
            })
        }
    });
    let leaves = futures::future::try_join_all(leaf_loads).await?;

    let documents = if let Some(docs) = partition.docs.legacy() {
        LoadedDocuments::Legacy(docs.clone())
    } else {
        let documents = partition.docs.modern().cloned().ok_or_else(|| {
            Error::internal("FTS partition contains neither legacy nor modern documents")
        })?;
        let materialize_selected = mask.max_len().is_some_and(|selected| {
            u128::from(selected).saturating_mul(100)
                <= u128::from(*FLAT_SEARCH_PERCENT_THRESHOLD)
                    .saturating_mul(documents.len() as u128)
        });
        let visibility = match documents.immediate_visibility(mask.clone(), materialize_selected) {
            Some(visibility) => visibility,
            None => {
                documents
                    .visibility(mask.clone(), materialize_selected)
                    .await?
            }
        };
        if visibility.is_empty() {
            return Ok(None);
        }
        let lengths = match documents.cached_lengths() {
            Some(lengths) => lengths,
            None => documents.lengths().await?,
        };
        let projection = documents.resident_address_projection();
        LoadedDocuments::Modern {
            documents,
            lengths,
            visibility,
            projection,
        }
    };

    Ok(Some(LoadedPartition {
        segment_ordinal,
        partition_ordinal,
        partition,
        documents,
        leaves,
        collection_start_doc: None,
    }))
}

async fn load_deferred_prohibited_postings(
    mut loaded: LoadedPartition,
    leaves: &[PreparedLeaf],
    metrics: Arc<dyn MetricsCollector>,
) -> Result<LoadedPartition> {
    if loaded.leaves.len() != leaves.len() {
        return Err(Error::internal(format!(
            "compound FTS loaded {} leaves for a {}-leaf plan",
            loaded.leaves.len(),
            leaves.len()
        )));
    }
    let segment_ordinal = loaded.segment_ordinal;
    let loads = loaded
        .leaves
        .iter()
        .enumerate()
        .filter(|(_, leaf)| leaf.postings.is_none())
        .map(|(index, loaded_leaf)| {
            let partition = loaded.partition.clone();
            let prepared = &leaves[index];
            let tokens = prepared.tokens_by_segment[segment_ordinal].clone();
            let params = prepared.params.clone();
            let metrics = metrics.clone();
            let operator = prepared.operator;
            let score_mode = prepared.score_mode;
            debug_assert_eq!(loaded_leaf.score_mode, CompoundScoreMode::CompleteNoScores);
            async move {
                if score_mode != CompoundScoreMode::CompleteNoScores {
                    return Err(Error::internal(format!(
                        "compound FTS deferred scoring leaf index {index}"
                    )));
                }
                let postings = if tokens.is_empty() {
                    Vec::new()
                } else {
                    partition
                        .load_membership_posting_lists(
                            tokens.as_ref(),
                            params.as_ref(),
                            operator,
                            metrics.as_ref(),
                        )
                        .await?
                };
                Result::Ok((index, postings))
            }
        });
    let postings = futures::future::try_join_all(loads).await?;
    let num_loads = postings.len();
    for (index, postings) in postings {
        let leaf = loaded.leaves.get_mut(index).ok_or_else(|| {
            Error::internal(format!(
                "compound FTS deferred leaf index {index} disappeared while loading"
            ))
        })?;
        leaf.postings = Some(postings);
    }
    if num_loads > 0 {
        metrics.record_compound_must_not_posting_loads(num_loads);
    }
    Ok(loaded)
}

struct DeferredCompoundRows {
    documents: Arc<PartitionDocuments>,
    rows: Vec<ScoredRow<DocId>>,
}

struct OverflowedCompoundPartition {
    segment_ordinal: usize,
    partition_ordinal: usize,
    partition: Arc<InvertedPartition>,
    documents: Arc<PartitionDocuments>,
    collection_start_doc: Option<u64>,
}

enum PartitionCollectionBoundary {
    NeedsProhibited(Vec<LoadedPartition>),
    Deferred(DeferredCompoundRows),
    Overflow(OverflowedCompoundPartition),
}

struct CollectedPartitions {
    collector: TopKCollector<u64>,
    remaining: VecDeque<LoadedPartition>,
    boundary: Option<PartitionCollectionBoundary>,
}

fn first_partition_competitive_doc<'a, D: WandDocuments + Sync>(
    documents: &'a D,
    leaves: &'a [LoadedLeaf],
    plan: &CompoundScorerPlan,
    metrics: &'a dyn MetricsCollector,
    min_score: f32,
) -> Result<Option<u64>> {
    let is_membership_gate = min_score == f32::NEG_INFINITY;
    let mut leaf_scorers = leaves
        .iter()
        .map(|leaf| -> Result<Option<BoxScorer<'a>>> {
            if leaf.score_mode == CompoundScoreMode::CompleteNoScores {
                return Ok(None);
            }
            let postings = leaf
                .postings
                .as_ref()
                .ok_or_else(|| Error::internal("compound FTS positive posting was not loaded"))?;
            let postings = postings
                .iter()
                .map(PostingIterator::fork_from_start)
                .collect::<Vec<_>>();
            let scorer: BoxScorer<'a> = if postings.is_empty() {
                Box::new(EmptyScorer)
            } else if is_membership_gate {
                Box::new(WandCursor::new_membership(
                    leaf.operator,
                    postings,
                    documents,
                    leaf.scorer.clone(),
                    leaf.params.as_ref(),
                    metrics,
                ))
            } else {
                Box::new(WandCursor::new(
                    leaf.operator,
                    postings,
                    documents,
                    leaf.scorer.clone(),
                    leaf.params.as_ref(),
                    metrics,
                ))
            };
            Ok(Some(scorer))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut scorer = plan.build_gate(&mut leaf_scorers, metrics)?;
    first_competitive_match(scorer.as_mut(), min_score)
}

fn collect_partition_with_documents<D, K>(
    documents: &D,
    leaves: Vec<LoadedLeaf>,
    plan: &CompoundScorerPlan,
    metrics: &dyn MetricsCollector,
    collector: &mut TopKCollector<K>,
    collection_start_doc: Option<u64>,
    mut map_document: impl FnMut(u64) -> Result<K>,
) -> Result<CollectionStatus>
where
    D: WandDocuments + Sync,
    K: Copy + Ord,
{
    let mut leaf_scorers = leaves
        .into_iter()
        .map(|leaf| -> Result<Option<BoxScorer<'_>>> {
            let postings = leaf.postings.ok_or_else(|| {
                Error::internal("compound FTS prohibited posting was not loaded before collection")
            })?;
            let scorer: BoxScorer<'_> = if postings.is_empty() {
                Box::new(EmptyScorer)
            } else if leaf.score_mode == CompoundScoreMode::CompleteNoScores {
                Box::new(WandCursor::new_membership(
                    leaf.operator,
                    postings,
                    documents,
                    leaf.scorer,
                    leaf.params.as_ref(),
                    metrics,
                ))
            } else {
                Box::new(WandCursor::new(
                    leaf.operator,
                    postings,
                    documents,
                    leaf.scorer,
                    leaf.params.as_ref(),
                    metrics,
                ))
            };
            Ok(Some(scorer))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut scorer = plan.build(&mut leaf_scorers, metrics)?;
    if leaf_scorers.iter().any(Option::is_some) {
        return Err(Error::internal(
            "compound FTS scorer did not consume every prepared leaf",
        ));
    }
    collector.collect_mapped_from(scorer.as_mut(), collection_start_doc, &mut map_document)
}

fn collect_loaded_partitions(
    mut partitions: VecDeque<LoadedPartition>,
    plan: &CompoundScorerPlan,
    may_have_deferred_prohibited: bool,
    mask: &RowAddrMask,
    metrics: &dyn MetricsCollector,
    mut collector: TopKCollector<u64>,
) -> Result<CollectedPartitions> {
    let mut needs_prohibited = Vec::with_capacity(DEFERRED_MUST_NOT_LOAD_BATCH_SIZE);
    let mut has_safe_score_gate = None;
    while let Some(mut partition) = partitions.pop_front() {
        if may_have_deferred_prohibited
            && partition.leaves.iter().any(|leaf| leaf.postings.is_none())
        {
            // Signed nested Boosts can make a positive-only score gate
            // underestimate the final score. The positive match set remains
            // exact, so use a membership-only gate for those plans instead of
            // eagerly loading every prohibited posting.
            let has_safe_score_gate =
                *has_safe_score_gate.get_or_insert_with(|| plan.positive_gate_is_safe());
            let min_score = if has_safe_score_gate {
                collector.competitive_score.get()
            } else {
                f32::NEG_INFINITY
            };
            let collection_start_doc = match &partition.documents {
                LoadedDocuments::Legacy(docs) => {
                    let documents = LegacyWandDocuments::new(docs.as_ref(), mask);
                    first_partition_competitive_doc(
                        &documents,
                        &partition.leaves,
                        plan,
                        metrics,
                        min_score,
                    )?
                }
                LoadedDocuments::Modern {
                    lengths,
                    visibility,
                    ..
                } => {
                    let documents = ModernWandDocuments::filtered(lengths.as_ref(), visibility);
                    first_partition_competitive_doc(
                        &documents,
                        &partition.leaves,
                        plan,
                        metrics,
                        min_score,
                    )?
                }
            };
            if let Some(collection_start_doc) = collection_start_doc {
                partition.collection_start_doc = Some(collection_start_doc);
                needs_prohibited.push(partition);
                if needs_prohibited.len() == DEFERRED_MUST_NOT_LOAD_BATCH_SIZE {
                    return Ok(CollectedPartitions {
                        collector,
                        remaining: partitions,
                        boundary: Some(PartitionCollectionBoundary::NeedsProhibited(
                            needs_prohibited,
                        )),
                    });
                }
            }
            continue;
        }
        if !needs_prohibited.is_empty() {
            partitions.push_front(partition);
            return Ok(CollectedPartitions {
                collector,
                remaining: partitions,
                boundary: Some(PartitionCollectionBoundary::NeedsProhibited(
                    needs_prohibited,
                )),
            });
        }
        let LoadedPartition {
            segment_ordinal,
            partition_ordinal,
            partition: source,
            documents,
            leaves,
            collection_start_doc,
        } = partition;
        match documents {
            LoadedDocuments::Legacy(docs) => {
                let documents = LegacyWandDocuments::new(docs.as_ref(), mask);
                let status = collect_partition_with_documents(
                    &documents,
                    leaves,
                    plan,
                    metrics,
                    &mut collector,
                    collection_start_doc,
                    Ok,
                )?;
                debug_assert_eq!(status, CollectionStatus::Complete);
            }
            LoadedDocuments::Modern {
                documents: partition_documents,
                lengths,
                visibility,
                projection,
            } => {
                let documents = ModernWandDocuments::filtered(lengths.as_ref(), &visibility);
                if let Some(projection) = projection {
                    let mut addresses_resolved = 0;
                    let status = collect_partition_with_documents(
                        &documents,
                        leaves,
                        plan,
                        metrics,
                        &mut collector,
                        collection_start_doc,
                        |doc_id| {
                            let doc_id = DocId::new(u32::try_from(doc_id).map_err(|_| {
                                Error::index(format!(
                                    "FTS DocId {doc_id} exceeds the modern u32 domain"
                                ))
                            })?);
                            let row_id = projection.address(doc_id).ok_or_else(|| {
                                Error::internal(format!(
                                    "compound FTS scorer returned non-visible DocId {} in segment {segment_ordinal}, partition {partition_ordinal}",
                                    doc_id.get()
                                ))
                            })?;
                            addresses_resolved += 1;
                            Ok(row_id)
                        },
                    )?;
                    debug_assert_eq!(status, CollectionStatus::Complete);
                    metrics.record_compound_addresses_resolved(addresses_resolved);
                } else {
                    let max_buffered = collector
                        .limit
                        .saturating_add(SCORE_FLOOR_RESOLUTION_BATCH_SIZE);
                    let mut local_collector = TopKCollector::retaining_score_floor(
                        collector.limit,
                        collector.competitive_score.clone(),
                        max_buffered,
                    );
                    let status = collect_partition_with_documents(
                        &documents,
                        leaves,
                        plan,
                        metrics,
                        &mut local_collector,
                        collection_start_doc,
                        |doc_id| {
                            Ok(DocId::new(u32::try_from(doc_id).map_err(|_| {
                                Error::index(format!(
                                    "FTS DocId {doc_id} exceeds the modern u32 domain"
                                ))
                            })?))
                        },
                    )?;
                    metrics.record_compound_peak_buffered_candidates(local_collector.peak_buffered);
                    let boundary = match status {
                        CollectionStatus::Complete => {
                            PartitionCollectionBoundary::Deferred(DeferredCompoundRows {
                                documents: partition_documents,
                                rows: local_collector.into_candidates(),
                            })
                        }
                        CollectionStatus::ScoreFloorOverflow => {
                            metrics.record_compound_score_floor_overflows(1);
                            PartitionCollectionBoundary::Overflow(OverflowedCompoundPartition {
                                segment_ordinal,
                                partition_ordinal,
                                partition: source,
                                documents: partition_documents,
                                collection_start_doc,
                            })
                        }
                    };
                    metrics.record_compound_peak_buffered_candidates(collector.peak_buffered);
                    return Ok(CollectedPartitions {
                        collector,
                        remaining: partitions,
                        boundary: Some(boundary),
                    });
                }
            }
        }
    }
    metrics.record_compound_peak_buffered_candidates(collector.peak_buffered);
    let boundary = (!needs_prohibited.is_empty()).then_some(
        PartitionCollectionBoundary::NeedsProhibited(needs_prohibited),
    );
    Ok(CollectedPartitions {
        collector,
        remaining: partitions,
        boundary,
    })
}

async fn merge_resolved_compound_rows(
    collector: &mut TopKCollector<u64>,
    deferred: DeferredCompoundRows,
    metrics: &dyn MetricsCollector,
) -> Result<()> {
    for rows in deferred.rows.chunks(SCORE_FLOOR_RESOLUTION_BATCH_SIZE) {
        let doc_ids = rows.iter().map(|row| row.row_id).collect::<Vec<_>>();
        let addresses = deferred.documents.resolve_addresses(&doc_ids).await?;
        if addresses.len() != rows.len() {
            return Err(Error::internal(format!(
                "compound FTS resolved {} addresses for {} DocIds",
                addresses.len(),
                rows.len()
            )));
        }
        metrics.record_compound_address_resolution_batches(1);
        metrics.record_compound_peak_address_resolution_batch_size(rows.len());
        metrics.record_compound_addresses_resolved(addresses.len());
        for (row, row_id) in rows.iter().zip(addresses) {
            let status = collector.insert(ScoredRow {
                row_id,
                score: row.score,
            });
            debug_assert_eq!(status, CollectionStatus::Complete);
        }
    }
    metrics.record_compound_peak_buffered_candidates(collector.peak_buffered);
    Ok(())
}

async fn reload_compound_partition_with_projection(
    overflow: OverflowedCompoundPartition,
    leaves: &[PreparedLeaf],
    mask: Arc<RowAddrMask>,
    metrics: Arc<dyn MetricsCollector>,
) -> Result<LoadedPartition> {
    let projection = overflow.documents.address_projection().await?;
    let mut loaded = load_compound_partition(
        overflow.segment_ordinal,
        overflow.partition_ordinal,
        overflow.partition,
        leaves,
        mask,
        metrics,
        PartitionLoadMode::All,
    )
    .await?
    .ok_or_else(|| {
        Error::internal(format!(
            "compound FTS retry lost visible documents in segment {}, partition {}",
            overflow.segment_ordinal, overflow.partition_ordinal
        ))
    })?;
    match &mut loaded.documents {
        LoadedDocuments::Modern {
            projection: loaded_projection,
            ..
        } => *loaded_projection = Some(projection),
        LoadedDocuments::Legacy(_) => {
            return Err(Error::internal(format!(
                "compound FTS retry changed segment {}, partition {} from modern to legacy documents",
                overflow.segment_ordinal, overflow.partition_ordinal
            )));
        }
    }
    loaded.collection_start_doc = overflow.collection_start_doc;
    Ok(loaded)
}

/// Search one-column compound FTS directly over posting-backed scorers.
///
/// The caller must provide all committed index segments for the column and a
/// ready prefilter. One collector owns the global top-k heap and propagates its
/// score floor through every partition-local scorer tree. Modern partitions
/// resolve candidates in bounded batches; an oversized kth-score tie is retried
/// against a resident row-address projection so final row-id ordering stays exact.
pub async fn compound_search(
    indices: &[Arc<InvertedIndex>],
    query: &FtsQuery,
    params: &FtsSearchParams,
    prefilter: Arc<dyn PreFilter>,
    metrics: Arc<dyn MetricsCollector>,
) -> Result<(Vec<u64>, Vec<f32>)> {
    compound_search_impl(indices, query, params, prefilter, metrics, None).await
}

/// Search one-column compound FTS with caller-supplied corpus-wide BM25 statistics.
///
/// The scorer must contain an entry for every token used by every query leaf,
/// including terms produced by fuzzy expansion. An incomplete scorer is rejected
/// instead of treating missing token statistics as zero.
pub async fn compound_search_with_base_scorer(
    indices: &[Arc<InvertedIndex>],
    query: &FtsQuery,
    params: &FtsSearchParams,
    prefilter: Arc<dyn PreFilter>,
    metrics: Arc<dyn MetricsCollector>,
    base_scorer: Arc<MemBM25Scorer>,
) -> Result<(Vec<u64>, Vec<f32>)> {
    compound_search_impl(
        indices,
        query,
        params,
        prefilter,
        metrics,
        Some(base_scorer),
    )
    .await
}

async fn compound_search_impl(
    indices: &[Arc<InvertedIndex>],
    query: &FtsQuery,
    params: &FtsSearchParams,
    prefilter: Arc<dyn PreFilter>,
    metrics: Arc<dyn MetricsCollector>,
    base_scorer: Option<Arc<MemBM25Scorer>>,
) -> Result<(Vec<u64>, Vec<f32>)> {
    let limit = params.limit.unwrap_or(usize::MAX);
    if limit == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let (plan, leaves) =
        prepare_compound_query(indices, query, params, metrics.as_ref(), base_scorer).await?;
    let has_prohibited_leaves = leaves
        .iter()
        .any(|leaf| leaf.score_mode == CompoundScoreMode::CompleteNoScores);
    prefilter.wait_for_ready().await?;
    let mask = prefilter.mask();
    let mut collector = TopKCollector::new(limit);

    for (segment_ordinal, index) in indices.iter().enumerate() {
        let loads =
            index
                .partitions
                .iter()
                .cloned()
                .enumerate()
                .map(|(partition_ordinal, partition)| {
                    load_compound_partition(
                        segment_ordinal,
                        partition_ordinal,
                        partition,
                        &leaves,
                        mask.clone(),
                        metrics.clone(),
                        PartitionLoadMode::ScoringOnly,
                    )
                });
        let mut loaded_partitions = stream::iter(loads)
            .buffer_unordered(get_num_compute_intensive_cpus().clamp(1, 32))
            .try_collect::<Vec<_>>()
            .await?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        if has_prohibited_leaves {
            loaded_partitions.sort_unstable_by_key(|partition| partition.partition_ordinal);
        }
        let mut partitions = VecDeque::from(loaded_partitions);
        while !partitions.is_empty() {
            let cpu_plan = plan.clone();
            let cpu_mask = mask.clone();
            let cpu_metrics = metrics.clone();
            let collected = spawn_cpu(move || {
                collect_loaded_partitions(
                    partitions,
                    &cpu_plan,
                    has_prohibited_leaves,
                    cpu_mask.as_ref(),
                    cpu_metrics.as_ref(),
                    collector,
                )
            })
            .await?;
            collector = collected.collector;
            partitions = collected.remaining;
            match collected.boundary {
                Some(PartitionCollectionBoundary::NeedsProhibited(pending)) => {
                    let loads = pending.into_iter().map(|partition| {
                        load_deferred_prohibited_postings(partition, &leaves, metrics.clone())
                    });
                    let loaded = futures::future::try_join_all(loads).await?;
                    let mut resumed = VecDeque::from(loaded);
                    resumed.append(&mut partitions);
                    partitions = resumed;
                }
                Some(PartitionCollectionBoundary::Deferred(deferred)) => {
                    merge_resolved_compound_rows(&mut collector, deferred, metrics.as_ref())
                        .await?;
                }
                Some(PartitionCollectionBoundary::Overflow(overflow)) => {
                    let retry = reload_compound_partition_with_projection(
                        overflow,
                        &leaves,
                        mask.clone(),
                        metrics.clone(),
                    )
                    .await?;
                    partitions.push_front(retry);
                }
                None => debug_assert!(partitions.is_empty()),
            }
        }
    }

    let rows = collector.into_rows();
    Ok(rows.into_iter().map(|row| (row.row_id, row.score)).unzip())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::AtomicUsize;

    use rand::{Rng, SeedableRng, rngs::SmallRng};

    use super::*;
    use crate::scalar::inverted::query::{BooleanQuery, BoostQuery, MultiMatchQuery, Occur};

    fn rows(values: &[(u64, f32)]) -> Vec<ScoredRow> {
        values
            .iter()
            .map(|(row_id, score)| ScoredRow::new(*row_id, *score).unwrap())
            .collect()
    }

    fn materialized(values: &[(u64, f32)]) -> Box<dyn ComposableScorer> {
        Box::new(MaterializedScorer::try_new(rows(values)).unwrap())
    }

    #[test]
    fn direct_prohibited_leaves_use_membership_mode() {
        let query = FtsQuery::Boolean(BooleanQuery::new([
            (
                Occur::Must,
                FtsQuery::Match(MatchQuery::new(String::from("required"))),
            ),
            (
                Occur::MustNot,
                FtsQuery::Match(MatchQuery::new(String::from("excluded"))),
            ),
            (
                Occur::MustNot,
                FtsQuery::Phrase(PhraseQuery::new(String::from("excluded phrase"))),
            ),
        ]));
        let mut leaves = Vec::new();

        collect_leaf_queries(&query, CompoundScoreMode::Scoring, &mut leaves).unwrap();

        assert!(matches!(
            &leaves[0],
            (LeafQuery::Match(_), CompoundScoreMode::Scoring)
        ));
        assert!(matches!(
            &leaves[1],
            (LeafQuery::Match(_), CompoundScoreMode::CompleteNoScores)
        ));
        assert!(matches!(
            &leaves[2],
            (LeafQuery::Phrase(_), CompoundScoreMode::CompleteNoScores)
        ));
    }

    #[test]
    fn prohibited_subtrees_recursively_use_membership_mode() {
        let nested = FtsQuery::Boolean(BooleanQuery::new([
            (
                Occur::Should,
                FtsQuery::Phrase(PhraseQuery::new(String::from("optional phrase"))),
            ),
            (
                Occur::Must,
                FtsQuery::MultiMatch(MultiMatchQuery {
                    match_queries: vec![
                        MatchQuery::new(String::from("multi a")),
                        MatchQuery::new(String::from("multi b")),
                    ],
                }),
            ),
            (
                Occur::Must,
                FtsQuery::Boost(BoostQuery::new(
                    FtsQuery::Match(MatchQuery::new(String::from("boost positive"))),
                    FtsQuery::Match(MatchQuery::new(String::from("boost negative"))),
                    Some(0.25),
                )),
            ),
            (
                Occur::MustNot,
                FtsQuery::Boolean(BooleanQuery::new([(
                    Occur::Should,
                    FtsQuery::Phrase(PhraseQuery::new(String::from("nested exclusion"))),
                )])),
            ),
        ]));
        let query = FtsQuery::Boolean(BooleanQuery::new([
            (
                Occur::Must,
                FtsQuery::Match(MatchQuery::new(String::from("required"))),
            ),
            (Occur::MustNot, nested),
        ]));
        let mut leaves = Vec::new();

        collect_leaf_queries(&query, CompoundScoreMode::Scoring, &mut leaves).unwrap();

        assert_eq!(
            leaves
                .iter()
                .map(|(leaf, _)| leaf.terms())
                .collect::<Vec<_>>(),
            vec![
                "required",
                "multi a",
                "multi b",
                "boost positive",
                "nested exclusion"
            ]
        );
        assert_eq!(leaves[0].1, CompoundScoreMode::Scoring);
        assert!(
            leaves[1..]
                .iter()
                .all(|(_, mode)| *mode == CompoundScoreMode::CompleteNoScores)
        );
    }

    #[test]
    fn prohibited_subtree_pruning_preserves_query_validation() {
        let invalid_optional =
            FtsQuery::Match(MatchQuery::new(String::from("invalid optional")).with_boost(f32::NAN));
        let prohibited = FtsQuery::Boolean(BooleanQuery::new([
            (
                Occur::Must,
                FtsQuery::Match(MatchQuery::new(String::from("required"))),
            ),
            (Occur::Should, invalid_optional),
        ]));
        let query = FtsQuery::Boolean(BooleanQuery::new([
            (
                Occur::Must,
                FtsQuery::Match(MatchQuery::new(String::from("positive"))),
            ),
            (Occur::MustNot, prohibited),
        ]));

        let error = validate_compound_query(&query).unwrap_err();
        assert!(matches!(&error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("MatchQuery boost must be finite and non-negative, got NaN")
        );

        let empty_multi_match = FtsQuery::Boolean(BooleanQuery::new([
            (
                Occur::Must,
                FtsQuery::Match(MatchQuery::new(String::from("positive"))),
            ),
            (
                Occur::MustNot,
                FtsQuery::Boost(BoostQuery::new(
                    FtsQuery::Match(MatchQuery::new(String::from("required"))),
                    FtsQuery::MultiMatch(MultiMatchQuery {
                        match_queries: Vec::new(),
                    }),
                    Some(0.5),
                )),
            ),
        ]));
        let error = validate_compound_query(&empty_multi_match).unwrap_err();
        assert!(matches!(&error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("MultiMatchQuery must have at least one match query")
        );
    }

    #[test]
    fn prohibited_subtrees_never_score_nested_compound_leaves() {
        let nested = FtsQuery::Boolean(BooleanQuery::new([
            (
                Occur::Should,
                FtsQuery::Match(MatchQuery::new(String::from("optional"))),
            ),
            (
                Occur::Must,
                FtsQuery::MultiMatch(MultiMatchQuery {
                    match_queries: vec![
                        MatchQuery::new(String::from("multi a")),
                        MatchQuery::new(String::from("multi b")),
                    ],
                }),
            ),
            (
                Occur::Must,
                FtsQuery::Boost(BoostQuery::new(
                    FtsQuery::Match(
                        MatchQuery::new(String::from("boost positive")).with_boost(2.0),
                    ),
                    FtsQuery::Match(MatchQuery::new(String::from("boost negative"))),
                    Some(0.5),
                )),
            ),
            (
                Occur::MustNot,
                FtsQuery::Boolean(BooleanQuery::new([(
                    Occur::Should,
                    FtsQuery::Match(MatchQuery::new(String::from("inner exclusion"))),
                )])),
            ),
        ]));
        let query = FtsQuery::Boolean(BooleanQuery::new([
            (
                Occur::Must,
                FtsQuery::Match(MatchQuery::new(String::from("required"))),
            ),
            (Occur::MustNot, nested),
        ]));
        let mut num_leaves = 0;
        let plan =
            CompoundScorerPlan::from_query(&query, &mut num_leaves, CompoundScoreMode::Scoring)
                .unwrap();
        assert_eq!(num_leaves, 5);

        let (positive, _) = instrumented(materialized(&[
            (0, 10.0),
            (1, 9.0),
            (2, 8.0),
            (3, 7.0),
            (4, 6.0),
        ]));
        let prohibited_rows = [
            &[(1, 1.0), (2, 1.0), (3, 1.0)][..],
            &[(2, 1.0), (4, 1.0)][..],
            &[(1, 3.0), (2, 3.0), (4, 3.0)][..],
        ];
        let metrics = BooleanMetrics::default();
        let mut prohibited_work = Vec::new();
        let mut leaves = vec![Some(positive)];
        for values in prohibited_rows {
            let (scorer, work) = instrumented(materialized(values));
            leaves.push(Some(scorer));
            prohibited_work.push(work);
        }
        let (inner_exclusion, _, inner_exclusion_confirmations) =
            two_phase(&[(2, 1.0), (4, 1.0)], vec![2, 4], Some(1.0));
        let (inner_exclusion, work) = instrumented(inner_exclusion);
        leaves.push(Some(inner_exclusion));
        prohibited_work.push(work);
        let mut scorer = plan.build(&mut leaves, &metrics).unwrap();

        assert_eq!(
            TopKCollector::new(10).collect(scorer.as_mut()).unwrap(),
            rows(&[(0, 10.0), (2, 8.0), (3, 7.0), (4, 6.0)])
        );
        assert!(leaves.iter().all(Option::is_none));
        assert_eq!(
            inner_exclusion_confirmations.load(AtomicOrdering::Relaxed),
            2
        );
        for work in prohibited_work {
            assert_eq!(work.scores.load(AtomicOrdering::Relaxed), 0);
            assert_eq!(work.shallow_advances.load(AtomicOrdering::Relaxed), 0);
            assert_eq!(work.bounds.load(AtomicOrdering::Relaxed), 0);
        }
    }

    #[test]
    fn positive_gate_keeps_floor_ties() {
        let mut tied = MaterializedScorer::try_new(rows(&[(0, 4.0), (1, 5.0)])).unwrap();
        assert_eq!(first_competitive_match(&mut tied, 5.0).unwrap(), Some(1));

        let mut below = MaterializedScorer::try_new(rows(&[(0, 4.0), (1, 5.0)])).unwrap();
        assert_eq!(first_competitive_match(&mut below, 6.0).unwrap(), None);

        let (mut unbounded, work) = instrumented(materialized(&[(0, 1.0)]));
        assert_eq!(
            first_competitive_match(unbounded.as_mut(), f32::NEG_INFINITY).unwrap(),
            Some(0)
        );
        assert_eq!(work.scores.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(work.shallow_advances.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(work.bounds.load(AtomicOrdering::Relaxed), 0);
    }

    #[test]
    fn collection_resumes_at_first_competitive_positive() {
        let values = [(0, 1.0), (1, 10.0), (2, 11.0)];
        let mut gate = MaterializedScorer::try_new(rows(&values)).unwrap();
        let start_doc = first_competitive_match(&mut gate, 10.0).unwrap();
        assert_eq!(start_doc, Some(1));

        let mut scorer = BooleanScorer::try_new(
            Vec::new(),
            vec![materialized(&values)],
            vec![materialized(&[(1, 1.0)])],
        )
        .unwrap();
        let competitive_score = Arc::new(CompetitiveScore::default());
        competitive_score.raise(10.0);
        let mut collector = TopKCollector::with_competitive_score(1, competitive_score);

        collector
            .collect_mapped_from(&mut scorer, start_doc, Ok)
            .unwrap();

        assert_eq!(collector.into_rows(), rows(&[(2, 11.0)]));
    }

    #[test]
    fn positive_gate_falls_back_for_signed_negative_subtrees() {
        let match_query = |terms: &str| FtsQuery::Match(MatchQuery::new(String::from(terms)));
        let signed_negative = FtsQuery::Boost(BoostQuery::new(
            match_query("inner positive"),
            match_query("inner negative"),
            Some(1.0),
        ));
        let unsafe_query = FtsQuery::Boost(BoostQuery::new(
            match_query("outer positive"),
            signed_negative.clone(),
            Some(0.5),
        ));
        let mut num_leaves = 0;
        let unsafe_plan = CompoundScorerPlan::from_query(
            &unsafe_query,
            &mut num_leaves,
            CompoundScoreMode::Scoring,
        )
        .unwrap();
        assert!(!unsafe_plan.positive_gate_is_safe());
        let metrics = BooleanMetrics::default();
        let mut leaves = vec![
            Some(materialized(&[(0, 1.0)])),
            Some(materialized(&[(0, 1.0)])),
            Some(materialized(&[(0, 11.0)])),
        ];
        let mut scorer = unsafe_plan.build(&mut leaves, &metrics).unwrap();
        assert_eq!(
            TopKCollector::new(1).collect(scorer.as_mut()).unwrap(),
            rows(&[(0, 6.0)])
        );

        let safe_query = FtsQuery::Boost(BoostQuery::new(
            match_query("outer positive"),
            signed_negative,
            Some(0.0),
        ));
        let mut num_leaves = 0;
        let safe_plan = CompoundScorerPlan::from_query(
            &safe_query,
            &mut num_leaves,
            CompoundScoreMode::Scoring,
        )
        .unwrap();
        assert!(safe_plan.positive_gate_is_safe());
    }

    fn should_maxscore<'a>(
        children: Vec<BoxScorer<'a>>,
        metrics: Option<&'a dyn MetricsCollector>,
    ) -> ShouldMaxScoreScorer<'a> {
        let global_bounds = ShouldMaxScoreScorer::global_bounds(&children).unwrap();
        ShouldMaxScoreScorer::new(children, global_bounds, metrics)
    }

    #[derive(Default)]
    struct ShouldMetrics {
        reports: AtomicUsize,
        skipped_windows: AtomicUsize,
        bound_recomputations: AtomicUsize,
        essential_evaluations: AtomicUsize,
        non_essential_evaluations: AtomicUsize,
    }

    impl MetricsCollector for ShouldMetrics {
        fn record_parts_loaded(&self, _num_parts: usize) {}

        fn record_index_loads(&self, _num_loads: usize) {}

        fn record_comparisons(&self, _num_comparisons: usize) {}

        fn record_compound_should_skipped_windows(&self, num_windows: usize) {
            self.reports.fetch_add(1, AtomicOrdering::Relaxed);
            self.skipped_windows
                .fetch_add(num_windows, AtomicOrdering::Relaxed);
        }

        fn record_compound_should_bound_recomputations(&self, num_recomputations: usize) {
            self.reports.fetch_add(1, AtomicOrdering::Relaxed);
            self.bound_recomputations
                .fetch_add(num_recomputations, AtomicOrdering::Relaxed);
        }

        fn record_compound_should_essential_evaluations(&self, num_evaluations: usize) {
            self.reports.fetch_add(1, AtomicOrdering::Relaxed);
            self.essential_evaluations
                .fetch_add(num_evaluations, AtomicOrdering::Relaxed);
        }

        fn record_compound_should_non_essential_evaluations(&self, num_evaluations: usize) {
            self.reports.fetch_add(1, AtomicOrdering::Relaxed);
            self.non_essential_evaluations
                .fetch_add(num_evaluations, AtomicOrdering::Relaxed);
        }
    }

    #[derive(Default)]
    struct BooleanMetrics {
        reports: AtomicUsize,
        positive_survivors: AtomicUsize,
        must_not_probes: AtomicUsize,
    }

    impl MetricsCollector for BooleanMetrics {
        fn record_parts_loaded(&self, _num_parts: usize) {}

        fn record_index_loads(&self, _num_loads: usize) {}

        fn record_comparisons(&self, _num_comparisons: usize) {}

        fn record_compound_positive_survivors(&self, num_candidates: usize) {
            self.reports.fetch_add(1, AtomicOrdering::Relaxed);
            self.positive_survivors
                .fetch_add(num_candidates, AtomicOrdering::Relaxed);
        }

        fn record_compound_must_not_probes(&self, num_probes: usize) {
            self.reports.fetch_add(1, AtomicOrdering::Relaxed);
            self.must_not_probes
                .fetch_add(num_probes, AtomicOrdering::Relaxed);
        }
    }

    #[test]
    fn score_bounds_are_conservative_under_nested_sum_and_boost() {
        let should = DisjunctionScorer::try_new(
            vec![
                materialized(&[(0, 0.1), (2, 2.0)]),
                materialized(&[(0, 0.2)]),
            ],
            DisjunctionScore::Sum,
        )
        .unwrap();
        let negative = materialized(&[(0, 0.3), (2, 5.0)]);
        let mut scorer = BoostScorer::try_new(Box::new(should), negative, 0.5).unwrap();

        assert_eq!(scorer.next().unwrap(), Some(0));
        let up_to = scorer.advance_shallow(0).unwrap();
        let bounds = scorer.score_bounds(up_to).unwrap();
        let first_score = scorer.score().unwrap();
        assert!(bounds.lower <= first_score);
        assert!(bounds.upper >= first_score);

        assert_eq!(scorer.next().unwrap(), Some(2));
        let second_score = scorer.score().unwrap();
        assert!(second_score.is_sign_negative());
        let up_to = scorer.advance_shallow(2).unwrap();
        let bounds = scorer.score_bounds(up_to).unwrap();
        assert!(bounds.lower <= second_score);
        assert!(bounds.upper >= second_score);
    }

    #[test]
    fn collector_propagates_threshold_across_partitions_and_keeps_ties() {
        let mut collector = TopKCollector::new(2);
        let mut first = MaterializedScorer::try_new(rows(&[(8, 9.0), (4, 10.0), (3, 9.0)]))
            .unwrap()
            .with_block_size(1);
        collector.collect_mapped(&mut first, Ok).unwrap();
        assert_eq!(collector.competitive_score.get(), 9.0);

        let mut second = MaterializedScorer::try_new(rows(&[(1, 1.0), (2, 9.0)]))
            .unwrap()
            .with_block_size(1);
        collector.collect_mapped(&mut second, Ok).unwrap();
        assert_eq!(
            collector.into_rows(),
            vec![
                ScoredRow {
                    row_id: 4,
                    score: 10.0
                },
                ScoredRow {
                    row_id: 2,
                    score: 9.0
                }
            ]
        );
    }

    #[test]
    fn collector_bounds_equal_score_candidates() {
        let limit = 1;
        let num_candidates = DEFAULT_BLOCK_SIZE * 4;
        let values = (0..num_candidates)
            .map(|row_id| (row_id as u64, 1.0))
            .collect::<Vec<_>>();
        let mut scorer = MaterializedScorer::try_new(rows(&values)).unwrap();
        let max_buffered = limit + SCORE_FLOOR_RESOLUTION_BATCH_SIZE;
        let mut collector = TopKCollector::retaining_score_floor(
            limit,
            Arc::new(CompetitiveScore::default()),
            max_buffered,
        );

        let status = collector.collect_mapped(&mut scorer, Ok).unwrap();

        assert_eq!(status, CollectionStatus::ScoreFloorOverflow);
        assert_eq!(collector.heap.len(), max_buffered);
    }

    #[test]
    fn collector_reclaims_obsolete_score_floor_before_overflowing() {
        let limit = 2;
        let max_buffered = 4;
        let values = [
            (0, 1.0),
            (1, 1.0),
            (2, 1.0),
            (3, 2.0),
            (4, 2.0),
            (5, 2.0),
            (6, 2.0),
            (7, 2.0),
        ];
        let mut scorer = MaterializedScorer::try_new(rows(&values)).unwrap();
        let competitive_score = Arc::new(CompetitiveScore::default());
        let mut collector =
            TopKCollector::retaining_score_floor(limit, competitive_score.clone(), max_buffered);

        let status = collector.collect_mapped(&mut scorer, Ok).unwrap();

        assert_eq!(status, CollectionStatus::ScoreFloorOverflow);
        assert_eq!(collector.heap.len(), max_buffered);
        assert!(collector.heap.iter().all(|row| row.0.score == 2.0));
        assert_eq!(competitive_score.get(), 2.0);
    }

    struct TwoPhaseScorer {
        inner: MaterializedScorer,
        accepted: Vec<u64>,
        match_cost: Option<f32>,
        approximations: Arc<AtomicUsize>,
        confirmations: Arc<AtomicUsize>,
    }

    impl ComposableScorer for TwoPhaseScorer {
        fn doc(&self) -> Option<u64> {
            self.inner.doc()
        }

        fn next(&mut self) -> Result<Option<u64>> {
            let doc = self.inner.next()?;
            if doc.is_some() {
                self.approximations.fetch_add(1, AtomicOrdering::Relaxed);
            }
            Ok(doc)
        }

        fn advance(&mut self, target: u64) -> Result<Option<u64>> {
            let doc = self.inner.advance(target)?;
            if doc.is_some() {
                self.approximations.fetch_add(1, AtomicOrdering::Relaxed);
            }
            Ok(doc)
        }

        fn cost(&self) -> usize {
            self.inner.cost()
        }

        fn score(&mut self) -> Result<f32> {
            self.inner.score()
        }

        fn advance_shallow(&mut self, target: u64) -> Result<u64> {
            self.inner.advance_shallow(target)
        }

        fn score_bounds(&mut self, up_to: u64) -> Result<ScoreBounds> {
            self.inner.score_bounds(up_to)
        }

        fn global_score_upper_bound(&self) -> Option<f32> {
            self.inner.global_score_upper_bound()
        }

        fn set_min_competitive_score(&mut self, min_score: f32) -> Result<()> {
            self.inner.set_min_competitive_score(min_score)
        }

        fn matches(&mut self) -> Result<bool> {
            self.confirmations.fetch_add(1, AtomicOrdering::Relaxed);
            Ok(self
                .doc()
                .is_some_and(|doc| self.accepted.binary_search(&doc).is_ok()))
        }

        fn match_cost(&self) -> Option<f32> {
            self.match_cost
        }

        fn scores_non_negative(&self) -> bool {
            true
        }
    }

    fn two_phase(
        values: &[(u64, f32)],
        accepted: Vec<u64>,
        match_cost: Option<f32>,
    ) -> (
        Box<dyn ComposableScorer>,
        Arc<AtomicUsize>,
        Arc<AtomicUsize>,
    ) {
        let approximations = Arc::new(AtomicUsize::new(0));
        let confirmations = Arc::new(AtomicUsize::new(0));
        let scorer = TwoPhaseScorer {
            inner: MaterializedScorer::try_new(rows(values)).unwrap(),
            accepted,
            match_cost,
            approximations: approximations.clone(),
            confirmations: confirmations.clone(),
        };
        (Box::new(scorer), approximations, confirmations)
    }

    #[derive(Default)]
    struct ScorerWork {
        advances: AtomicUsize,
        confirmations: AtomicUsize,
        scores: AtomicUsize,
        shallow_advances: AtomicUsize,
        bounds: AtomicUsize,
    }

    struct InstrumentedScorer<'a> {
        inner: BoxScorer<'a>,
        work: Arc<ScorerWork>,
    }

    impl ComposableScorer for InstrumentedScorer<'_> {
        fn doc(&self) -> Option<u64> {
            self.inner.doc()
        }

        fn document_key(&self) -> Option<u64> {
            self.inner.document_key()
        }

        fn next(&mut self) -> Result<Option<u64>> {
            let doc = self.inner.next()?;
            if doc.is_some() {
                self.work.advances.fetch_add(1, AtomicOrdering::Relaxed);
            }
            Ok(doc)
        }

        fn advance(&mut self, target: u64) -> Result<Option<u64>> {
            let doc = self.inner.advance(target)?;
            if doc.is_some() {
                self.work.advances.fetch_add(1, AtomicOrdering::Relaxed);
            }
            Ok(doc)
        }

        fn cost(&self) -> usize {
            self.inner.cost()
        }

        fn score(&mut self) -> Result<f32> {
            self.work.scores.fetch_add(1, AtomicOrdering::Relaxed);
            self.inner.score()
        }

        fn advance_shallow(&mut self, target: u64) -> Result<u64> {
            self.work
                .shallow_advances
                .fetch_add(1, AtomicOrdering::Relaxed);
            self.inner.advance_shallow(target)
        }

        fn score_bounds(&mut self, up_to: u64) -> Result<ScoreBounds> {
            self.work.bounds.fetch_add(1, AtomicOrdering::Relaxed);
            self.inner.score_bounds(up_to)
        }

        fn global_score_upper_bound(&self) -> Option<f32> {
            self.inner.global_score_upper_bound()
        }

        fn set_min_competitive_score(&mut self, min_score: f32) -> Result<()> {
            self.inner.set_min_competitive_score(min_score)
        }

        fn matches(&mut self) -> Result<bool> {
            self.work
                .confirmations
                .fetch_add(1, AtomicOrdering::Relaxed);
            self.inner.matches()
        }

        fn match_cost(&self) -> Option<f32> {
            self.inner.match_cost()
        }

        fn scores_non_negative(&self) -> bool {
            self.inner.scores_non_negative()
        }
    }

    fn instrumented<'a>(inner: BoxScorer<'a>) -> (BoxScorer<'a>, Arc<ScorerWork>) {
        let work = Arc::new(ScorerWork::default());
        (
            Box::new(InstrumentedScorer {
                inner,
                work: work.clone(),
            }),
            work,
        )
    }

    struct UnboundedScorer {
        inner: MaterializedScorer,
    }

    impl ComposableScorer for UnboundedScorer {
        fn doc(&self) -> Option<u64> {
            self.inner.doc()
        }

        fn next(&mut self) -> Result<Option<u64>> {
            self.inner.next()
        }

        fn advance(&mut self, target: u64) -> Result<Option<u64>> {
            self.inner.advance(target)
        }

        fn cost(&self) -> usize {
            self.inner.cost()
        }

        fn score(&mut self) -> Result<f32> {
            self.inner.score()
        }

        fn advance_shallow(&mut self, target: u64) -> Result<u64> {
            self.inner.advance_shallow(target)
        }

        fn score_bounds(&mut self, _up_to: u64) -> Result<ScoreBounds> {
            Ok(ScoreBounds::UNBOUNDED)
        }

        fn set_min_competitive_score(&mut self, min_score: f32) -> Result<()> {
            self.inner.set_min_competitive_score(min_score)
        }

        fn scores_non_negative(&self) -> bool {
            true
        }
    }

    struct CountingScorer {
        inner: MaterializedScorer,
        cost: usize,
        advance_calls: Arc<AtomicUsize>,
    }

    impl ComposableScorer for CountingScorer {
        fn doc(&self) -> Option<u64> {
            self.inner.doc()
        }

        fn next(&mut self) -> Result<Option<u64>> {
            self.inner.next()
        }

        fn advance(&mut self, target: u64) -> Result<Option<u64>> {
            self.advance_calls.fetch_add(1, AtomicOrdering::Relaxed);
            self.inner.advance(target)
        }

        fn cost(&self) -> usize {
            self.cost
        }

        fn score(&mut self) -> Result<f32> {
            self.inner.score()
        }

        fn advance_shallow(&mut self, target: u64) -> Result<u64> {
            self.inner.advance_shallow(target)
        }

        fn score_bounds(&mut self, up_to: u64) -> Result<ScoreBounds> {
            self.inner.score_bounds(up_to)
        }

        fn global_score_upper_bound(&self) -> Option<f32> {
            self.inner.global_score_upper_bound()
        }

        fn set_min_competitive_score(&mut self, min_score: f32) -> Result<()> {
            self.inner.set_min_competitive_score(min_score)
        }

        fn matches(&mut self) -> Result<bool> {
            self.inner.matches()
        }

        fn scores_non_negative(&self) -> bool {
            self.inner.scores_non_negative()
        }
    }

    fn counting(
        values: &[(u64, f32)],
        cost: usize,
    ) -> (Box<dyn ComposableScorer>, Arc<AtomicUsize>) {
        let advance_calls = Arc::new(AtomicUsize::new(0));
        let scorer = CountingScorer {
            inner: MaterializedScorer::try_new(rows(values)).unwrap(),
            cost,
            advance_calls: advance_calls.clone(),
        };
        (Box::new(scorer), advance_calls)
    }

    #[test]
    fn collector_confirms_two_phase_matches_without_a_cost_hint() {
        let (mut scorer, approximations, confirmations) =
            two_phase(&[(1, 100.0), (2, 2.0), (3, 1.0)], vec![2, 3], None);
        let results = TopKCollector::new(2).collect(scorer.as_mut()).unwrap();
        assert_eq!(results, rows(&[(2, 2.0), (3, 1.0)]));
        assert_eq!(approximations.load(AtomicOrdering::Relaxed), 3);
        assert_eq!(confirmations.load(AtomicOrdering::Relaxed), 3);
        assert_eq!(scorer.match_cost(), None);
    }

    #[test]
    fn required_conjunction_confirms_cheapest_first_and_short_circuits() {
        let values = (0..100).map(|doc| (doc, 1.0)).collect::<Vec<_>>();
        let accepted_by_cheap = (0..100).step_by(5).collect::<Vec<_>>();

        let (expensive, expensive_approximations, expensive_confirmations) =
            two_phase(&values, (0..100).collect(), Some(10.0));
        let (cheap, cheap_approximations, cheap_confirmations) =
            two_phase(&values, accepted_by_cheap.clone(), Some(1.0));
        let mut scorer = RequiredConjunctionScorer::try_new(vec![expensive, cheap]).unwrap();
        assert_eq!(scorer.confirmation_order.as_deref(), Some(&[1, 0][..]));

        let results = TopKCollector::new(100).collect(&mut scorer).unwrap();
        let expected = accepted_by_cheap
            .iter()
            .map(|doc| (*doc, 2.0))
            .collect::<Vec<_>>();
        assert_eq!(results, rows(&expected));
        assert_eq!(cheap_confirmations.load(AtomicOrdering::Relaxed), 100);
        assert_eq!(expensive_confirmations.load(AtomicOrdering::Relaxed), 20);
        let approximations = cheap_approximations.load(AtomicOrdering::Relaxed)
            + expensive_approximations.load(AtomicOrdering::Relaxed);
        let confirmations = cheap_confirmations.load(AtomicOrdering::Relaxed)
            + expensive_confirmations.load(AtomicOrdering::Relaxed);
        assert_eq!(approximations, 200);
        assert_eq!(confirmations, 120);
        assert!(
            confirmations * 5 <= approximations * 4,
            "confirmation ordering should reduce work by at least 20%: {confirmations}/{approximations}"
        );

        let (cheap, _, _) = two_phase(&values, accepted_by_cheap, Some(1.0));
        let (expensive, _, _) = two_phase(&values, (0..100).collect(), Some(10.0));
        let scorer = RequiredConjunctionScorer::try_new(vec![cheap, expensive]).unwrap();
        assert!(scorer.confirmation_order.is_none());
    }

    #[test]
    fn required_conjunction_confirms_children_without_cost_hints() {
        let (unknown, _, unknown_confirmations) = two_phase(&[(0, 1.0)], Vec::new(), None);
        let (costed, _, costed_confirmations) = two_phase(&[(0, 1.0)], vec![0], Some(1.0));
        let mut scorer = RequiredConjunctionScorer::try_new(vec![unknown, costed]).unwrap();

        assert_eq!(scorer.confirmation_order.as_deref(), Some(&[1, 0][..]));
        assert!(
            TopKCollector::new(1)
                .collect(&mut scorer)
                .unwrap()
                .is_empty()
        );
        assert_eq!(costed_confirmations.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(unknown_confirmations.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    fn required_conjunction_rejects_invalid_match_cost() {
        let (invalid, _, _) = two_phase(&[(0, 1.0)], vec![0], Some(f32::NAN));

        let error = RequiredConjunctionScorer::try_new(vec![invalid])
            .err()
            .unwrap();
        assert!(matches!(error, Error::Internal { .. }));
        assert!(
            error
                .to_string()
                .contains("child 0 reported invalid two-phase match cost: NaN")
        );
    }

    #[test]
    fn required_conjunction_uses_all_must_scores_for_competitive_bounds() {
        let left = Box::new(
            MaterializedScorer::try_new(rows(&[(1, 3.0), (3, 1.0)]))
                .unwrap()
                .with_block_size(1),
        );
        let right = Box::new(
            MaterializedScorer::try_new(rows(&[(1, 30.0), (3, 10.0)]))
                .unwrap()
                .with_block_size(1),
        );
        let mut scorer = RequiredConjunctionScorer::try_new(vec![left, right]).unwrap();
        let competitive_score = Arc::new(CompetitiveScore::default());
        competitive_score.raise(10.0);

        let results = TopKCollector::with_competitive_score(10, competitive_score)
            .collect(&mut scorer)
            .unwrap();

        assert_eq!(results, rows(&[(1, 33.0), (3, 11.0)]));
    }

    #[test]
    fn required_conjunction_aligns_cheapest_clause_first() {
        let dense_rows = (0..=100).map(|row_id| (row_id, 1.0)).collect::<Vec<_>>();
        let (dense, dense_advance_calls) = counting(&dense_rows, 101);
        let (rare, rare_advance_calls) = counting(&[(50, 1.0)], 1);
        let mut scorer = RequiredConjunctionScorer::try_new(vec![dense, rare]).unwrap();
        assert_eq!(scorer.approximation_order.as_deref(), Some(&[1, 0][..]));

        assert_eq!(scorer.next().unwrap(), Some(50));
        assert_eq!(scorer.next().unwrap(), None);
        assert_eq!(dense_advance_calls.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(rare_advance_calls.load(AtomicOrdering::Relaxed), 2);

        let (rare, _) = counting(&[(50, 1.0)], 1);
        let (dense, _) = counting(&dense_rows, 101);
        let scorer = RequiredConjunctionScorer::try_new(vec![rare, dense]).unwrap();
        assert!(scorer.approximation_order.is_none());
    }

    #[test]
    fn required_conjunction_preserves_query_score_order() {
        let (large, _) = counting(&[(0, 16_777_216.0)], 3);
        let (first_small, _) = counting(&[(0, 1.0)], 1);
        let (second_small, _) = counting(&[(0, 1.0)], 2);
        let mut scorer =
            RequiredConjunctionScorer::try_new(vec![large, first_small, second_small]).unwrap();

        assert_eq!(scorer.next().unwrap(), Some(0));
        assert_eq!(scorer.score().unwrap(), 16_777_216.0);
    }

    #[test]
    fn boolean_sums_all_matching_clause_scores() {
        let must = vec![
            materialized(&[(1, 3.0), (2, 2.0), (3, 1.0)]),
            materialized(&[(1, 30.0), (3, 10.0)]),
        ];
        let should = vec![
            materialized(&[(1, 0.5), (3, 4.0)]),
            materialized(&[(3, 2.0)]),
        ];
        let must_not = vec![materialized(&[(1, 9.0)])];
        let mut boolean = BooleanScorer::try_new(should, must, must_not).unwrap();
        let results = TopKCollector::new(10).collect(&mut boolean).unwrap();
        assert_eq!(
            results,
            vec![ScoredRow {
                row_id: 3,
                score: 17.0
            }]
        );

        let mut dismax = DisjunctionScorer::try_new(
            vec![
                materialized(&[(1, 2.0), (3, 3.0)]),
                materialized(&[(1, 4.0), (2, 4.0)]),
            ],
            DisjunctionScore::Max,
        )
        .unwrap();
        let results = TopKCollector::new(2).collect(&mut dismax).unwrap();
        assert_eq!(results, rows(&[(1, 4.0), (2, 4.0)]));
    }

    #[test]
    fn boolean_delays_must_not_until_exact_positive_score_is_competitive() {
        let positive_rows = [(0, 0.5), (1, 50.0)];
        let positive = || {
            Box::new(MaterializedScorer::try_new(rows(&positive_rows)).unwrap()) as BoxScorer<'_>
        };
        let (prohibited, prohibited_approximations, prohibited_confirmations) =
            two_phase(&[(0, 0.0), (1, 0.0)], Vec::new(), Some(1.0));
        let mut scorer =
            BooleanScorer::try_new(Vec::new(), vec![positive(), positive()], vec![prohibited])
                .unwrap();
        let competitive_score = Arc::new(CompetitiveScore::default());
        competitive_score.raise(50.0);

        let results = TopKCollector::with_competitive_score(1, competitive_score)
            .collect(&mut scorer)
            .unwrap();

        assert_eq!(results, rows(&[(1, 100.0)]));
        assert_eq!(
            prohibited_approximations.load(AtomicOrdering::Relaxed),
            1,
            "the low-scoring document shares a high block bound but must not probe MUST_NOT"
        );
        assert_eq!(prohibited_confirmations.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    fn boolean_delays_selective_must_not_probes() {
        let positive_rows = (0..100)
            .map(|doc| (doc, if doc == 99 { 50.0 } else { 0.5 }))
            .collect::<Vec<_>>();
        let prohibited_rows = (0..100).map(|doc| (doc, 0.0)).collect::<Vec<_>>();
        let eager_probes = positive_rows.len();
        let (prohibited, prohibited_approximations, prohibited_confirmations) =
            two_phase(&prohibited_rows, Vec::new(), Some(1.0));
        let metrics = BooleanMetrics::default();
        let results = {
            let positive = || {
                Box::new(
                    MaterializedScorer::try_new(rows(&positive_rows))
                        .unwrap()
                        .with_block_size(1),
                ) as BoxScorer<'_>
            };
            let mut scorer = BooleanScorer::try_new_with_metrics(
                Vec::new(),
                vec![positive(), positive()],
                vec![prohibited],
                Some(&metrics),
            )
            .unwrap();
            let competitive_score = Arc::new(CompetitiveScore::default());
            competitive_score.raise(50.0);
            TopKCollector::with_competitive_score(1, competitive_score)
                .collect(&mut scorer)
                .unwrap()
        };
        let probes = prohibited_approximations.load(AtomicOrdering::Relaxed);

        assert_eq!(results, rows(&[(99, 100.0)]));
        assert_eq!(probes, 1);
        assert_eq!(prohibited_confirmations.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(metrics.reports.load(AtomicOrdering::Relaxed), 2);
        assert_eq!(metrics.positive_survivors.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(metrics.must_not_probes.load(AtomicOrdering::Relaxed), 1);
        assert!(
            probes * 5 <= eager_probes * 4,
            "delayed MUST_NOT should reduce probes by at least 20%: {probes}/{eager_probes}"
        );
    }

    #[test]
    fn boolean_caches_must_not_decision_for_current_document() {
        let (prohibited, prohibited_approximations, prohibited_confirmations) =
            two_phase(&[(0, 0.0)], Vec::new(), Some(10.0));
        let mut scorer = BooleanScorer::try_new(
            Vec::new(),
            vec![materialized(&[(0, 1.0)])],
            vec![prohibited],
        )
        .unwrap();

        assert_eq!(scorer.next().unwrap(), Some(0));
        assert!(scorer.matches().unwrap());
        assert_eq!(scorer.score().unwrap(), 1.0);
        assert!(scorer.matches().unwrap());
        assert_eq!(scorer.score().unwrap(), 1.0);
        assert_eq!(prohibited_approximations.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(prohibited_confirmations.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    fn boolean_with_unbounded_floor_rejects_prohibited_candidates_before_scoring() {
        let positive_rows = (0..100).map(|doc| (doc, 1.0)).collect::<Vec<_>>();
        let (positive, positive_work) = instrumented(materialized(&positive_rows));
        let prohibited_rows = (0..100).map(|doc| (doc, 0.0)).collect::<Vec<_>>();
        let (prohibited, prohibited_approximations, prohibited_confirmations) =
            two_phase(&prohibited_rows, (0..100).collect(), Some(1.0));
        let mut scorer =
            BooleanScorer::try_new(Vec::new(), vec![positive], vec![prohibited]).unwrap();

        assert!(
            TopKCollector::new(1)
                .collect(&mut scorer)
                .unwrap()
                .is_empty()
        );
        assert_eq!(positive_work.scores.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(prohibited_approximations.load(AtomicOrdering::Relaxed), 100);
        assert_eq!(prohibited_confirmations.load(AtomicOrdering::Relaxed), 100);
    }

    #[test]
    fn boolean_switches_to_delayed_probes_after_raising_the_score_floor() {
        let positive_rows = (0..100)
            .map(|doc| {
                let score = match doc {
                    0 => 50.0,
                    99 => 50.5,
                    _ => 0.5,
                };
                (doc, score)
            })
            .collect::<Vec<_>>();
        let positive = || materialized(&positive_rows);
        let prohibited_rows = (0..100).map(|doc| (doc, 0.0)).collect::<Vec<_>>();
        let (prohibited, prohibited_approximations, prohibited_confirmations) =
            two_phase(&prohibited_rows, (1..100).collect(), Some(1.0));
        let metrics = BooleanMetrics::default();
        let results = {
            let mut scorer = BooleanScorer::try_new_with_metrics(
                Vec::new(),
                vec![positive(), positive()],
                vec![prohibited],
                Some(&metrics),
            )
            .unwrap();
            TopKCollector::new(1).collect(&mut scorer).unwrap()
        };

        assert_eq!(results, rows(&[(0, 100.0)]));
        assert_eq!(prohibited_approximations.load(AtomicOrdering::Relaxed), 2);
        assert_eq!(prohibited_confirmations.load(AtomicOrdering::Relaxed), 2);
        assert_eq!(metrics.positive_survivors.load(AtomicOrdering::Relaxed), 2);
        assert_eq!(metrics.must_not_probes.load(AtomicOrdering::Relaxed), 2);
    }

    #[test]
    fn boolean_skips_must_not_for_positive_confirmation_rejects() {
        let (positive, _, positive_confirmations) =
            two_phase(&[(0, 1.0), (1, 2.0)], vec![1], Some(1.0));
        let (prohibited, prohibited_approximations, _) =
            two_phase(&[(0, 0.0), (1, 0.0)], Vec::new(), Some(10.0));
        let mut scorer =
            BooleanScorer::try_new(Vec::new(), vec![positive], vec![prohibited]).unwrap();

        assert_eq!(
            TopKCollector::new(2).collect(&mut scorer).unwrap(),
            rows(&[(1, 2.0)])
        );
        assert_eq!(positive_confirmations.load(AtomicOrdering::Relaxed), 2);
        assert_eq!(prohibited_approximations.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    fn boolean_without_must_not_keeps_confirmed_iteration_and_skips_metrics() {
        let metrics = BooleanMetrics::default();
        let (positive, positive_approximations, positive_confirmations) =
            two_phase(&[(0, 1.0), (1, 2.0)], vec![1], Some(1.0));
        let (positive, positive_work) = instrumented(positive);
        {
            let mut scorer = BooleanScorer::try_new_with_metrics(
                Vec::new(),
                vec![positive],
                Vec::new(),
                Some(&metrics),
            )
            .unwrap();

            assert_eq!(scorer.next().unwrap(), Some(1));
            assert_eq!(positive_work.scores.load(AtomicOrdering::Relaxed), 0);
            assert!(scorer.matches().unwrap());
            assert_eq!(positive_work.scores.load(AtomicOrdering::Relaxed), 0);
            assert_eq!(scorer.score().unwrap(), 2.0);
            assert_eq!(positive_work.scores.load(AtomicOrdering::Relaxed), 1);
            assert_eq!(scorer.next().unwrap(), None);
        }

        assert_eq!(positive_approximations.load(AtomicOrdering::Relaxed), 2);
        assert_eq!(positive_confirmations.load(AtomicOrdering::Relaxed), 2);
        assert_eq!(metrics.reports.load(AtomicOrdering::Relaxed), 0);
    }

    #[test]
    fn boolean_without_must_not_positions_signed_optional_after_confirmation() {
        let (required, _, required_confirmations) =
            two_phase(&[(0, 1.0), (1, 2.0)], vec![1], Some(1.0));
        let signed_optional = Box::new(
            BoostScorer::try_new(
                materialized(&[(0, 4.0), (1, 4.0)]),
                materialized(&[(0, 1.0), (1, 1.0)]),
                0.5,
            )
            .unwrap(),
        );
        let (signed_optional, optional_work) = instrumented(signed_optional);
        let mut scorer =
            BooleanScorer::try_new(vec![signed_optional], vec![required], Vec::new()).unwrap();

        assert_eq!(
            TopKCollector::new(1).collect(&mut scorer).unwrap(),
            rows(&[(1, 5.5)])
        );
        assert_eq!(required_confirmations.load(AtomicOrdering::Relaxed), 2);
        assert_eq!(optional_work.advances.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    fn outer_boolean_without_must_not_preserves_inner_delayed_probes() {
        let metrics = BooleanMetrics::default();
        let (prohibited, prohibited_approximations, prohibited_confirmations) =
            two_phase(&[(0, 0.0), (1, 0.0)], Vec::new(), Some(1.0));
        let inner = BooleanScorer::try_new_with_metrics(
            Vec::new(),
            vec![materialized(&[(0, 1.0), (1, 50.0)])],
            vec![prohibited],
            Some(&metrics),
        )
        .unwrap();
        let results = {
            let mut outer = BooleanScorer::try_new_with_metrics(
                Vec::new(),
                vec![Box::new(inner)],
                Vec::new(),
                Some(&metrics),
            )
            .unwrap();
            let competitive_score = Arc::new(CompetitiveScore::default());
            competitive_score.raise(50.0);
            TopKCollector::with_competitive_score(1, competitive_score)
                .collect(&mut outer)
                .unwrap()
        };

        assert_eq!(results, rows(&[(1, 50.0)]));
        assert_eq!(prohibited_approximations.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(prohibited_confirmations.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(metrics.reports.load(AtomicOrdering::Relaxed), 2);
        assert_eq!(metrics.positive_survivors.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(metrics.must_not_probes.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    fn reqopt_delays_sparse_optional_probes() {
        let values = (0..100).map(|doc| (doc, 1.0)).collect::<Vec<_>>();
        let build = || {
            let required =
                Box::new(RequiredConjunctionScorer::try_new(vec![materialized(&values)]).unwrap());
            let optional = Box::new(
                DisjunctionScorer::try_new(
                    vec![materialized(&[(99, 100.0)])],
                    DisjunctionScore::Sum,
                )
                .unwrap(),
            );
            let (required, required_work) = instrumented(required);
            let (optional, optional_work) = instrumented(optional);
            (required, required_work, optional, optional_work)
        };

        let (required, _, optional, eager_optional_work) = build();
        let mut eager = BooleanScorer {
            driver: required,
            optional: Some(optional),
            prohibited: None,
            score_mode: CompoundScoreMode::Scoring,
            current: None,
            optional_matches: false,
            min_competitive_score: f32::NEG_INFINITY,
            positive_checked_doc: None,
            positive_score: None,
            positive_survivor_doc: None,
            prohibited_checked_doc: None,
            prohibited_matches: false,
            metrics: None,
            work: BooleanWork::default(),
        };
        let eager_results = TopKCollector::new(1).collect(&mut eager).unwrap();

        let (required, required_work, optional, optional_work) = build();
        let mut scorer = ReqOptScorer::new(required, optional);
        let results = TopKCollector::new(1).collect(&mut scorer).unwrap();

        assert_eq!(eager_results, rows(&[(99, 101.0)]));
        assert_eq!(results, rows(&[(99, 101.0)]));
        let required_advances = required_work.advances.load(AtomicOrdering::Relaxed);
        let eager_optional_probes = eager_optional_work.advances.load(AtomicOrdering::Relaxed);
        let optional_probes = optional_work.advances.load(AtomicOrdering::Relaxed);
        assert_eq!(required_advances, 100);
        assert_eq!(eager_optional_probes, 100);
        assert_eq!(optional_probes, 1);
        assert_eq!(
            required_work.shallow_advances.load(AtomicOrdering::Relaxed),
            2
        );
        assert_eq!(required_work.bounds.load(AtomicOrdering::Relaxed), 2);
        assert_eq!(
            required_work.confirmations.load(AtomicOrdering::Relaxed),
            100
        );
        assert_eq!(optional_work.confirmations.load(AtomicOrdering::Relaxed), 1);
        assert!(
            optional_probes * 5 <= eager_optional_probes * 4,
            "lazy required-plus-optional scoring should reduce optional probes by at least 20%: \
             {optional_probes}/{eager_optional_probes}"
        );
    }

    #[test]
    fn reqopt_temporarily_requires_optional_contribution() {
        let values = (0..100).map(|doc| (doc, 1.0)).collect::<Vec<_>>();
        let (required, required_work) = instrumented(materialized(&values));
        let (optional, optional_work) = instrumented(materialized(&[(50, 10.0)]));
        let mut scorer = ReqOptScorer::new(required, optional);
        let competitive_score = Arc::new(CompetitiveScore::default());
        competitive_score.raise(10.0);

        let results = TopKCollector::with_competitive_score(1, competitive_score)
            .collect(&mut scorer)
            .unwrap();

        assert_eq!(results, rows(&[(50, 11.0)]));
        assert!(required_work.advances.load(AtomicOrdering::Relaxed) < 10);
        assert_eq!(required_work.confirmations.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(optional_work.advances.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(optional_work.confirmations.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    fn boolean_reqopt_keeps_current_confirmation_stable_after_bounds() {
        let mut scorer = BooleanScorer::try_new(
            vec![materialized(&[(1, 10.0)])],
            vec![materialized(&[(0, 1.0), (1, 1.0)])],
            Vec::new(),
        )
        .unwrap();
        let competitive_score = Arc::new(CompetitiveScore::default());
        competitive_score.raise(5.0);

        let results = TopKCollector::with_competitive_score(1, competitive_score)
            .collect(&mut scorer)
            .unwrap();

        assert_eq!(results, rows(&[(1, 11.0)]));
    }

    #[test]
    fn boolean_gates_reqopt_and_signed_boost_uses_exact_fallback() {
        let supported = BooleanScorer::try_new(
            vec![materialized(&[(1, 4.0)])],
            vec![materialized(&[(1, 2.0)])],
            Vec::new(),
        )
        .unwrap();
        assert!(supported.optional.is_none());

        let signed_optional = Box::new(
            BoostScorer::try_new(
                materialized(&[(1, 4.0), (2, 1.0)]),
                materialized(&[(1, 1.0), (2, 4.0)]),
                0.5,
            )
            .unwrap(),
        );
        let mut fallback = BooleanScorer::try_new(
            vec![signed_optional],
            vec![materialized(&[(1, 2.0), (2, 2.0)])],
            Vec::new(),
        )
        .unwrap();
        assert!(fallback.optional.is_some());
        assert_eq!(
            TopKCollector::new(2).collect(&mut fallback).unwrap(),
            rows(&[(1, 5.5), (2, 1.0)])
        );
    }

    #[test]
    fn reqopt_uses_exact_iteration_for_unbounded_scorers() {
        let required = Box::new(UnboundedScorer {
            inner: MaterializedScorer::try_new(rows(&[(1, 1.0), (2, 1.0)])).unwrap(),
        });
        let optional = materialized(&[(2, 10.0)]);
        let mut scorer = ReqOptScorer::new(required, optional);
        let competitive_score = Arc::new(CompetitiveScore::default());
        competitive_score.raise(5.0);

        let results = TopKCollector::with_competitive_score(1, competitive_score)
            .collect(&mut scorer)
            .unwrap();

        assert_eq!(results, rows(&[(2, 11.0)]));
    }

    fn pure_should_canary_children() -> (Vec<BoxScorer<'static>>, Vec<Arc<ScorerWork>>) {
        let mut children = Vec::new();
        let mut work = Vec::new();
        let mut push = |child: BoxScorer<'static>| {
            let (child, child_work) = instrumented(child);
            children.push(child);
            work.push(child_work);
        };

        push(materialized(&[(0, 2.0)]));
        let dense = (1..=1024).map(|doc| (doc, 0.125)).collect::<Vec<_>>();
        for _ in 0..8 {
            push(materialized(&dense));
        }
        let sparse = (127..=1023)
            .step_by(128)
            .map(|doc| (doc, 1.5))
            .collect::<Vec<_>>();
        push(materialized(&sparse));
        (children, work)
    }

    fn scorer_advances(work: &[Arc<ScorerWork>]) -> usize {
        work.iter()
            .map(|work| work.advances.load(AtomicOrdering::Relaxed))
            .sum()
    }

    fn exhaustive_should_top_k(children: &[Vec<(u64, f32)>], limit: usize) -> Vec<ScoredRow> {
        let mut scores = HashMap::<u64, f32>::new();
        for child in children {
            for (doc, score) in child {
                *scores.entry(*doc).or_default() += *score;
            }
        }
        let mut rows = scores
            .into_iter()
            .map(|(doc, score)| ScoredRow::new(doc, score).unwrap())
            .collect::<Vec<_>>();
        rows.sort_unstable_by(compare_scored_rows);
        rows.truncate(limit);
        rows
    }

    #[test]
    fn pure_should_maxscore_reduces_posting_comparisons() {
        let (children, eager_work) = pure_should_canary_children();
        let mut eager = DisjunctionScorer::try_new(children, DisjunctionScore::Sum).unwrap();
        let eager_results = TopKCollector::new(1).collect(&mut eager).unwrap();
        let eager_comparisons = scorer_advances(&eager_work);

        let metrics = ShouldMetrics::default();
        let (children, optimized_work) = pure_should_canary_children();
        let optimized_results = {
            let mut optimized = should_maxscore(children, Some(&metrics));
            TopKCollector::new(1).collect(&mut optimized).unwrap()
        };
        let optimized_comparisons = scorer_advances(&optimized_work);

        assert_eq!(eager_results, rows(&[(127, 2.5)]));
        assert_eq!(optimized_results, eager_results);
        assert!(eager_comparisons > 0);
        assert!(
            optimized_comparisons * 5 <= eager_comparisons * 4,
            "pure-SHOULD MAXSCORE should reduce posting candidate probes by at least 20%: \
             optimized={optimized_comparisons} eager={eager_comparisons}"
        );
        assert_eq!(metrics.reports.load(AtomicOrdering::Relaxed), 4);
        assert!(metrics.skipped_windows.load(AtomicOrdering::Relaxed) > 0);
        assert!(metrics.bound_recomputations.load(AtomicOrdering::Relaxed) > 0);
        assert!(metrics.essential_evaluations.load(AtomicOrdering::Relaxed) > 0);
        assert!(
            metrics
                .non_essential_evaluations
                .load(AtomicOrdering::Relaxed)
                > 0
        );
    }

    #[test]
    fn pure_should_maxscore_matches_randomized_exhaustive_top_k() {
        for seed in 0..8 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let children = (0..6)
                .map(|_| {
                    (0..256)
                        .filter_map(|doc| {
                            if rng.random_bool(0.35) {
                                let score = rng.random_range(1..=16) as f32 * 0.25;
                                Some((doc, score))
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            for limit in [1, 7, 31, 512] {
                let expected = exhaustive_should_top_k(&children, limit);

                let mut optimized = should_maxscore(
                    children.iter().map(|values| materialized(values)).collect(),
                    None,
                );
                let actual = TopKCollector::new(limit).collect(&mut optimized).unwrap();
                assert_eq!(actual, expected, "seed={seed} limit={limit}");
            }
        }
    }

    #[test]
    fn pure_should_maxscore_confirms_two_phase_children_before_scoring() {
        let (phrase, _, confirmations) = two_phase(&[(1, 100.0)], Vec::new(), Some(10.0));
        let metrics = ShouldMetrics::default();
        let competitive_score = Arc::new(CompetitiveScore::default());
        competitive_score.raise(10.0);
        let results = {
            let mut scorer = should_maxscore(
                vec![
                    materialized(&[(0, 10.0)]),
                    phrase,
                    materialized(&[(1, 6.0)]),
                    materialized(&[(1, 5.0)]),
                ],
                Some(&metrics),
            );
            TopKCollector::with_competitive_score(1, competitive_score)
                .collect(&mut scorer)
                .unwrap()
        };

        assert_eq!(results, rows(&[(1, 11.0)]));
        assert_eq!(confirmations.load(AtomicOrdering::Relaxed), 1);
        assert!(
            metrics
                .non_essential_evaluations
                .load(AtomicOrdering::Relaxed)
                > 0
        );
    }

    #[test]
    fn pure_should_maxscore_preserves_query_score_order_and_terminal_doc() {
        let mut scorer = should_maxscore(
            vec![
                materialized(&[(u64::MAX, 16_777_216.0)]),
                materialized(&[(u64::MAX, 1.0)]),
                materialized(&[(u64::MAX, 1.0)]),
                materialized(&[]),
            ],
            None,
        );

        assert_eq!(
            TopKCollector::new(1).collect(&mut scorer).unwrap(),
            rows(&[(u64::MAX, 16_777_216.0)])
        );
    }

    #[test]
    fn pure_should_maxscore_keeps_equal_floor_across_bound_ordering() {
        let scores = [
            f32::from_bits(0x4783_798b),
            f32::from_bits(0x4dd3_8b75),
            f32::from_bits(0x48e7_7236),
            f32::from_bits(0x418e_5b26),
            f32::from_bits(0x4241_b1eb),
        ];
        let exact_score = scores
            .into_iter()
            .fold(0.0_f32, |total, score| total + score);
        assert_eq!(exact_score.to_bits(), 0x4dd3_cd8d);

        let mut scorer = should_maxscore(
            scores
                .into_iter()
                .map(|score| materialized(&[(7, score)]))
                .collect(),
            None,
        );
        let competitive_score = Arc::new(CompetitiveScore::default());
        competitive_score.raise(exact_score);

        assert_eq!(
            TopKCollector::with_competitive_score(1, competitive_score)
                .collect(&mut scorer)
                .unwrap(),
            rows(&[(7, exact_score)])
        );
    }

    #[test]
    fn pure_should_maxscore_supports_nested_non_negative_children() {
        let nested_dismax = Box::new(
            DisjunctionScorer::try_new(
                vec![
                    materialized(&[(0, 1.0), (1, 5.0)]),
                    materialized(&[(0, 3.0), (2, 4.0)]),
                ],
                DisjunctionScore::Max,
            )
            .unwrap(),
        );
        let nested_boolean = Box::new(
            BooleanScorer::try_new(
                Vec::new(),
                vec![materialized(&[(0, 2.0), (1, 2.0), (2, 2.0)])],
                vec![materialized(&[(1, 0.0)])],
            )
            .unwrap(),
        );
        let metrics = ShouldMetrics::default();
        let results = {
            let mut scorer = BooleanScorer::try_new_with_metrics(
                vec![
                    nested_dismax,
                    nested_boolean,
                    materialized(&[(0, 0.5), (1, 0.5), (2, 0.5)]),
                ],
                Vec::new(),
                Vec::new(),
                Some(&metrics),
            )
            .unwrap();
            TopKCollector::new(1).collect(&mut scorer).unwrap()
        };

        assert_eq!(results, rows(&[(2, 6.5)]));
        assert_eq!(metrics.reports.load(AtomicOrdering::Relaxed), 4);
    }

    #[test]
    fn pure_should_maxscore_applies_must_not_before_raising_the_floor() {
        let metrics = ShouldMetrics::default();
        let results = {
            let mut scorer = BooleanScorer::try_new_with_metrics(
                vec![
                    materialized(&[(0, 10.0), (1, 5.0)]),
                    materialized(&[(0, 1.0), (1, 1.0)]),
                    materialized(&[(2, 8.0)]),
                ],
                Vec::new(),
                vec![materialized(&[(0, 1.0)])],
                Some(&metrics),
            )
            .unwrap();
            TopKCollector::new(1).collect(&mut scorer).unwrap()
        };

        assert_eq!(results, rows(&[(2, 8.0)]));
        assert_eq!(metrics.reports.load(AtomicOrdering::Relaxed), 4);
    }

    #[test]
    fn pure_should_uses_exact_fallback_for_unsupported_shapes() {
        let signed_metrics = ShouldMetrics::default();
        let signed_results = {
            let signed = Box::new(
                BoostScorer::try_new(
                    materialized(&[(0, 5.0), (1, 1.0)]),
                    materialized(&[(0, 2.0), (1, 4.0)]),
                    1.0,
                )
                .unwrap(),
            );
            let mut scorer = BooleanScorer::try_new_with_metrics(
                vec![
                    signed,
                    materialized(&[(0, 1.0), (1, 1.0)]),
                    materialized(&[(1, 5.0)]),
                ],
                Vec::new(),
                Vec::new(),
                Some(&signed_metrics),
            )
            .unwrap();
            TopKCollector::new(2).collect(&mut scorer).unwrap()
        };
        assert_eq!(signed_results, rows(&[(0, 4.0), (1, 3.0)]));
        assert_eq!(signed_metrics.reports.load(AtomicOrdering::Relaxed), 0);

        let unbounded_metrics = ShouldMetrics::default();
        let unbounded_results = {
            let unbounded = Box::new(UnboundedScorer {
                inner: MaterializedScorer::try_new(rows(&[(0, 1.0), (2, 3.0)])).unwrap(),
            });
            let mut scorer = BooleanScorer::try_new_with_metrics(
                vec![
                    unbounded,
                    materialized(&[(0, 2.0), (1, 2.0)]),
                    materialized(&[(1, 4.0)]),
                ],
                Vec::new(),
                Vec::new(),
                Some(&unbounded_metrics),
            )
            .unwrap();
            TopKCollector::new(3).collect(&mut scorer).unwrap()
        };
        assert_eq!(unbounded_results, rows(&[(1, 6.0), (0, 3.0), (2, 3.0)]));
        assert_eq!(unbounded_metrics.reports.load(AtomicOrdering::Relaxed), 0);

        let low_count_metrics = ShouldMetrics::default();
        {
            let mut scorer = BooleanScorer::try_new_with_metrics(
                vec![materialized(&[(0, 1.0)]), materialized(&[(1, 2.0)])],
                Vec::new(),
                Vec::new(),
                Some(&low_count_metrics),
            )
            .unwrap();
            assert_eq!(
                TopKCollector::new(2).collect(&mut scorer).unwrap(),
                rows(&[(1, 2.0), (0, 1.0)])
            );
        }
        assert_eq!(low_count_metrics.reports.load(AtomicOrdering::Relaxed), 0);

        let overflow_metrics = ShouldMetrics::default();
        let large_score = f32::MAX / 2.0;
        {
            let mut scorer = BooleanScorer::try_new_with_metrics(
                vec![
                    materialized(&[(0, large_score)]),
                    materialized(&[(1, large_score)]),
                    materialized(&[(2, large_score)]),
                ],
                Vec::new(),
                Vec::new(),
                Some(&overflow_metrics),
            )
            .unwrap();
            assert_eq!(
                TopKCollector::new(3).collect(&mut scorer).unwrap(),
                rows(&[(0, large_score), (1, large_score), (2, large_score)])
            );
        }
        assert_eq!(overflow_metrics.reports.load(AtomicOrdering::Relaxed), 0);
    }
}
