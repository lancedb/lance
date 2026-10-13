// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lexical evidence and source excerpts for already selected FTS results.

use std::collections::{BTreeSet, VecDeque};
use std::ops::Range;
use std::sync::Arc;

use arrow::buffer::OffsetBuffer;
use arrow_array::{ArrayRef, ListArray, StringArray, StructArray, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field, Fields};
use icu_segmenter::{GraphemeClusterSegmenter, SentenceSegmenter};
use lance_core::{Error, Result};
use lance_tokenizer::Token;
use serde::{Deserialize, Serialize};

use super::InvertedIndex;
use super::query::{FtsQuery, Operator, Tokens};
use super::query::{FtsSearchParams, uses_fuzzy_expansion};
use super::tokenizer::document_tokenizer::LanceTokenizer;

/// Result column containing plain source excerpts and UTF-8 byte ranges.
pub const HIGHLIGHTS_COL: &str = "_highlights";

/// Options for highlighting final full-text search results.
///
/// No markup is inserted into source text. Match offsets are half-open UTF-8
/// byte ranges, not character or JavaScript UTF-16 indices.
///
/// ```
/// use lance_index::scalar::inverted::highlight::HighlightOptions;
/// let options = HighlightOptions::new(["title", "body"])
///     .with_max_fragments(2)
///     .with_fragment_size(160);
/// options.validate().unwrap();
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HighlightOptions {
    /// Unique positive-query source fields to highlight, in output order.
    pub fields: Vec<String>,
    /// Maximum excerpts per field across all its values. Must be positive.
    pub max_fragments: usize,
    /// Target excerpt length in grapheme clusters. Complete phrases may exceed it.
    pub fragment_size: usize,
    /// Maximum logical source bytes analyzed per returned row. Exceeding it fails the query.
    pub max_analyzed_bytes: usize,
}

impl Default for HighlightOptions {
    fn default() -> Self {
        Self {
            fields: Vec::new(),
            max_fragments: 2,
            fragment_size: 160,
            max_analyzed_bytes: 1024 * 1024,
        }
    }
}

impl HighlightOptions {
    /// Select positive lexical query fields. At least one field is required.
    pub fn new(fields: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            fields: fields.into_iter().map(Into::into).collect(),
            ..Self::default()
        }
    }

    /// Set the positive maximum number of excerpts per field.
    pub fn with_max_fragments(mut self, value: usize) -> Self {
        self.max_fragments = value;
        self
    }
    /// Set the positive target excerpt length in grapheme clusters.
    pub fn with_fragment_size(mut self, value: usize) -> Self {
        self.fragment_size = value;
        self
    }
    /// Set the positive logical source-byte limit per returned row.
    pub fn with_max_analyzed_bytes(mut self, value: usize) -> Self {
        self.max_analyzed_bytes = value;
        self
    }

    /// Validate option values, independently of the query and source schema.
    pub fn validate(&self) -> Result<()> {
        if self.fields.is_empty() || self.fields.iter().any(String::is_empty) {
            return Err(Error::invalid_input(
                "highlight fields must contain at least one non-empty field",
            ));
        }
        if self.fields.iter().collect::<BTreeSet<_>>().len() != self.fields.len() {
            return Err(Error::invalid_input("highlight fields must be unique"));
        }
        for (name, value) in [
            ("max_fragments", self.max_fragments),
            ("fragment_size", self.fragment_size),
            ("max_analyzed_bytes", self.max_analyzed_bytes),
        ] {
            if value == 0 {
                return Err(Error::invalid_input(format!(
                    "highlight {name} must be positive, got {value}"
                )));
            }
        }
        Ok(())
    }
}

/// A half-open UTF-8 byte range relative to a highlight excerpt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HighlightMatch {
    /// Inclusive start byte.
    pub start: u64,
    /// Exclusive end byte.
    pub end: u64,
}

/// An unchanged slice of an original source value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HighlightFragment {
    /// Resolved query field path.
    pub field: String,
    /// Nested list ordinals; empty for a scalar.
    pub value_index: Vec<u32>,
    /// Inclusive byte offset in the original source value.
    pub start: u64,
    /// Plain source text; escape it before adding HTML markup.
    pub text: String,
    /// Sorted, non-overlapping display ranges relative to `text`.
    pub matches: Vec<HighlightMatch>,
}

/// One source value and its coordinates within a result row.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct HighlightValue {
    pub value_index: Vec<u32>,
    pub text: String,
}

/// Source values of one logical field, selected at the search snapshot.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct HighlightDocument {
    pub field: String,
    pub values: Vec<HighlightValue>,
    /// Whether this row used the posting-backed vocabulary rather than flat matching.
    pub indexed: bool,
}

/// Resolved matching terms for a leaf. Alternatives share a query position.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct HighlightTerms {
    pub tokens: Arc<Tokens>,
    pub has_all_query_positions: bool,
}

impl HighlightTerms {
    /// Resolve the same deterministic vocabulary as posting-backed search.
    pub fn for_indices(
        indices: &[Arc<InvertedIndex>],
        tokens: Tokens,
        params: &FtsSearchParams,
    ) -> Result<Self> {
        if uses_fuzzy_expansion(params.fuzziness) {
            let expanded = Arc::new(super::final_query_tokens(indices, &tokens, params)?);
            let has_all_query_positions = super::has_all_query_positions(&tokens, &expanded);
            Ok(Self {
                tokens: expanded,
                has_all_query_positions,
            })
        } else {
            Ok(Self {
                tokens: Arc::new(tokens),
                has_all_query_positions: true,
            })
        }
    }
}

/// Prepared leaf state in depth-first query order; Boost's negative arm is omitted.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct HighlightLeaf {
    pub fields: Vec<String>,
    pub tokenizer: Arc<dyn LanceTokenizer>,
    pub flat: HighlightTerms,
    pub indexed: HighlightTerms,
    pub operator: Operator,
    pub phrase_slop: Option<u32>,
}

#[derive(Debug, Clone)]
struct Evidence {
    field: String,
    value: usize,
    range: Range<usize>,
    coverage: (usize, u32),
}

/// A query-local highlighter; it never changes retrieval scores or ordering.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct PreparedHighlightQuery {
    query: FtsQuery,
    leaves: Vec<HighlightLeaf>,
    options: HighlightOptions,
}

impl PreparedHighlightQuery {
    pub fn try_new(
        query: FtsQuery,
        leaves: Vec<HighlightLeaf>,
        options: HighlightOptions,
    ) -> Result<Self> {
        options.validate()?;
        let mut positives = BTreeSet::new();
        positive_fields(&query, &mut positives);
        for field in &options.fields {
            if !positives.contains(field) {
                return Err(Error::invalid_input(format!(
                    "highlight field '{field}' is not a positive FTS query field"
                )));
            }
        }
        Ok(Self {
            query,
            leaves,
            options,
        })
    }

    pub fn highlight(&self, documents: &[HighlightDocument]) -> Result<Vec<HighlightFragment>> {
        let bytes = documents.iter().try_fold(0usize, |total, doc| {
            let logical_bytes = doc.values.iter().try_fold(0usize, |length, value| {
                length
                    .checked_add(usize::from(length > 0))
                    .and_then(|length| length.checked_add(value.text.len()))
                    .ok_or_else(|| Error::invalid_input("highlight source length overflow"))
            })?;
            total
                .checked_add(logical_bytes)
                .ok_or_else(|| Error::invalid_input("highlight source length overflow"))
        })?;
        if bytes > self.options.max_analyzed_bytes {
            return Err(Error::invalid_input(format!(
                "highlight source has {bytes} bytes, exceeding max_analyzed_bytes={}",
                self.options.max_analyzed_bytes
            )));
        }
        let mut analyzed = Vec::new();
        for document in documents {
            for leaf in self
                .leaves
                .iter()
                .filter(|leaf| leaf.fields.contains(&document.field))
            {
                if !analyzed.iter().any(|previous: &AnalyzedDocument| {
                    previous.field == document.field
                        && Arc::ptr_eq(&previous.tokenizer, &leaf.tokenizer)
                }) {
                    analyzed.push(analyze_document(
                        document,
                        &self.leaves,
                        leaf.tokenizer.clone(),
                    )?);
                }
            }
        }
        let results = self
            .leaves
            .iter()
            .enumerate()
            .map(|(index, leaf)| match_leaf(leaf, index, &analyzed))
            .collect::<Result<Vec<_>>>()?;
        let mut cursor = 0;
        let (matched, evidence) = evaluate(&self.query, &results, &mut cursor)?;
        if cursor != results.len() {
            return Err(Error::internal(
                "highlight prepared leaf count does not match query",
            ));
        }
        if !matched {
            return Err(Error::internal(
                "highlight source replay did not match the selected FTS result",
            ));
        }
        let mut fragments = Vec::new();
        for field in &self.options.fields {
            let doc = documents
                .iter()
                .find(|doc| &doc.field == field)
                .ok_or_else(|| {
                    Error::internal(format!("highlight source field '{field}' is missing"))
                })?;
            let mut candidates = Vec::new();
            for (value, source) in doc.values.iter().enumerate() {
                let matches = evidence
                    .iter()
                    .filter(|hit| &hit.field == field && hit.value == value)
                    .collect::<Vec<_>>();
                candidates.extend(passages(
                    source,
                    value,
                    &matches,
                    self.options.fragment_size,
                ));
            }
            let mut selected = Vec::new();
            let mut covered = BTreeSet::new();
            while !candidates.is_empty() && selected.len() < self.options.max_fragments {
                candidates.sort_by_key(|candidate| {
                    (
                        std::cmp::Reverse(candidate.coverage.difference(&covered).count()),
                        candidate.range.len(),
                        candidate.value,
                        candidate.range.start,
                    )
                });
                let candidate = candidates.remove(0);
                covered.extend(candidate.coverage.iter().copied());
                candidates.retain(|other| {
                    other.value != candidate.value
                        || other.range.end <= candidate.range.start
                        || other.range.start >= candidate.range.end
                });
                selected.push(candidate);
            }
            selected.sort_by_key(|candidate| (candidate.value, candidate.range.start));
            for passage in selected {
                let source = &doc.values[passage.value];
                fragments.push(HighlightFragment {
                    field: field.clone(),
                    value_index: source.value_index.clone(),
                    start: passage.range.start as u64,
                    text: source.text[passage.range.clone()].to_owned(),
                    matches: passage
                        .matches
                        .into_iter()
                        .map(|range| HighlightMatch {
                            start: (range.start - passage.range.start) as u64,
                            end: (range.end - passage.range.start) as u64,
                        })
                        .collect(),
                });
            }
        }
        Ok(fragments)
    }
}

fn positive_fields(query: &FtsQuery, fields: &mut BTreeSet<String>) {
    match query {
        FtsQuery::Match(query) => fields.extend(query.column.iter().cloned()),
        FtsQuery::Phrase(query) => fields.extend(query.column.iter().cloned()),
        FtsQuery::Boost(query) => positive_fields(&query.positive, fields),
        FtsQuery::MultiMatch(query) => {
            for query in &query.match_queries {
                fields.extend(query.column.iter().cloned());
            }
        }
        FtsQuery::CombinedFields(query) => fields.extend(query.column_names().map(str::to_owned)),
        FtsQuery::Boolean(query) => {
            for query in query.must.iter().chain(&query.should) {
                positive_fields(query, fields);
            }
        }
    }
}

type MatchResult = (bool, Vec<Evidence>);

fn evaluate(query: &FtsQuery, results: &[MatchResult], cursor: &mut usize) -> Result<MatchResult> {
    match query {
        FtsQuery::Match(_) | FtsQuery::Phrase(_) | FtsQuery::CombinedFields(_) => {
            let result = results
                .get(*cursor)
                .ok_or_else(|| Error::internal("highlight prepared leaf is missing"))?
                .clone();
            *cursor += 1;
            Ok(result)
        }
        FtsQuery::Boost(query) => evaluate(&query.positive, results, cursor),
        FtsQuery::MultiMatch(query) => {
            let mut matched = false;
            let mut evidence = Vec::new();
            for query in &query.match_queries {
                let (hit, spans) = evaluate(&FtsQuery::Match(query.clone()), results, cursor)?;
                matched |= hit;
                if hit {
                    evidence.extend(spans);
                }
            }
            Ok((matched, evidence))
        }
        FtsQuery::Boolean(query) => {
            let mut must_match = true;
            let mut should_match = false;
            let mut excluded = false;
            let mut evidence = Vec::new();
            for child in &query.must {
                let (matched, spans) = evaluate(child, results, cursor)?;
                must_match &= matched;
                if matched {
                    evidence.extend(spans);
                }
            }
            for child in &query.should {
                let (matched, spans) = evaluate(child, results, cursor)?;
                should_match |= matched;
                if matched {
                    evidence.extend(spans);
                }
            }
            for child in &query.must_not {
                excluded |= evaluate(child, results, cursor)?.0;
            }
            let matched = must_match && !excluded && (!query.must.is_empty() || should_match);
            Ok((matched, if matched { evidence } else { Vec::new() }))
        }
    }
}

#[derive(Debug, Clone)]
struct Occurrence {
    position: u32,
    range: Range<usize>,
}

struct AnalyzedDocument {
    tokenizer: Arc<dyn LanceTokenizer>,
    field: String,
    indexed: bool,
    offsets: Vec<Range<usize>>,
    tokens: Vec<Token>,
}

fn analyze_document(
    document: &HighlightDocument,
    leaves: &[HighlightLeaf],
    tokenizer: Arc<dyn LanceTokenizer>,
) -> Result<AnalyzedDocument> {
    let relevant = leaves
        .iter()
        .filter(|leaf| {
            leaf.fields.contains(&document.field) && Arc::ptr_eq(&leaf.tokenizer, &tokenizer)
        })
        .collect::<Vec<_>>();
    let vocabulary = relevant
        .iter()
        .flat_map(|leaf| {
            let terms = if document.indexed {
                &leaf.indexed
            } else {
                &leaf.flat
            };
            (0..terms.tokens.len()).map(|index| terms.tokens.get_token(index))
        })
        .collect::<BTreeSet<_>>();
    let mut logical = String::new();
    let mut offsets = Vec::with_capacity(document.values.len());
    for value in &document.values {
        if !logical.is_empty() {
            logical.push(' ');
        }
        let start = logical.len();
        logical.push_str(&value.text);
        offsets.push(start..logical.len());
    }
    if document.values.is_empty() {
        return Ok(AnalyzedDocument {
            tokenizer,
            field: document.field.clone(),
            indexed: document.indexed,
            offsets,
            tokens: Vec::new(),
        });
    }
    let mut analyzer = tokenizer.box_clone();
    let mut stream = analyzer.token_stream_for_doc(&logical);
    let mut tokens = Vec::new();
    while let Some(token) = stream.try_next().map_err(Error::invalid_input)? {
        if token.offset_from > token.offset_to
            || !logical.is_char_boundary(token.offset_from)
            || !logical.is_char_boundary(token.offset_to)
        {
            return Err(Error::invalid_input(format!(
                "highlight analyzer returned invalid offsets {}..{} for field '{}'",
                token.offset_from, token.offset_to, document.field
            )));
        }
        if vocabulary.contains(token.text.as_str()) {
            tokens.push(token.clone());
        }
    }
    Ok(AnalyzedDocument {
        tokenizer,
        field: document.field.clone(),
        indexed: document.indexed,
        offsets,
        tokens,
    })
}

fn match_leaf(
    leaf: &HighlightLeaf,
    leaf_id: usize,
    documents: &[AnalyzedDocument],
) -> Result<MatchResult> {
    let mut found = BTreeSet::new();
    let mut required = BTreeSet::new();
    let mut complete = true;
    let mut matched_phrase = false;
    let mut evidence = Vec::new();
    for field in &leaf.fields {
        let doc = documents
            .iter()
            .find(|doc| &doc.field == field && Arc::ptr_eq(&doc.tokenizer, &leaf.tokenizer))
            .ok_or_else(|| {
                Error::internal(format!("highlight source field '{field}' is missing"))
            })?;
        let terms = if doc.indexed {
            &leaf.indexed
        } else {
            &leaf.flat
        };
        complete &= terms.has_all_query_positions;
        let tokens = terms.tokens.as_ref();
        for i in 0..tokens.len() {
            required.insert(tokens.position(i));
        }
        let mut occurrences = vec![Vec::new(); tokens.len()];
        for token in &doc.tokens {
            for (i, hits) in occurrences.iter_mut().enumerate() {
                if tokens.get_token(i) == token.text {
                    found.insert(tokens.position(i));
                    hits.push(Occurrence {
                        position: u32::try_from(token.position).map_err(|_| {
                            Error::invalid_input("highlight token position exceeds u32")
                        })?,
                        range: token.offset_from..token.offset_to,
                    });
                }
            }
        }
        let ranges = if let Some(slop) = leaf.phrase_slop {
            phrase_witnesses(tokens, &occurrences, slop)
                .into_iter()
                .map(|range| (range, 0))
                .collect::<Vec<_>>()
        } else {
            occurrences
                .iter()
                .enumerate()
                .flat_map(|(i, hits)| {
                    hits.iter()
                        .map(move |hit| (hit.range.clone(), tokens.position(i)))
                })
                .collect()
        };
        matched_phrase |= leaf.phrase_slop.is_some() && !ranges.is_empty();
        for (range, position) in ranges {
            let first = doc
                .offsets
                .partition_point(|source| source.end <= range.start);
            for (value, source) in doc
                .offsets
                .iter()
                .enumerate()
                .skip(first)
                .take_while(|(_, source)| source.start < range.end)
            {
                let start = range.start.max(source.start);
                let end = range.end.min(source.end);
                if start < end {
                    evidence.push(Evidence {
                        field: field.clone(),
                        value,
                        range: start - source.start..end - source.start,
                        coverage: (leaf_id, position),
                    });
                }
            }
        }
    }
    let matched = if leaf.phrase_slop.is_some() {
        matched_phrase
    } else {
        !found.is_empty() && (leaf.operator == Operator::Or || (complete && found == required))
    };
    Ok((matched, if matched { evidence } else { Vec::new() }))
}

/// Shared ordered, per-gap phrase predicate. Uses wide arithmetic for positions near u32::MAX.
pub(super) fn phrase_position_follows(previous: u32, current: u32, delta: u32, slop: u32) -> bool {
    let least = u64::from(previous) + u64::from(delta);
    least <= u64::from(current) && u64::from(current) <= least + u64::from(slop)
}

fn phrase_witnesses(
    tokens: &Tokens,
    positions: &[Vec<Occurrence>],
    slop: u32,
) -> Vec<Range<usize>> {
    let Some(last) = positions.last() else {
        return Vec::new();
    };
    let mut suffixes = last
        .iter()
        .map(|hit| Some(hit.range.clone()))
        .collect::<Vec<_>>();
    // Each occurrence retains its earliest complete suffix. A monotonic queue
    // joins adjacent position lists in linear time, including repeated terms.
    for index in (0..positions.len().saturating_sub(1)).rev() {
        let Some(delta) = tokens
            .position(index + 1)
            .checked_sub(tokens.position(index))
        else {
            return Vec::new();
        };
        let next = &positions[index + 1];
        let mut end = 0;
        let mut eligible = VecDeque::<usize>::new();
        let mut current = Vec::with_capacity(positions[index].len());
        for hit in &positions[index] {
            let minimum = u64::from(hit.position) + u64::from(delta);
            let maximum = minimum + u64::from(slop);
            while end < next.len() && u64::from(next[end].position) <= maximum {
                if let Some(suffix) = &suffixes[end] {
                    while eligible.back().is_some_and(|previous| {
                        suffixes[*previous]
                            .as_ref()
                            .is_some_and(|previous| previous.end >= suffix.end)
                    }) {
                        eligible.pop_back();
                    }
                    eligible.push_back(end);
                }
                end += 1;
            }
            while eligible
                .front()
                .is_some_and(|index| u64::from(next[*index].position) < minimum)
            {
                eligible.pop_front();
            }
            current.push(
                eligible
                    .front()
                    .and_then(|index| suffixes[*index].as_ref())
                    .map(|suffix| hit.range.start.min(suffix.start)..hit.range.end.max(suffix.end)),
            );
        }
        suffixes = current;
    }
    let mut witnesses = suffixes.into_iter().flatten().collect::<Vec<_>>();
    witnesses.sort_by_key(|range| (range.start, range.end));
    witnesses.dedup();
    witnesses
}

#[derive(Debug)]
struct Passage {
    value: usize,
    range: Range<usize>,
    matches: Vec<Range<usize>>,
    coverage: BTreeSet<(usize, u32)>,
}

fn passages(
    source: &HighlightValue,
    value: usize,
    evidence: &[&Evidence],
    size: usize,
) -> Vec<Passage> {
    if evidence.is_empty() {
        return Vec::new();
    }
    let graphemes = GraphemeClusterSegmenter::new()
        .segment_str(&source.text)
        .collect::<Vec<_>>();
    let sentences = SentenceSegmenter::new(Default::default())
        .segment_str(&source.text)
        .collect::<Vec<_>>();
    let mut windows = Vec::<Range<usize>>::new();
    let mut display = Vec::new();
    let mut evidence = evidence.to_vec();
    evidence.sort_by_key(|hit| (hit.range.start, hit.range.end));
    let mut covered_until = 0;
    for hit in evidence {
        let start = graphemes
            .partition_point(|offset| *offset <= hit.range.start)
            .saturating_sub(1);
        let end = graphemes.partition_point(|offset| *offset < hit.range.end);
        let available = size.saturating_sub(end - start);
        let window_start = start.saturating_sub(available / 2);
        let window_end = end
            .saturating_add(available - (start - window_start))
            .min(graphemes.len() - 1);
        let mut window = graphemes[window_start]..graphemes[window_end];
        let sentence_start = sentences
            .partition_point(|offset| *offset <= hit.range.start)
            .saturating_sub(1);
        let sentence_end = sentences.partition_point(|offset| *offset < hit.range.end);
        if let (Some(&left), Some(&right)) =
            (sentences.get(sentence_start), sentences.get(sentence_end))
        {
            let count = graphemes.partition_point(|offset| *offset < right)
                - graphemes.partition_point(|offset| *offset < left);
            if count <= size {
                window = left..right;
            }
        }
        if hit.range.end > covered_until {
            covered_until = window.end;
            windows.push(window);
        }
        display.push((graphemes[start]..graphemes[end], hit.coverage));
    }
    windows.sort_by_key(|range| (range.start, range.end));
    windows.dedup();
    // Keep candidate windows bounded. Suppress overlapping selections below rather
    // than allowing chains of neighboring hits to expand an excerpt to the whole value.
    let mut passages = Vec::new();
    for range in windows {
        let first = display.partition_point(|(hit, _)| hit.start < range.start);
        let last = display.partition_point(|(hit, _)| hit.start < range.end);
        let display = &display[first..last];
        let mut matches = display
            .iter()
            .filter(|(hit, _)| hit.start >= range.start && hit.end <= range.end)
            .map(|(hit, _)| hit.clone())
            .collect::<Vec<_>>();
        matches.sort_by_key(|hit| (hit.start, hit.end));
        let mut merged = Vec::<Range<usize>>::new();
        for hit in matches {
            if let Some(last) = merged.last_mut()
                && hit.start <= last.end
            {
                last.end = last.end.max(hit.end);
            } else {
                merged.push(hit);
            }
        }
        let coverage = display
            .iter()
            .filter(|(hit, _)| hit.start >= range.start && hit.end <= range.end)
            .map(|(_, coverage)| *coverage)
            .collect();
        passages.push(Passage {
            value,
            range,
            matches: merged,
            coverage,
        });
    }
    passages
}

fn match_fields() -> Fields {
    vec![
        Field::new("start", DataType::UInt64, false),
        Field::new("end", DataType::UInt64, false),
    ]
    .into()
}

fn fragment_fields() -> Fields {
    vec![
        Field::new("field", DataType::Utf8, false),
        Field::new(
            "value_index",
            DataType::List(Arc::new(Field::new("item", DataType::UInt32, false))),
            false,
        ),
        Field::new("start", DataType::UInt64, false),
        Field::new("text", DataType::Utf8, false),
        Field::new(
            "matches",
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::Struct(match_fields()),
                false,
            ))),
            false,
        ),
    ]
    .into()
}

/// Arrow field appended to an opted-in FTS result.
pub fn highlights_field() -> Field {
    Field::new(
        HIGHLIGHTS_COL,
        DataType::List(Arc::new(Field::new(
            "item",
            DataType::Struct(fragment_fields()),
            false,
        ))),
        false,
    )
}

/// Encode excerpts without converting source text or byte offsets through JSON.
#[doc(hidden)]
pub fn highlights_array(rows: &[Vec<HighlightFragment>]) -> Result<ArrayRef> {
    let fragments = rows.iter().flatten().collect::<Vec<_>>();
    let matches = fragments
        .iter()
        .flat_map(|fragment| &fragment.matches)
        .collect::<Vec<_>>();
    let matched = StructArray::try_new(
        match_fields(),
        vec![
            Arc::new(UInt64Array::from_iter_values(
                matches.iter().map(|hit| hit.start),
            )),
            Arc::new(UInt64Array::from_iter_values(
                matches.iter().map(|hit| hit.end),
            )),
        ],
        None,
    )?;
    let list = |lengths: Vec<usize>, values: ArrayRef| -> Result<ArrayRef> {
        let mut offsets = vec![0i32];
        let mut total = 0i32;
        for length in lengths {
            total = total
                .checked_add(
                    i32::try_from(length)
                        .map_err(|_| Error::invalid_input("highlight array exceeds i32 offsets"))?,
                )
                .ok_or_else(|| Error::invalid_input("highlight array exceeds i32 offsets"))?;
            offsets.push(total);
        }
        Ok(Arc::new(ListArray::try_new(
            Arc::new(Field::new("item", values.data_type().clone(), false)),
            OffsetBuffer::new(offsets.into()),
            values,
            None,
        )?))
    };
    let values = StructArray::try_new(
        fragment_fields(),
        vec![
            Arc::new(StringArray::from_iter_values(
                fragments.iter().map(|fragment| &fragment.field),
            )),
            list(
                fragments
                    .iter()
                    .map(|fragment| fragment.value_index.len())
                    .collect(),
                Arc::new(UInt32Array::from_iter_values(
                    fragments
                        .iter()
                        .flat_map(|fragment| fragment.value_index.iter().copied()),
                )),
            )?,
            Arc::new(UInt64Array::from_iter_values(
                fragments.iter().map(|fragment| fragment.start),
            )),
            Arc::new(StringArray::from_iter_values(
                fragments.iter().map(|fragment| &fragment.text),
            )),
            list(
                fragments
                    .iter()
                    .map(|fragment| fragment.matches.len())
                    .collect(),
                Arc::new(matched),
            )?,
        ],
        None,
    )?;
    list(rows.iter().map(Vec::len).collect(), Arc::new(values))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::inverted::query::{
        BooleanQuery, MatchQuery, Occur, PhraseQuery, try_collect_query_tokens,
    };
    use crate::scalar::inverted::tokenizer::document_tokenizer::{DocType, TextTokenizer};
    use arrow_array::{Array, cast::AsArray};
    use lance_tokenizer::{
        AsciiFoldingFilter, Language, LowerCaser, SimpleTokenizer, Stemmer, StopWordFilter,
        TextAnalyzer,
    };
    use rstest::rstest;

    fn analyzer() -> Box<dyn LanceTokenizer> {
        Box::new(TextTokenizer::new(
            TextAnalyzer::builder(SimpleTokenizer::default())
                .filter(LowerCaser)
                .filter(AsciiFoldingFilter)
                .filter(Stemmer::new(Language::English))
                .filter(StopWordFilter::remove(vec!["the".into()]))
                .build(),
        ))
    }

    fn term(text: &str) -> FtsQuery {
        MatchQuery::new(text.into())
            .with_column(Some("body".into()))
            .into()
    }

    fn leaf(text: &str, slop: Option<u32>) -> HighlightLeaf {
        let mut tokenizer = analyzer();
        let tokens = Arc::new(try_collect_query_tokens(text, &mut tokenizer).unwrap());
        let terms = HighlightTerms {
            tokens,
            has_all_query_positions: true,
        };
        HighlightLeaf {
            fields: vec!["body".into()],
            tokenizer: tokenizer.into(),
            flat: terms.clone(),
            indexed: terms,
            operator: Operator::Or,
            phrase_slop: slop,
        }
    }

    fn documents(values: &[&str]) -> Vec<HighlightDocument> {
        vec![HighlightDocument {
            field: "body".into(),
            indexed: true,
            values: values
                .iter()
                .enumerate()
                .map(|(i, text)| HighlightValue {
                    value_index: if values.len() == 1 {
                        vec![]
                    } else {
                        vec![i as u32]
                    },
                    text: (*text).into(),
                })
                .collect(),
        }]
    }

    fn marked(fragments: &[HighlightFragment]) -> Vec<&str> {
        fragments
            .iter()
            .flat_map(|fragment| {
                fragment
                    .matches
                    .iter()
                    .map(|range| &fragment.text[range.start as usize..range.end as usize])
            })
            .collect()
    }

    #[rstest]
    #[case::stem("run", "She was running quickly.", "running")]
    #[case::fold("cafe", "A café résumé.", "café")]
    #[case::combining("cafe", "Cafe\u{301}", "Cafe\u{301}")]
    #[case::unicode("rust", "中文🙂 Rust", "Rust")]
    #[case::html("rust", "<script>alert(1)</script> Rust & storage", "Rust")]
    fn source_offsets(#[case] text: &str, #[case] source: &str, #[case] expected: &str) {
        let highlighter = PreparedHighlightQuery::try_new(
            term(text),
            vec![leaf(text, None)],
            HighlightOptions::new(["body"]),
        )
        .unwrap();
        let fragments = highlighter.highlight(&documents(&[source])).unwrap();
        assert_eq!(marked(&fragments), vec![expected]);
        for fragment in &fragments {
            assert_eq!(
                fragment.text,
                source[fragment.start as usize..fragment.start as usize + fragment.text.len()]
            );
        }
        assert_eq!(highlights_array(&[fragments, vec![]]).unwrap().len(), 2);
    }

    #[test]
    fn failed_and_excluded_branches_emit_no_evidence() {
        let branch = BooleanQuery::new([(Occur::Must, term("alpha")), (Occur::Must, term("beta"))]);
        let excluded = BooleanQuery::new([
            (Occur::Must, term("alpha")),
            (Occur::MustNot, term("gamma")),
        ]);
        let query = BooleanQuery::new([
            (Occur::Should, branch.into()),
            (Occur::Should, excluded.into()),
            (Occur::Should, term("gamma")),
        ]);
        let leaves = ["alpha", "beta", "alpha", "gamma", "gamma"]
            .map(|text| leaf(text, None))
            .to_vec();
        let highlighter =
            PreparedHighlightQuery::try_new(query.into(), leaves, HighlightOptions::new(["body"]))
                .unwrap();
        assert_eq!(
            marked(&highlighter.highlight(&documents(&["alpha gamma"])).unwrap()),
            vec!["gamma"]
        );
    }

    #[rstest]
    #[case::gaps("new the york", "new fast york. new york.", 0, "new fast york")]
    #[case::slop("new york", "new fast york. new very fast york.", 1, "new fast york")]
    fn phrase_witnesses_preserve_position_gaps(
        #[case] text: &str,
        #[case] source: &str,
        #[case] slop: u32,
        #[case] expected: &str,
    ) {
        let query = PhraseQuery::new(text.into()).with_column(Some("body".into()));
        let highlighter = PreparedHighlightQuery::try_new(
            query.into(),
            vec![leaf(text, Some(slop))],
            HighlightOptions::new(["body"]),
        )
        .unwrap();
        assert_eq!(
            marked(&highlighter.highlight(&documents(&[source])).unwrap()),
            vec![expected]
        );
    }

    #[test]
    fn row_phrase_preserves_array_coordinates() {
        let query = PhraseQuery::new("new york".into()).with_column(Some("body".into()));
        let highlighter = PreparedHighlightQuery::try_new(
            query.into(),
            vec![leaf("new york", Some(0))],
            HighlightOptions::new(["body"]),
        )
        .unwrap();
        let fragments = highlighter.highlight(&documents(&["new", "york"])).unwrap();
        assert_eq!(marked(&fragments), vec!["new", "york"]);
        assert_eq!(
            fragments
                .iter()
                .map(|fragment| &fragment.value_index)
                .collect::<Vec<_>>(),
            vec![&vec![0], &vec![1]]
        );
    }

    #[test]
    fn vocabulary_depends_on_search_path() {
        let mut leaf = leaf("cat", None);
        leaf.indexed.tokens = Arc::new(Tokens::new(vec!["bat".into()], DocType::Text));
        let highlighter = PreparedHighlightQuery::try_new(
            term("cat"),
            vec![leaf],
            HighlightOptions::new(["body"]),
        )
        .unwrap();
        let mut source = documents(&["bat cat"]);
        assert_eq!(
            marked(&highlighter.highlight(&source).unwrap()),
            vec!["bat"]
        );
        source[0].indexed = false;
        assert_eq!(
            marked(&highlighter.highlight(&source).unwrap()),
            vec!["cat"]
        );
    }

    #[test]
    fn incomplete_fuzzy_conjunction_cannot_match() {
        let mut leaf = leaf("cat", None);
        leaf.operator = Operator::And;
        leaf.indexed.has_all_query_positions = false;
        let highlighter = PreparedHighlightQuery::try_new(
            term("cat"),
            vec![leaf],
            HighlightOptions::new(["body"]),
        )
        .unwrap();
        assert!(matches!(
            highlighter.highlight(&documents(&["cat"])).unwrap_err(),
            Error::Internal { .. }
        ));
    }

    #[test]
    fn analysis_limit_is_an_error_not_a_truncated_match() {
        let highlighter = PreparedHighlightQuery::try_new(
            term("needle"),
            vec![leaf("needle", None)],
            HighlightOptions::new(["body"]).with_max_analyzed_bytes(16),
        )
        .unwrap();
        let error = highlighter
            .highlight(&documents(&["padding padding padding needle"]))
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("max_analyzed_bytes=16"));
    }

    #[test]
    fn passages_prefer_new_coverage_and_do_not_overlap() {
        let highlighter = PreparedHighlightQuery::try_new(
            term("alpha beta"),
            vec![leaf("alpha beta", None)],
            HighlightOptions::new(["body"])
                .with_max_fragments(1)
                .with_fragment_size(20),
        )
        .unwrap();
        let fragments = highlighter
            .highlight(&documents(&["Alpha alpha. Alpha beta!! "]))
            .unwrap();
        assert_eq!(marked(&fragments), vec!["Alpha", "beta"]);
        assert_eq!(fragments[0].text, "Alpha beta!! ");
    }

    #[test]
    fn arrow_empty_rows_and_schema_are_stable() {
        let array = highlights_array(&[vec![], vec![]]).unwrap();
        assert_eq!(array.data_type(), highlights_field().data_type());
        assert_eq!(array.null_count(), 0);
        assert_eq!(array.as_list::<i32>().value_length(1), 0);
        assert_eq!(highlights_array(&[]).unwrap().len(), 0);
    }

    #[test]
    fn options_reject_invalid_or_unknown_values() {
        let error = HighlightOptions::new(["body", "body"])
            .validate()
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("unique"));
        assert!(
            serde_json::from_str::<HighlightOptions>(r#"{"fields":["body"],"unknown_option":20}"#)
                .is_err()
        );
    }
    #[test]
    fn repeated_phrase_and_dense_matches_have_bounded_passages() {
        let text = format!("{}{}", "alpha ".repeat(10_000), "padding ".repeat(10));
        let query = PhraseQuery::new("alpha alpha".into()).with_column(Some("body".into()));
        let highlighter = PreparedHighlightQuery::try_new(
            query.into(),
            vec![leaf("alpha alpha", Some(0))],
            HighlightOptions::new(["body"]).with_fragment_size(40),
        )
        .unwrap();
        let fragments = highlighter.highlight(&documents(&[&text])).unwrap();
        assert_eq!(fragments.len(), 2);
        assert!(fragments.iter().all(|fragment| fragment.text.len() <= 40));
        assert!(
            fragments
                .iter()
                .all(|fragment| !fragment.matches.is_empty())
        );
        assert!(
            fragments
                .windows(2)
                .all(|pair| { pair[0].start + pair[0].text.len() as u64 <= pair[1].start })
        );
    }
    #[test]
    fn empty_raw_value_matches_without_display_ranges_but_null_does_not() {
        let tokenizer: Arc<dyn LanceTokenizer> = super::super::InvertedIndexParams::default()
            .base_tokenizer("raw".into())
            .build()
            .unwrap()
            .into();
        let terms = HighlightTerms {
            tokens: Arc::new(Tokens::new(vec!["".into()], DocType::Text)),
            has_all_query_positions: true,
        };
        let leaf = HighlightLeaf {
            fields: vec!["body".into()],
            tokenizer,
            flat: terms.clone(),
            indexed: terms,
            operator: Operator::Or,
            phrase_slop: None,
        };
        let query =
            PreparedHighlightQuery::try_new(term(""), vec![leaf], HighlightOptions::new(["body"]))
                .unwrap();
        assert!(query.highlight(&documents(&[""])).unwrap().is_empty());
        assert!(matches!(
            query.highlight(&documents(&[])),
            Err(Error::Internal { .. })
        ));
    }
}
