// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashSet;

use lance_core::{Error, Result};
use snafu::location;

#[derive(Debug, Clone)]
pub struct FtsSearchParams {
    pub limit: usize,
    pub wand_factor: f32,
}

impl FtsSearchParams {
    pub fn new() -> Self {
        Self {
            limit: usize::MAX,
            wand_factor: 1.0,
        }
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    pub fn with_wand_factor(mut self, factor: f32) -> Self {
        self.wand_factor = factor;
        self
    }
}

impl Default for FtsSearchParams {
    fn default() -> Self {
        Self::new()
    }
}

pub trait FtsQueryNode {
    fn fields(&self) -> HashSet<String>;
}

#[derive(Debug, Clone, PartialEq)]
pub enum FtsQuery {
    Match(MatchQuery),
    Phrase(PhraseQuery),
}

impl std::fmt::Display for FtsQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Match(query) => write!(f, "Match({:?})", query),
            Self::Phrase(query) => write!(f, "Phrase({:?})", query),
        }
    }
}

impl FtsQueryNode for FtsQuery {
    fn fields(&self) -> HashSet<String> {
        match self {
            Self::Match(query) => query.fields(),
            Self::Phrase(query) => query.fields(),
        }
    }
}

impl FtsQuery {
    pub fn query(&self) -> &str {
        match self {
            Self::Match(query) => &query.terms,
            Self::Phrase(query) => &query.terms,
        }
    }

    pub fn is_missing_field(&self) -> bool {
        match self {
            Self::Match(query) => query.field.is_none(),
            Self::Phrase(query) => query.field.is_none(),
        }
    }

    pub fn with_field(self, field: String) -> Self {
        match self {
            Self::Match(query) => Self::Match(query.with_field(Some(field))),
            Self::Phrase(query) => Self::Phrase(query.with_field(Some(field))),
        }
    }
}

impl From<MatchQuery> for FtsQuery {
    fn from(query: MatchQuery) -> Self {
        Self::Match(query)
    }
}

impl From<PhraseQuery> for FtsQuery {
    fn from(query: PhraseQuery) -> Self {
        Self::Phrase(query)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchQuery {
    // The field to search in.
    // If None, it will be determined at query time.
    pub field: Option<String>,
    pub terms: String,
    pub boost: f32,

    // The max edit distance for fuzzy matching.
    // If Some(0), it will be exact match.
    // If None, it will be determined automatically by the rules:
    // - 0 for terms with length <= 2
    // - 1 for terms with length <= 5
    // - 2 for terms with length > 5
    pub fuzziness: Option<u32>,
}

impl MatchQuery {
    pub fn new(terms: String) -> Self {
        Self {
            field: None,
            terms,
            boost: 1.0,
            fuzziness: Some(0),
        }
    }

    pub fn with_field(mut self, field: Option<String>) -> Self {
        self.field = field;
        self
    }

    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    pub fn with_fuzziness(mut self, fuzziness: Option<u32>) -> Self {
        self.fuzziness = fuzziness;
        self
    }

    pub fn auto_fuzziness(token: &str) -> u32 {
        match token.len() {
            0..=2 => 0,
            3..=5 => 1,
            _ => 2,
        }
    }
}

impl FtsQueryNode for MatchQuery {
    fn fields(&self) -> HashSet<String> {
        let mut fields = HashSet::new();
        if let Some(field) = &self.field {
            fields.insert(field.clone());
        }
        fields
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhraseQuery {
    // The field to search in.
    // If None, it will be determined at query time.
    pub field: Option<String>,
    pub terms: String,
}

impl PhraseQuery {
    pub fn new(terms: String) -> Self {
        Self { field: None, terms }
    }

    pub fn with_field(mut self, field: Option<String>) -> Self {
        self.field = field;
        self
    }
}

impl FtsQueryNode for PhraseQuery {
    fn fields(&self) -> HashSet<String> {
        let mut fields = HashSet::new();
        if let Some(field) = &self.field {
            fields.insert(field.clone());
        }
        fields
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CompoundQuery {
    Leaf(FtsQuery),
    Boost(BoostQuery),
    MultiMatch(MultiMatchQuery),
}

impl From<FtsQuery> for CompoundQuery {
    fn from(query: FtsQuery) -> Self {
        Self::Leaf(query)
    }
}

impl From<BoostQuery> for CompoundQuery {
    fn from(query: BoostQuery) -> Self {
        Self::Boost(query)
    }
}

impl From<MultiMatchQuery> for CompoundQuery {
    fn from(query: MultiMatchQuery) -> Self {
        Self::MultiMatch(query)
    }
}

impl std::fmt::Display for CompoundQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Leaf(query) => write!(f, "Leaf({})", query),
            Self::Boost(query) => write!(
                f,
                "Boosting(positive={}, negative={}, negative_boost={})",
                query.positive, query.negative, query.negative_boost
            ),
            Self::MultiMatch(query) => write!(f, "MultiMatch({:?})", query),
        }
    }
}

impl FtsQueryNode for CompoundQuery {
    fn fields(&self) -> HashSet<String> {
        match self {
            Self::Leaf(query) => query.fields(),
            Self::Boost(query) => {
                let mut fields = query.positive.fields();
                fields.extend(query.negative.fields());
                fields
            }
            Self::MultiMatch(query) => {
                let mut fields = HashSet::new();
                for match_query in &query.match_queries {
                    fields.extend(match_query.fields());
                }
                fields
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoostQuery {
    pub positive: Box<CompoundQuery>,
    pub negative: Box<CompoundQuery>,
    pub negative_boost: f32,
}

impl BoostQuery {
    pub fn new(
        positive: CompoundQuery,
        negative: CompoundQuery,
        negative_boost: Option<f32>,
    ) -> Self {
        Self {
            positive: Box::new(positive),
            negative: Box::new(negative),
            negative_boost: negative_boost.unwrap_or(0.5),
        }
    }
}

impl FtsQueryNode for BoostQuery {
    fn fields(&self) -> HashSet<String> {
        let mut fields = self.positive.fields();
        fields.extend(self.negative.fields());
        fields
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MultiMatchQuery {
    // each query must be a match query with specified field
    pub match_queries: Vec<CompoundQuery>,
}

impl MultiMatchQuery {
    pub fn new(query: String, fields: Vec<String>) -> Self {
        let match_queries = fields
            .into_iter()
            .map(|field| {
                FtsQuery::Match(MatchQuery::new(query.clone()).with_field(Some(field))).into()
            })
            .collect();
        Self { match_queries }
    }

    pub fn with_boosts(query: String, fields: Vec<String>, boosts: Vec<f32>) -> Self {
        let match_queries = fields
            .into_iter()
            .zip(boosts)
            .map(|(field, boost)| {
                FtsQuery::Match(
                    MatchQuery::new(query.clone())
                        .with_field(Some(field))
                        .with_boost(boost),
                )
                .into()
            })
            .collect();
        Self { match_queries }
    }
}

impl FtsQueryNode for MultiMatchQuery {
    fn fields(&self) -> HashSet<String> {
        let mut fields = HashSet::with_capacity(self.match_queries.len());
        for query in &self.match_queries {
            fields.extend(query.fields());
        }
        fields
    }
}

pub fn collect_tokens(
    text: &str,
    tokenizer: &mut tantivy::tokenizer::TextAnalyzer,
    inclusive: Option<&HashSet<String>>,
) -> Vec<String> {
    let mut stream = tokenizer.token_stream(text);
    let mut tokens = Vec::new();
    while let Some(token) = stream.next() {
        if let Some(inclusive) = inclusive {
            if !inclusive.contains(&token.text) {
                continue;
            }
        }
        tokens.push(token.text.to_owned());
    }
    tokens
}

pub fn fill_fts_query_field(
    query: &CompoundQuery,
    columns: &[String],
    replace: bool,
) -> Result<CompoundQuery> {
    match query {
        CompoundQuery::Leaf(leaf) => {
            if !leaf.is_missing_field() && !replace {
                return Ok(query.clone());
            }

            match columns.len() {
                0 => {
                    Err(Error::invalid_input(
                        "Cannot perform full text search unless an INVERTED index has been created on at least one column".to_string(),
                        location!(),
                    ))
                }
                1 => {
                    let field = columns[0].clone();
                    let query = leaf.clone().with_field(field);
                    Ok(CompoundQuery::Leaf(query))
                }
                _ => {
                    // if there are multiple columns, we need to create a MultiMatch query
                    let multi_match_query =
                        MultiMatchQuery::new(leaf.query().to_owned(), columns.to_vec());
                    Ok(CompoundQuery::MultiMatch(multi_match_query))
                }
            }
        }
        // for compound queries, we require the users to specify the field
        _ => Err(Error::invalid_input(
            "the field must be specified in the query".to_string(),
            location!(),
        )),
    }
}
