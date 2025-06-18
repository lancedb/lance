// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashSet;

use lance_core::{Error, Result};
use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize};
use snafu::location;

#[derive(Debug, Clone)]
pub struct FtsSearchParams {
    pub limit: Option<usize>,
    pub wand_factor: f32,
    pub fuzziness: Option<u32>,
    pub max_expansions: usize,
    // None means not a phrase query
    // Some(n) means a phrase query with slop n
    pub phrase_slop: Option<u32>,
    /// The number of beginning characters being unchanged for fuzzy matching.
    pub prefix_length: u32,
}

impl FtsSearchParams {
    pub fn new() -> Self {
        Self {
            limit: None,
            wand_factor: 1.0,
            fuzziness: Some(0),
            max_expansions: 50,
            phrase_slop: None,
            prefix_length: 0,
        }
    }

    pub fn with_limit(mut self, limit: Option<usize>) -> Self {
        self.limit = limit;
        self
    }

    pub fn with_wand_factor(mut self, factor: f32) -> Self {
        self.wand_factor = factor;
        self
    }

    pub fn with_fuzziness(mut self, fuzziness: Option<u32>) -> Self {
        self.fuzziness = fuzziness;
        self
    }

    pub fn with_max_expansions(mut self, max_expansions: usize) -> Self {
        self.max_expansions = max_expansions;
        self
    }

    pub fn with_phrase_slop(mut self, phrase_slop: Option<u32>) -> Self {
        self.phrase_slop = phrase_slop;
        self
    }

    pub fn with_prefix_length(mut self, prefix_length: u32) -> Self {
        self.prefix_length = prefix_length;
        self
    }
}

impl Default for FtsSearchParams {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Operator {
    And,
    Or,
}

impl Default for Operator {
    fn default() -> Self {
        Self::Or
    }
}

impl TryFrom<&str> for Operator {
    type Error = Error;
    fn try_from(value: &str) -> Result<Self> {
        match value.to_ascii_uppercase().as_str() {
            "AND" => Ok(Self::And),
            "OR" => Ok(Self::Or),
            _ => Err(Error::invalid_input(
                format!("Invalid operator: {}", value),
                location!(),
            )),
        }
    }
}

impl From<Operator> for &'static str {
    fn from(operator: Operator) -> Self {
        match operator {
            Operator::And => "AND",
            Operator::Or => "OR",
        }
    }
}

pub trait FtsQueryNode {
    fn columns(&self) -> HashSet<String>;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FtsQuery {
    // leaf queries
    Match(MatchQuery),
    Phrase(PhraseQuery),

    // compound queries
    Boost(BoostQuery),
    MultiMatch(MultiMatchQuery),
    Boolean(BooleanQuery),
}

impl std::fmt::Display for FtsQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Match(query) => write!(f, "Match({:?})", query),
            Self::Phrase(query) => write!(f, "Phrase({:?})", query),
            Self::Boost(query) => write!(
                f,
                "Boosting(positive={}, negative={}, negative_boost={})",
                query.positive, query.negative, query.negative_boost
            ),
            Self::MultiMatch(query) => write!(f, "MultiMatch({:?})", query),
            Self::Boolean(query) => {
                write!(
                    f,
                    "Boolean(must={:?}, should={:?})",
                    query.must, query.should
                )
            }
        }
    }
}

impl FtsQueryNode for FtsQuery {
    fn columns(&self) -> HashSet<String> {
        match self {
            Self::Match(query) => query.columns(),
            Self::Phrase(query) => query.columns(),
            Self::Boost(query) => {
                let mut columns = query.positive.columns();
                columns.extend(query.negative.columns());
                columns
            }
            Self::MultiMatch(query) => {
                let mut columns = HashSet::new();
                for match_query in &query.match_queries {
                    columns.extend(match_query.columns());
                }
                columns
            }
            Self::Boolean(query) => {
                let mut columns = HashSet::new();
                for query in &query.must {
                    columns.extend(query.columns());
                }
                for query in &query.should {
                    columns.extend(query.columns());
                }
                columns
            }
        }
    }
}

impl FtsQuery {
    pub fn query(&self) -> String {
        match self {
            Self::Match(query) => query.terms.clone(),
            Self::Phrase(query) => format!("\"{}\"", query.terms), // Phrase queries are quoted
            Self::Boost(query) => query.positive.query(),
            Self::MultiMatch(query) => query.match_queries[0].terms.clone(),
            Self::Boolean(_) => {
                // Bool queries don't have a single query string, they are composed of multiple queries
                String::new()
            }
        }
    }

    pub fn is_missing_column(&self) -> bool {
        match self {
            Self::Match(query) => query.column.is_none(),
            Self::Phrase(query) => query.column.is_none(),
            Self::Boost(query) => {
                query.positive.is_missing_column() || query.negative.is_missing_column()
            }
            Self::MultiMatch(query) => query.match_queries.iter().any(|q| q.column.is_none()),
            Self::Boolean(query) => {
                query.must.iter().any(|q| q.is_missing_column())
                    || query.should.iter().any(|q| q.is_missing_column())
            }
        }
    }

    pub fn with_column(self, column: String) -> Self {
        match self {
            Self::Match(query) => Self::Match(query.with_column(Some(column))),
            Self::Phrase(query) => Self::Phrase(query.with_column(Some(column))),
            Self::Boost(query) => {
                let positive = query.positive.with_column(column.clone());
                let negative = query.negative.with_column(column);
                Self::Boost(BoostQuery {
                    positive: Box::new(positive),
                    negative: Box::new(negative),
                    negative_boost: query.negative_boost,
                })
            }
            Self::MultiMatch(query) => {
                let match_queries = query
                    .match_queries
                    .into_iter()
                    .map(|q| q.with_column(Some(column.clone())))
                    .collect();
                Self::MultiMatch(MultiMatchQuery { match_queries })
            }
            Self::Boolean(query) => {
                let must = query
                    .must
                    .into_iter()
                    .map(|q| q.with_column(column.clone()))
                    .collect();
                let should = query
                    .should
                    .into_iter()
                    .map(|q| q.with_column(column.clone()))
                    .collect();
                let must_not = query
                    .must_not
                    .into_iter()
                    .map(|q| q.with_column(column.clone()))
                    .collect();
                Self::Boolean(BooleanQuery {
                    must,
                    should,
                    must_not,
                })
            }
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

impl From<BoostQuery> for FtsQuery {
    fn from(query: BoostQuery) -> Self {
        Self::Boost(query)
    }
}

impl From<MultiMatchQuery> for FtsQuery {
    fn from(query: MultiMatchQuery) -> Self {
        Self::MultiMatch(query)
    }
}

impl From<BooleanQuery> for FtsQuery {
    fn from(query: BooleanQuery) -> Self {
        Self::Boolean(query)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchQuery {
    // The column to search in.
    // If None, it will be determined at query time.
    pub column: Option<String>,
    pub terms: String,

    // literal default is not supported so we set it by function
    #[serde(default = "MatchQuery::default_boost")]
    pub boost: f32,

    // The max edit distance for fuzzy matching.
    // If Some(0), it will be exact match.
    // If None, it will be determined automatically by the rules:
    // - 0 for terms with length <= 2
    // - 1 for terms with length <= 5
    // - 2 for terms with length > 5
    pub fuzziness: Option<u32>,

    /// The maximum number of terms to expand for fuzzy matching.
    /// Default to 50.
    #[serde(default = "MatchQuery::default_max_expansions")]
    pub max_expansions: usize,

    /// The operator to use for combining terms.
    /// This can be either `And` or `Or`, it's 'Or' by default.
    /// - `And`: All terms must match.
    /// - `Or`: At least one term must match.
    #[serde(default)]
    pub operator: Operator,

    /// The number of beginning characters being unchanged for fuzzy matching.
    /// Default to 0.
    #[serde(default)]
    pub prefix_length: u32,
}

impl MatchQuery {
    pub fn new(terms: String) -> Self {
        Self {
            column: None,
            terms,
            boost: 1.0,
            fuzziness: Some(0),
            max_expansions: 50,
            operator: Operator::Or,
            prefix_length: 0,
        }
    }

    fn default_boost() -> f32 {
        1.0
    }

    fn default_max_expansions() -> usize {
        50
    }

    pub fn with_column(mut self, column: Option<String>) -> Self {
        self.column = column;
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

    pub fn with_max_expansions(mut self, max_expansions: usize) -> Self {
        self.max_expansions = max_expansions;
        self
    }

    pub fn with_operator(mut self, operator: Operator) -> Self {
        self.operator = operator;
        self
    }

    pub fn with_prefix_length(mut self, prefix_length: u32) -> Self {
        self.prefix_length = prefix_length;
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
    fn columns(&self) -> HashSet<String> {
        let mut columns = HashSet::new();
        if let Some(column) = &self.column {
            columns.insert(column.clone());
        }
        columns
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhraseQuery {
    // The column to search in.
    // If None, it will be determined at query time.
    pub column: Option<String>,
    pub terms: String,
    #[serde(default = "u32::default")]
    pub slop: u32,
}

impl PhraseQuery {
    pub fn new(terms: String) -> Self {
        Self {
            column: None,
            terms,
            slop: 0,
        }
    }

    pub fn with_column(mut self, column: Option<String>) -> Self {
        self.column = column;
        self
    }

    pub fn with_slop(mut self, slop: u32) -> Self {
        self.slop = slop;
        self
    }
}

impl FtsQueryNode for PhraseQuery {
    fn columns(&self) -> HashSet<String> {
        let mut columns = HashSet::new();
        if let Some(column) = &self.column {
            columns.insert(column.clone());
        }
        columns
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoostQuery {
    pub positive: Box<FtsQuery>,
    pub negative: Box<FtsQuery>,
    #[serde(default = "BoostQuery::default_negative_boost")]
    pub negative_boost: f32,
}

impl BoostQuery {
    pub fn new(positive: FtsQuery, negative: FtsQuery, negative_boost: Option<f32>) -> Self {
        Self {
            positive: Box::new(positive),
            negative: Box::new(negative),
            negative_boost: negative_boost.unwrap_or(0.5),
        }
    }

    fn default_negative_boost() -> f32 {
        0.5
    }
}

impl FtsQueryNode for BoostQuery {
    fn columns(&self) -> HashSet<String> {
        let mut columns = self.positive.columns();
        columns.extend(self.negative.columns());
        columns
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MultiMatchQuery {
    // each query must be a match query with specified column
    pub match_queries: Vec<MatchQuery>,
}

impl Serialize for MultiMatchQuery {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut map = serializer.serialize_map(Some(3))?;

        let query = self.match_queries.first().ok_or(serde::ser::Error::custom(
            "MultiMatchQuery must have at least one MatchQuery".to_string(),
        ))?;
        map.serialize_entry("query", &query.terms)?;
        let columns = self
            .match_queries
            .iter()
            .map(|q| q.column.as_ref().unwrap().clone())
            .collect::<Vec<String>>();
        map.serialize_entry("columns", &columns)?;
        let boosts = self
            .match_queries
            .iter()
            .map(|q| q.boost)
            .collect::<Vec<f32>>();
        map.serialize_entry("boost", &boosts)?;
        map.end()
    }
}

impl<'de> Deserialize<'de> for MultiMatchQuery {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct MultiMatchQueryData {
            query: String,
            columns: Vec<String>,
            boost: Option<Vec<f32>>,
        }

        let data = MultiMatchQueryData::deserialize(deserializer)?;
        let boosts = data.boost.unwrap_or(vec![1.0; data.columns.len()]);

        Self::try_new(data.query, data.columns)
            .map_err(serde::de::Error::custom)?
            .try_with_boosts(boosts)
            .map_err(serde::de::Error::custom)
    }
}

impl MultiMatchQuery {
    pub fn try_new(query: String, columns: Vec<String>) -> Result<Self> {
        if columns.is_empty() {
            return Err(Error::invalid_input(
                "Cannot create MultiMatchQuery with no columns".to_string(),
                location!(),
            ));
        }

        let match_queries = columns
            .into_iter()
            .map(|column| MatchQuery::new(query.clone()).with_column(Some(column)))
            .collect();
        Ok(Self { match_queries })
    }

    pub fn try_with_boosts(mut self, boosts: Vec<f32>) -> Result<Self> {
        if boosts.len() != self.match_queries.len() {
            return Err(Error::invalid_input(
                "The number of boosts must match the number of queries".to_string(),
                location!(),
            ));
        }

        for (query, boost) in self.match_queries.iter_mut().zip(boosts) {
            query.boost = boost;
        }
        Ok(self)
    }

    pub fn with_operator(mut self, operator: Operator) -> Self {
        for query in &mut self.match_queries {
            query.operator = operator;
        }
        self
    }
}

impl FtsQueryNode for MultiMatchQuery {
    fn columns(&self) -> HashSet<String> {
        let mut columns = HashSet::with_capacity(self.match_queries.len());
        for query in &self.match_queries {
            columns.extend(query.columns());
        }
        columns
    }
}

pub enum Occur {
    Should,
    Must,
    MustNot,
}

impl TryFrom<&str> for Occur {
    type Error = Error;
    fn try_from(value: &str) -> Result<Self> {
        match value.to_ascii_uppercase().as_str() {
            "SHOULD" => Ok(Self::Should),
            "MUST" => Ok(Self::Must),
            "MUST_NOT" => Ok(Self::MustNot),
            _ => Err(Error::invalid_input(
                format!("Invalid occur value: {}", value),
                location!(),
            )),
        }
    }
}

impl From<Occur> for &'static str {
    fn from(occur: Occur) -> Self {
        match occur {
            Occur::Should => "SHOULD",
            Occur::Must => "MUST",
            Occur::MustNot => "MUST_NOT",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BooleanQuery {
    pub should: Vec<FtsQuery>,
    pub must: Vec<FtsQuery>,
    pub must_not: Vec<FtsQuery>,
}

impl BooleanQuery {
    pub fn new(iter: impl IntoIterator<Item = (Occur, FtsQuery)>) -> Self {
        let mut should = Vec::new();
        let mut must = Vec::new();
        let mut must_not = Vec::new();
        for (occur, query) in iter {
            match occur {
                Occur::Should => should.push(query),
                Occur::Must => must.push(query),
                Occur::MustNot => must_not.push(query),
            }
        }
        Self {
            should,
            must,
            must_not,
        }
    }

    pub fn with_should(mut self, query: FtsQuery) -> Self {
        self.should.push(query);
        self
    }

    pub fn with_must(mut self, query: FtsQuery) -> Self {
        self.must.push(query);
        self
    }

    pub fn with_must_not(mut self, query: FtsQuery) -> Self {
        self.must_not.push(query);
        self
    }
}

impl FtsQueryNode for BooleanQuery {
    fn columns(&self) -> HashSet<String> {
        let mut columns = HashSet::new();
        for query in &self.should {
            columns.extend(query.columns());
        }
        for query in &self.must {
            columns.extend(query.columns());
        }
        for query in &self.must_not {
            columns.extend(query.columns());
        }
        columns
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

pub fn fill_fts_query_column(
    query: &FtsQuery,
    columns: &[String],
    replace: bool,
) -> Result<FtsQuery> {
    if !query.is_missing_column() && !replace {
        return Ok(query.clone());
    }
    match query {
        FtsQuery::Match(match_query) => {
            match columns.len() {
                0 => {
                    Err(Error::invalid_input(
                        "Cannot perform full text search unless an INVERTED index has been created on at least one column".to_string(),
                        location!(),
                    ))
                }
                1 => {
                    let column = columns[0].clone();
                    let query = match_query.clone().with_column(Some(column));
                    Ok(FtsQuery::Match(query))
                }
                _ => {
                    // if there are multiple columns, we need to create a MultiMatch query
                    let multi_match_query =
                        MultiMatchQuery::try_new(match_query.terms.clone(), columns.to_vec())?;
                    Ok(FtsQuery::MultiMatch(multi_match_query))
                }
            }
        }
        FtsQuery::Phrase(phrase_query) => {
            match columns.len() {
                0 => {
                    Err(Error::invalid_input(
                        "Cannot perform full text search unless an INVERTED index has been created on at least one column".to_string(),
                        location!(),
                    ))
                }
                1 => {
                    let column = columns[0].clone();
                    let query = phrase_query.clone().with_column(Some(column));
                    Ok(FtsQuery::Phrase(query))
                }
                _ => {
                    Err(Error::invalid_input(
                        "the column must be specified in the query".to_string(),
                        location!(),
                    ))
                }
            }
        }
       FtsQuery::Boost(boost_query) => {
            let positive = fill_fts_query_column(&boost_query.positive, columns, replace)?;
            let negative = fill_fts_query_column(&boost_query.negative, columns, replace)?;
            Ok(FtsQuery::Boost(BoostQuery {
                positive: Box::new(positive),
                negative: Box::new(negative),
                negative_boost: boost_query.negative_boost,
            }))
        }
        FtsQuery::MultiMatch(multi_match_query) => {
            let match_queries = multi_match_query
                .match_queries
                .iter()
                .map(|query| fill_fts_query_column(&FtsQuery::Match(query.clone()), columns, replace))
                .map(|result| {
                    result.map(|query| {
                        if let FtsQuery::Match(match_query) = query {
                            match_query
                        } else {
                            unreachable!("Expected MatchQuery")
                        }
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(FtsQuery::MultiMatch(MultiMatchQuery { match_queries }))
       }
        FtsQuery::Boolean(bool_query) => {
            let must = bool_query
                .must
                .iter()
                .map(|query| fill_fts_query_column(query, columns, replace))
                .collect::<Result<Vec<_>>>()?;
            let should = bool_query
                .should
                .iter()
                .map(|query| fill_fts_query_column(query, columns, replace))
                .collect::<Result<Vec<_>>>()?;
            let must_not = bool_query
                .must_not
                .iter()
                .map(|query| fill_fts_query_column(query, columns, replace))
                .collect::<Result<Vec<_>>>()?;
            Ok(FtsQuery::Boolean(BooleanQuery { must, should, must_not }))
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_match_query_serde() {
        use super::*;
        use serde_json::json;

        let query = MatchQuery::new("hello world".to_string())
            .with_column(Some("text".to_string()))
            .with_boost(2.0)
            .with_fuzziness(Some(1))
            .with_max_expansions(10)
            .with_operator(Operator::And);

        let serialized = serde_json::to_value(&query).unwrap();
        let expected = json!({
            "column": "text",
            "terms": "hello world",
            "boost": 2.0,
            "fuzziness": 1,
            "max_expansions": 10,
            "operator": "And",
            "prefix_length": 0,
        });
        assert_eq!(serialized, expected);

        let expected = json!({
            "column": "text",
            "terms": "hello world",
            "fuzziness": 0,
        });
        let query = serde_json::from_str::<MatchQuery>(&expected.to_string()).unwrap();
        assert_eq!(query.column, Some("text".to_owned()));
        assert_eq!(query.terms, "hello world");
        assert_eq!(query.boost, 1.0);
        assert_eq!(query.fuzziness, Some(0));
        assert_eq!(query.max_expansions, 50);
        assert_eq!(query.operator, Operator::Or);
        assert_eq!(query.prefix_length, 0);
    }

    #[test]
    fn test_phrase_query_serde() {
        use super::*;
        use serde_json::json;

        let query = json!({
            "terms": "hello world",
        });
        let expected = PhraseQuery::new("hello world".to_string());
        let query: PhraseQuery = serde_json::from_value(query).unwrap();
        assert_eq!(query, expected);

        let query = json!({
            "terms": "hello world",
            "column": "text",
            "slop": 2,
        });
        let expected = PhraseQuery::new("hello world".to_string())
            .with_column(Some("text".to_string()))
            .with_slop(2);
        let query: PhraseQuery = serde_json::from_value(query).unwrap();
        assert_eq!(query, expected);
    }
}
