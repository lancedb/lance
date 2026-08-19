// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stage 0: normalize the scanner's queries before the plan is built.
//!
//! This stage exists only because of full-text search, and it is the one place the design doc's
//! four-stage model needed extending. An [`FtsQuery`](lance_index::scalar::inverted::query::FtsQuery)
//! may omit both the column it searches and its document granularity, and filling them in means
//! asking which columns carry an inverted index — I/O. Stage 1 is synchronous, and a node's
//! output schema depends on the granularity, so the question cannot be deferred into a rule
//! either.
//!
//! The vector path needs nothing here: a `Query` always names its column.

use lance_index::scalar::FullTextSearchQuery;
use lance_index::vector::Query;

use super::super::QueryFilter;
use crate::Result;
use crate::dataset::Scanner;

/// The scanner's search queries, with every implicit field resolved.
#[derive(Debug, Default)]
pub struct PreparedQueries {
    pub full_text: Option<FullTextSearchQuery>,
    pub fts_filter: Option<FullTextSearchQuery>,
    pub vector_filter: Option<Query>,
    /// Whether the full-text query scores list elements rather than whole rows. Decides whether
    /// `_doc_index` is part of the output.
    pub element_granularity: bool,
}

impl PreparedQueries {
    pub async fn resolve(scanner: &Scanner) -> Result<Self> {
        let full_text = match &scanner.full_text_query {
            Some(query) => Some(scanner.resolve_full_text_search_query(query).await?),
            None => None,
        };
        let (fts_filter, vector_filter) = match &scanner.filter.query_filter {
            Some(QueryFilter::Fts(query)) => (
                Some(scanner.resolve_full_text_search_query(query).await?),
                None,
            ),
            Some(QueryFilter::Vector(query)) => (None, Some(query.clone())),
            None => (None, None),
        };

        let element_granularity = match &full_text {
            Some(query) => scanner
                .fts_document_granularity(&query.query)?
                .is_list_element(),
            None => false,
        };

        Ok(Self {
            full_text,
            fts_filter,
            vector_filter,
            element_granularity,
        })
    }
}
