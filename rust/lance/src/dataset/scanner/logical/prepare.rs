// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stage 0: normalize the scanner's queries before the plan is built.
//!
//! Stage 1 is synchronous, and a search query may carry fields that take I/O to fill in, so
//! whatever resolving they need happens here.

use lance_index::vector::Query;

use super::super::QueryFilter;
use crate::Result;
use crate::dataset::Scanner;

/// The scanner's search queries, with every implicit field resolved.
#[derive(Debug, Default)]
pub struct PreparedQueries {
    pub vector_filter: Option<Query>,
}

impl PreparedQueries {
    pub async fn resolve(scanner: &Scanner) -> Result<Self> {
        let vector_filter = match &scanner.filter.query_filter {
            Some(QueryFilter::Vector(query)) => Some(query.clone()),
            _ => None,
        };
        Ok(Self { vector_filter })
    }
}
