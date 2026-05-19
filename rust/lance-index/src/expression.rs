// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Plan-time expression parsing for scalar and aggregate index pushdown.
//!
//! Both halves split a user expression into an index-evaluable leaf plus the
//! residual computation: [`scalar`] parses `WHERE` clauses, [`aggregate`]
//! parses `SELECT`-list aggregates. The execute-time consumers live under
//! `lance::io::exec::{scalar_index, aggregate_index}`.

pub mod aggregate;
pub mod scalar;
