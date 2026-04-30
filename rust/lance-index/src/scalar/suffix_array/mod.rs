// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Suffix array index for n-gram counting and text search.
//!
//! This module implements a suffix array-based scalar index that supports
//! efficient n-gram counting and text search operations on tokenized data.

mod builder;
mod index;
mod plugin;
mod query;

pub use index::SuffixArrayIndex;
pub use plugin::SuffixArrayIndexPlugin;
pub use query::SuffixArrayQuery;
