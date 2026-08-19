// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Full-text search, as logical nodes, rules, and lowering.
//!
//! Everything FTS-specific lives under this directory rather than being spread across the
//! framework's `builder` / `rules` / `planner` modules. That is deliberate: the design doc asks
//! whether an index type could one day ship its own planning support, and keeping one index's
//! nodes, rules, prefetch, and lowering together is the closest thing to an answer we have. The
//! rest of the module reaches in through five entry points — [`build_source`], [`prefetch`],
//! [`analyzer_rules`] / [`optimizer_rules`], [`plan_extension`], and [`collect_requests`].
//!
//! [`vector`](super::vector) is deliberately arranged the same way. Two index types sharing one
//! shape is what makes the boundary a claim rather than an artifact of a single file.
//!
//! # Shape
//!
//! An [`FtsQuery`](lance_index::scalar::inverted::query::FtsQuery) is a recursive IR, so it maps
//! onto a subtree rather than a single node:
//!
//! ```text
//! FtsCompound{Boolean}                 <- Boost / MultiMatch / Boolean
//!   FtsLeaf{Match, via=index}          <- one per Match / Phrase
//!     Filter / TableScan               <- prefilter source, or the text to scan
//!   FtsLeaf{Match, via=flat}
//!     Filter / TableScan
//! ```
//!
//! Each leaf resolves independently, which is what lets a partially-indexed column be handled by
//! the same split-into-two-branches rewrite the vector path uses.

mod builder;
mod compound;
mod leaf;
mod match_filter;
mod planner;
mod prefetch;
mod rules;
mod scorer;

pub use builder::*;
pub use compound::*;
pub use leaf::*;
pub use match_filter::*;
pub use planner::*;
pub use prefetch::*;
pub use rules::*;
pub use scorer::*;

use datafusion::logical_expr::{Extension, LogicalPlan, UserDefinedLogicalNodeCore};
use std::sync::Arc;

/// Wrap a node as a [`LogicalPlan::Extension`].
fn extension(node: impl UserDefinedLogicalNodeCore) -> LogicalPlan {
    LogicalPlan::Extension(Extension {
        node: Arc::new(node),
    })
}
