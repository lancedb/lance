// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Vector search, as logical nodes, rules, and lowering.
//!
//! Arranged deliberately like [`fts`](super::fts): the same five entry points, so that the
//! plugin boundary the design doc proposes is tested by two index types rather than asserted
//! by one.

mod node;
mod planner;
mod rerank;
mod rules;

pub use node::*;
pub use planner::*;
pub use rerank::*;
pub use rules::*;
