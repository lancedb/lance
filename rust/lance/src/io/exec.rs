// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Execution nodes
//!
//! WARNING: Internal API with no stability guarantees.

mod filter;
pub mod filtered_read;
pub mod fts;
pub(crate) mod knn;
mod optimizer;
mod projection;
mod pushdown_scan;
mod rowids;
pub mod scalar_index;
mod scan;
mod take;
#[cfg(test)]
pub mod testing;
pub mod utils;

pub use filter::LanceFilterExec;
pub use knn::{ANNIvfPartitionExec, ANNIvfSubIndexExec, KNNVectorDistanceExec};
pub use lance_datafusion::planner::Planner;
pub use lance_index::scalar::expression::FilterPlan;
pub use optimizer::get_physical_optimizer;
pub use projection::project;
pub use pushdown_scan::{LancePushdownScanExec, ScanConfig};
pub use rowids::{AddRowAddrExec, AddRowOffsetExec};
pub use scan::{LanceScanConfig, LanceScanExec};
pub use take::TakeExec;
pub use utils::PreFilterSource;
