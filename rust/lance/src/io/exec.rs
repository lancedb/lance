// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Execution nodes
//!
//! WARNING: Internal API with no stability guarantees.

pub(crate) mod knn;
mod optimizer;
mod projection;
mod pushdown_scan;
pub mod scalar_index;
mod scan;
mod take;
#[cfg(test)]
pub mod testing;
pub mod utils;

pub use knn::{ANNIvfPartitionExec, ANNIvfSubIndexExec, KNNVectorDistanceExec, PreFilterSource};
pub use lance_datafusion::planner::Planner;
pub use lance_index::scalar::expression::FilterPlan;
pub use optimizer::get_physical_optimizer;
pub use projection::ProjectionExec;
pub use pushdown_scan::{LancePushdownScanExec, ScanConfig};
pub use scan::LanceScanExec;
pub use take::TakeExec;
