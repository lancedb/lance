// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Geographic indexing module
//!
//! This module contains implementations for spatial/geographic indexing:
//! - BKD: Block K-Dimensional tree for efficient spatial partitioning (core data structure)
//! - BkdTree: Geographic index built on top of BKD trees for GeoArrow data

pub mod bkd;
pub mod bkdtree;

pub use bkd::*;
pub use bkdtree::*;
