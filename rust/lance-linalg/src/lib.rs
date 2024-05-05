// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! High-performance [Apache Arrow](https://docs.rs/arrow/latest/arrow/) native Linear Algebra algorithms.

#![deny(clippy::unused_async)]

pub mod distance;
pub mod kernels;
pub mod kmeans;
pub mod matrix;
pub mod simd;

#[cfg(test)]
pub(crate) mod test_utils;

pub use matrix::MatrixView;

use arrow_schema::ArrowError;

type Error = ArrowError;
pub type Result<T> = std::result::Result<T, Error>;
