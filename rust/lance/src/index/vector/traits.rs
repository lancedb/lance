// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Traits for vector index.
//!

use arrow_array::{types::Float32Type, FixedSizeListArray};
use async_trait::async_trait;

use lance_core::Result;
use lance_io::object_writer::ObjectWriter;
use lance_linalg::MatrixView;

use crate::index::pb::Transform;

/// Transformer on vectors.
#[async_trait]
pub trait Transformer: std::fmt::Debug + Sync + Send {
    /// Train the transformer.
    ///
    /// Parameters:
    /// - *data*: training vectors.
    async fn train(&mut self, data: &MatrixView<Float32Type>) -> Result<()>;

    /// Apply transform on the matrix `data`.
    ///
    /// Returns a new Matrix instead.
    async fn transform(&self, data: &FixedSizeListArray) -> Result<FixedSizeListArray>;

    async fn save(&self, writer: &mut ObjectWriter) -> Result<Transform>;
}
