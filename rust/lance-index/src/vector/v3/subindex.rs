// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fmt::Debug;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use snafu::{location, Location};

use crate::vector::storage::VectorStore;
use crate::vector::{flat, hnsw};
use crate::{prefilter::PreFilter, vector::Query};
/// A sub index for IVF index
pub trait IvfSubIndex: Send + Sync + Debug + DeepSizeOf {
    type QueryParams: Send + Sync + for<'a> From<&'a Query>;
    type BuildParams: Clone;

    /// Load the sub index from a record batch with a single row
    fn load(data: RecordBatch) -> Result<Self>
    where
        Self: Sized;

    fn use_residual() -> bool;

    fn name() -> &'static str;

    fn metadata_key() -> &'static str;

    /// Return the schema of the sub index
    fn schema() -> arrow_schema::SchemaRef;

    /// Search the sub index for nearest neighbors.
    /// # Arguments:
    /// * `query` - The query vector
    /// * `k` - The number of nearest neighbors to return
    /// * `params` - The query parameters
    /// * `prefilter` - The prefilter object indicating which vectors to skip
    fn search(
        &self,
        query: ArrayRef,
        k: usize,
        params: Self::QueryParams,
        storage: &impl VectorStore,
        prefilter: Arc<dyn PreFilter>,
    ) -> Result<RecordBatch>;

    /// Given a vector storage, containing all the data for the IVF partition, build the sub index.
    fn index_vectors(storage: &impl VectorStore, params: Self::BuildParams) -> Result<Self>
    where
        Self: Sized;

    /// Encode the sub index into a record batch
    fn to_batch(&self) -> Result<RecordBatch>;

    fn stats(&self) -> serde_json::Value;
}

pub enum SubIndexType {
    Flat,
    Hnsw,
}

impl std::fmt::Display for SubIndexType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Flat => write!(f, "{}", flat::index::FlatIndex::name()),
            Self::Hnsw => write!(f, "{}", hnsw::builder::HNSW::name()),
        }
    }
}

impl TryFrom<&str> for SubIndexType {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self> {
        match value {
            "FLAT" => Ok(Self::Flat),
            "HNSW" => Ok(Self::Hnsw),
            _ => Err(Error::Index {
                message: format!("unknown sub index type {}", value),
                location: location!(),
            }),
        }
    }
}
