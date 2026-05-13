// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! HNSW primitives for Lance's in-memory write path.
//!
//! This crate takes the hnswlib layout as the performance baseline but keeps
//! Lance-specific constraints explicit:
//!
//! - Vector data is supplied by a [`VectorSource`] instead of copied into the
//!   graph. [`ArrowFixedSizeListVectorStore`] is a MemTable-friendly
//!   implementation that holds `Arc<FixedSizeListArray>` references.
//! - The graph supports a multi-reader / single-writer lifecycle. Batch
//!   insertion may use worker threads internally, but a new contiguous id range
//!   is published to readers only after the writer operation completes.
//! - [`HnswGraph::to_lance_hnsw_batch`] emits the HNSW sub-index batch used by
//!   Lance's `IVF_HNSW_*` readers, so a MemTable flush can reuse the same
//!   on-disk format.

mod graph;
mod storage;

pub use graph::{
    BuildParams, HnswGraph, LanceHnswMetadata, ScoredPoint, SearchParams, SearchResult,
};
pub use storage::{
    ArrowFixedSizeListVectorStore, VectorSource, VectorStoreSnapshot, compute_f32_distance,
};
