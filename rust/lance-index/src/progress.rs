// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use async_trait::async_trait;
use lance_core::Result;
use std::sync::Arc;

/// Progress callback for index building.
///
/// Called at stage boundaries during index construction. Stages are sequential:
/// `stage_complete` is always called before the next `stage_start`, so only one
/// stage is active at a time. Stage names are index-type-specific (e.g.
/// "train_ivf", "shuffle", "build_partitions" for vector indices; "load_data",
/// "build_pages" for scalar indices).
///
/// Methods take `&self` to allow concurrent calls from within a single stage.
/// Implementations must be thread-safe.
#[async_trait]
pub trait IndexBuildProgress: std::fmt::Debug + Sync + Send {
    /// A named stage has started.
    ///
    /// `total` is the number of work units if known, and `unit` describes
    /// what is being counted (e.g. "partitions", "batches", "rows").
    async fn stage_start(&self, stage: &str, total: Option<u64>, unit: &str) -> Result<()>;

    /// Progress within the current stage.
    async fn stage_progress(&self, stage: &str, completed: u64) -> Result<()>;

    /// A named stage has completed.
    async fn stage_complete(&self, stage: &str) -> Result<()>;
}

#[derive(Debug, Clone, Default)]
pub struct NoopIndexBuildProgress;

#[async_trait]
impl IndexBuildProgress for NoopIndexBuildProgress {
    async fn stage_start(&self, _: &str, _: Option<u64>, _: &str) -> Result<()> {
        Ok(())
    }
    async fn stage_progress(&self, _: &str, _: u64) -> Result<()> {
        Ok(())
    }
    async fn stage_complete(&self, _: &str) -> Result<()> {
        Ok(())
    }
}

/// Helper to create a default noop progress instance.
pub fn noop_progress() -> Arc<dyn IndexBuildProgress> {
    Arc::new(NoopIndexBuildProgress)
}
