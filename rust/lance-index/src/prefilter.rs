// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use async_trait::async_trait;
use lance_core::utils::mask::RowIdMask;
use lance_core::Result;

/// A trait to be implemented by anything supplying a prefilter row id mask
///
/// This trait is for internal use only and has no stability guarantees.
#[async_trait]
pub trait FilterLoader: Send + 'static {
    async fn load(self: Box<Self>) -> Result<RowIdMask>;
}

/// Filter out row ids that we know are not relevant to the query.
///
/// This could be both rows that are deleted or a prefilter
/// that should be applied to the search
///
/// <section class="warning">
/// Internal use only. No API stability guarantees.
/// </section>
#[async_trait]
pub trait PreFilter: Send + Sync {
    /// Waits for the prefilter to be fully loaded
    ///
    /// The prefilter loads in the background while the rest of the index
    /// search is running.  When you are ready to use the prefilter you
    /// must first call this method to ensure it is fully loaded.  This
    /// allows `filter_row_ids` to be a synchronous method.
    async fn wait_for_ready(&self) -> Result<()>;

    /// If the filter is empty.
    fn is_empty(&self) -> bool;

    /// Check whether a slice of row ids should be included in a query.
    ///
    /// Returns a vector of indices into the input slice that should be included,
    /// also known as a selection vector.
    ///
    /// This method must be called after `wait_for_ready`
    fn filter_row_ids(&self, row_ids: &[u64]) -> Vec<u64>;
}
