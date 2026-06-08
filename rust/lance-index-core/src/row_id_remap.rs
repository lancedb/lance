// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::RecordBatch;
use lance_core::Result;
use lance_select::RowAddrTreeMap;
use roaring::RoaringTreemap;

/// Trait for remapping row IDs at index load time.
///
/// When fragments are compacted after an index is built, the row IDs stored
/// in that index become stale. Implementors of this trait know how to map
/// an old row ID to the current row ID (or `None` if the row was deleted).
///
/// This is injected into index loading so that indices can update their
/// in-memory state without being rebuilt.
pub trait RowIdRemapper: Send + Sync + std::fmt::Debug {
    fn remap_row_id(&self, row_id: u64) -> Option<u64>;
    fn remap_row_addrs_tree_map(&self, row_addrs: &RowAddrTreeMap) -> RowAddrTreeMap;
    fn remap_row_ids_roaring_tree_map(&self, row_ids: &RoaringTreemap) -> RoaringTreemap;
    fn remap_row_ids_record_batch(
        &self,
        batch: RecordBatch,
        row_id_idx: usize,
    ) -> Result<RecordBatch>;
}
