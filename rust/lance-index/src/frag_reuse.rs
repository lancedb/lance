// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `Index`-trait adapter for the fragment-reuse system index.
//!
//! The data structures and table-format logic live in
//! [`lance_table::system_index::frag_reuse`]; this module re-exports them and
//! implements the local [`Index`] trait for [`FragReuseIndex`].

use std::any::Any;
use std::sync::Arc;

use arrow_array::RecordBatch;
use async_trait::async_trait;
use lance_core::{Error, Result};
use lance_select::RowAddrTreeMap;
use roaring::{RoaringBitmap, RoaringTreemap};
use serde::Serialize;

pub use lance_table::system_index::frag_reuse::*;

use crate::scalar::RowIdRemapper;
use crate::{Index, IndexType};

impl RowIdRemapper for FragReuseIndex {
    fn remap_row_id(&self, row_id: u64) -> Option<u64> {
        self.remap_row_id(row_id)
    }

    fn remap_row_addrs_tree_map(&self, row_addrs: &RowAddrTreeMap) -> RowAddrTreeMap {
        self.remap_row_addrs_tree_map(row_addrs)
    }

    fn remap_row_ids_roaring_tree_map(&self, row_ids: &RoaringTreemap) -> RoaringTreemap {
        self.remap_row_ids_roaring_tree_map(row_ids)
    }

    fn remap_row_ids_record_batch(
        &self,
        batch: RecordBatch,
        row_id_idx: usize,
    ) -> Result<RecordBatch> {
        self.remap_row_ids_record_batch(batch, row_id_idx)
    }
}

#[derive(Serialize)]
struct FragReuseStatistics {
    num_versions: usize,
}

#[async_trait]
impl Index for FragReuseIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        let stats = FragReuseStatistics {
            num_versions: self.details.versions.len(),
        };
        serde_json::to_value(stats).map_err(|e| {
            Error::internal(format!(
                "failed to serialize fragment reuse index statistics: {}",
                e
            ))
        })
    }

    async fn prewarm(&self) -> Result<()> {
        Ok(())
    }

    fn index_type(&self) -> IndexType {
        IndexType::FragmentReuse
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        unimplemented!()
    }
}
