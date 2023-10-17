// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Secondary Index pre-filter
//!
//! Based on the query, we might have information about which fragment ids and
//! row ids can be excluded from the search.

use std::collections::HashSet;
use std::sync::Arc;

use futures::future;
use futures::stream;
use futures::{StreamExt, TryStreamExt};
use roaring::{RoaringBitmap, RoaringTreemap};
use tracing::instrument;
use tracing::Instrument;

use crate::error::Result;
use crate::format::Index;
use crate::format::RowAddress;
use crate::utils::future::SharedPrerequisite;
use crate::Dataset;

/// Filter out row ids that we know are not relevant to the query. This currently
/// is just deleted rows.
pub struct PreFilter {
    dataset: Arc<Dataset>,
    has_deletion_vectors: bool,
    has_missing_fragments: bool,
    block_list: Arc<SharedPrerequisite<Arc<RoaringTreemap>>>,
}

impl PreFilter {
    pub fn new(dataset: Arc<Dataset>, index: Index) -> Self {
        let dataset_ref = dataset.as_ref();
        let mut has_fragment = Vec::new();
        let mut has_deletion_vectors = false;
        has_fragment.resize(
            (dataset
                .manifest
                .max_fragment_id()
                .map(|id| id + 1)
                .unwrap_or(0)) as usize,
            false,
        );
        for frag in dataset_ref.manifest.fragments.iter() {
            has_fragment[frag.id as usize] = true;
            has_deletion_vectors |= frag.deletion_file.is_some();
        }
        let has_missing_fragments = has_fragment.iter().any(|&x| !x);
        let dataset_clone = dataset.clone();
        let block_list = SharedPrerequisite::spawn(
            Self::load_block_list(dataset_clone, index).in_current_span(),
        );
        Self {
            dataset,
            has_deletion_vectors,
            has_missing_fragments,
            block_list,
        }
    }

    pub fn is_empty(&self) -> bool {
        !self.has_deletion_vectors && !self.has_missing_fragments
    }

    /// Check whether a single row id should be included in the query.
    pub async fn check_one(&self, row_id: u64) -> Result<bool> {
        let fragment_id = (row_id >> 32) as u32;
        // If the fragment isn't found, then it must have been deleted.
        let Some(fragment) = self.dataset.get_fragment(fragment_id as usize) else {
            return Ok(false);
        };
        // If the fragment has no deletion vector, then the row must be there.
        let Some(deletion_vector) = fragment.get_deletion_vector().await? else {
            return Ok(true);
        };
        let local_row_id = row_id as u32;
        Ok(!deletion_vector.contains(local_row_id))
    }

    #[instrument(level = "debug", skip_all)]
    async fn load_block_list(dataset: Arc<Dataset>, index: Index) -> Result<Arc<RoaringTreemap>> {
        let fragments = dataset.get_fragments();
        let frag_id_deletion_vectors = stream::iter(fragments.iter())
            .map(|frag| async move {
                let id = frag.id() as u32;
                let deletion_vector = frag.get_deletion_vector().await;
                (id, deletion_vector)
            })
            .collect::<Vec<_>>()
            .await;
        let frag_id_deletion_vectors = stream::iter(frag_id_deletion_vectors)
            .buffer_unordered(num_cpus::get())
            .filter_map(|(id, maybe_deletion_vector)| {
                let val = if let Ok(deletion_vector) = maybe_deletion_vector {
                    deletion_vector.map(|deletion_vector| {
                        Ok((id, RoaringBitmap::from(deletion_vector.as_ref())))
                    })
                } else {
                    Some(Err(maybe_deletion_vector.unwrap_err()))
                };
                future::ready(val)
            })
            .try_collect::<Vec<_>>()
            .await?;
        let mut block_list = RoaringTreemap::from_bitmaps(frag_id_deletion_vectors);

        let frag_ids_in_dataset: HashSet<u32> =
            HashSet::from_iter(fragments.iter().map(|frag| frag.id() as u32));
        if let Some(fragment_bitmap) = index.fragment_bitmap {
            for frag_id in fragment_bitmap.into_iter() {
                if !frag_ids_in_dataset.contains(&frag_id) {
                    // Entire fragment has been deleted
                    block_list.insert_range(RowAddress::fragment_range(frag_id));
                }
            }
        }
        Ok(Arc::new(block_list))
    }

    /// Waits for the prefilter to be fully loaded
    ///
    /// The prefilter loads in the background while the rest of the index
    /// search is running.  When you are ready to use the prefilter you
    /// must first call this method to ensure it is fully loaded.  This
    /// allows `filter_row_ids` to be a synchronous method.
    pub async fn wait_for_ready(&self) -> Result<()> {
        self.block_list.wait_ready().await
    }

    /// Check whether a slice of row ids should be included in a query.
    ///
    /// Returns a vector of indices into the input slice that should be included,
    /// also known as a selection vector.
    ///
    /// This method must be called after `wait_for_ready`
    #[instrument(level = "debug", skip_all)]
    pub fn filter_row_ids(&self, row_ids: &[u64]) -> Vec<u64> {
        let block_list = self.block_list.get_ready();
        row_ids
            .iter()
            .enumerate()
            .filter(|(_, row_id)| !block_list.contains(**row_id))
            .map(|(idx, _)| idx as u64)
            .collect::<Vec<_>>()
    }
}
