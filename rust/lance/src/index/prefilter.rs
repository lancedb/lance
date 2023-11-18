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

use std::cell::OnceCell;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::Mutex;

use async_trait::async_trait;
use futures::future;
use futures::stream;
use futures::StreamExt;
use lance_core::utils::mask::RowIdMask;
use lance_core::utils::mask::RowIdTreeMap;
use roaring::RoaringBitmap;
use tracing::instrument;
use tracing::Instrument;

use crate::error::Result;
use crate::format::Index;
use crate::utils::future::SharedPrerequisite;
use crate::Dataset;

/// A trait to be implemented by anything supplying a prefilter row id mask
#[async_trait]
pub trait FilterLoader: Send + 'static {
    async fn load(self: Box<Self>) -> Result<RowIdMask>;
}

/// Filter out row ids that we know are not relevant to the query.
///
/// This could be both rows that are deleted or a prefilter
/// that should be applied to the search
pub struct PreFilter {
    // Expressing these as tasks allows us to start calculating the block list
    // and allow list at the same time we start searching the query.  We will await
    // these tasks only when we've done as much work as we can without them.
    deleted_ids: Option<Arc<SharedPrerequisite<Arc<RowIdTreeMap>>>>,
    filtered_ids: Option<Arc<SharedPrerequisite<RowIdMask>>>,
    // When the tasks are finished this is the combined filter
    final_mask: Mutex<OnceCell<RowIdMask>>,
}

impl PreFilter {
    pub fn new(dataset: Arc<Dataset>, index: Index, filter: Option<Box<dyn FilterLoader>>) -> Self {
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
        let deleted_ids = if has_missing_fragments || has_deletion_vectors {
            Some(SharedPrerequisite::spawn(
                Self::load_deleted_ids(dataset_clone, index).in_current_span(),
            ))
        } else {
            None
        };
        let filtered_ids = filter
            .map(|filtered_ids| SharedPrerequisite::spawn(filtered_ids.load().in_current_span()));
        Self {
            deleted_ids,
            filtered_ids,
            final_mask: Mutex::new(OnceCell::new()),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.deleted_ids.is_none() && self.filtered_ids.is_none()
    }

    /// Check whether a single row id should be included in the query.
    pub fn check_one(&self, row_id: u64) -> bool {
        let final_mask = self.final_mask.lock().unwrap();
        final_mask
            .get()
            .expect("check_one called before wait_ready")
            .selected(row_id)
    }

    #[instrument(level = "debug", skip_all)]
    async fn load_deleted_ids(dataset: Arc<Dataset>, index: Index) -> Result<Arc<RowIdTreeMap>> {
        let fragments = dataset.get_fragments();
        let frag_id_deletion_vectors = stream::iter(fragments.iter())
            .map(|frag| async move {
                let id = frag.id() as u32;
                let deletion_vector = frag.get_deletion_vector().await;
                (id, deletion_vector)
            })
            .collect::<Vec<_>>()
            .await;
        let mut frag_id_deletion_vectors = stream::iter(frag_id_deletion_vectors)
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
            });

        let mut deleted_ids = RowIdTreeMap::new();
        while let Some(res) = frag_id_deletion_vectors.next().await {
            let (id, deletion_vector) = res?;
            deleted_ids.insert_bitmap(id, deletion_vector);
        }

        let frag_ids_in_dataset: HashSet<u32> =
            HashSet::from_iter(fragments.iter().map(|frag| frag.id() as u32));
        if let Some(fragment_bitmap) = index.fragment_bitmap {
            for frag_id in fragment_bitmap.into_iter() {
                if !frag_ids_in_dataset.contains(&frag_id) {
                    // Entire fragment has been deleted
                    deleted_ids.insert_fragment(frag_id);
                }
            }
        }
        Ok(Arc::new(deleted_ids))
    }

    /// Waits for the prefilter to be fully loaded
    ///
    /// The prefilter loads in the background while the rest of the index
    /// search is running.  When you are ready to use the prefilter you
    /// must first call this method to ensure it is fully loaded.  This
    /// allows `filter_row_ids` to be a synchronous method.
    pub async fn wait_for_ready(&self) -> Result<()> {
        if let Some(filtered_ids) = &self.filtered_ids {
            filtered_ids.wait_ready().await?;
        }
        if let Some(deleted_ids) = &self.deleted_ids {
            deleted_ids.wait_ready().await?;
        }
        let final_mask = self.final_mask.lock().unwrap();
        final_mask.get_or_init(|| {
            let mut combined = RowIdMask::default();
            if let Some(filtered_ids) = &self.filtered_ids {
                combined = combined & filtered_ids.get_ready();
            }
            if let Some(deleted_ids) = &self.deleted_ids {
                combined = combined.also_block((*deleted_ids.get_ready()).clone());
            }
            combined
        });
        Ok(())
    }

    /// Check whether a slice of row ids should be included in a query.
    ///
    /// Returns a vector of indices into the input slice that should be included,
    /// also known as a selection vector.
    ///
    /// This method must be called after `wait_for_ready`
    #[instrument(level = "debug", skip_all)]
    pub fn filter_row_ids(&self, row_ids: &[u64]) -> Vec<u64> {
        let final_mask = self.final_mask.lock().unwrap();
        final_mask
            .get()
            .expect("filter_row_ids called without call to wait_for_ready")
            .selected_indices(row_ids)
    }
}
