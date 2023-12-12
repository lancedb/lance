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
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use async_trait::async_trait;
use futures::future::BoxFuture;
use futures::stream;
use futures::FutureExt;
use futures::StreamExt;
use futures::TryStreamExt;
use lance_core::format::Fragment;
use lance_core::utils::mask::RowIdMask;
use lance_core::utils::mask::RowIdTreeMap;
use roaring::RoaringBitmap;
use tracing::instrument;
use tracing::Instrument;

use crate::dataset::fragment::FileFragment;
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
        let fragments = index.fragment_bitmap.unwrap_or_else(|| {
            // If the index doesn't have a fragment bitmap then we don't know which fragments were covered
            // by the index so we have to be very conservative and figure that any fragment that has ever
            // been deleted might be part of the index
            let mut bitmap = RoaringBitmap::new();
            bitmap.insert_range(0..dataset.manifest.max_fragment_id);
            bitmap
        });
        let deleted_ids =
            Self::create_deletion_mask(dataset.clone(), fragments).map(SharedPrerequisite::spawn);
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
    async fn do_create_deletion_mask(
        dataset: Arc<Dataset>,
        missing_frags: Vec<u32>,
        frags_with_deletion_files: Vec<u32>,
    ) -> Result<Arc<RowIdTreeMap>> {
        let fragments = dataset.get_fragments();
        let frag_map: Arc<HashMap<u32, &FileFragment>> = Arc::new(HashMap::from_iter(
            fragments.iter().map(|frag| (frag.id() as u32, frag)),
        ));
        let frag_id_deletion_vectors = stream::iter(
            frags_with_deletion_files
                .iter()
                .map(|frag_id| (frag_id, frag_map.clone())),
        )
        .map(|(frag_id, frag_map)| async move {
            let frag = frag_map.get(frag_id).unwrap();
            frag.get_deletion_vector()
                .await
                .transpose()
                .unwrap()
                .map(|deletion_vector| (*frag_id, RoaringBitmap::from(deletion_vector.as_ref())))
        })
        .collect::<Vec<_>>()
        .await;
        let mut frag_id_deletion_vectors =
            stream::iter(frag_id_deletion_vectors).buffer_unordered(num_cpus::get());

        let mut deleted_ids = RowIdTreeMap::new();
        while let Some((id, deletion_vector)) = frag_id_deletion_vectors.try_next().await? {
            deleted_ids.insert_bitmap(id, deletion_vector);
        }

        for frag_id in missing_frags.into_iter() {
            deleted_ids.insert_fragment(frag_id);
        }
        Ok(Arc::new(deleted_ids))
    }

    /// Creates a task to load deleted row ids in `fragments`
    ///
    /// If it can be synchronously determined that there are no missing row ids then
    /// this function return None
    pub fn create_deletion_mask(
        dataset: Arc<Dataset>,
        fragments: RoaringBitmap,
    ) -> Option<BoxFuture<'static, Result<Arc<RowIdTreeMap>>>> {
        let mut missing_frags = Vec::new();
        let mut frags_with_deletion_files = Vec::new();
        let frag_map: HashMap<u32, &Fragment> = HashMap::from_iter(
            dataset
                .manifest
                .fragments
                .iter()
                .map(|frag| (frag.id as u32, frag)),
        );
        for frag_id in fragments.iter() {
            let frag = frag_map.get(&frag_id);
            if let Some(frag) = frag {
                if frag.deletion_file.is_some() {
                    frags_with_deletion_files.push(frag_id);
                }
            } else {
                missing_frags.push(frag_id);
            }
        }
        if missing_frags.is_empty() && frags_with_deletion_files.is_empty() {
            None
        } else {
            Some(
                Self::do_create_deletion_mask(dataset, missing_frags, frags_with_deletion_files)
                    .boxed(),
            )
        }
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
