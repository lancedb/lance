// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

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
use lance_core::utils::mask::RowAddressTreeMap;
use lance_core::utils::mask::RowIdMask;
use lance_table::format::Fragment;
use lance_table::format::Index;
use roaring::RoaringBitmap;
use tokio::join;
use tracing::instrument;
use tracing::Instrument;

use crate::dataset::fragment::FileFragment;
use crate::dataset::rowids::load_row_id_sequence;
use crate::error::Result;
use crate::utils::future::SharedPrerequisite;
use crate::Dataset;

pub use lance_index::prefilter::{FilterLoader, PreFilter};

/// Filter out row ids that we know are not relevant to the query.
///
/// This could be both rows that are deleted or a prefilter
/// that should be applied to the search
///
/// This struct is for internal use only and has no stability guarantees.
pub struct DatasetPreFilter {
    // Expressing these as tasks allows us to start calculating the block list
    // and allow list at the same time we start searching the query.  We will await
    // these tasks only when we've done as much work as we can without them.
    pub(super) deleted_ids: Option<Arc<SharedPrerequisite<Arc<RowIdMask>>>>,
    pub(super) filtered_ids: Option<Arc<SharedPrerequisite<RowIdMask>>>,
    // When the tasks are finished this is the combined filter
    pub(super) final_mask: Mutex<OnceCell<RowIdMask>>,
}

impl DatasetPreFilter {
    pub fn new(
        dataset: Arc<Dataset>,
        indices: &[Index],
        filter: Option<Box<dyn FilterLoader>>,
    ) -> Self {
        let mut fragments = RoaringBitmap::new();
        if indices.iter().any(|idx| idx.fragment_bitmap.is_none()) {
            fragments.insert_range(0..dataset.manifest.max_fragment_id);
        } else {
            indices.iter().for_each(|idx| {
                fragments |= idx.fragment_bitmap.as_ref().unwrap();
            });
        }
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

    #[instrument(level = "debug", skip_all)]
    async fn do_create_deletion_mask(
        dataset: Arc<Dataset>,
        missing_frags: Vec<u32>,
        frags_with_deletion_files: Vec<u32>,
    ) -> Result<Arc<RowIdMask>> {
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

        let mut deleted_ids = RowAddressTreeMap::new();
        while let Some((id, deletion_vector)) = frag_id_deletion_vectors.try_next().await? {
            deleted_ids.insert_bitmap(id, deletion_vector);
        }

        for frag_id in missing_frags.into_iter() {
            deleted_ids.insert_fragment(frag_id);
        }
        Ok(Arc::new(RowIdMask::from_block(deleted_ids)))
    }

    #[instrument(level = "debug", skip_all)]
    async fn do_create_deletion_mask_row_id(dataset: Arc<Dataset>) -> Result<Arc<RowIdMask>> {
        // This can only be computed as an allow list, since we have no idea
        // what the row ids were in the missing fragments.

        // For each fragment, compute which row ids are still in use.
        let dataset_ref = dataset.as_ref();

        let row_ids_and_deletions = stream::iter(dataset.get_fragments())
            .map(|frag| async move {
                let row_ids = load_row_id_sequence(dataset_ref, frag.metadata());
                let deletion_vector = frag.get_deletion_vector();
                let (row_ids, deletion_vector) = join!(row_ids, deletion_vector);
                Ok::<_, crate::Error>((row_ids?, deletion_vector?))
            })
            .buffer_unordered(10)
            .try_collect::<Vec<_>>()
            .await?;

        // The process of computing the final mask is CPU-bound, so we spawn it
        // on a blocking thread.
        let allow_list = tokio::task::spawn_blocking(move || {
            row_ids_and_deletions.into_iter().fold(
                RowAddressTreeMap::new(),
                |mut allow_list, (row_ids, deletion_vector)| {
                    let row_ids = if let Some(deletion_vector) = deletion_vector {
                        // We have to mask the row ids
                        row_ids.as_ref().iter().enumerate().fold(
                            RowAddressTreeMap::new(),
                            |mut allow_list, (idx, row_id)| {
                                if !deletion_vector.contains(idx as u32) {
                                    allow_list.insert(row_id);
                                }
                                allow_list
                            },
                        )
                    } else {
                        // Can do a direct translation
                        RowAddressTreeMap::from(row_ids.as_ref())
                    };
                    allow_list |= row_ids;
                    allow_list
                },
            )
        })
        .await?;

        // TODO: We should see if we can cache this for the dataset, since it should
        // be re-usable.
        Ok(Arc::new(RowIdMask::from_allowed(allow_list)))
    }

    /// Creates a task to load mask to filter out deleted rows.
    ///
    /// Sometimes this will be a block list of row ids that are deleted, based
    /// on the deletion files in the fragments. If stable row ids are used and
    /// there are missing fragments, this may instead be an allow list, since
    /// we can't easily compute the block list.
    ///
    /// If it can be synchronously determined that there are no missing row ids then
    /// this function return None
    pub fn create_deletion_mask(
        dataset: Arc<Dataset>,
        fragments: RoaringBitmap,
    ) -> Option<BoxFuture<'static, Result<Arc<RowIdMask>>>> {
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
        } else if dataset.manifest.uses_move_stable_row_ids() {
            Some(Self::do_create_deletion_mask_row_id(dataset.clone()).boxed())
        } else {
            Some(
                Self::do_create_deletion_mask(dataset, missing_frags, frags_with_deletion_files)
                    .boxed(),
            )
        }
    }
}

#[async_trait]
impl PreFilter for DatasetPreFilter {
    /// Waits for the prefilter to be fully loaded
    ///
    /// The prefilter loads in the background while the rest of the index
    /// search is running.  When you are ready to use the prefilter you
    /// must first call this method to ensure it is fully loaded.  This
    /// allows `filter_row_ids` to be a synchronous method.
    async fn wait_for_ready(&self) -> Result<()> {
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
                combined = combined & (*deleted_ids.get_ready()).clone();
            }
            combined
        });

        Ok(())
    }

    fn is_empty(&self) -> bool {
        self.deleted_ids.is_none() && self.filtered_ids.is_none()
    }

    /// Check whether a slice of row ids should be included in a query.
    ///
    /// Returns a vector of indices into the input slice that should be included,
    /// also known as a selection vector.
    ///
    /// This method must be called after `wait_for_ready`
    #[instrument(level = "debug", skip_all)]
    fn filter_row_ids<'a>(&self, row_ids: Box<dyn Iterator<Item = &'a u64> + 'a>) -> Vec<u64> {
        let final_mask = self.final_mask.lock().unwrap();
        final_mask
            .get()
            .expect("filter_row_ids called without call to wait_for_ready")
            .selected_indices(row_ids)
    }
}

#[cfg(test)]
mod test {
    use lance_testing::datagen::{BatchGenerator, IncrementingInt32};

    use crate::dataset::WriteParams;

    use super::*;

    struct TestDatasets {
        no_deletions: Arc<Dataset>,
        deletions_no_missing_frags: Arc<Dataset>,
        deletions_missing_frags: Arc<Dataset>,
        only_missing_frags: Arc<Dataset>,
    }

    async fn test_datasets(use_stable_row_id: bool) -> TestDatasets {
        let test_data = BatchGenerator::new()
            .col(Box::new(IncrementingInt32::new().named("x")))
            .batch(9);
        let mut dataset = Dataset::write(
            test_data,
            "memory://test",
            Some(WriteParams {
                max_rows_per_file: 3,
                enable_move_stable_row_ids: use_stable_row_id,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let no_deletions = Arc::new(dataset.clone());

        // This will add a deletion file.
        dataset.delete("x = 8").await.unwrap();
        let deletions_no_missing_frags = Arc::new(dataset.clone());

        dataset.delete("x >= 3 and x <= 5").await.unwrap();
        assert_eq!(dataset.get_fragments().len(), 2);
        let deletions_missing_frags = Arc::new(dataset.clone());

        dataset.delete("x >= 3").await.unwrap();
        assert_eq!(dataset.get_fragments().len(), 1);
        assert!(dataset.get_fragments()[0]
            .metadata()
            .deletion_file
            .is_none());
        let only_missing_frags = Arc::new(dataset.clone());

        TestDatasets {
            no_deletions,
            deletions_no_missing_frags,
            deletions_missing_frags,
            only_missing_frags,
        }
    }

    #[tokio::test]
    async fn test_deletion_mask() {
        let datasets = test_datasets(false).await;

        // If there are no deletions, we should get None
        let mask = DatasetPreFilter::create_deletion_mask(
            datasets.no_deletions.clone(),
            RoaringBitmap::from_iter(0..3),
        );
        assert!(mask.is_none());

        // If there are deletions, we should get a mask
        let mask = DatasetPreFilter::create_deletion_mask(
            datasets.deletions_no_missing_frags.clone(),
            RoaringBitmap::from_iter(0..3),
        );
        assert!(mask.is_some());
        let mask = mask.unwrap().await.unwrap();
        assert_eq!(mask.block_list.as_ref().and_then(|x| x.len()), Some(1)); // There was just one row deleted.

        // If there are deletions and missing fragments, we should get a mask
        let mask = DatasetPreFilter::create_deletion_mask(
            datasets.deletions_missing_frags.clone(),
            RoaringBitmap::from_iter(0..3),
        );
        assert!(mask.is_some());
        let mask = mask.unwrap().await.unwrap();
        let mut expected = RowAddressTreeMap::from_iter(vec![(2 << 32) + 2]);
        expected.insert_fragment(1);
        assert_eq!(&mask.block_list, &Some(expected));

        // If we don't pass the missing fragment id, we should get a smaller mask.
        let mask = DatasetPreFilter::create_deletion_mask(
            datasets.deletions_missing_frags.clone(),
            RoaringBitmap::from_iter(2..3),
        );
        assert!(mask.is_some());
        let mask = mask.unwrap().await.unwrap();
        assert_eq!(mask.block_list.as_ref().and_then(|x| x.len()), Some(1));

        // If there are only missing fragments, we should still get a mask
        let mask = DatasetPreFilter::create_deletion_mask(
            datasets.only_missing_frags.clone(),
            RoaringBitmap::from_iter(0..3),
        );
        assert!(mask.is_some());
        let mask = mask.unwrap().await.unwrap();
        let mut expected = RowAddressTreeMap::new();
        expected.insert_fragment(1);
        expected.insert_fragment(2);
        assert_eq!(&mask.block_list, &Some(expected));
    }

    #[tokio::test]
    async fn test_deletion_mask_stable_row_id() {
        // Here, behavior is different.
        let datasets = test_datasets(true).await;

        // If there are no deletions, we should get None
        let mask = DatasetPreFilter::create_deletion_mask(
            datasets.no_deletions.clone(),
            RoaringBitmap::from_iter(0..3),
        );
        assert!(mask.is_none());

        // If there are deletions but no missing files, we should get a block list
        let mask = DatasetPreFilter::create_deletion_mask(
            datasets.deletions_no_missing_frags.clone(),
            RoaringBitmap::from_iter(0..3),
        );
        assert!(mask.is_some());
        let mask = mask.unwrap().await.unwrap();
        let expected = RowAddressTreeMap::from_iter(0..8);
        assert_eq!(mask.allow_list, Some(expected)); // There was just one row deleted.

        // If there are deletions and missing fragments, we should get an allow list
        let mask = DatasetPreFilter::create_deletion_mask(
            datasets.deletions_missing_frags.clone(),
            RoaringBitmap::from_iter(0..2),
        );
        assert!(mask.is_some());
        let mask = mask.unwrap().await.unwrap();
        assert_eq!(mask.allow_list.as_ref().and_then(|x| x.len()), Some(5)); // There were five rows left over;

        // If there are only missing fragments, we should get an allow list
        let mask = DatasetPreFilter::create_deletion_mask(
            datasets.only_missing_frags.clone(),
            RoaringBitmap::from_iter(0..3),
        );
        assert!(mask.is_some());
        let mask = mask.unwrap().await.unwrap();
        assert_eq!(mask.allow_list.as_ref().and_then(|x| x.len()), Some(3)); // There were three rows left over;
    }
}
