// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::session::caches::DeletionFileKey;
use crate::Dataset;
use lance_core::utils::deletion::DeletionVector;
use lance_table::format::DeletionFile;
use lance_table::io::deletion::read_deletion_file;
use std::sync::Arc;

pub async fn read_dataset_deletion_file(
    dataset: &Dataset,
    fragment_id: u64,
    deletion_file: &DeletionFile,
) -> lance_core::Result<Arc<DeletionVector>> {
    let key = DeletionFileKey {
        fragment_id,
        deletion_file,
    };

    if let Some(cached) = dataset.metadata_cache.get_with_key(&key).await {
        Ok(cached)
    } else {
        let deletion_vector = Arc::new(
            read_deletion_file(
                fragment_id,
                deletion_file,
                &dataset.base,
                dataset.object_store.as_ref(),
            )
            .await?,
        );

        dataset
            .metadata_cache
            .insert_with_key(&key, deletion_vector.clone())
            .await;

        Ok(deletion_vector)
    }
}
