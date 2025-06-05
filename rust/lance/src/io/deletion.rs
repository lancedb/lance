// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::io::commit::deletion_file_cache_key;
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
    let cache_key = deletion_file_cache_key(fragment_id, deletion_file);
    dataset
        .metadata_cache
        .get_or_insert(cache_key, |_| {
            read_deletion_file(
                fragment_id,
                deletion_file,
                &dataset.base,
                dataset.object_store.as_ref(),
            )
        })
        .await
}
