use crate::Dataset;
use lance_core::utils::deletion::DeletionVector;
use lance_table::format::DeletionFile;
use lance_table::io::deletion::{deletion_file_path, read_deletion_file};
use std::sync::Arc;

pub async fn read_dataset_deletion_file(
    dataset: &Dataset,
    fragment_id: u64,
    deletion_file: &DeletionFile,
) -> lance_core::Result<Arc<DeletionVector>> {
    let path = deletion_file_path(&dataset.base, fragment_id, deletion_file);
    dataset
        .session
        .file_metadata_cache
        .get_or_insert(&path, |_| {
            read_deletion_file(
                fragment_id,
                deletion_file,
                &dataset.base,
                dataset.object_store.as_ref(),
            )
        })
        .await
}
