// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::dataset::transaction::{Operation, Transaction};
use crate::index::frag_reuse::{build_frag_reuse_index_metadata, load_frag_reuse_index_details};
use crate::Dataset;
use lance_core::Error;
use lance_index::frag_reuse::{FragReuseIndexDetails, FragReuseVersion, FRAG_REUSE_INDEX_NAME};
use lance_index::is_system_index;
use lance_table::format::IndexMetadata;
use lance_table::io::manifest::read_manifest_indexes;
use log::warn;
use roaring::RoaringBitmap;
use snafu::location;

/// Cleanup a fragment reuse index based on the current condition of the indices.
/// If all the indices currently available are already caught up to as a specific reuse version,
/// all older reuse versions (inclusive) can be cleaned up.
///
/// An index is considered caught up against a specific reuse version if
/// 1. the index is created after or at the same dataset version as the reuse version
/// 2. there is no old fragment in the version that is covered by the index and can be remapped.
///    If an index's fragment bitmap is missing, we will consider it as caught up.
///    Otherwise, we will never be able to clean up the reuse version.
///
/// Note that there could be a race condition that an index is being added during the cleanup,
/// This will make that specific index not efficient until the next reindex,
/// but it will not cause any correctness problem.
pub async fn cleanup_frag_reuse_index(dataset: &mut Dataset) -> lance_core::Result<()> {
    // check against index metadata before auto-remap
    let indices = read_manifest_indexes(
        &dataset.object_store,
        &dataset.manifest_location,
        &dataset.manifest,
    )
    .await?;
    let Some(frag_reuse_index_meta) = indices.iter().find(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
    else {
        return Ok(());
    };

    let frag_reuse_details = load_frag_reuse_index_details(dataset, frag_reuse_index_meta)
        .await
        .unwrap();

    let mut retained_versions = Vec::new();
    let mut fragment_bitmaps = RoaringBitmap::new();
    for version in frag_reuse_details.versions.iter() {
        let check_results = indices
            .iter()
            .map(|idx| is_index_remap_caught_up(version, idx))
            .collect::<Vec<_>>();

        if check_results
            .iter()
            .any(|r| matches!(r, Err(Error::InvalidInput { .. })))
        {
            // If the check fails, the reuse version is likely corrupted, do not retain it.
            continue;
        }

        if !check_results.into_iter().all(|r| r.unwrap()) {
            fragment_bitmaps.extend(version.new_frag_bitmap());
            retained_versions.push(version.clone());
        }
    }

    // Return early if there is nothing to cleanup
    if retained_versions.len() == frag_reuse_details.versions.len() {
        return Ok(());
    }

    let frag_reuse_index_details = FragReuseIndexDetails {
        versions: retained_versions,
    };

    let new_index_meta = build_frag_reuse_index_metadata(
        dataset,
        Some(frag_reuse_index_meta),
        frag_reuse_index_details,
        fragment_bitmaps,
    )
    .await?;

    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::CreateIndex {
            new_indices: vec![new_index_meta],
            removed_indices: vec![frag_reuse_index_meta.clone()],
        },
        None,
    );

    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await?;

    Ok(())
}

fn is_index_remap_caught_up(
    frag_reuse_version: &FragReuseVersion,
    index_meta: &IndexMetadata,
) -> lance_core::Result<bool> {
    if is_system_index(index_meta) {
        return Ok(true);
    }

    if index_meta.dataset_version < frag_reuse_version.dataset_version {
        return Ok(false);
    }

    match index_meta.fragment_bitmap.clone() {
        Some(index_frag_bitmap) => {
            for group in frag_reuse_version.groups.iter() {
                let mut old_frag_in_index = 0;
                for old_frag in group.old_frags.iter() {
                    if index_frag_bitmap.contains(old_frag.id as u32) {
                        old_frag_in_index += 1;
                    }
                }

                if old_frag_in_index > 0 {
                    if old_frag_in_index != group.old_frags.len() {
                        // This should never happen because we always commit a full rewrite group
                        // and we always reindex either the entire group or nothing.
                        // We use invalid input to be consistent with
                        // dataset::transaction::recalculate_fragment_bitmap
                        return Err(Error::invalid_input(
                            format!("The compaction plan included a rewrite group that was a split of indexed and non-indexed data: {:?}",
                                    group.old_frags),
                            location!()));
                    }
                    return Ok(false);
                }
            }
            Ok(true)
        }
        None => {
            warn!(
                "Index {} ({}) missing fragment bitmap, cannot determine if it is caught up with the fragment reuse version, consider retraining the index",
                index_meta.name, index_meta.uuid
            );
            Ok(true)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::optimize::{compact_files, remapping, CompactionOptions};
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use all_asserts::{assert_false, assert_true};
    use arrow_array::types::{Float32Type, Int32Type};
    use lance_datagen::Dimension;
    use lance_index::scalar::ScalarIndexParams;
    use lance_index::{DatasetIndexExt, IndexType};

    #[tokio::test]
    async fn test_cleanup_frag_reuse_index() {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
            .await
            .unwrap();

        // Create an index to be remapped
        let index_name = Some("scalar".into());
        dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                index_name.clone(),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // Compact and check index not caught up
        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 2_000,
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        let Some(frag_reuse_index_meta) = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
        else {
            panic!("Fragment reuse index must be available");
        };
        let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap();
        assert_eq!(frag_reuse_details.versions.len(), 1);
        let indices = dataset.load_indices().await.unwrap();
        let scalar_index = indices.iter().find(|idx| idx.name == "scalar").unwrap();
        // Should not be considered caught up because index was created at an old dataset version
        assert_false!(
            is_index_remap_caught_up(&frag_reuse_details.versions[0], scalar_index).unwrap()
        );

        // Remap and check index is caught up
        remapping::remap_column_index(&mut dataset, &["i"], index_name.clone())
            .await
            .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let scalar_index = indices.iter().find(|idx| idx.name == "scalar").unwrap();
        assert_true!(
            is_index_remap_caught_up(&frag_reuse_details.versions[0], scalar_index).unwrap()
        );

        // Cleanup frag reuse index and check there is no reuse version
        let mut dataset_clone = dataset.clone();
        cleanup_frag_reuse_index(&mut dataset).await.unwrap();
        let Some(frag_reuse_index_meta) = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
        else {
            panic!("Fragment reuse index must be available");
        };
        let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap();
        assert_eq!(frag_reuse_details.versions.len(), 0);

        // Try doing a concurrent cleanup should fail with conflict
        assert!(matches!(
            cleanup_frag_reuse_index(&mut dataset_clone).await,
            Err(Error::RetryableCommitConflict { .. })
        ));
    }
}
