// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! MemWAL Index operations.
//!
//! The MemWAL Index stores:
//! - Configuration (region_specs, maintained_indexes)
//! - Merge progress (merged_generations per region)
//! - Region state snapshots (eventually consistent)
//!
//! Writers no longer update the index on every write. Instead, they update
//! region manifests directly. This module provides functions to:
//! - Load the MemWAL index
//! - Update merged generations (called during merge-insert commits)

use std::sync::Arc;

use lance_core::{Error, Result};
use lance_index::mem_wal::{MemWalIndex, MemWalIndexDetails, MergedGeneration, MEM_WAL_INDEX_NAME};
use lance_table::format::{pb, IndexMetadata};
use snafu::location;
use uuid::Uuid;

/// Load MemWalIndexDetails from an IndexMetadata.
pub(crate) fn load_mem_wal_index_details(index: IndexMetadata) -> Result<MemWalIndexDetails> {
    if let Some(details_any) = index.index_details.as_ref() {
        if !details_any.type_url.ends_with("MemWalIndexDetails") {
            return Err(Error::Index {
                message: format!(
                    "Index details is not for the MemWAL index, but {}",
                    details_any.type_url
                ),
                location: location!(),
            });
        }

        Ok(MemWalIndexDetails::try_from(
            details_any.to_msg::<pb::MemWalIndexDetails>()?,
        )?)
    } else {
        Err(Error::Index {
            message: "Index details not found for the MemWAL index".into(),
            location: location!(),
        })
    }
}

/// Open the MemWAL index from its metadata.
pub(crate) fn open_mem_wal_index(index: IndexMetadata) -> Result<Arc<MemWalIndex>> {
    Ok(Arc::new(MemWalIndex::new(load_mem_wal_index_details(
        index,
    )?)))
}

/// Update merged_generations in the MemWAL index.
/// This is called during merge-insert commits to atomically record which
/// generations have been merged to the base table.
pub(crate) fn update_mem_wal_index_merged_generations(
    indices: &mut Vec<IndexMetadata>,
    dataset_version: u64,
    new_merged_generations: Vec<MergedGeneration>,
) -> Result<()> {
    if new_merged_generations.is_empty() {
        return Ok(());
    }

    let pos = indices
        .iter()
        .position(|idx| idx.name == MEM_WAL_INDEX_NAME);

    let new_meta = if let Some(pos) = pos {
        let current_meta = indices.remove(pos);
        let mut details = load_mem_wal_index_details(current_meta)?;

        // Update merged_generations - for each region, keep the higher generation
        for new_mg in new_merged_generations {
            if let Some(existing) = details
                .merged_generations
                .iter_mut()
                .find(|mg| mg.region_id == new_mg.region_id)
            {
                if new_mg.generation > existing.generation {
                    existing.generation = new_mg.generation;
                }
            } else {
                details.merged_generations.push(new_mg);
            }
        }

        new_mem_wal_index_meta(dataset_version, details)?
    } else {
        // Create new MemWAL index with just the merged generations
        let details = MemWalIndexDetails {
            merged_generations: new_merged_generations,
            ..Default::default()
        };
        new_mem_wal_index_meta(dataset_version, details)?
    };

    indices.push(new_meta);
    Ok(())
}

/// Create a new MemWAL index metadata entry.
pub(crate) fn new_mem_wal_index_meta(
    dataset_version: u64,
    details: MemWalIndexDetails,
) -> Result<IndexMetadata> {
    Ok(IndexMetadata {
        uuid: Uuid::new_v4(),
        name: MEM_WAL_INDEX_NAME.to_string(),
        fields: vec![],
        dataset_version,
        fragment_bitmap: None,
        index_details: Some(Arc::new(prost_types::Any::from_msg(
            &pb::MemWalIndexDetails::from(&details),
        )?)),
        index_version: 0,
        created_at: Some(chrono::Utc::now()),
        base_id: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use lance_index::DatasetIndexExt;

    use crate::dataset::transaction::{Operation, Transaction};
    use crate::dataset::{CommitBuilder, InsertBuilder, WriteParams};

    async fn test_dataset() -> crate::Dataset {
        let write_params = WriteParams {
            max_rows_per_file: 10,
            ..Default::default()
        };
        let data = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("a", DataType::Int32, false),
                Field::new("b", DataType::Int32, true),
            ])),
            vec![
                Arc::new(Int32Array::from_iter_values(0..10_i32)),
                Arc::new(Int32Array::from_iter_values(std::iter::repeat_n(0, 10))),
            ],
        )
        .unwrap();
        InsertBuilder::new("memory://test_mem_wal")
            .with_params(&write_params)
            .execute(vec![data])
            .await
            .unwrap()
    }

    /// Test that UpdateMemWalState with lower generation than committed fails without retry.
    /// Per spec: If committed_generation >= to_commit_generation, abort without retry.
    #[tokio::test]
    async fn test_update_mem_wal_state_conflict_lower_generation_no_retry() {
        let dataset = test_dataset().await;
        let region = Uuid::new_v4();

        // First commit UpdateMemWalState with generation 10
        let txn1 = Transaction::new(
            dataset.manifest.version,
            Operation::UpdateMemWalState {
                merged_generations: vec![MergedGeneration::new(region, 10)],
            },
            None,
        );
        let dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(txn1)
            .await
            .unwrap();

        // Try to commit UpdateMemWalState with generation 5 (lower than 10)
        // This should fail with non-retryable conflict
        let txn2 = Transaction::new(
            dataset.manifest.version - 1, // Based on old version
            Operation::UpdateMemWalState {
                merged_generations: vec![MergedGeneration::new(region, 5)],
            },
            None,
        );
        let result = CommitBuilder::new(Arc::new(dataset)).execute(txn2).await;

        assert!(
            matches!(result, Err(crate::Error::CommitConflict { .. })),
            "Expected non-retryable CommitConflict for lower generation, got {:?}",
            result
        );
    }

    /// Test that UpdateMemWalState with equal generation as committed fails without retry.
    #[tokio::test]
    async fn test_update_mem_wal_state_conflict_equal_generation_no_retry() {
        let dataset = test_dataset().await;
        let region = Uuid::new_v4();

        // First commit UpdateMemWalState with generation 10
        let txn1 = Transaction::new(
            dataset.manifest.version,
            Operation::UpdateMemWalState {
                merged_generations: vec![MergedGeneration::new(region, 10)],
            },
            None,
        );
        let dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(txn1)
            .await
            .unwrap();

        // Try to commit UpdateMemWalState with generation 10 (equal)
        let txn2 = Transaction::new(
            dataset.manifest.version - 1, // Based on old version
            Operation::UpdateMemWalState {
                merged_generations: vec![MergedGeneration::new(region, 10)],
            },
            None,
        );
        let result = CommitBuilder::new(Arc::new(dataset)).execute(txn2).await;

        assert!(
            matches!(result, Err(crate::Error::CommitConflict { .. })),
            "Expected non-retryable CommitConflict for equal generation, got {:?}",
            result
        );
    }

    /// Test that UpdateMemWalState with higher generation than committed is retryable.
    /// Per spec: If committed_generation < to_commit_generation, retry is allowed.
    #[tokio::test]
    async fn test_update_mem_wal_state_conflict_higher_generation_retryable() {
        let dataset = test_dataset().await;
        let region = Uuid::new_v4();

        // First commit UpdateMemWalState with generation 5
        let txn1 = Transaction::new(
            dataset.manifest.version,
            Operation::UpdateMemWalState {
                merged_generations: vec![MergedGeneration::new(region, 5)],
            },
            None,
        );
        let dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(txn1)
            .await
            .unwrap();

        // Try to commit UpdateMemWalState with generation 10 (higher than 5)
        // This should fail with retryable conflict
        let txn2 = Transaction::new(
            dataset.manifest.version - 1, // Based on old version
            Operation::UpdateMemWalState {
                merged_generations: vec![MergedGeneration::new(region, 10)],
            },
            None,
        );
        let result = CommitBuilder::new(Arc::new(dataset)).execute(txn2).await;

        assert!(
            matches!(result, Err(crate::Error::RetryableCommitConflict { .. })),
            "Expected retryable conflict for higher generation, got {:?}",
            result
        );
    }

    /// Test that UpdateMemWalState on different regions don't conflict.
    #[tokio::test]
    async fn test_update_mem_wal_state_different_regions_no_conflict() {
        let dataset = test_dataset().await;
        let region1 = Uuid::new_v4();
        let region2 = Uuid::new_v4();

        // First commit UpdateMemWalState for region1
        let txn1 = Transaction::new(
            dataset.manifest.version,
            Operation::UpdateMemWalState {
                merged_generations: vec![MergedGeneration::new(region1, 10)],
            },
            None,
        );
        let dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(txn1)
            .await
            .unwrap();

        // Commit UpdateMemWalState for region2 based on old version
        // This should succeed because different regions don't conflict
        let txn2 = Transaction::new(
            dataset.manifest.version - 1, // Based on old version
            Operation::UpdateMemWalState {
                merged_generations: vec![MergedGeneration::new(region2, 5)],
            },
            None,
        );
        let result = CommitBuilder::new(Arc::new(dataset)).execute(txn2).await;

        assert!(
            result.is_ok(),
            "Expected success for different regions, got {:?}",
            result
        );

        // Verify both regions are in the index
        let dataset = result.unwrap();
        let mem_wal_idx = dataset
            .load_indices()
            .await
            .unwrap()
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .unwrap()
            .clone();
        let details = load_mem_wal_index_details(mem_wal_idx).unwrap();
        assert_eq!(details.merged_generations.len(), 2);
    }

    /// Test that CreateIndex of MemWalIndex can be rebased against UpdateMemWalState.
    /// The merged_generations from UpdateMemWalState should be merged into CreateIndex.
    #[tokio::test]
    async fn test_create_index_rebase_against_update_mem_wal_state() {
        let dataset = test_dataset().await;
        let region = Uuid::new_v4();

        // First commit UpdateMemWalState with generation 10
        let txn1 = Transaction::new(
            dataset.manifest.version,
            Operation::UpdateMemWalState {
                merged_generations: vec![MergedGeneration::new(region, 10)],
            },
            None,
        );
        let dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(txn1)
            .await
            .unwrap();

        // CreateIndex of MemWalIndex based on old version (before UpdateMemWalState)
        // This should succeed and merge the generations
        let details = MemWalIndexDetails {
            num_regions: 1,
            ..Default::default()
        };
        let mem_wal_index = new_mem_wal_index_meta(dataset.manifest.version - 1, details).unwrap();

        let txn2 = Transaction::new(
            dataset.manifest.version - 1, // Based on old version
            Operation::CreateIndex {
                new_indices: vec![mem_wal_index],
                removed_indices: vec![],
            },
            None,
        );
        let result = CommitBuilder::new(Arc::new(dataset)).execute(txn2).await;

        assert!(
            result.is_ok(),
            "Expected CreateIndex to succeed with rebase, got {:?}",
            result
        );

        // Verify the merged_generations from UpdateMemWalState were merged into CreateIndex
        let dataset = result.unwrap();
        let mem_wal_idx = dataset
            .load_indices()
            .await
            .unwrap()
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .unwrap()
            .clone();
        let details = load_mem_wal_index_details(mem_wal_idx).unwrap();
        assert_eq!(details.merged_generations.len(), 1);
        assert_eq!(details.merged_generations[0].region_id, region);
        assert_eq!(details.merged_generations[0].generation, 10);
        assert_eq!(details.num_regions, 1); // Config from CreateIndex preserved
    }

    /// Test that UpdateMemWalState against CreateIndex of MemWalIndex checks generations.
    #[tokio::test]
    async fn test_update_mem_wal_state_against_create_index_lower_generation() {
        let dataset = test_dataset().await;
        let region = Uuid::new_v4();

        // First commit CreateIndex of MemWalIndex with merged_generations
        let details = MemWalIndexDetails {
            merged_generations: vec![MergedGeneration::new(region, 10)],
            ..Default::default()
        };
        let mem_wal_index = new_mem_wal_index_meta(dataset.manifest.version, details).unwrap();

        let txn1 = Transaction::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![mem_wal_index],
                removed_indices: vec![],
            },
            None,
        );
        let dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(txn1)
            .await
            .unwrap();

        // Try UpdateMemWalState with lower generation
        let txn2 = Transaction::new(
            dataset.manifest.version - 1, // Based on old version
            Operation::UpdateMemWalState {
                merged_generations: vec![MergedGeneration::new(region, 5)],
            },
            None,
        );
        let result = CommitBuilder::new(Arc::new(dataset)).execute(txn2).await;

        assert!(
            matches!(result, Err(crate::Error::CommitConflict { .. })),
            "Expected non-retryable CommitConflict when UpdateMemWalState generation is lower than CreateIndex, got {:?}",
            result
        );
    }

    #[test]
    fn test_update_merged_generations() {
        let mut indices = Vec::new();
        let region1 = Uuid::new_v4();
        let region2 = Uuid::new_v4();

        // First update - creates new index
        update_mem_wal_index_merged_generations(
            &mut indices,
            1,
            vec![MergedGeneration::new(region1, 5)],
        )
        .unwrap();

        assert_eq!(indices.len(), 1);
        let details = load_mem_wal_index_details(indices[0].clone()).unwrap();
        assert_eq!(details.merged_generations.len(), 1);
        assert_eq!(details.merged_generations[0].region_id, region1);
        assert_eq!(details.merged_generations[0].generation, 5);

        // Second update - updates existing region
        update_mem_wal_index_merged_generations(
            &mut indices,
            2,
            vec![MergedGeneration::new(region1, 10)],
        )
        .unwrap();

        assert_eq!(indices.len(), 1);
        let details = load_mem_wal_index_details(indices[0].clone()).unwrap();
        assert_eq!(details.merged_generations.len(), 1);
        assert_eq!(details.merged_generations[0].generation, 10);

        // Third update - adds new region
        update_mem_wal_index_merged_generations(
            &mut indices,
            3,
            vec![MergedGeneration::new(region2, 3)],
        )
        .unwrap();

        assert_eq!(indices.len(), 1);
        let details = load_mem_wal_index_details(indices[0].clone()).unwrap();
        assert_eq!(details.merged_generations.len(), 2);

        // Fourth update - lower generation should not update
        update_mem_wal_index_merged_generations(
            &mut indices,
            4,
            vec![MergedGeneration::new(region1, 8)], // lower than 10
        )
        .unwrap();

        let details = load_mem_wal_index_details(indices[0].clone()).unwrap();
        let r1_mg = details
            .merged_generations
            .iter()
            .find(|mg| mg.region_id == region1)
            .unwrap();
        assert_eq!(r1_mg.generation, 10); // Should still be 10
    }

    #[test]
    fn test_empty_merged_generations_noop() {
        let mut indices = Vec::new();

        // Empty update should be a no-op
        update_mem_wal_index_merged_generations(&mut indices, 1, vec![]).unwrap();

        assert!(indices.is_empty());
    }
}
