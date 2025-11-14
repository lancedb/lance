// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::dataset::transaction::{Operation, Transaction};
use crate::index::DatasetIndexInternalExt;
use crate::Dataset;
use lance_core::{Error, Result};
use lance_index::mem_wal::{MemWal, MemWalId, MemWalIndex, MemWalIndexDetails, MEM_WAL_INDEX_NAME};
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::{is_system_index, DatasetIndexExt};
use lance_table::format::{pb, IndexMetadata};
use prost::Message;
use snafu::location;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use uuid::Uuid;

fn load_mem_wal_index_details(index: IndexMetadata) -> Result<MemWalIndexDetails> {
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

pub(crate) fn open_mem_wal_index(index: IndexMetadata) -> Result<Arc<MemWalIndex>> {
    Ok(Arc::new(MemWalIndex::new(load_mem_wal_index_details(
        index,
    )?)))
}

/// Find the latest generation
pub async fn find_latest_mem_wal_generation(
    dataset: &Dataset,
    region: &str,
) -> Result<Option<MemWal>> {
    let Some(mem_wal_index) = dataset.open_mem_wal_index(&NoOpMetricsCollector).await? else {
        return Ok(None);
    };

    let Some(generations) = mem_wal_index.mem_wal_map.get(region) else {
        return Ok(None);
    };

    // MemWALs of the same region is ordered increasingly by its generation
    if let Some(latest_mem_wal) = generations.values().last() {
        Ok(Some(latest_mem_wal.clone()))
    } else {
        Err(Error::Internal {
            message: format!("Encountered MemWAL index mapping that has a region with an empty list of generations: {}", region),
            location: location!(),
        })
    }
}

pub async fn create_mem_wal_generation(
    dataset: &mut Dataset,
    region: &str,
    generation: u64,
    new_mem_table_location: &str,
    new_wal_location: &str,
    owner_id: &str,
) -> Result<MemWal> {
    let mem_wal = MemWal::new_empty(
        MemWalId::new(region, generation),
        new_mem_table_location,
        new_wal_location,
        owner_id,
    );
    let txn = Transaction::new(
        dataset.manifest.version,
        Operation::UpdateMemWalState {
            added: vec![mem_wal.clone()],
            updated: vec![],
            removed: vec![],
        },
        None,
    );

    dataset
        .apply_commit(txn, &Default::default(), &Default::default())
        .await?;

    Ok(mem_wal)
}

/// Advance the generation of the MemWAL for the given region.
/// If the MemWAL does not exist, create one with generation 0, and
/// `expected_owner_id` should be None in this case.
/// If the MemWAL exists, seal the one with the latest generation,
/// and open one with the same name and the next generation.
/// If the MemWALIndex structure does not exist, create it along the way.
pub async fn advance_mem_wal_generation(
    dataset: &mut Dataset,
    region: &str,
    new_mem_table_location: &str,
    new_wal_location: &str,
    expected_owner_id: Option<&str>,
    new_owner_id: &str,
) -> Result<()> {
    let transaction = if let Some(mem_wal_index) =
        dataset.open_mem_wal_index(&NoOpMetricsCollector).await?
    {
        let (added_mem_wal, updated_mem_wal, removed_mem_wal) = if let Some(generations) =
            mem_wal_index.mem_wal_map.get(region)
        {
            if let Some(latest_mem_wal) = generations.values().last() {
                // TODO: technically should check against all WAL locations
                if latest_mem_wal.wal_location == new_wal_location {
                    return Err(Error::invalid_input(
                        format!(
                            "Must use a different WAL location from current: {}",
                            latest_mem_wal.wal_location
                        ),
                        location!(),
                    ));
                }

                if let Some(expected_owner_id) = expected_owner_id {
                    latest_mem_wal.check_expected_owner_id(expected_owner_id)?;
                } else {
                    return Err(Error::invalid_input(
                        format!(
                            "Expected creating generation 0 for MemWAL region {}, but found current latest MemWAL: {:?}",
                            region, latest_mem_wal
                        ),
                        location!()));
                }

                if latest_mem_wal.mem_table_location == new_mem_table_location {
                    return Err(Error::invalid_input(
                        format!(
                            "Must use a different MemTable location from current: {}",
                            latest_mem_wal.mem_table_location
                        ),
                        location!(),
                    ));
                }

                let (updated_mem_wal, removed_mem_wal) =
                    if latest_mem_wal.state == lance_index::mem_wal::State::Open {
                        let mut updated_mem_wal = latest_mem_wal.clone();
                        updated_mem_wal.state = lance_index::mem_wal::State::Sealed;
                        (Some(updated_mem_wal), Some(latest_mem_wal.clone()))
                    } else {
                        (None, None)
                    };

                let added_mem_wal = MemWal::new_empty(
                    MemWalId::new(region, latest_mem_wal.id.generation + 1),
                    new_mem_table_location,
                    new_wal_location,
                    new_owner_id,
                );

                Ok((added_mem_wal, updated_mem_wal, removed_mem_wal))
            } else {
                Err(Error::Internal {
                    message: format!("Encountered MemWAL index mapping that has a region with an empty list of generations: {}", region),
                    location: location!(),
                })
            }
        } else {
            if let Some(expected_owner_id) = expected_owner_id {
                return Err(Error::invalid_input(
                    format!(
                        "Expected advancing MemWAL region {} from owner ID {}, but found no generation yet",
                        region, expected_owner_id
                    ),
                    location!()));
            }

            Ok((
                MemWal::new_empty(
                    MemWalId::new(region, 0),
                    new_mem_table_location,
                    new_wal_location,
                    new_owner_id,
                ),
                None,
                None,
            ))
        }?;

        Transaction::new(
            dataset.manifest.version,
            Operation::UpdateMemWalState {
                added: vec![added_mem_wal],
                updated: updated_mem_wal.into_iter().collect(),
                removed: removed_mem_wal.into_iter().collect(),
            },
            None,
        )
    } else {
        // this is the first time the MemWAL index is created
        if let Some(expected_owner_id) = expected_owner_id {
            return Err(Error::invalid_input(
                format!(
                    "Expected advancing MemWAL region {} from owner ID {}, but found no MemWAL index",
                    region, expected_owner_id
                ),
                location!()));
        }

        Transaction::new(
            dataset.manifest.version,
            Operation::UpdateMemWalState {
                added: vec![MemWal::new_empty(
                    MemWalId::new(region, 0),
                    new_mem_table_location,
                    new_wal_location,
                    new_owner_id,
                )],
                updated: vec![],
                removed: vec![],
            },
            None,
        )
    };

    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await
}

/// Add a new entry to the MemWAL
pub async fn append_mem_wal_entry(
    dataset: &mut Dataset,
    mem_wal_region: &str,
    mem_wal_generation: u64,
    entry_id: u64,
    expected_owner_id: &str,
) -> Result<MemWal> {
    let mutate = |mem_wal: &MemWal| -> Result<MemWal> {
        // Can only append to open MemWALs
        mem_wal.check_state(lance_index::mem_wal::State::Open)?;
        mem_wal.check_expected_owner_id(expected_owner_id)?;

        let mut updated_mem_wal = mem_wal.clone();
        let wal_entries = updated_mem_wal.wal_entries();
        updated_mem_wal.wal_entries =
            pb::U64Segment::from(wal_entries.with_new_high(entry_id)?).encode_to_vec();
        Ok(updated_mem_wal)
    };

    mutate_mem_wal(dataset, mem_wal_region, mem_wal_generation, mutate).await
}

/// Mark the specific MemWAL as sealed.
/// Typically, it is recommended to call [`advance_mem_wal_generation`] instead.
/// But this will always keep the table in a state with an unsealed MemTable.
/// Calling this function will only seal the current latest MemWAL without opening the next one.
pub async fn mark_mem_wal_as_sealed(
    dataset: &mut Dataset,
    mem_wal_region: &str,
    mem_wal_generation: u64,
    expected_owner_id: &str,
) -> Result<MemWal> {
    let mutate = |mem_wal: &MemWal| -> Result<MemWal> {
        // Can only seal open MemWALs
        mem_wal.check_state(lance_index::mem_wal::State::Open)?;
        mem_wal.check_expected_owner_id(expected_owner_id)?;

        let mut updated_mem_wal = mem_wal.clone();
        updated_mem_wal.state = lance_index::mem_wal::State::Sealed;
        Ok(updated_mem_wal)
    };

    mutate_mem_wal(dataset, mem_wal_region, mem_wal_generation, mutate).await
}

/// Mark the specific MemWAL as flushed (data on disk but not merged)
pub async fn mark_mem_wal_as_flushed(
    dataset: &mut Dataset,
    mem_wal_region: &str,
    mem_wal_generation: u64,
    expected_owner_id: &str,
) -> Result<MemWal> {
    let mutate = |mem_wal: &MemWal| -> Result<MemWal> {
        // Can only flush sealed MemWALs
        mem_wal.check_state(lance_index::mem_wal::State::Sealed)?;
        mem_wal.check_expected_owner_id(expected_owner_id)?;

        let mut updated_mem_wal = mem_wal.clone();
        updated_mem_wal.state = lance_index::mem_wal::State::Flushed;
        Ok(updated_mem_wal)
    };

    mutate_mem_wal(dataset, mem_wal_region, mem_wal_generation, mutate).await
}

/// Mark the specific MemWAL as merged (data merged into source table)
pub async fn mark_mem_wal_as_merged(
    dataset: &mut Dataset,
    mem_wal_region: &str,
    mem_wal_generation: u64,
    expected_owner_id: &str,
) -> Result<MemWal> {
    let mutate = |mem_wal: &MemWal| -> Result<MemWal> {
        // Can only merge flushed MemWALs
        mem_wal.check_state(lance_index::mem_wal::State::Flushed)?;
        mem_wal.check_expected_owner_id(expected_owner_id)?;

        let mut updated_mem_wal = mem_wal.clone();
        updated_mem_wal.state = lance_index::mem_wal::State::Merged;
        Ok(updated_mem_wal)
    };

    mutate_mem_wal(dataset, mem_wal_region, mem_wal_generation, mutate).await
}

/// Mark the specific MemWAL as flushed, in the list of indices in the dataset.
/// This is intended to be used as a part of the Update transaction after resolving all conflicts.
pub(crate) fn update_mem_wal_index_in_indices_list(
    dataset_read_version: u64,
    dataset_new_version: u64,
    indices: &mut Vec<IndexMetadata>,
    added: Vec<MemWal>,
    updated: Vec<MemWal>,
    removed: Vec<MemWal>,
) -> Result<()> {
    let new_meta = if let Some(pos) = indices
        .iter()
        .position(|idx| idx.name == MEM_WAL_INDEX_NAME)
    {
        let current_meta = indices.remove(pos);
        let mut details = load_mem_wal_index_details(current_meta)?;
        let removed_set = removed
            .iter()
            .map(|rm| rm.id.clone())
            .collect::<HashSet<_>>();
        details
            .mem_wal_list
            .retain(|m| !removed_set.contains(&m.id));

        for mut mem_wal in added.into_iter() {
            mem_wal.last_updated_dataset_version = dataset_new_version;
            details.mem_wal_list.push(mem_wal);
        }

        for mut mem_wal in updated.into_iter() {
            mem_wal.last_updated_dataset_version = dataset_new_version;
            details.mem_wal_list.push(mem_wal);
        }

        new_mem_wal_index_meta(dataset_read_version, details.mem_wal_list)?
    } else {
        // This should only happen with new index creation when opening the first MemWAL
        if !updated.is_empty() || !removed.is_empty() {
            return Err(Error::invalid_input(
                "Cannot update MemWAL state without a MemWAL index",
                location!(),
            ));
        }

        let mut added_with_version = Vec::with_capacity(added.len());
        for mut mem_wal in added.into_iter() {
            mem_wal.last_updated_dataset_version = dataset_new_version;
            added_with_version.push(mem_wal);
        }

        new_mem_wal_index_meta(dataset_read_version, added_with_version)?
    };

    indices.push(new_meta);
    Ok(())
}

/// Owner ID serves as a pre-check that the MemWAL has not changed owner before commit.
/// Each writer is required to keep an invariant of its owner ID for a MemWAL.
/// At any point in time, there should be only 1 writer that owns the right to mutate the MemWAL,
/// and the owner ID serves as the optimistic lock for it.
/// Specifically, before a writer starts to replay a WAL, it should call this method to claim
/// ownership and stop any additional writes to the MemWAL from other writers.
///
/// Consider a distributed cluster which currently has node A writing to the table's MemWAL.
/// A network partition happens, node A is not dead but fails the health check.
/// Node B is newly assigned and starts the WAL replay process which modifies the owner ID.
/// In this case, if node A is doing a modification to the same MemWAL including adding an entry,
/// sealing or flushing, advancing the MemWAL generation, it will receive a commit conflict failure.
/// In theory, all the writes from node A should abort after seeing this failure without retrying.
/// However, if the writer decides to retry the operation for any reason (e.g. a bug), without the check,
/// the retry would succeed. The `expected_owner_id` in all write functions serves as the guard to
/// make sure it continues to fail until the write traffic is fully redirected to node B.
pub async fn update_mem_wal_owner(
    dataset: &mut Dataset,
    region: &str,
    generation: u64,
    new_owner_id: &str,
    new_mem_table_location: Option<&str>,
) -> Result<MemWal> {
    let mutate = |mem_wal: &MemWal| -> Result<MemWal> {
        if new_owner_id == mem_wal.owner_id {
            return Err(Error::invalid_input(
                format!(
                    "Must use a different owner ID from current: {}",
                    mem_wal.owner_id
                ),
                location!(),
            ));
        }

        if let Some(new_mem_table_location) = new_mem_table_location {
            if new_mem_table_location == mem_wal.mem_table_location {
                return Err(Error::invalid_input(
                    format!(
                        "Must use a different MemTable location from current: {}",
                        mem_wal.mem_table_location
                    ),
                    location!(),
                ));
            }
        }

        let mut updated_mem_wal = mem_wal.clone();
        updated_mem_wal.owner_id = new_owner_id.to_owned();
        if let Some(new_mem_table_location) = new_mem_table_location {
            updated_mem_wal.mem_table_location = new_mem_table_location.to_owned();
        }
        Ok(updated_mem_wal)
    };

    mutate_mem_wal(dataset, region, generation, mutate).await
}

/// Trim all the MemWALs that are already merged.
pub async fn trim_mem_wal_index(dataset: &mut Dataset) -> Result<()> {
    if let Some(mem_wal_index) = dataset.open_mem_wal_index(&NoOpMetricsCollector).await? {
        let indices = dataset.load_indices().await?;

        // group by name to get the latest version of each index
        // For delta indices, we take the highest dataset version
        let mut index_versions = HashMap::new();
        for index in indices.iter() {
            if !is_system_index(index) {
                let current_version = index_versions.entry(index.name.clone()).or_insert(0);
                *current_version = (*current_version).max(index.dataset_version);
            }
        }

        let min_index_dataset_version = index_versions.values().min().copied().unwrap_or(u64::MAX);

        let mut removed = Vec::new();
        for (_, generations) in mem_wal_index.mem_wal_map.iter() {
            for (_, mem_wal) in generations.iter() {
                if mem_wal.state == lance_index::mem_wal::State::Merged {
                    // all indices are caught up, can trim it
                    if mem_wal.last_updated_dataset_version <= min_index_dataset_version {
                        removed.push(mem_wal.clone());
                    }
                }
            }
        }

        let transaction = Transaction::new(
            dataset.manifest.version,
            Operation::UpdateMemWalState {
                added: vec![],
                updated: vec![],
                removed,
            },
            None,
        );

        dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await
    } else {
        Err(Error::NotSupported {
            source: "MemWAL is not enabled".into(),
            location: location!(),
        })
    }
}

async fn mutate_mem_wal<F>(
    dataset: &mut Dataset,
    region: &str,
    generation: u64,
    mutate: F,
) -> Result<MemWal>
where
    F: Fn(&MemWal) -> Result<MemWal>,
{
    if let Some(mem_wal_index) = dataset.open_mem_wal_index(&NoOpMetricsCollector).await? {
        if let Some(generations) = mem_wal_index.mem_wal_map.get(region) {
            if let Some(mem_wal) = generations.get(&generation) {
                let updated_mem_wal = mutate(mem_wal)?;

                let transaction = Transaction::new(
                    dataset.manifest.version,
                    Operation::UpdateMemWalState {
                        added: vec![],
                        updated: vec![updated_mem_wal.clone()],
                        removed: vec![mem_wal.clone()],
                    },
                    None,
                );

                dataset
                    .apply_commit(transaction, &Default::default(), &Default::default())
                    .await?;

                Ok(updated_mem_wal)
            } else {
                Err(Error::invalid_input(
                    format!(
                        "Cannot find MemWAL generation {} for region {}",
                        generation, region
                    ),
                    location!(),
                ))
            }
        } else {
            Err(Error::invalid_input(
                format!("Cannot find MemWAL for region {}", region),
                location!(),
            ))
        }
    } else {
        Err(Error::NotSupported {
            source: "MemWAL is not enabled".into(),
            location: location!(),
        })
    }
}

pub(crate) fn new_mem_wal_index_meta(
    dataset_version: u64,
    new_mem_wal_list: Vec<MemWal>,
) -> Result<IndexMetadata> {
    Ok(IndexMetadata {
        uuid: Uuid::new_v4(),
        name: MEM_WAL_INDEX_NAME.to_string(),
        fields: vec![],
        dataset_version,
        fragment_bitmap: None,
        index_details: Some(Arc::new(prost_types::Any::from_msg(
            &pb::MemWalIndexDetails::from(&MemWalIndexDetails {
                mem_wal_list: new_mem_wal_list,
            }),
        )?)),
        index_version: 0,
        created_at: Some(chrono::Utc::now()),
        base_id: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{WriteDestination, WriteMode, WriteParams};
    use crate::index::vector::VectorIndexParams;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use arrow_array::types::{Float32Type, Int32Type};
    use lance_datafusion::datagen::DatafusionDatagenExt;
    use lance_datagen::{BatchCount, Dimension, RowCount};
    use lance_index::mem_wal::{MemWalId, MEM_WAL_INDEX_NAME};
    use lance_index::optimize::OptimizeOptions;
    use lance_index::{DatasetIndexExt, Index};
    use lance_linalg::distance::MetricType;

    #[tokio::test]
    async fn test_advance_mem_wal_generation() {
        // Create a dataset with some data
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(1000))
            .await
            .unwrap();

        // Initially, there should be no MemWAL index
        let indices = dataset.load_indices().await.unwrap();
        assert!(!indices.iter().any(|idx| idx.name == MEM_WAL_INDEX_NAME));

        // First call to advance_mem_wal_generation should create the MemWAL index and generation 0
        let initial_version = dataset.manifest.version;
        advance_mem_wal_generation(
            &mut dataset,
            "GLOBAL",
            "mem_table_location_0",
            "wal_location_0",
            None,
            "owner_0",
        )
        .await
        .unwrap();

        // Verify the MemWAL index was created
        let indices = dataset.load_indices().await.unwrap();
        let mem_wal_index_meta = indices
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .expect("MemWAL index should be created");

        // Load and verify the MemWAL index details
        let mem_wal_details = load_mem_wal_index_details(mem_wal_index_meta.clone()).unwrap();
        assert_eq!(mem_wal_details.mem_wal_list.len(), 1);
        let mem_wal_index = open_mem_wal_index(mem_wal_index_meta.clone()).unwrap();
        let stats = mem_wal_index.statistics().unwrap();
        assert_eq!(
            serde_json::to_string(&stats).unwrap(),
            dataset.index_statistics(MEM_WAL_INDEX_NAME).await.unwrap()
        );

        let mem_wal = &mem_wal_details.mem_wal_list[0];
        assert_eq!(mem_wal.id.region, "GLOBAL");
        assert_eq!(mem_wal.id.generation, 0);
        assert_eq!(mem_wal.mem_table_location, "mem_table_location_0");
        assert_eq!(mem_wal.wal_location, "wal_location_0");
        assert_eq!(mem_wal.state, lance_index::mem_wal::State::Open);
        assert_eq!(mem_wal.last_updated_dataset_version, initial_version + 1);

        // Second call to advance_mem_wal_generation should seal generation 0 and create generation 1
        let version_before_second_advance = dataset.manifest.version;
        advance_mem_wal_generation(
            &mut dataset,
            "GLOBAL",
            "mem_table_location_1",
            "wal_location_1",
            Some("owner_0"),
            "owner_1",
        )
        .await
        .unwrap();

        // Verify the MemWAL index now has two generations
        let indices = dataset.load_indices().await.unwrap();
        let mem_wal_index_meta = indices
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .expect("MemWAL index should still exist");

        let mem_wal_details = load_mem_wal_index_details(mem_wal_index_meta.clone()).unwrap();
        assert_eq!(mem_wal_details.mem_wal_list.len(), 2);

        // Find generation 0 (should be sealed) and generation 1 (should be unsealed)
        let gen_0 = mem_wal_details
            .mem_wal_list
            .iter()
            .find(|m| m.id.generation == 0)
            .expect("Generation 0 should exist");
        let gen_1 = mem_wal_details
            .mem_wal_list
            .iter()
            .find(|m| m.id.generation == 1)
            .expect("Generation 1 should exist");

        // Verify generation 0 is sealed
        assert_eq!(gen_0.id.region, "GLOBAL");
        assert_eq!(gen_0.id.generation, 0);
        assert_eq!(gen_0.mem_table_location, "mem_table_location_0");
        assert_eq!(gen_0.wal_location, "wal_location_0");
        assert_eq!(gen_0.state, lance_index::mem_wal::State::Sealed);
        // Verify the sealed MemWAL has updated version
        assert_eq!(
            gen_0.last_updated_dataset_version,
            version_before_second_advance + 1
        );

        // Verify generation 1 is unsealed
        assert_eq!(gen_1.id.region, "GLOBAL");
        assert_eq!(gen_1.id.generation, 1);
        assert_eq!(gen_1.mem_table_location, "mem_table_location_1");
        assert_eq!(gen_1.wal_location, "wal_location_1");
        assert_eq!(gen_1.state, lance_index::mem_wal::State::Open);
        // Verify the new MemWAL has correct version
        assert_eq!(
            gen_1.last_updated_dataset_version,
            version_before_second_advance + 1
        );

        // Test that using the same MemTable location should fail
        let result = advance_mem_wal_generation(
            &mut dataset,
            "GLOBAL",
            "mem_table_location_1", // Same as current generation
            "wal_location_2",       // Different WAL location
            Some("owner_1"),
            "owner_2",
        )
        .await;
        assert!(
            result.is_err(),
            "Should fail when using same MemTable location as current generation"
        );

        // Test that using the same WAL location should fail
        let result = advance_mem_wal_generation(
            &mut dataset,
            "GLOBAL",
            "mem_table_location_2", // Different MemTable location
            "wal_location_1",       // Same as current generation
            Some("owner_1"),
            "owner_2",
        )
        .await;
        assert!(
            result.is_err(),
            "Should fail when using same WAL location as current generation"
        );
    }

    #[tokio::test]
    async fn test_append_new_entry_to_mem_wal() {
        // Create a dataset with some data
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(1000))
            .await
            .unwrap();

        // Test failure case: MemWAL is not enabled
        let result = append_mem_wal_entry(&mut dataset, "GLOBAL", 0, 123, "owner_0").await;
        assert!(result.is_err(), "Should fail when MemWAL is not enabled");

        // Create MemWAL index and generation 0
        advance_mem_wal_generation(
            &mut dataset,
            "GLOBAL",
            "mem_table_location_0",
            "wal_location_0",
            None,
            "owner_0",
        )
        .await
        .unwrap();

        // Test failure case: region doesn't exist
        let result = append_mem_wal_entry(&mut dataset, "NONEXISTENT", 0, 123, "owner_0").await;
        assert!(result.is_err(), "Should fail when region doesn't exist");

        // Test failure case: generation doesn't exist
        let result = append_mem_wal_entry(&mut dataset, "GLOBAL", 999, 123, "owner_0").await;
        assert!(result.is_err(), "Should fail when generation doesn't exist");

        // Test success case: append entry to generation 0
        let version_before_append = dataset.manifest.version;
        append_mem_wal_entry(&mut dataset, "GLOBAL", 0, 123, "owner_0")
            .await
            .unwrap();

        // Verify the entry was added
        let indices = dataset.load_indices().await.unwrap();
        let mem_wal_index_meta = indices
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .expect("MemWAL index should exist");

        let mem_wal_details = load_mem_wal_index_details(mem_wal_index_meta.clone()).unwrap();
        let mem_wal = &mem_wal_details.mem_wal_list[0];

        // Check that the WAL entries contain the entry_id
        let wal_entries = mem_wal.wal_entries();
        assert!(
            wal_entries.contains(123),
            "WAL entries should contain entry_id 123"
        );
        // Verify the MemWAL version was updated after append
        assert_eq!(
            mem_wal.last_updated_dataset_version,
            version_before_append + 1
        );

        // Test appending multiple entries
        append_mem_wal_entry(&mut dataset, "GLOBAL", 0, 456, "owner_0")
            .await
            .unwrap();
        let version_after_second_append = dataset.manifest.version;
        append_mem_wal_entry(&mut dataset, "GLOBAL", 0, 789, "owner_0")
            .await
            .unwrap();

        // Verify all entries were added
        let indices = dataset.load_indices().await.unwrap();
        let mem_wal_index_meta = indices
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .expect("MemWAL index should exist");

        let mem_wal_details = load_mem_wal_index_details(mem_wal_index_meta.clone()).unwrap();
        let mem_wal = &mem_wal_details.mem_wal_list[0];

        let wal_entries = mem_wal.wal_entries();
        assert!(
            wal_entries.contains(123),
            "WAL entries should contain entry_id 123"
        );
        assert!(
            wal_entries.contains(456),
            "WAL entries should contain entry_id 456"
        );
        assert!(
            wal_entries.contains(789),
            "WAL entries should contain entry_id 789"
        );
        // Verify the MemWAL version was updated after the last append
        assert_eq!(
            mem_wal.last_updated_dataset_version,
            version_after_second_append + 1
        );

        // Test failure case: cannot append to sealed MemWAL
        mark_mem_wal_as_sealed(&mut dataset, "GLOBAL", 0, "owner_0")
            .await
            .unwrap();
        let result = append_mem_wal_entry(&mut dataset, "GLOBAL", 0, 999, "owner_0").await;
        assert!(
            result.is_err(),
            "Should fail when trying to append to sealed MemWAL"
        );

        // Check the specific error message
        let error = result.unwrap_err();
        assert!(error.to_string().contains("MemWAL MemWalId { region: \"GLOBAL\", generation: 0 } is in state Sealed, but expected Open"), 
                "Error message should indicate the MemWAL is sealed, got: {}", error);

        // Test failure case: cannot append to flushed MemWAL
        mark_mem_wal_as_flushed(&mut dataset, "GLOBAL", 0, "owner_0")
            .await
            .unwrap();
        let result = append_mem_wal_entry(&mut dataset, "GLOBAL", 0, 999, "owner_0").await;
        assert!(
            result.is_err(),
            "Should fail when trying to append to flushed MemWAL"
        );

        // Check the specific error message
        let error = result.unwrap_err();
        assert!(error.to_string().contains("MemWAL MemWalId { region: \"GLOBAL\", generation: 0 } is in state Flushed, but expected Open"), 
                "Error message should indicate the MemWAL is flushed, got: {}", error);
    }

    #[tokio::test]
    async fn test_seal_mem_wal() {
        // Create a dataset with some data
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(1000))
            .await
            .unwrap();

        // Test failure case: MemWAL is not enabled
        let result = mark_mem_wal_as_sealed(&mut dataset, "GLOBAL", 0, "owner_0").await;
        assert!(result.is_err(), "Should fail when MemWAL is not enabled");

        // Create MemWAL index and generation 0
        advance_mem_wal_generation(
            &mut dataset,
            "GLOBAL",
            "mem_table_location_0",
            "wal_location_0",
            None,
            "owner_0",
        )
        .await
        .unwrap();

        // Test failure case: region doesn't exist
        let result = mark_mem_wal_as_sealed(&mut dataset, "NONEXISTENT", 0, "owner_0").await;
        assert!(result.is_err(), "Should fail when region doesn't exist");

        // Test failure case: generation doesn't exist
        let result = mark_mem_wal_as_sealed(&mut dataset, "GLOBAL", 999, "owner_0").await;
        assert!(result.is_err(), "Should fail when generation doesn't exist");

        // Verify generation 0 is initially unsealed
        let indices = dataset.load_indices().await.unwrap();
        let mem_wal_index_meta = indices
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .expect("MemWAL index should exist");

        let mem_wal_details = load_mem_wal_index_details(mem_wal_index_meta.clone()).unwrap();
        let mem_wal = &mem_wal_details.mem_wal_list[0];
        assert_eq!(
            mem_wal.state,
            lance_index::mem_wal::State::Open,
            "Generation 0 should initially be open"
        );

        // Test success case: seal generation 0
        let version_before_seal = dataset.manifest.version;
        mark_mem_wal_as_sealed(&mut dataset, "GLOBAL", 0, "owner_0")
            .await
            .unwrap();

        // Verify generation 0 is now sealed
        let indices = dataset.load_indices().await.unwrap();
        let mem_wal_index_meta = indices
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .expect("MemWAL index should exist");

        let mem_wal_details = load_mem_wal_index_details(mem_wal_index_meta.clone()).unwrap();
        let mem_wal = &mem_wal_details.mem_wal_list[0];
        assert_eq!(
            mem_wal.state,
            lance_index::mem_wal::State::Sealed,
            "Generation 0 should now be sealed"
        );
        // Verify the MemWAL version was updated after sealing
        assert_eq!(
            mem_wal.last_updated_dataset_version,
            version_before_seal + 1
        );

        // Create a new generation and test sealing it
        advance_mem_wal_generation(
            &mut dataset,
            "GLOBAL",
            "mem_table_location_1",
            "wal_location_1",
            Some("owner_0"),
            "owner_1",
        )
        .await
        .unwrap();

        // Verify generation 1 is unsealed
        let indices = dataset.load_indices().await.unwrap();
        let mem_wal_index_meta = indices
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .expect("MemWAL index should exist");

        let mem_wal_details = load_mem_wal_index_details(mem_wal_index_meta.clone()).unwrap();
        let gen_1 = mem_wal_details
            .mem_wal_list
            .iter()
            .find(|m| m.id.generation == 1)
            .expect("Generation 1 should exist");

        assert_eq!(
            gen_1.state,
            lance_index::mem_wal::State::Open,
            "Generation 1 should be open"
        );

        // Seal generation 1
        let version_before_seal_gen1 = dataset.manifest.version;
        mark_mem_wal_as_sealed(&mut dataset, "GLOBAL", 1, "owner_1")
            .await
            .unwrap();

        // Verify it's sealed
        let indices = dataset.load_indices().await.unwrap();
        let mem_wal_index_meta = indices
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .expect("MemWAL index should exist");

        let mem_wal_details = load_mem_wal_index_details(mem_wal_index_meta.clone()).unwrap();
        let gen_1 = mem_wal_details
            .mem_wal_list
            .iter()
            .find(|m| m.id.generation == 1)
            .expect("Generation 1 should exist");

        assert_eq!(
            gen_1.state,
            lance_index::mem_wal::State::Sealed,
            "Generation 1 should be sealed"
        );
        // Verify the MemWAL version was updated after sealing generation 1
        assert_eq!(
            gen_1.last_updated_dataset_version,
            version_before_seal_gen1 + 1
        );

        // Test that sealing an already sealed MemWAL should fail
        let result = mark_mem_wal_as_sealed(&mut dataset, "GLOBAL", 1, "owner_1").await;
        assert!(
            result.is_err(),
            "Should fail when trying to seal an already sealed MemWAL"
        );

        // Check the specific error message
        let error = result.unwrap_err();
        assert!(error.to_string().contains("MemWAL MemWalId { region: \"GLOBAL\", generation: 1 } is in state Sealed, but expected Open"), 
                "Error message should indicate the MemWAL is not open, got: {}", error);

        // Test that sealing an already flushed MemWAL should fail
        mark_mem_wal_as_flushed(&mut dataset, "GLOBAL", 0, "owner_0")
            .await
            .unwrap();
        let result = mark_mem_wal_as_sealed(&mut dataset, "GLOBAL", 0, "owner_0").await;
        assert!(
            result.is_err(),
            "Should fail when trying to seal an already flushed MemWAL"
        );

        // Check the specific error message
        let error = result.unwrap_err();
        assert!(error.to_string().contains("MemWAL MemWalId { region: \"GLOBAL\", generation: 0 } is in state Flushed, but expected Open"),
                "Error message should indicate the MemWAL is already flushed, got: {}", error);
    }

    #[tokio::test]
    async fn test_flush_and_merge_mem_wal() {
        // Create a dataset with some data
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(1000))
            .await
            .unwrap();

        // Test failure case: MemWAL is not enabled
        let result = mark_mem_wal_as_flushed(&mut dataset, "GLOBAL", 0, "owner_0").await;
        assert!(result.is_err(), "Should fail when MemWAL is not enabled");

        // Create MemWAL index and generation 0
        advance_mem_wal_generation(
            &mut dataset,
            "GLOBAL",
            "mem_table_location_0",
            "wal_location_0",
            None,
            "owner_0",
        )
        .await
        .unwrap();

        // Test failure case: region doesn't exist
        let result = mark_mem_wal_as_flushed(&mut dataset, "NONEXISTENT", 0, "owner_0").await;
        assert!(result.is_err(), "Should fail when region doesn't exist");

        // Test failure case: generation doesn't exist
        let result = mark_mem_wal_as_flushed(&mut dataset, "GLOBAL", 999, "owner_0").await;
        assert!(result.is_err(), "Should fail when generation doesn't exist");

        // Verify generation 0 is initially unflushed
        let indices = dataset.load_indices().await.unwrap();
        let mem_wal_index_meta = indices
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .expect("MemWAL index should exist");

        let mem_wal_details = load_mem_wal_index_details(mem_wal_index_meta.clone()).unwrap();
        let mem_wal = &mem_wal_details.mem_wal_list[0];
        assert_eq!(
            mem_wal.state,
            lance_index::mem_wal::State::Open,
            "Generation 0 should initially be open"
        );

        // Test failure case: cannot flush unsealed MemWAL
        let result = mark_mem_wal_as_flushed(&mut dataset, "GLOBAL", 0, "owner_0").await;
        assert!(
            result.is_err(),
            "Should fail when trying to flush unsealed MemWAL"
        );

        // Check the specific error message
        let error = result.unwrap_err();
        assert!(error.to_string().contains("MemWAL MemWalId { region: \"GLOBAL\", generation: 0 } is in state Open, but expected Sealed"), 
                "Error message should indicate the MemWAL is not sealed, got: {}", error);

        // Seal generation 0 first
        mark_mem_wal_as_sealed(&mut dataset, "GLOBAL", 0, "owner_0")
            .await
            .unwrap();

        // Test success case: mark sealed generation 0 as flushed
        let version_before_flush = dataset.manifest.version;
        mark_mem_wal_as_flushed(&mut dataset, "GLOBAL", 0, "owner_0")
            .await
            .unwrap();

        // Verify generation 0 is now flushed
        let indices = dataset.load_indices().await.unwrap();
        let mem_wal_index_meta = indices
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .expect("MemWAL index should exist");

        let mem_wal_details = load_mem_wal_index_details(mem_wal_index_meta.clone()).unwrap();
        let mem_wal = &mem_wal_details.mem_wal_list[0];
        assert_eq!(
            mem_wal.state,
            lance_index::mem_wal::State::Flushed,
            "Generation 0 should now be flushed"
        );
        // Verify the MemWAL version was updated after flushing
        assert_eq!(
            mem_wal.last_updated_dataset_version,
            version_before_flush + 1
        );

        // Test failure case: cannot flush already flushed MemWAL
        let result = mark_mem_wal_as_flushed(&mut dataset, "GLOBAL", 0, "owner_0").await;
        assert!(
            result.is_err(),
            "Should fail when trying to flush already flushed MemWAL"
        );

        // Check the specific error message
        let error = result.unwrap_err();
        assert!(error.to_string().contains("MemWAL MemWalId { region: \"GLOBAL\", generation: 0 } is in state Flushed, but expected Sealed"), 
                "Error message should indicate the MemWAL is already flushed, got: {}", error);

        // Test success case: mark flushed generation 0 as merged
        let version_before_merge = dataset.manifest.version;
        mark_mem_wal_as_merged(&mut dataset, "GLOBAL", 0, "owner_0")
            .await
            .unwrap();

        // Verify generation 0 is now merged
        let indices = dataset.load_indices().await.unwrap();
        let mem_wal_index_meta = indices
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .expect("MemWAL index should exist");

        let mem_wal_details = load_mem_wal_index_details(mem_wal_index_meta.clone()).unwrap();
        let mem_wal = &mem_wal_details.mem_wal_list[0];
        assert_eq!(
            mem_wal.state,
            lance_index::mem_wal::State::Merged,
            "Generation 0 should now be merged"
        );
        // Verify the MemWAL version was updated after merging
        assert_eq!(
            mem_wal.last_updated_dataset_version,
            version_before_merge + 1
        );

        // Test failure case: cannot merge already merged MemWAL
        let result = mark_mem_wal_as_merged(&mut dataset, "GLOBAL", 0, "owner_0").await;
        assert!(
            result.is_err(),
            "Should fail when trying to merge already merged MemWAL"
        );

        // Check the specific error message
        let error = result.unwrap_err();
        assert!(error.to_string().contains("MemWAL MemWalId { region: \"GLOBAL\", generation: 0 } is in state Merged, but expected Flushed"), 
                "Error message should indicate the MemWAL is already merged, got: {}", error);
    }

    #[tokio::test]
    async fn test_update_mem_wal_owner() {
        // Create a dataset with some data
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(1000))
            .await
            .unwrap();

        // Test failure case: MemWAL is not enabled
        let result = update_mem_wal_owner(
            &mut dataset,
            "GLOBAL",
            0,
            "new_owner_id",
            Some("new_mem_table_location"),
        )
        .await;
        assert!(result.is_err(), "Should fail when MemWAL is not enabled");

        // Create MemWAL index and generation 0
        advance_mem_wal_generation(
            &mut dataset,
            "GLOBAL",
            "mem_table_location_0",
            "wal_location_0",
            None,
            "owner_0",
        )
        .await
        .unwrap();

        // Test failure case: region doesn't exist
        let result = update_mem_wal_owner(
            &mut dataset,
            "NONEXISTENT",
            0,
            "new_owner_id",
            Some("new_mem_table_location"),
        )
        .await;
        assert!(result.is_err(), "Should fail when region doesn't exist");

        // Test failure case: generation doesn't exist
        let result = update_mem_wal_owner(
            &mut dataset,
            "GLOBAL",
            999,
            "new_owner_id",
            Some("new_mem_table_location"),
        )
        .await;
        assert!(result.is_err(), "Should fail when generation doesn't exist");

        // Test failure case: cannot replay with same MemTable location
        let result = update_mem_wal_owner(
            &mut dataset,
            "GLOBAL",
            0,
            "new_owner_id",
            Some("mem_table_location_0"),
        )
        .await;
        assert!(
            result.is_err(),
            "Should fail when using same MemTable location"
        );

        // Check the specific error message
        let error = result.unwrap_err();
        assert!(
            error.to_string().contains(
                "Must use a different MemTable location from current: mem_table_location_0"
            ),
            "Error message should indicate the MemTable location must be different, got: {}",
            error
        );

        // Test success case: start replay with different MemTable location
        let version_before_owner_update = dataset.manifest.version;
        update_mem_wal_owner(
            &mut dataset,
            "GLOBAL",
            0,
            "new_owner_id",
            Some("new_mem_table_location"),
        )
        .await
        .unwrap();

        // Verify the MemTable location was updated
        let indices = dataset.load_indices().await.unwrap();
        let mem_wal_index_meta = indices
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .expect("MemWAL index should exist");

        let mem_wal_details = load_mem_wal_index_details(mem_wal_index_meta.clone()).unwrap();
        let mem_wal = &mem_wal_details.mem_wal_list[0];
        assert_eq!(
            mem_wal.mem_table_location, "new_mem_table_location",
            "MemTable location should be updated"
        );
        // Verify the MemWAL version was updated after owner change
        assert_eq!(
            mem_wal.last_updated_dataset_version,
            version_before_owner_update + 1
        );

        // Test success case: can replay generation 1
        advance_mem_wal_generation(
            &mut dataset,
            "GLOBAL",
            "new_mem_table_location_1",
            "wal_location_1",
            Some("new_owner_id"),
            "owner_1",
        )
        .await
        .unwrap();

        let version_before_gen1_owner_update = dataset.manifest.version;
        update_mem_wal_owner(
            &mut dataset,
            "GLOBAL",
            1,
            "owner_1_new",
            Some("mem_table_location_1"),
        )
        .await
        .unwrap();

        // Verify the MemTable location was updated for generation 1
        let indices = dataset.load_indices().await.unwrap();
        let mem_wal_index_meta = indices
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .expect("MemWAL index should exist");

        let mem_wal_details = load_mem_wal_index_details(mem_wal_index_meta.clone()).unwrap();
        let gen_1 = mem_wal_details
            .mem_wal_list
            .iter()
            .find(|m| m.id.generation == 1)
            .expect("Generation 1 should exist");

        assert_eq!(
            gen_1.mem_table_location, "mem_table_location_1",
            "Generation 1 MemTable location should be updated"
        );
        // Verify the MemWAL version was updated after generation 1 owner change
        assert_eq!(
            gen_1.last_updated_dataset_version,
            version_before_gen1_owner_update + 1
        );
    }

    #[tokio::test]
    async fn test_trim_mem_wal_index_with_reindex() {
        // Create a dataset with some data
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(1000))
            .await
            .unwrap();

        // Test failure case: MemWAL is not enabled
        let result = trim_mem_wal_index(&mut dataset).await;
        assert!(result.is_err(), "Should fail when MemWAL is not enabled");

        // Create MemWAL index and multiple generations
        advance_mem_wal_generation(
            &mut dataset,
            "GLOBAL",
            "mem_table_location_0",
            "wal_location_0",
            None,
            "owner_0",
        )
        .await
        .unwrap();

        advance_mem_wal_generation(
            &mut dataset,
            "GLOBAL",
            "mem_table_location_1",
            "wal_location_1",
            Some("owner_0"),
            "owner_1",
        )
        .await
        .unwrap();

        advance_mem_wal_generation(
            &mut dataset,
            "GLOBAL",
            "mem_table_location_2",
            "wal_location_2",
            Some("owner_1"),
            "owner_2",
        )
        .await
        .unwrap();

        // Verify we have 3 generations initially
        let indices = dataset.load_indices().await.unwrap();
        let mem_wal_index_meta = indices
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .expect("MemWAL index should exist");

        let mem_wal_details = load_mem_wal_index_details(mem_wal_index_meta.clone()).unwrap();
        assert_eq!(
            mem_wal_details.mem_wal_list.len(),
            3,
            "Should have 3 generations initially"
        );

        // flush and merge generation 0
        mark_mem_wal_as_flushed(&mut dataset, "GLOBAL", 0, "owner_0")
            .await
            .unwrap();
        mark_mem_wal_as_merged(&mut dataset, "GLOBAL", 0, "owner_0")
            .await
            .unwrap();

        // Test case 1: No indices exist (besides MemWAL index itself)
        // Should trim merged MemWAL since no other indices exist
        trim_mem_wal_index(&mut dataset).await.unwrap();

        let indices = dataset.load_indices().await.unwrap();
        let mem_wal_index_meta = indices
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .expect("MemWAL index should still exist");

        let mem_wal_details = load_mem_wal_index_details(mem_wal_index_meta.clone()).unwrap();
        assert_eq!(
            mem_wal_details.mem_wal_list.len(),
            2,
            "Should have 2 generations after trimming (no other indices)"
        );

        // Verify generation 0 was removed
        let gen_0_exists = mem_wal_details
            .mem_wal_list
            .iter()
            .any(|m| m.id.generation == 0);
        assert!(!gen_0_exists, "Generation 0 should be removed");

        // Test case 2: Create index after MemWAL flush, then flush another generation
        advance_mem_wal_generation(
            &mut dataset,
            "GLOBAL",
            "mem_table_location_3",
            "wal_location_3",
            Some("owner_2"),
            "owner_3",
        )
        .await
        .unwrap();

        // Seal, flush and merge generation 1
        mark_mem_wal_as_flushed(&mut dataset, "GLOBAL", 1, "owner_1")
            .await
            .unwrap();
        mark_mem_wal_as_merged(&mut dataset, "GLOBAL", 1, "owner_1")
            .await
            .unwrap();

        // Create an index after the MemWAL was merged
        dataset
            .create_index(
                &["i"],
                lance_index::IndexType::Scalar,
                Some("scalar_after".into()),
                &lance_index::scalar::ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // Should trim the merged MemWAL since the index was created after it
        trim_mem_wal_index(&mut dataset).await.unwrap();

        let indices = dataset.load_indices().await.unwrap();
        let mem_wal_index_meta = indices
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .expect("MemWAL index should still exist");

        let mem_wal_details = load_mem_wal_index_details(mem_wal_index_meta.clone()).unwrap();
        assert_eq!(
            mem_wal_details.mem_wal_list.len(),
            2,
            "Should have 2 generations after trimming (index created after MemWAL)"
        );

        // Verify generation 1 was removed
        let gen_1_exists = mem_wal_details
            .mem_wal_list
            .iter()
            .any(|m| m.id.generation == 1);
        assert!(!gen_1_exists, "Generation 1 should be removed");

        // Test case 3: Create index before MemWAL flush
        // Create another index before flushing the next generation
        dataset
            .create_index(
                &["i"],
                lance_index::IndexType::Scalar,
                Some("scalar_before".into()),
                &lance_index::scalar::ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // Now flush and merge generation 2 (created before the vector index)
        mark_mem_wal_as_flushed(&mut dataset, "GLOBAL", 2, "owner_2")
            .await
            .unwrap();
        mark_mem_wal_as_merged(&mut dataset, "GLOBAL", 2, "owner_2")
            .await
            .unwrap();

        // Should NOT trim generation 2 since the index was created before it
        trim_mem_wal_index(&mut dataset).await.unwrap();

        let indices = dataset.load_indices().await.unwrap();
        let mem_wal_index_meta = indices
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .expect("MemWAL index should still exist");

        let mem_wal_details = load_mem_wal_index_details(mem_wal_index_meta.clone()).unwrap();
        assert_eq!(
            mem_wal_details.mem_wal_list.len(),
            2,
            "Should still have 2 generations (index created before MemWAL, so cannot trim)"
        );

        // Verify generation 2 still exists
        let gen_2_exists = mem_wal_details
            .mem_wal_list
            .iter()
            .any(|m| m.id.generation == 2);
        assert!(gen_2_exists, "Generation 2 should still exist");
    }

    #[tokio::test]
    async fn test_trim_mem_wal_index_with_delta_index() {
        // Create a dataset with enough data for vector index clustering
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(5), FragmentRowCount::from(100))
            .await
            .unwrap();

        // Create initial vector index
        dataset
            .create_index(
                &["vec"],
                lance_index::IndexType::Vector,
                Some("vector_index".into()),
                &VectorIndexParams::ivf_pq(8, 8, 8, MetricType::Cosine, 50),
                false,
            )
            .await
            .unwrap();

        // Create MemWAL index and generation 0
        advance_mem_wal_generation(
            &mut dataset,
            "GLOBAL",
            "mem_table_location_0",
            "wal_location_0",
            None,
            "owner_0",
        )
        .await
        .unwrap();

        // Seal the MemWAL
        mark_mem_wal_as_sealed(&mut dataset, "GLOBAL", 0, "owner_0")
            .await
            .unwrap();

        // Append new data files to the dataset (without rewriting existing files)
        let new_data = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col(
                "i",
                lance_datagen::array::step_custom::<Int32Type>(500, 1000),
            )
            .into_reader_rows(RowCount::from(100), BatchCount::from(5));

        // Append some new data
        let write_params = WriteParams {
            mode: WriteMode::Append,
            ..WriteParams::default()
        };
        dataset = Dataset::write(
            new_data,
            WriteDestination::Dataset(Arc::new(dataset)),
            Some(write_params),
        )
        .await
        .unwrap();

        // Flush and merge the MemWAL separately
        mark_mem_wal_as_flushed(&mut dataset, "GLOBAL", 0, "owner_0")
            .await
            .unwrap();
        mark_mem_wal_as_merged(&mut dataset, "GLOBAL", 0, "owner_0")
            .await
            .unwrap();

        // Verify the MemWAL is now merged
        let indices = dataset.load_indices().await.unwrap();
        let mem_wal_index_meta = indices
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .expect("MemWAL index should exist");
        let mem_wal_details = load_mem_wal_index_details(mem_wal_index_meta.clone()).unwrap();
        assert_eq!(mem_wal_details.mem_wal_list.len(), 1);
        let mem_wal = &mem_wal_details.mem_wal_list[0];
        assert_eq!(mem_wal.state, lance_index::mem_wal::State::Merged);

        // Now use optimize_indices to create delta index (this is how delta indices are actually created)
        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();

        // Verify we now have multiple indices with the same name (delta indices)
        let indices = dataset.load_indices().await.unwrap();
        let vector_indices: Vec<_> = indices
            .iter()
            .filter(|idx| idx.name == "vector_index")
            .collect();
        assert_eq!(vector_indices.len(), 2);
        // If we have delta indices, verify they work correctly
        // Verify the delta index has a higher dataset version than the original
        let mut versions: Vec<_> = vector_indices
            .iter()
            .map(|idx| idx.dataset_version)
            .collect();
        versions.sort();
        assert!(
            versions[versions.len() - 1] > versions[0],
            "Latest delta index should have higher dataset version than original"
        );

        // Now the MemWAL should be trimmed because the delta index was created after the merge
        // Our logic should take the maximum dataset version for each index name
        trim_mem_wal_index(&mut dataset).await.unwrap();

        let indices = dataset.load_indices().await.unwrap();
        let mem_wal_index_meta = indices
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .expect("MemWAL index should still exist");
        let mem_wal_details = load_mem_wal_index_details(mem_wal_index_meta.clone()).unwrap();
        assert_eq!(
            mem_wal_details.mem_wal_list.len(),
            0,
            "MemWAL should be trimmed because delta index was created after flush"
        );
    }

    #[tokio::test]
    async fn test_flush_mem_wal_through_merge_insert() {
        // Create a dataset with some data
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(1000))
            .await
            .unwrap();

        // Create MemWAL index and generation 0
        advance_mem_wal_generation(
            &mut dataset,
            "GLOBAL",
            "mem_table_location_0",
            "wal_location_0",
            None,
            "owner_0",
        )
        .await
        .unwrap();

        // Add some entries to the MemWAL
        append_mem_wal_entry(&mut dataset, "GLOBAL", 0, 123, "owner_0")
            .await
            .unwrap();
        append_mem_wal_entry(&mut dataset, "GLOBAL", 0, 456, "owner_0")
            .await
            .unwrap();

        // Seal and flush the MemWAL (required before merging)
        mark_mem_wal_as_sealed(&mut dataset, "GLOBAL", 0, "owner_0")
            .await
            .unwrap();
        mark_mem_wal_as_flushed(&mut dataset, "GLOBAL", 0, "owner_0")
            .await
            .unwrap();

        // Verify the MemWAL is flushed but not merged
        let indices = dataset.load_indices().await.unwrap();
        let mem_wal_index_meta = indices
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .expect("MemWAL index should exist");

        let mem_wal_details = load_mem_wal_index_details(mem_wal_index_meta.clone()).unwrap();
        let mem_wal = &mem_wal_details.mem_wal_list[0];
        assert_eq!(
            mem_wal.state,
            lance_index::mem_wal::State::Flushed,
            "MemWAL should be flushed but not merged yet"
        );

        // Create new data for merge insert
        let new_data = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("i", lance_datagen::array::step_custom::<Int32Type>(1000, 1))
            .into_df_stream(RowCount::from(100), BatchCount::from(10));

        // Create merge insert job that will merge the MemWAL
        let merge_insert_job = crate::dataset::MergeInsertBuilder::try_new(
            Arc::new(dataset.clone()),
            vec!["i".to_string()],
        )
        .unwrap()
        .when_matched(crate::dataset::WhenMatched::UpdateAll)
        .when_not_matched(crate::dataset::WhenNotMatched::InsertAll)
        .mark_mem_wal_as_merged(MemWalId::new("GLOBAL", 0), "owner_0")
        .await
        .unwrap()
        .try_build()
        .unwrap();

        // Execute the merge insert
        let (updated_dataset, _stats) = merge_insert_job.execute_reader(new_data).await.unwrap();

        // Verify that the MemWAL is now marked as merged
        let indices = updated_dataset.load_indices().await.unwrap();
        let mem_wal_index_meta = indices
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .expect("MemWAL index should still exist");

        let mem_wal_details = load_mem_wal_index_details(mem_wal_index_meta.clone()).unwrap();
        let mem_wal = &mem_wal_details.mem_wal_list[0];
        assert_eq!(
            mem_wal.state,
            lance_index::mem_wal::State::Merged,
            "MemWAL should now be merged"
        );

        // Test that trying to mark a non-existent MemWAL as merged fails
        let mut merge_insert_job = crate::dataset::MergeInsertBuilder::try_new(
            updated_dataset.clone(),
            vec!["i".to_string()],
        )
        .unwrap();
        merge_insert_job
            .when_matched(crate::dataset::WhenMatched::UpdateAll)
            .when_not_matched(crate::dataset::WhenNotMatched::InsertAll);

        let result = merge_insert_job
            .mark_mem_wal_as_merged(MemWalId::new("GLOBAL", 999), "owner_0")
            .await;
        assert!(
            result.is_err(),
            "Should fail when trying to mark non-existent MemWAL as merged"
        );

        // Test that trying to mark a MemWAL from non-existent region fails
        let result = merge_insert_job
            .mark_mem_wal_as_merged(MemWalId::new("NONEXISTENT", 0), "owner_0")
            .await;
        assert!(
            result.is_err(),
            "Should fail when trying to mark MemWAL from non-existent region as merged"
        );

        // Test that trying to mark an unflushed MemWAL as merged fails
        // First, create a new generation that is unsealed
        let mut dataset_for_advance = updated_dataset.as_ref().clone();
        advance_mem_wal_generation(
            &mut dataset_for_advance,
            "GLOBAL",
            "mem_table_location_1",
            "wal_location_1",
            Some("owner_0"),
            "owner_1",
        )
        .await
        .unwrap();

        // Update our reference to use the new dataset
        let updated_dataset = Arc::new(dataset_for_advance);

        // Verify that generation 1 exists and is unsealed
        let indices = updated_dataset.load_indices().await.unwrap();
        let mem_wal_index_meta = indices
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .expect("MemWAL index should exist");

        let mem_wal_details = load_mem_wal_index_details(mem_wal_index_meta.clone()).unwrap();
        let gen_1 = mem_wal_details
            .mem_wal_list
            .iter()
            .find(|m| m.id.generation == 1)
            .expect("Generation 1 should exist");
        assert_eq!(
            gen_1.state,
            lance_index::mem_wal::State::Open,
            "Generation 1 should be open"
        );

        let mut merge_insert_job_unsealed = crate::dataset::MergeInsertBuilder::try_new(
            updated_dataset.clone(),
            vec!["i".to_string()],
        )
        .unwrap();
        merge_insert_job_unsealed
            .when_matched(crate::dataset::WhenMatched::UpdateAll)
            .when_not_matched(crate::dataset::WhenNotMatched::InsertAll);

        let result = merge_insert_job_unsealed
            .mark_mem_wal_as_merged(MemWalId::new("GLOBAL", 1), "owner_1")
            .await;
        assert!(
            result.is_err(),
            "Should fail when trying to mark unsealed MemWAL as merged"
        );

        // Check the specific error message
        let error = result.unwrap_err();
        assert!(error.to_string().contains("MemWAL MemWalId { region: \"GLOBAL\", generation: 1 } is in state Open, but expected Flushed"),
                "Error message should indicate the MemWAL is not flushed, got: {}", error);

        // Test that trying to mark an already merged MemWAL as merged fails
        let mut merge_insert_job_merged = crate::dataset::MergeInsertBuilder::try_new(
            updated_dataset.clone(),
            vec!["i".to_string()],
        )
        .unwrap();
        merge_insert_job_merged
            .when_matched(crate::dataset::WhenMatched::UpdateAll)
            .when_not_matched(crate::dataset::WhenNotMatched::InsertAll);

        let result = merge_insert_job_merged
            .mark_mem_wal_as_merged(MemWalId::new("GLOBAL", 0), "owner_1")
            .await;
        assert!(
            result.is_err(),
            "Should fail when trying to mark already merged MemWAL as merged"
        );

        // Check the specific error message
        let error = result.unwrap_err();
        assert!(error.to_string().contains("MemWAL MemWalId { region: \"GLOBAL\", generation: 0 } is in state Merged, but expected Flushed"),
                "Error message should indicate the MemWAL is already merged, got: {}", error);

        // Test that merge insert with mark_mem_wal_as_merged works correctly when MemWAL is in proper state
        // Seal and flush generation 1 and then test the merge insert
        let mut dataset_for_seal = updated_dataset.as_ref().clone();
        mark_mem_wal_as_sealed(&mut dataset_for_seal, "GLOBAL", 1, "owner_1")
            .await
            .unwrap();
        mark_mem_wal_as_flushed(&mut dataset_for_seal, "GLOBAL", 1, "owner_1")
            .await
            .unwrap();
        let updated_dataset = Arc::new(dataset_for_seal);

        // Verify generation 1 is now flushed but not merged
        let indices = updated_dataset.load_indices().await.unwrap();
        let mem_wal_index_meta = indices
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .expect("MemWAL index should exist");

        let mem_wal_details = load_mem_wal_index_details(mem_wal_index_meta.clone()).unwrap();
        let gen_1 = mem_wal_details
            .mem_wal_list
            .iter()
            .find(|m| m.id.generation == 1)
            .expect("Generation 1 should exist");
        assert_eq!(
            gen_1.state,
            lance_index::mem_wal::State::Flushed,
            "Generation 1 should be flushed"
        );

        // Create merge insert that merges generation 1
        let new_data_valid = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("i", lance_datagen::array::step_custom::<Int32Type>(4000, 1))
            .into_df_stream(RowCount::from(75), BatchCount::from(5));

        let merge_insert_job_valid = crate::dataset::MergeInsertBuilder::try_new(
            updated_dataset.clone(),
            vec!["i".to_string()],
        )
        .unwrap()
        .when_matched(crate::dataset::WhenMatched::UpdateAll)
        .when_not_matched(crate::dataset::WhenNotMatched::InsertAll)
        .mark_mem_wal_as_merged(MemWalId::new("GLOBAL", 1), "owner_1")
        .await
        .unwrap()
        .try_build()
        .unwrap();

        // Execute the merge insert - this should succeed
        let (final_dataset, _stats) = merge_insert_job_valid
            .execute_reader(new_data_valid)
            .await
            .unwrap();

        // Verify that the MemWAL is now marked as merged
        let indices = final_dataset.load_indices().await.unwrap();
        let mem_wal_index_meta = indices
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .expect("MemWAL index should still exist");

        let mem_wal_details = load_mem_wal_index_details(mem_wal_index_meta.clone()).unwrap();
        let gen_1 = mem_wal_details
            .mem_wal_list
            .iter()
            .find(|m| m.id.generation == 1)
            .expect("Generation 1 should still exist");
        assert_eq!(
            gen_1.state,
            lance_index::mem_wal::State::Merged,
            "Generation 1 should now be merged"
        );
    }

    #[tokio::test]
    async fn test_replay_mem_wal_with_split_brain_writer() {
        // Create a dataset with some data
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(1000))
            .await
            .unwrap();

        // Create MemWAL index and generation 0
        advance_mem_wal_generation(
            &mut dataset,
            "GLOBAL",
            "mem_table_location_0",
            "wal_location_0",
            None,
            "owner_0",
        )
        .await
        .unwrap();

        // Add some entries to the MemWAL
        append_mem_wal_entry(&mut dataset, "GLOBAL", 0, 123, "owner_0")
            .await
            .unwrap();
        append_mem_wal_entry(&mut dataset, "GLOBAL", 0, 456, "owner_0")
            .await
            .unwrap();

        // Simulate a network partition scenario where another node starts replay
        // This changes the MemTable location from "mem_table_location_0" to "new_mem_table_location"
        update_mem_wal_owner(
            &mut dataset,
            "GLOBAL",
            0,
            "new_owner_id",
            Some("new_mem_table_location"),
        )
        .await
        .unwrap();

        // Verify the MemTable location was updated
        let indices = dataset.load_indices().await.unwrap();
        let mem_wal_index_meta = indices
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .expect("MemWAL index should exist");

        let mem_wal_details = load_mem_wal_index_details(mem_wal_index_meta.clone()).unwrap();
        let mem_wal = &mem_wal_details.mem_wal_list[0];
        assert_eq!(
            mem_wal.mem_table_location, "new_mem_table_location",
            "MemTable location should be updated after replay"
        );

        // Now simulate a split-brain scenario where the original writer (node A)
        // tries to perform operations using the old MemTable location

        // Test 1: append_mem_wal_entry with old owner_id should fail
        let result = append_mem_wal_entry(&mut dataset, "GLOBAL", 0, 789, "owner_0").await;
        assert!(
            result.is_err(),
            "Should fail when using old owner_id for append"
        );

        // Check the specific error message
        let error = result.unwrap_err();
        assert!(error.to_string().contains("MemWAL MemWalId { region: \"GLOBAL\", generation: 0 } has owner_id: new_owner_id, but expected owner_0"), 
                "Error message should indicate owner_id mismatch, got: {}", error);

        // Test 2: mark_mem_wal_as_sealed with old owner_id should fail
        let result = mark_mem_wal_as_sealed(&mut dataset, "GLOBAL", 0, "owner_0").await;
        assert!(
            result.is_err(),
            "Should fail when using old owner_id for seal"
        );

        // Check the specific error message
        let error = result.unwrap_err();
        assert!(error.to_string().contains("MemWAL MemWalId { region: \"GLOBAL\", generation: 0 } has owner_id: new_owner_id, but expected owner_0"), 
                "Error message should indicate owner_id mismatch, got: {}", error);

        // Test 3: mark_mem_wal_as_flushed with old owner_id should fail
        // First seal the MemWAL using the correct owner_id
        mark_mem_wal_as_sealed(&mut dataset, "GLOBAL", 0, "new_owner_id")
            .await
            .unwrap();

        let result = mark_mem_wal_as_flushed(&mut dataset, "GLOBAL", 0, "owner_0").await;
        assert!(
            result.is_err(),
            "Should fail when using old owner_id for flush"
        );

        // Check the specific error message
        let error = result.unwrap_err();
        assert!(error.to_string().contains("MemWAL MemWalId { region: \"GLOBAL\", generation: 0 } has owner_id: new_owner_id, but expected owner_0"), 
                "Error message should indicate owner_id mismatch, got: {}", error);

        // Test 4: advance_mem_wal_generation with old owner_id should fail
        let result = advance_mem_wal_generation(
            &mut dataset,
            "GLOBAL",
            "mem_table_location_1",
            "wal_location_1",
            Some("owner_0"), // Using old owner_id
            "owner_1",
        )
        .await;
        assert!(
            result.is_err(),
            "Should fail when using old owner_id for advance generation"
        );

        // Check the specific error message
        let error = result.unwrap_err();
        assert!(error.to_string().contains("MemWAL MemWalId { region: \"GLOBAL\", generation: 0 } has owner_id: new_owner_id, but expected owner_0"), 
                "Error message should indicate owner_id mismatch, got: {}", error);

        // Test 5: merge_insert with mark_mem_wal_as_merged using old owner_id should fail
        // First flush the MemWAL using the correct owner_id so it's ready for merging
        mark_mem_wal_as_flushed(&mut dataset, "GLOBAL", 0, "new_owner_id")
            .await
            .unwrap();

        // Try to create merge insert job that merges using the old owner_id
        let mut merge_insert_job_builder = crate::dataset::MergeInsertBuilder::try_new(
            Arc::new(dataset.clone()),
            vec!["i".to_string()],
        )
        .unwrap();

        let build_result = merge_insert_job_builder
            .when_matched(crate::dataset::WhenMatched::UpdateAll)
            .when_not_matched(crate::dataset::WhenNotMatched::InsertAll)
            .mark_mem_wal_as_merged(MemWalId::new("GLOBAL", 0), "owner_0") // Using old owner_id
            .await;

        assert!(
            build_result.is_err(),
            "Should fail when using old owner_id for merge insert merge"
        );

        // Check the specific error message
        let error = build_result.unwrap_err();
        assert!(error.to_string().contains("MemWAL MemWalId { region: \"GLOBAL\", generation: 0 } has owner_id: new_owner_id, but expected owner_0"), 
                "Error message should indicate owner_id mismatch for merge insert, got: {}", error);
    }

    #[tokio::test]
    async fn test_concurrent_mem_wal_replay_and_modifications() {
        // Create a dataset with some data
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(1000))
            .await
            .unwrap();

        // Create MemWAL index and generation 0
        advance_mem_wal_generation(
            &mut dataset,
            "GLOBAL",
            "mem_table_location_0",
            "wal_location_0",
            None,
            "owner_0",
        )
        .await
        .unwrap();

        // Add some entries to the MemWAL
        append_mem_wal_entry(&mut dataset, "GLOBAL", 0, 123, "owner_0")
            .await
            .unwrap();
        append_mem_wal_entry(&mut dataset, "GLOBAL", 0, 456, "owner_0")
            .await
            .unwrap();

        // Clone the dataset multiple times to simulate concurrent operations
        let mut dataset_clone_append = dataset.clone();
        let mut dataset_clone_seal = dataset.clone();
        let mut dataset_clone_flush = dataset.clone();
        let mut dataset_clone_advance = dataset.clone();

        // Start replay operation on the original dataset
        let replay_result = update_mem_wal_owner(
            &mut dataset,
            "GLOBAL",
            0,
            "new_owner_id",
            Some("new_mem_table_location"),
        )
        .await;

        // Test all concurrent operations against the replay
        let append_result =
            append_mem_wal_entry(&mut dataset_clone_append, "GLOBAL", 0, 789, "owner_0").await;
        let seal_result =
            mark_mem_wal_as_sealed(&mut dataset_clone_seal, "GLOBAL", 0, "owner_0").await;
        let flush_result =
            mark_mem_wal_as_flushed(&mut dataset_clone_flush, "GLOBAL", 0, "owner_0").await;
        let advance_result = advance_mem_wal_generation(
            &mut dataset_clone_advance,
            "GLOBAL",
            "mem_table_location_1",
            "wal_location_1",
            Some("owner_0"),
            "owner_1",
        )
        .await;

        // Test merge_insert merge operation separately (requires flushed MemWAL)
        // Advance to a new generation and seal it for merge insert test
        advance_mem_wal_generation(
            &mut dataset,
            "GLOBAL",
            "mem_table_location_1",
            "wal_location_1",
            Some("new_owner_id"),
            "owner_1",
        )
        .await
        .unwrap();

        // Seal and flush the new generation
        mark_mem_wal_as_sealed(&mut dataset, "GLOBAL", 1, "owner_1")
            .await
            .unwrap();
        mark_mem_wal_as_flushed(&mut dataset, "GLOBAL", 1, "owner_1")
            .await
            .unwrap();

        let dataset_clone_merge_insert = dataset.clone();

        // Start replay operation on the new generation
        let replay_result_merge_insert = update_mem_wal_owner(
            &mut dataset,
            "GLOBAL",
            1,
            "new_owner_id",
            Some("new_mem_table_location_merge"),
        )
        .await;

        // Test merge_insert merge operation
        let mut merge_insert_job_builder = crate::dataset::MergeInsertBuilder::try_new(
            Arc::new(dataset_clone_merge_insert),
            vec!["i".to_string()],
        )
        .unwrap();

        let merge_insert_job = merge_insert_job_builder
            .when_matched(crate::dataset::WhenMatched::UpdateAll)
            .when_not_matched(crate::dataset::WhenNotMatched::InsertAll)
            .mark_mem_wal_as_merged(MemWalId::new("GLOBAL", 1), "owner_1")
            .await
            .unwrap()
            .try_build()
            .unwrap();

        // Create some data for the merge insert
        let new_data = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("i", lance_datagen::array::step_custom::<Int32Type>(2000, 1))
            .into_df_stream(RowCount::from(50), BatchCount::from(5));

        // Execute the merge insert (this should fail due to version conflict)
        let merge_insert_result = merge_insert_job.execute_reader(new_data).await;

        // Replay should succeed and all other operations should fail due to version conflict
        assert!(replay_result.is_ok(), "Replay operation should succeed");
        assert!(
            append_result.is_err(),
            "Append operation should fail due to version conflict"
        );
        assert!(
            seal_result.is_err(),
            "Seal operation should fail due to version conflict"
        );
        assert!(
            flush_result.is_err(),
            "Flush operation should fail due to version conflict"
        );
        assert!(
            advance_result.is_err(),
            "Advance generation operation should fail due to version conflict"
        );

        // For merge insert test, replay should succeed and merge insert should fail
        assert!(
            replay_result_merge_insert.is_ok(),
            "Replay operation for merge insert test should succeed"
        );
        assert!(
            merge_insert_result.is_err(),
            "Merge insert flush operation should fail due to version conflict"
        );
    }

    #[tokio::test]
    async fn test_concurrent_mem_wal_append_and_merge_insert_flush() {
        // Create a dataset with some data
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(1000))
            .await
            .unwrap();

        // Create MemWAL index and generation 0
        advance_mem_wal_generation(
            &mut dataset,
            "GLOBAL",
            "mem_table_location_0",
            "wal_location_0",
            None,
            "owner_0",
        )
        .await
        .unwrap();

        // Add some entries to generation 0
        append_mem_wal_entry(&mut dataset, "GLOBAL", 0, 123, "owner_0")
            .await
            .unwrap();
        append_mem_wal_entry(&mut dataset, "GLOBAL", 0, 456, "owner_0")
            .await
            .unwrap();

        // Seal and flush generation 0 (required for merge insert merge)
        mark_mem_wal_as_sealed(&mut dataset, "GLOBAL", 0, "owner_0")
            .await
            .unwrap();
        mark_mem_wal_as_flushed(&mut dataset, "GLOBAL", 0, "owner_0")
            .await
            .unwrap();

        // Advance to generation 1
        advance_mem_wal_generation(
            &mut dataset,
            "GLOBAL",
            "mem_table_location_1",
            "wal_location_1",
            Some("owner_0"),
            "owner_1",
        )
        .await
        .unwrap();

        // Add some entries to generation 1
        append_mem_wal_entry(&mut dataset, "GLOBAL", 1, 789, "owner_1")
            .await
            .unwrap();
        append_mem_wal_entry(&mut dataset, "GLOBAL", 1, 790, "owner_1")
            .await
            .unwrap();

        // Clone the dataset to simulate concurrent operations
        let mut dataset_clone_append = dataset.clone();
        let dataset_clone_merge_insert = dataset.clone();

        // Test concurrent operations: append to generation 1 and merge_insert merge generation 0
        let append_result =
            append_mem_wal_entry(&mut dataset_clone_append, "GLOBAL", 1, 791, "owner_1").await;

        // Create merge insert job that merges generation 0
        let mut merge_insert_job_builder = crate::dataset::MergeInsertBuilder::try_new(
            Arc::new(dataset_clone_merge_insert),
            vec!["i".to_string()],
        )
        .unwrap();

        let merge_insert_job = merge_insert_job_builder
            .when_matched(crate::dataset::WhenMatched::UpdateAll)
            .when_not_matched(crate::dataset::WhenNotMatched::InsertAll)
            .mark_mem_wal_as_merged(MemWalId::new("GLOBAL", 0), "owner_0")
            .await
            .unwrap()
            .try_build()
            .unwrap();

        // Create some data for the merge insert
        let new_data = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("i", lance_datagen::array::step_custom::<Int32Type>(2000, 1))
            .into_df_stream(RowCount::from(50), BatchCount::from(5));

        // Execute the merge insert
        let merge_insert_result = merge_insert_job.execute_reader(new_data).await;

        // Both operations should succeed since they operate on different generations
        assert!(
            append_result.is_ok(),
            "Append to generation 1 should succeed"
        );
        assert!(
            merge_insert_result.is_ok(),
            "Merge insert flush of generation 0 should succeed"
        );

        // Get the updated dataset from the merge insert result
        let (updated_dataset, _stats) = merge_insert_result.unwrap();

        // Verify the final state using the updated dataset
        let indices = updated_dataset.load_indices().await.unwrap();
        let mem_wal_index_meta = indices
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .expect("MemWAL index should exist");

        let mem_wal_details = load_mem_wal_index_details(mem_wal_index_meta.clone()).unwrap();

        // Find generation 0 and generation 1
        let gen_0 = mem_wal_details
            .mem_wal_list
            .iter()
            .find(|m| m.id.generation == 0)
            .expect("Generation 0 should exist");
        let gen_1 = mem_wal_details
            .mem_wal_list
            .iter()
            .find(|m| m.id.generation == 1)
            .expect("Generation 1 should exist");

        // Verify generation 0 is merged (after merge_insert)
        assert_eq!(
            gen_0.state,
            lance_index::mem_wal::State::Merged,
            "Generation 0 should be merged"
        );

        // Verify generation 1 is unsealed and unflushed
        assert_eq!(
            gen_1.state,
            lance_index::mem_wal::State::Open,
            "Generation 1 should be open"
        );

        // Verify that generation 1 has the new entry
        let wal_entries = gen_1.wal_entries();
        assert!(
            wal_entries.contains(791),
            "Generation 1 should contain the new entry 791"
        );
    }

    #[tokio::test]
    async fn test_concurrent_mem_wal_advance_and_merge_insert_flush() {
        // Create a dataset with some data
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(1000))
            .await
            .unwrap();

        // Create MemWAL index and generation 0
        advance_mem_wal_generation(
            &mut dataset,
            "GLOBAL",
            "mem_table_location_0",
            "wal_location_0",
            None,
            "owner_0",
        )
        .await
        .unwrap();

        // Add some entries to generation 0
        append_mem_wal_entry(&mut dataset, "GLOBAL", 0, 123, "owner_0")
            .await
            .unwrap();
        append_mem_wal_entry(&mut dataset, "GLOBAL", 0, 456, "owner_0")
            .await
            .unwrap();

        // Seal and flush generation 0 (required for merge insert merge)
        mark_mem_wal_as_sealed(&mut dataset, "GLOBAL", 0, "owner_0")
            .await
            .unwrap();
        mark_mem_wal_as_flushed(&mut dataset, "GLOBAL", 0, "owner_0")
            .await
            .unwrap();

        // Advance to generation 1
        advance_mem_wal_generation(
            &mut dataset,
            "GLOBAL",
            "mem_table_location_1",
            "wal_location_1",
            Some("owner_0"),
            "owner_1",
        )
        .await
        .unwrap();

        // Add some entries to generation 1
        append_mem_wal_entry(&mut dataset, "GLOBAL", 1, 789, "owner_1")
            .await
            .unwrap();
        append_mem_wal_entry(&mut dataset, "GLOBAL", 1, 790, "owner_1")
            .await
            .unwrap();

        // Clone the dataset to simulate concurrent operations
        let mut dataset_clone_advance = dataset.clone();
        let dataset_clone_merge_insert = dataset.clone();

        // Test concurrent operations: advance to generation 2 and merge_insert flush generation 0
        let advance_result = advance_mem_wal_generation(
            &mut dataset_clone_advance,
            "GLOBAL",
            "mem_table_location_2",
            "wal_location_2",
            Some("owner_1"),
            "owner_2",
        )
        .await;

        // Create merge insert job that merges generation 0
        let mut merge_insert_job_builder = crate::dataset::MergeInsertBuilder::try_new(
            Arc::new(dataset_clone_merge_insert),
            vec!["i".to_string()],
        )
        .unwrap();

        let merge_insert_job = merge_insert_job_builder
            .when_matched(crate::dataset::WhenMatched::UpdateAll)
            .when_not_matched(crate::dataset::WhenNotMatched::InsertAll)
            .mark_mem_wal_as_merged(MemWalId::new("GLOBAL", 0), "owner_0")
            .await
            .unwrap()
            .try_build()
            .unwrap();

        // Create some data for the merge insert
        let new_data = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("i", lance_datagen::array::step_custom::<Int32Type>(2000, 1))
            .into_df_stream(RowCount::from(50), BatchCount::from(5));

        // Execute the merge insert
        let merge_insert_result = merge_insert_job.execute_reader(new_data).await;

        // Both operations should succeed since they operate on different generations
        assert!(
            advance_result.is_ok(),
            "Advance to generation 2 should succeed"
        );
        assert!(
            merge_insert_result.is_ok(),
            "Merge insert flush of generation 0 should succeed"
        );

        // Get the updated dataset from the merge insert result
        let (updated_dataset, _stats) = merge_insert_result.unwrap();

        // Verify the final state using the updated dataset
        let indices = updated_dataset.load_indices().await.unwrap();
        let mem_wal_index_meta = indices
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .expect("MemWAL index should exist");

        let mem_wal_details = load_mem_wal_index_details(mem_wal_index_meta.clone()).unwrap();

        // Find all generations
        let gen_0 = mem_wal_details
            .mem_wal_list
            .iter()
            .find(|m| m.id.generation == 0)
            .expect("Generation 0 should exist");
        let gen_1 = mem_wal_details
            .mem_wal_list
            .iter()
            .find(|m| m.id.generation == 1)
            .expect("Generation 1 should exist");
        let gen_2 = mem_wal_details
            .mem_wal_list
            .iter()
            .find(|m| m.id.generation == 2)
            .expect("Generation 2 should exist");

        // Verify generation 0 is merged (after merge_insert)
        assert_eq!(
            gen_0.state,
            lance_index::mem_wal::State::Merged,
            "Generation 0 should be merged"
        );

        // Verify generation 1 is sealed (due to advance) but unflushed
        assert_eq!(
            gen_1.state,
            lance_index::mem_wal::State::Sealed,
            "Generation 1 should be sealed due to advance"
        );

        // Verify generation 2 is unsealed and unflushed
        assert_eq!(
            gen_2.state,
            lance_index::mem_wal::State::Open,
            "Generation 2 should be open"
        );

        // Verify that generation 1 has the expected entries
        let wal_entries = gen_1.wal_entries();
        assert!(
            wal_entries.contains(789),
            "Generation 1 should contain entry 789"
        );
        assert!(
            wal_entries.contains(790),
            "Generation 1 should contain entry 790"
        );
    }
}
