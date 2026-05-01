// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;

use bytes::Bytes;
use lance::Dataset;
use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use object_store::{PutMode, PutOptions};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

const GROUP_OFFSET_PREFIX: &str = "lance_queue.group";
const GROUP_COMMIT_DELIMITER: &str = ".commits";
const GROUP_COMMIT_PREFIX_DELIMITER: &str = ".commits.";
const GROUP_OFFSET_SUFFIX: &str = ".next_entry_position";
const MAX_OFFSET_COMMIT_RETRIES: usize = 10;

/// Consumer start position when no committed offset exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StartPosition {
    /// Start from the first WAL entry.
    Earliest,
    /// Start after the current last WAL entry.
    Latest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConsumerGroupOffset {
    pub partition_id: u32,
    pub producer_id: u32,
    pub next_entry_position: u64,
}

pub fn read_all_group_offsets(
    table_metadata: &HashMap<String, String>,
    group_id: &str,
) -> Result<Vec<ConsumerGroupOffset>> {
    validate_group_id(group_id)?;
    let prefix = group_offset_key_prefix(group_id);
    let mut offsets = table_metadata
        .iter()
        .filter_map(|(key, value)| {
            let shard_key = key
                .strip_prefix(&prefix)?
                .strip_suffix(GROUP_OFFSET_SUFFIX)?;
            Some((shard_key, value, key))
        })
        .map(|(shard_key, value, key)| {
            let (partition, producer) = shard_key.split_once('.').ok_or_else(|| {
                Error::invalid_input(format!(
                    "failed to parse consumer group offset metadata key '{}': expected partition.producer",
                    key
                ))
            })?;
            let partition_id = partition.parse::<u32>().map_err(|e| {
                Error::invalid_input(format!(
                    "failed to parse consumer group offset partition from metadata key '{}': {}",
                    key, e
                ))
            })?;
            let producer_id = producer.parse::<u32>().map_err(|e| {
                Error::invalid_input(format!(
                    "failed to parse consumer group offset producer from metadata key '{}': {}",
                    key, e
                ))
            })?;
            let next_entry_position = value.parse::<u64>().map_err(|e| {
                Error::invalid_input(format!(
                    "failed to parse consumer group offset value for metadata key '{}': {}",
                    key, e
                ))
            })?;
            Ok(ConsumerGroupOffset {
                partition_id,
                producer_id,
                next_entry_position,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    offsets.sort_by_key(|offset| (offset.partition_id, offset.producer_id));
    Ok(offsets)
}

pub async fn write_group_offsets(
    dataset: &Dataset,
    group_id: &str,
    offsets: &[ConsumerGroupOffset],
) -> Result<()> {
    validate_group_id(group_id)?;
    if offsets.is_empty() {
        return Ok(());
    }

    for attempt in 0..MAX_OFFSET_COMMIT_RETRIES {
        let mut dataset = dataset.clone();
        dataset.checkout_latest().await?;
        let existing_offsets = read_all_group_offsets(dataset.metadata(), group_id)?;
        let updates = offset_updates(group_id, &existing_offsets, offsets);
        if updates.is_empty() {
            return Ok(());
        }

        match dataset.update_metadata(updates).await {
            Ok(_) => return Ok(()),
            Err(error)
                if attempt + 1 < MAX_OFFSET_COMMIT_RETRIES
                    && is_retryable_metadata_commit_error(&error) =>
            {
                continue;
            }
            Err(error) => return Err(error),
        }
    }

    Err(Error::io(format!(
        "failed to commit consumer group offsets for group_id '{}' after {} retries",
        group_id, MAX_OFFSET_COMMIT_RETRIES
    )))
}

pub fn validate_group_id(group_id: &str) -> Result<()> {
    if group_id.is_empty() {
        return Err(Error::invalid_input("consumer group_id cannot be empty"));
    }
    if group_id == "." || group_id == ".." {
        return Err(Error::invalid_input(format!(
            "consumer group_id '{}' cannot be a relative path segment",
            group_id
        )));
    }
    if group_id.contains('/') || group_id.contains('\\') {
        return Err(Error::invalid_input(format!(
            "consumer group_id '{}' cannot contain path separators",
            group_id
        )));
    }
    if group_id.ends_with(GROUP_COMMIT_DELIMITER)
        || group_id.contains(GROUP_COMMIT_PREFIX_DELIMITER)
    {
        return Err(Error::invalid_input(format!(
            "consumer group_id '{}' cannot contain the reserved metadata delimiter '{}'",
            group_id, GROUP_COMMIT_DELIMITER
        )));
    }
    Ok(())
}

fn offset_updates(
    group_id: &str,
    existing_offsets: &[ConsumerGroupOffset],
    requested_offsets: &[ConsumerGroupOffset],
) -> Vec<(String, String)> {
    requested_offsets
        .iter()
        .filter_map(|requested| {
            let existing = existing_offsets
                .iter()
                .find(|offset| {
                    offset.partition_id == requested.partition_id
                        && offset.producer_id == requested.producer_id
                })
                .map(|offset| offset.next_entry_position)
                .unwrap_or(0);
            if requested.next_entry_position < existing {
                tracing::warn!(
                    group_id,
                    partition_id = requested.partition_id,
                    producer_id = requested.producer_id,
                    existing_next_entry_position = existing,
                    requested_next_entry_position = requested.next_entry_position,
                    "dropping stale consumer group offset commit"
                );
            }
            let next_entry_position = existing.max(requested.next_entry_position);
            (next_entry_position > existing).then(|| {
                (
                    group_offset_key(group_id, requested.partition_id, requested.producer_id),
                    next_entry_position.to_string(),
                )
            })
        })
        .collect()
}

fn group_offset_key_prefix(group_id: &str) -> String {
    format!("{GROUP_OFFSET_PREFIX}.{group_id}.commits.")
}

fn group_offset_key(group_id: &str, partition_id: u32, producer_id: u32) -> String {
    format!(
        "{}{}.{}{}",
        group_offset_key_prefix(group_id),
        partition_id,
        producer_id,
        GROUP_OFFSET_SUFFIX
    )
}

fn is_retryable_metadata_commit_error(error: &Error) -> bool {
    matches!(
        error,
        Error::CommitConflict { .. }
            | Error::RetryableCommitConflict { .. }
            | Error::IncompatibleTransaction { .. }
            | Error::VersionConflict { .. }
    )
}

pub enum PutCreateError {
    AlreadyExists,
    Other(Error),
}

pub async fn put_create(
    object_store: &ObjectStore,
    dir: &Path,
    filename: &str,
    bytes: Bytes,
) -> std::result::Result<(), PutCreateError> {
    let path = dir.child(filename);
    if object_store.is_local() {
        let temp_filename = format!("{}.tmp.{}", filename, Uuid::new_v4());
        let temp_path = dir.child(temp_filename);
        object_store
            .inner
            .put(&temp_path, bytes.into())
            .await
            .map_err(|e| {
                PutCreateError::Other(Error::io(format!(
                    "failed to write temp file for {}: {}",
                    path, e
                )))
            })?;

        match object_store
            .inner
            .rename_if_not_exists(&temp_path, &path)
            .await
        {
            Ok(()) => Ok(()),
            Err(object_store::Error::AlreadyExists { .. }) => {
                let _ = object_store.delete(&temp_path).await;
                Err(PutCreateError::AlreadyExists)
            }
            Err(e) => {
                let _ = object_store.delete(&temp_path).await;
                Err(PutCreateError::Other(Error::io(format!(
                    "failed to create {} atomically: {}",
                    path, e
                ))))
            }
        }
    } else {
        object_store
            .inner
            .put_opts(
                &path,
                bytes.into(),
                PutOptions {
                    mode: PutMode::Create,
                    ..Default::default()
                },
            )
            .await
            .map_err(|e| match e {
                object_store::Error::AlreadyExists { .. }
                | object_store::Error::Precondition { .. } => PutCreateError::AlreadyExists,
                _ => PutCreateError::Other(Error::io(format!(
                    "failed to create {} atomically: {}",
                    path, e
                ))),
            })?;
        Ok(())
    }
}
