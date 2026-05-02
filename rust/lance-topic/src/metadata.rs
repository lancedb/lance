// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use bytes::Bytes;
use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use object_store::{PutMode, PutOptions};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

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
    if group_id.contains('$') {
        return Err(Error::invalid_input(format!(
            "consumer group_id '{}' cannot contain '$'",
            group_id
        )));
    }
    Ok(())
}

pub fn validate_consumer_id(consumer_id: &str) -> Result<()> {
    if consumer_id.is_empty() {
        return Err(Error::invalid_input("consumer_id cannot be empty"));
    }
    Ok(())
}

pub enum AppendError {
    AlreadyExists,
    Other(Error),
}

pub async fn append(
    object_store: &ObjectStore,
    dir: &Path,
    filename: &str,
    bytes: Bytes,
) -> std::result::Result<(), AppendError> {
    let path = dir.child(filename);
    if object_store.is_local() {
        let temp_filename = format!("{}.tmp.{}", filename, Uuid::new_v4());
        let temp_path = dir.child(temp_filename);
        object_store
            .inner
            .put(&temp_path, bytes.into())
            .await
            .map_err(|e| {
                AppendError::Other(Error::io(format!(
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
                Err(AppendError::AlreadyExists)
            }
            Err(e) => {
                let _ = object_store.delete(&temp_path).await;
                Err(AppendError::Other(Error::io(format!(
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
                | object_store::Error::Precondition { .. } => AppendError::AlreadyExists,
                _ => AppendError::Other(Error::io(format!(
                    "failed to create {} atomically: {}",
                    path, e
                ))),
            })?;
        Ok(())
    }
}
