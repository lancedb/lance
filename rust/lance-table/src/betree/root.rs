// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! PROTOTYPE (discussion #7499): the root node.
//!
//! "Root node = the manifest we already have." Here it is a small protobuf
//! object holding table metadata + child refs + the inline ε-buffer. A commit
//! appends actions to the buffer and rewrites only this object.

use prost::Message;
use serde::{Deserialize, Serialize};

use crate::betree::action;
use crate::format::pb;
use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use object_store::{GetOptions, ObjectStore as OSObjectStore, PutOptions, PutPayload};

/// In-memory mirror of the last-written root object.
#[derive(Debug, Clone)]
pub struct RootState {
    pub version: u64,
    pub children: Vec<pb::ChildRef>,
    /// The inline ε-buffer.
    pub buffer: Vec<pb::FragmentAction>,
    pub buffer_cap_bytes: u64,
    pub schema_pb: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
struct VersionHint {
    version: u64,
}

impl RootState {
    /// Encoded byte size of the current ε-buffer (for cap accounting).
    pub fn buffer_encoded_len(&self) -> usize {
        self.buffer.iter().map(action::encoded_len).sum()
    }

    fn to_pb(&self) -> pb::BeTreeRoot {
        pb::BeTreeRoot {
            version: self.version,
            children: self.children.clone(),
            fragment_actions: self.buffer.clone(),
            buffer_cap_bytes: self.buffer_cap_bytes,
            schema_pb: self.schema_pb.clone(),
        }
    }

    fn from_pb(p: pb::BeTreeRoot) -> Self {
        Self {
            version: p.version,
            children: p.children,
            buffer: p.fragment_actions,
            buffer_cap_bytes: p.buffer_cap_bytes,
            schema_pb: p.schema_pb,
        }
    }
}

pub fn root_dir(base: &Path) -> Path {
    base.clone().join("_bt").join("root")
}

pub fn root_path(base: &Path, version: u64) -> Path {
    root_dir(base).join(format!("{version}.root"))
}

pub fn hint_path(base: &Path) -> Path {
    root_dir(base).join("latest_hint.json")
}

pub fn child_path(base: &Path, name: &str) -> Path {
    base.clone()
        .join("_bt")
        .join("child")
        .join(format!("{name}.lance"))
}

/// Write the root object at its version path and update the latest-version hint.
/// Returns the number of bytes written for the root object.
pub async fn write_root(object_store: &ObjectStore, base: &Path, root: &RootState) -> Result<u64> {
    let bytes = root.to_pb().encode_to_vec();
    let size = bytes.len() as u64;
    object_store
        .inner
        .put_opts(
            &root_path(base, root.version),
            PutPayload::from(bytes),
            PutOptions::default(),
        )
        .await?;
    let hint = serde_json::to_vec(&VersionHint {
        version: root.version,
    })
    .map_err(|e| Error::invalid_input(format!("failed to encode hint: {e}")))?;
    object_store
        .inner
        .put_opts(
            &hint_path(base),
            PutPayload::from(hint),
            PutOptions::default(),
        )
        .await?;
    Ok(size)
}

/// Read the latest committed root version from the hint file.
pub async fn read_latest_version(object_store: &ObjectStore, base: &Path) -> Result<u64> {
    let bytes = object_store
        .inner
        .get_opts(&hint_path(base), GetOptions::default())
        .await?
        .bytes()
        .await?;
    let hint: VersionHint = serde_json::from_slice(&bytes)
        .map_err(|e| Error::invalid_input(format!("failed to decode hint: {e}")))?;
    Ok(hint.version)
}

/// Read a root object at a specific version.
pub async fn read_root(object_store: &ObjectStore, base: &Path, version: u64) -> Result<RootState> {
    let bytes = object_store
        .inner
        .get_opts(&root_path(base, version), GetOptions::default())
        .await?
        .bytes()
        .await?;
    Ok(RootState::from_pb(pb::BeTreeRoot::decode(bytes)?))
}
