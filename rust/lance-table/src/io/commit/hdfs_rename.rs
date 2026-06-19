// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! HDFS manifest commit via libhdfs [`hdrs::Client::rename_file`] (NameNode rename RPC).
//!
//! Matches [`super::RenameCommitHandler`] semantics: write staging manifest, then rename into
//! place. On HDFS, `rename` fails when the destination path already exists, which we treat as
//! [`super::CommitError::CommitConflict`] for transaction retry.

use std::collections::HashMap;
use std::io;
use std::sync::Arc;

use hdrs::ClientBuilder;
use object_store::path::Path;
use tokio::task;
use url::Url;

use lance_core::{Error, Result};

use super::{
    CommitError, CommitHandler, ManifestLocation, ManifestNamingScheme, ManifestWriter,
    make_staging_manifest_path,
};
use crate::format::{IndexMetadata, Manifest, Transaction};
use lance_io::object_store::ObjectStore;
use lance_io::object_store::ObjectStoreParams;

/// [`CommitHandler`] for HDFS: staging write + atomic rename via libhdfs (same idea as
/// [`super::RenameCommitHandler`], but uses `hdfsRename` instead of object_store's
/// `rename_if_not_exists`).
#[derive(Debug)]
pub struct HdfsRenameCommitHandler {
    client: Arc<hdrs::Client>,
}

impl HdfsRenameCommitHandler {
    pub fn new(client: Arc<hdrs::Client>) -> Self {
        Self { client }
    }
}

/// Connect a libhdfs client for commit operations. Must match the same NameNode / user as the
/// OpenDAL HDFS [`ObjectStore`](lance_io::object_store::ObjectStore) used for I/O.
pub async fn connect_hdrs_client(
    url: &Url,
    params: &ObjectStoreParams,
) -> Result<Arc<hdrs::Client>> {
    let (name_node, user) = resolve_hdfs_namenode_and_user(url, params)?;
    let join = task::spawn_blocking(move || {
        let user_ref: Option<&str> = user.as_deref();
        let mut builder = ClientBuilder::new(&name_node);
        if let Some(u) = user_ref {
            builder = builder.with_user(u);
        }
        builder.connect().map_err(|e| {
            Error::io(format!(
                "Failed to connect HDFS client for manifest commit: {e}"
            ))
        })
    })
    .await
    .map_err(|e| Error::io(format!("HDFS connect task failed: {e}")))?;
    Ok(Arc::new(join?))
}

fn resolve_hdfs_namenode_and_user(
    url: &Url,
    params: &ObjectStoreParams,
) -> Result<(String, Option<String>)> {
    let namenode = url
        .host_str()
        .ok_or_else(|| Error::invalid_input("HDFS URL must contain namenode host"))?
        .to_string();

    let storage_map: HashMap<String, String> = params
        .storage_options()
        .cloned()
        .unwrap_or_default();

    let hadoop_conf_dir = std::env::var("HADOOP_CONF_DIR").ok();

    let name_node_url = if let Some(port) = url.port() {
        format!("hdfs://{namenode}:{port}")
    } else {
        format!("hdfs://{namenode}")
    };

    let is_logical_name = url.port().is_none() && !namenode.contains('.');

    let name_node = if let Some(nn) = storage_map
        .get("hdfs_name_node")
        .filter(|v| !v.is_empty())
    {
        nn.clone()
    } else if let Ok(nn) = std::env::var("HDFS_NAME_NODE") {
        if nn.is_empty() {
            name_node_url
        } else {
            nn
        }
    } else {
        if is_logical_name && hadoop_conf_dir.is_none() {
            return Err(Error::invalid_input(format!(
                "HDFS HA logical name '{namenode}' cannot be resolved without HADOOP_CONF_DIR or hdfs_name_node"
            )));
        }
        name_node_url
    };

    let user = if let Some(u) = storage_map
        .get("hdfs_user")
        .filter(|v| !v.is_empty())
    {
        Some(u.clone())
    } else if let Ok(u) = std::env::var("HADOOP_USER_NAME") {
        if u.is_empty() { None } else { Some(u) }
    } else if let Ok(u) = std::env::var("HDFS_USER") {
        if u.is_empty() { None } else { Some(u) }
    } else {
        std::env::var("USER")
            .or_else(|_| std::env::var("LOGNAME"))
            .ok()
            .filter(|s| !s.is_empty())
    };

    Ok((name_node, user))
}

fn object_store_path_to_hdfs_abs(path: &Path) -> String {
    let s = path.as_ref();
    if s.is_empty() {
        "/".to_string()
    } else if s.starts_with('/') {
        s.to_string()
    } else {
        format!("/{s}")
    }
}

fn hdfs_rename_implies_conflict(err: &io::Error) -> bool {
    if err.kind() == io::ErrorKind::AlreadyExists {
        return true;
    }
    let msg = err.to_string().to_lowercase();
    msg.contains("already exists")
        || msg.contains("file exists")
        || msg.contains("filealreadyexists")
        || msg.contains("rename destination")
}

#[async_trait::async_trait]
impl CommitHandler for HdfsRenameCommitHandler {
    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<IndexMetadata>>,
        base_path: &Path,
        object_store: &ObjectStore,
        manifest_writer: ManifestWriter,
        naming_scheme: ManifestNamingScheme,
        transaction: Option<Transaction>,
    ) -> std::result::Result<ManifestLocation, CommitError> {
        let path = naming_scheme.manifest_path(base_path, manifest.version);
        let tmp_path = make_staging_manifest_path(&path)?;

        let res = manifest_writer(object_store, manifest, indices, &tmp_path, transaction).await?;

        let client = Arc::clone(&self.client);
        let tmp_abs = object_store_path_to_hdfs_abs(&tmp_path);
        let final_abs = object_store_path_to_hdfs_abs(&path);

        let rename_result =
            task::spawn_blocking(move || client.rename_file(&tmp_abs, &final_abs)).await;

        match rename_result {
            Ok(Ok(())) => Ok(ManifestLocation {
                version: manifest.version,
                path,
                size: Some(res.size as u64),
                naming_scheme,
                e_tag: None,
            }),
            Ok(Err(e)) if hdfs_rename_implies_conflict(&e) => {
                let _ = object_store.delete(&tmp_path).await;
                Err(CommitError::CommitConflict)
            }
            Ok(Err(e)) => Err(CommitError::OtherError(Error::io(format!(
                "HDFS rename failed (staging -> manifest): {e}"
            )))),
            Err(e) => Err(CommitError::OtherError(Error::io(format!(
                "HDFS rename task join error: {e}"
            )))),
        }
    }
}
