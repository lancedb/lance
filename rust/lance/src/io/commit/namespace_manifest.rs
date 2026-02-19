// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use async_trait::async_trait;
use lance_core::Result;
use lance_namespace::models::{
    CreateTableVersionRequest, DescribeTableVersionRequest, ListTableVersionsRequest,
};
use lance_namespace::LanceNamespace;
use lance_table::io::commit::external_manifest::ExternalManifestStore;
use lance_table::io::commit::{ManifestLocation, ManifestNamingScheme};
use object_store::path::Path;
use object_store::ObjectStore as OSObjectStore;

#[derive(Debug)]
pub struct LanceNamespaceExternalManifestStore {
    namespace: Arc<dyn LanceNamespace>,
    table_id: Vec<String>,
}

impl LanceNamespaceExternalManifestStore {
    pub fn new(namespace: Arc<dyn LanceNamespace>, table_id: Vec<String>) -> Self {
        Self {
            namespace,
            table_id,
        }
    }
}

#[async_trait]
impl ExternalManifestStore for LanceNamespaceExternalManifestStore {
    async fn get(&self, _base_uri: &str, version: u64) -> Result<String> {
        let request = DescribeTableVersionRequest {
            id: Some(self.table_id.clone()),
            version: Some(version as i64),
            ..Default::default()
        };

        let response = self.namespace.describe_table_version(request).await?;

        // Namespace returns full path (relative to object store root)
        Ok(response.version.manifest_path)
    }

    async fn get_latest_version(&self, _base_uri: &str) -> Result<Option<(u64, String)>> {
        let request = ListTableVersionsRequest {
            id: Some(self.table_id.clone()),
            descending: Some(true),
            limit: Some(1),
            ..Default::default()
        };

        let response = self.namespace.list_table_versions(request).await?;

        if response.versions.is_empty() {
            return Ok(None);
        }

        let version = &response.versions[0];

        // Namespace returns full path (relative to object store root)
        Ok(Some((
            version.version as u64,
            version.manifest_path.clone(),
        )))
    }

    /// Direct-write commit: reads staging manifest and writes directly to final location.
    async fn commit(
        &self,
        base_path: &Path,
        version: u64,
        staging_path: &Path,
        size: u64,
        e_tag: Option<String>,
        object_store: &dyn OSObjectStore,
        naming_scheme: ManifestNamingScheme,
    ) -> Result<ManifestLocation> {
        // create_table_version reads staging manifest and writes to final location
        let request = CreateTableVersionRequest {
            id: Some(self.table_id.clone()),
            version: version as i64,
            manifest_path: staging_path.to_string(),
            manifest_size: Some(size as i64),
            e_tag: e_tag.clone(),
            ..Default::default()
        };

        self.namespace.create_table_version(request).await?;

        // Delete staging manifest (it's been copied to final location)
        let _ = object_store.delete(staging_path).await;

        // Return final manifest location (full path relative to object store root)
        let final_path = naming_scheme.manifest_path(base_path, version);

        Ok(ManifestLocation {
            version,
            path: final_path,
            size: Some(size),
            naming_scheme,
            e_tag,
        })
    }

    /// Not used when commit() is overridden.
    async fn put_if_not_exists(
        &self,
        _base_uri: &str,
        _version: u64,
        _path: &str,
        _size: u64,
        _e_tag: Option<String>,
    ) -> Result<()> {
        Ok(())
    }

    /// Not used when commit() is overridden.
    async fn put_if_exists(
        &self,
        _base_uri: &str,
        _version: u64,
        _path: &str,
        _size: u64,
        _e_tag: Option<String>,
    ) -> Result<()> {
        Ok(())
    }
}
