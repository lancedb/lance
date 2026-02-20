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

    /// Put the manifest to the namespace store.
    async fn put(
        &self,
        _base_path: &Path,
        version: u64,
        staging_path: &Path,
        size: u64,
        e_tag: Option<String>,
        _object_store: &dyn OSObjectStore,
        naming_scheme: ManifestNamingScheme,
    ) -> Result<ManifestLocation> {
        // create_table_version reads staging manifest and writes to final location
        let naming_scheme_str = match naming_scheme {
            ManifestNamingScheme::V1 => "V1",
            ManifestNamingScheme::V2 => "V2",
        };

        let request = CreateTableVersionRequest {
            id: Some(self.table_id.clone()),
            version: version as i64,
            manifest_path: staging_path.to_string(),
            manifest_size: Some(size as i64),
            e_tag: e_tag.clone(),
            naming_scheme: Some(naming_scheme_str.to_string()),
            ..Default::default()
        };

        let response = self.namespace.create_table_version(request).await?;

        // Get version info from response
        let version_info = response
            .version
            .ok_or_else(|| lance_core::Error::Internal {
                message: "create_table_version response missing version info".to_string(),
                location: snafu::location!(),
            })?;

        Ok(ManifestLocation {
            version: version_info.version as u64,
            path: Path::from(version_info.manifest_path),
            size: version_info.manifest_size.map(|s| s as u64),
            naming_scheme,
            e_tag: version_info.e_tag,
        })
    }

    async fn put_if_not_exists(
        &self,
        _base_uri: &str,
        _version: u64,
        _path: &str,
        _size: u64,
        _e_tag: Option<String>,
    ) -> Result<()> {
        Err(lance_core::Error::NotSupported {
            source: "put_if_not_exists is not supported for namespace-backed stores".into(),
            location: snafu::location!(),
        })
    }

    async fn put_if_exists(
        &self,
        _base_uri: &str,
        _version: u64,
        _path: &str,
        _size: u64,
        _e_tag: Option<String>,
    ) -> Result<()> {
        Err(lance_core::Error::NotSupported {
            source: "put_if_exists is not supported for namespace-backed stores".into(),
            location: snafu::location!(),
        })
    }
}
