// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use async_trait::async_trait;
use lance_core::Result;
use lance_namespace::LanceNamespace;
use lance_namespace::models::{
    BatchDeleteTableVersionsRequest, CreateTableVersionRequest, DescribeTableVersionRequest,
    ListTableVersionsRequest, TableVersion, VersionRange,
};
use lance_table::io::commit::external_manifest::ExternalManifestStore;
use lance_table::io::commit::{ManifestLocation, ManifestNamingScheme};
use object_store::ObjectStore as OSObjectStore;
use object_store::path::Path;

use lance_namespace::error::NamespaceError;

use crate::dataset::branch_location::BranchLocation;

/// Whether `e` says the requested chain (table or branch) does not exist, as
/// opposed to a failure talking to the namespace.
fn is_chain_not_found(e: &lance_core::Error) -> bool {
    if let lance_core::Error::Namespace { source, .. } = e
        && let Some(ns_err) = source.downcast_ref::<NamespaceError>()
    {
        return matches!(
            ns_err,
            NamespaceError::TableNotFound { .. } | NamespaceError::TableBranchNotFound { .. }
        );
    }
    false
}

#[derive(Debug)]
pub struct LanceNamespaceExternalManifestStore {
    namespace_client: Arc<dyn LanceNamespace>,
    table_id: Vec<String>,
    /// Object-store path of the table root (the main branch). The base path the
    /// trait methods receive is resolved against this to derive which branch a
    /// request targets, so a single store serves every branch of the table.
    table_root: Path,
    /// Reservation incarnation token from `declare_table`; the namespace
    /// validates it where the bootstrap publish lands.
    reservation_token: Option<String>,
}

impl LanceNamespaceExternalManifestStore {
    pub fn new(
        namespace_client: Arc<dyn LanceNamespace>,
        table_id: Vec<String>,
        table_root: Path,
    ) -> Self {
        Self {
            namespace_client,
            table_id,
            table_root,
            reservation_token: None,
        }
    }

    /// Attach the reservation token this publisher acts under.
    pub fn with_reservation_token(mut self, token: Option<String>) -> Self {
        self.reservation_token = token;
        self
    }

    /// Build a store for the table rooted at `table_uri`, resolving the root
    /// path from the uri without initializing an object store.
    pub fn for_table_uri(
        namespace_client: Arc<dyn LanceNamespace>,
        table_id: Vec<String>,
        table_uri: &str,
    ) -> Result<Self> {
        let table_root = lance_io::object_store::ObjectStore::extract_path_from_uri(
            Arc::new(lance_io::object_store::ObjectStoreRegistry::default()),
            table_uri,
        )?;
        Ok(Self::new(namespace_client, table_id, table_root))
    }

    /// Derive the branch targeted by `base` (the table root for main, or a
    /// branch chain produced by `BranchLocation::find_branch`). The branch
    /// path layout is owned by [`BranchLocation`]; this store never parses or
    /// constructs it directly.
    fn branch_for_base(&self, base: &str) -> Result<Option<String>> {
        BranchLocation::branch_of(self.table_root.as_ref(), base)
    }

    fn manifest_location(version: TableVersion) -> Result<ManifestLocation> {
        let path = Path::parse(&version.manifest_path).map_err(|e| {
            lance_core::Error::invalid_input(format!(
                "Invalid manifest path '{}': {}",
                version.manifest_path, e
            ))
        })?;
        let filename = path.filename().ok_or_else(|| {
            lance_core::Error::invalid_input(format!(
                "Manifest path '{}' has no filename",
                version.manifest_path
            ))
        })?;
        let naming_scheme = ManifestNamingScheme::detect_scheme(filename).ok_or_else(|| {
            lance_core::Error::invalid_input(format!(
                "Manifest path '{}' does not use a recognized naming scheme",
                version.manifest_path
            ))
        })?;
        Ok(ManifestLocation {
            version: version.version as u64,
            path,
            size: version.manifest_size.map(|size| size as u64),
            naming_scheme,
            e_tag: version.e_tag,
        })
    }
}

#[async_trait]
impl ExternalManifestStore for LanceNamespaceExternalManifestStore {
    async fn get(&self, base_uri: &str, version: u64) -> Result<String> {
        let request = DescribeTableVersionRequest {
            id: Some(self.table_id.clone()),
            version: Some(version as i64),
            branch: self.branch_for_base(base_uri)?,
            ..Default::default()
        };

        let response = self
            .namespace_client
            .describe_table_version(request)
            .await?;

        // Namespace returns full path (relative to object store root)
        Ok(response.version.manifest_path)
    }

    async fn get_latest_version(&self, base_uri: &str) -> Result<Option<(u64, String)>> {
        let request = ListTableVersionsRequest {
            id: Some(self.table_id.clone()),
            descending: Some(true),
            limit: Some(1),
            branch: self.branch_for_base(base_uri)?,
            ..Default::default()
        };

        let response = match self.namespace_client.list_table_versions(request).await {
            Ok(response) => response,
            // A chain that does not exist yet (e.g. probing a branch location
            // before the branch is created) has no latest version; the
            // ExternalManifestStore contract reports that as None, not an
            // error, so existence checks can treat it as a missing dataset.
            Err(e) if is_chain_not_found(&e) => return Ok(None),
            Err(e) => return Err(e),
        };

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

    async fn get_manifest_location(
        &self,
        base_uri: &str,
        version: u64,
    ) -> Result<ManifestLocation> {
        let request = DescribeTableVersionRequest {
            id: Some(self.table_id.clone()),
            version: Some(version as i64),
            branch: self.branch_for_base(base_uri)?,
            ..Default::default()
        };
        let response = self
            .namespace_client
            .describe_table_version(request)
            .await?;
        Self::manifest_location(*response.version)
    }

    async fn list_versions(
        &self,
        base_uri: &str,
        since: Option<u64>,
    ) -> Result<Option<Vec<ManifestLocation>>> {
        let branch = self.branch_for_base(base_uri)?;
        let mut page_token = None;
        let mut seen_page_tokens = std::collections::HashSet::new();
        let mut locations = Vec::new();

        loop {
            let request = ListTableVersionsRequest {
                id: Some(self.table_id.clone()),
                page_token,
                limit: Some(1_000),
                descending: Some(true),
                branch: branch.clone(),
                ..Default::default()
            };
            let response = match self.namespace_client.list_table_versions(request).await {
                Ok(response) => response,
                Err(error) if is_chain_not_found(&error) => return Ok(Some(Vec::new())),
                Err(error) => return Err(error),
            };

            for version in response.versions {
                if since.is_none_or(|since| version.version as u64 > since) {
                    locations.push(Self::manifest_location(version)?);
                }
            }

            page_token = response.page_token.filter(|token| !token.is_empty());
            if let Some(token) = &page_token
                && !seen_page_tokens.insert(token.clone())
            {
                return Err(lance_core::Error::internal(format!(
                    "Namespace returned repeated table-version page token '{token}'"
                )));
            }
            if page_token.is_none() {
                break;
            }
        }

        Ok(Some(locations))
    }

    async fn forget_version(
        &self,
        base_uri: &str,
        version: u64,
        manifest_path: &str,
    ) -> Result<()> {
        let version = i64::try_from(version).map_err(|_| {
            lance_core::Error::invalid_input(format!(
                "Table version {version} does not fit the namespace API"
            ))
        })?;
        let end_version = version.checked_add(1).ok_or_else(|| {
            lance_core::Error::invalid_input(format!(
                "Table version {version} cannot form a deletion range"
            ))
        })?;
        self.namespace_client
            .batch_delete_table_versions(BatchDeleteTableVersionsRequest {
                id: Some(self.table_id.clone()),
                branch: self.branch_for_base(base_uri)?,
                ranges: vec![VersionRange::new(version, end_version)],
                context: Some(std::collections::HashMap::from([(
                    lance_namespace::TABLE_VERSION_IDENTITY_KEY.to_string(),
                    manifest_path.to_string(),
                )])),
                ..Default::default()
            })
            .await?;
        Ok(())
    }

    /// Put the manifest to the namespace store.
    async fn put(
        &self,
        base_path: &Path,
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
            branch: self.branch_for_base(base_path.as_ref())?,
            context: self.reservation_token.clone().map(|token| {
                std::collections::HashMap::from([(
                    lance_namespace::RESERVATION_TOKEN_KEY.to_string(),
                    token,
                )])
            }),
            ..Default::default()
        };

        let response = self.namespace_client.create_table_version(request).await?;

        // Get version info from response
        let version_info = response.version.ok_or_else(|| {
            lance_core::Error::internal(
                "create_table_version response missing version info".to_string(),
            )
        })?;

        let mut location = Self::manifest_location(*version_info)?;
        location.naming_scheme = naming_scheme;
        Ok(location)
    }

    async fn put_if_not_exists(
        &self,
        _base_uri: &str,
        _version: u64,
        _path: &str,
        _size: u64,
        _e_tag: Option<String>,
    ) -> Result<()> {
        Err(lance_core::Error::not_supported_source(
            "put_if_not_exists is not supported for namespace-backed stores".into(),
        ))
    }

    async fn put_if_exists(
        &self,
        _base_uri: &str,
        _version: u64,
        _path: &str,
        _size: u64,
        _e_tag: Option<String>,
    ) -> Result<()> {
        Err(lance_core::Error::not_supported_source(
            "put_if_exists is not supported for namespace-backed stores".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_namespace::models::ListTableVersionsResponse;

    /// A namespace whose list_table_versions always fails with the configured
    /// error, to pin how get_latest_version classifies failures.
    #[derive(Debug)]
    struct FailingNamespace {
        error: fn() -> lance_core::Error,
    }

    #[async_trait]
    impl LanceNamespace for FailingNamespace {
        fn namespace_id(&self) -> String {
            "failing".to_string()
        }

        async fn list_table_versions(
            &self,
            _request: ListTableVersionsRequest,
        ) -> Result<ListTableVersionsResponse> {
            Err((self.error)())
        }
    }

    fn store_with(error: fn() -> lance_core::Error) -> LanceNamespaceExternalManifestStore {
        LanceNamespaceExternalManifestStore::new(
            Arc::new(FailingNamespace { error }),
            vec!["t".to_string()],
            Path::parse("data/t.lance").unwrap(),
        )
    }

    /// A chain that does not exist (missing table or branch) has no latest
    /// version; everything else is a real failure and must propagate so an
    /// outage is never mistaken for an absent dataset.
    #[tokio::test]
    async fn test_get_latest_version_error_classification() {
        use lance_namespace::error::NamespaceError;

        let absent = [
            store_with(|| {
                NamespaceError::TableNotFound {
                    message: "missing table".to_string(),
                }
                .into()
            }),
            store_with(|| {
                NamespaceError::TableBranchNotFound {
                    message: "missing branch".to_string(),
                }
                .into()
            }),
        ];
        for store in absent {
            let latest = store.get_latest_version("data/t.lance/tree/dev").await;
            assert!(
                matches!(latest, Ok(None)),
                "a missing chain must read as no latest version, got: {:?}",
                latest
            );
        }

        let failures = [
            store_with(|| {
                NamespaceError::Internal {
                    message: "server error".to_string(),
                }
                .into()
            }),
            store_with(|| {
                NamespaceError::Throttling {
                    message: "slow down".to_string(),
                }
                .into()
            }),
            store_with(|| lance_core::Error::io("connection reset".to_string())),
        ];
        for store in failures {
            let latest = store.get_latest_version("data/t.lance/tree/dev").await;
            assert!(
                latest.is_err(),
                "a real failure must propagate, got: {:?}",
                latest
            );
        }
    }
}
