// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Azure credential vending using SAS tokens.
//!
//! This module provides credential vending for Azure Blob Storage by generating
//! SAS (Shared Access Signature) tokens with user delegation keys.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use azure_core::auth::TokenCredential;
use azure_identity::DefaultAzureCredential;
use azure_storage::prelude::*;
use azure_storage_blobs::prelude::*;
use lance_core::{Error, Result};
use lance_io::object_store::uri_to_url;
use log::{debug, info, warn};

use super::{
    redact_credential, CredentialVendor, VendedCredentials, VendedPermission,
    DEFAULT_CREDENTIAL_DURATION_MILLIS,
};

/// Configuration for Azure credential vending.
#[derive(Debug, Clone)]
pub struct AzureCredentialVendorConfig {
    /// Optional tenant ID for authentication.
    pub tenant_id: Option<String>,

    /// Storage account name. Required for credential vending.
    pub account_name: Option<String>,

    /// Duration for vended credentials in milliseconds.
    /// Default: 3600000 (1 hour). Azure allows up to 7 days for SAS tokens.
    pub duration_millis: u64,

    /// Permission level for vended credentials.
    /// Default: Read (full read access)
    pub permission: VendedPermission,
}

impl Default for AzureCredentialVendorConfig {
    fn default() -> Self {
        Self {
            tenant_id: None,
            account_name: None,
            duration_millis: DEFAULT_CREDENTIAL_DURATION_MILLIS,
            permission: VendedPermission::default(),
        }
    }
}

impl AzureCredentialVendorConfig {
    /// Create a new default config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the tenant ID.
    pub fn with_tenant_id(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }

    /// Set the storage account name.
    pub fn with_account_name(mut self, account_name: impl Into<String>) -> Self {
        self.account_name = Some(account_name.into());
        self
    }

    /// Set the credential duration in milliseconds.
    pub fn with_duration_millis(mut self, millis: u64) -> Self {
        self.duration_millis = millis;
        self
    }

    /// Set the permission level for vended credentials.
    pub fn with_permission(mut self, permission: VendedPermission) -> Self {
        self.permission = permission;
        self
    }
}

/// Azure credential vendor that generates SAS tokens.
#[derive(Debug)]
pub struct AzureCredentialVendor {
    config: AzureCredentialVendorConfig,
}

impl AzureCredentialVendor {
    /// Create a new Azure credential vendor with the specified configuration.
    pub fn new(config: AzureCredentialVendorConfig) -> Self {
        Self { config }
    }

    /// Build SAS permissions based on the VendedPermission level.
    ///
    /// - Read: read + list
    /// - Write: read + list + write + add + create
    /// - Admin: read + list + write + add + create + delete
    #[allow(clippy::field_reassign_with_default)]
    fn build_sas_permissions(permission: VendedPermission) -> BlobSasPermissions {
        let mut p = BlobSasPermissions::default();

        // All permission levels have read access
        p.read = true;
        p.list = true;

        // Write and Admin have write access
        if permission.can_write() {
            p.write = true;
            p.add = true;
            p.create = true;
        }

        // Admin has delete access
        if permission.can_delete() {
            p.delete = true;
        }

        p
    }

    /// Generate a SAS token for the specified container.
    async fn generate_sas_token(&self, account: &str, container: &str) -> Result<(String, u64)> {
        let credential =
            DefaultAzureCredential::create(azure_identity::TokenCredentialOptions::default())
                .map_err(|e| Error::IO {
                    source: Box::new(std::io::Error::other(format!(
                        "Failed to create Azure credentials: {}",
                        e
                    ))),
                    location: snafu::location!(),
                })?;

        let credential: Arc<dyn TokenCredential> = Arc::new(credential);

        let blob_service_client = BlobServiceClient::new(account, credential.clone());

        // Calculate times using time crate (which Azure SDK uses)
        let now = time::OffsetDateTime::now_utc();
        let duration_millis = self.config.duration_millis as i64;
        let end_time = now + time::Duration::milliseconds(duration_millis);

        // Azure limits user delegation key to 7 days
        let max_key_end = now + time::Duration::days(7) - time::Duration::seconds(60);
        let key_end_time = if end_time > max_key_end {
            max_key_end
        } else {
            end_time
        };

        // Get user delegation key (note: typo in the library method name)
        let user_delegation_key = blob_service_client
            .get_user_deligation_key(now, key_end_time)
            .await
            .map_err(|e| Error::IO {
                source: Box::new(std::io::Error::other(format!(
                    "Failed to get user delegation key for account '{}': {}",
                    account, e
                ))),
                location: snafu::location!(),
            })?;

        let permissions = Self::build_sas_permissions(self.config.permission);

        // Generate SAS token for the container
        let container_client = blob_service_client.container_client(container);

        let sas_token = container_client
            .user_delegation_shared_access_signature(
                permissions,
                &user_delegation_key.user_deligation_key,
            )
            .await
            .map_err(|e| Error::IO {
                source: Box::new(std::io::Error::other(format!(
                    "Failed to generate SAS token for container '{}': {}",
                    container, e
                ))),
                location: snafu::location!(),
            })?;

        let expires_at_millis =
            (end_time.unix_timestamp() * 1000 + end_time.millisecond() as i64) as u64;

        let token = sas_token.token().map_err(|e| Error::IO {
            source: Box::new(std::io::Error::other(format!(
                "Failed to get SAS token: {}",
                e
            ))),
            location: snafu::location!(),
        })?;

        Ok((token, expires_at_millis))
    }
}

#[async_trait]
impl CredentialVendor for AzureCredentialVendor {
    async fn vend_credentials(&self, table_location: &str) -> Result<VendedCredentials> {
        debug!(
            "Azure credential vending: location={}, permission={}",
            table_location, self.config.permission
        );

        let url = uri_to_url(table_location)?;

        let container = url.host_str().ok_or_else(|| Error::InvalidInput {
            source: format!("Azure URI '{}' missing container", table_location).into(),
            location: snafu::location!(),
        })?;

        // Check if path extends beyond container level
        let path = url.path().trim_start_matches('/');
        if !path.is_empty() {
            warn!(
                "Azure SAS tokens are scoped to container level only. \
                 Credentials for '{}' will have access to entire container '{}', not just path '{}'",
                table_location, container, path
            );
        }

        let account =
            self.config
                .account_name
                .as_ref()
                .ok_or_else(|| Error::InvalidInput {
                    source: "Azure credential vending requires 'credential_vendor.azure_account_name' to be set in configuration".into(),
                    location: snafu::location!(),
                })?;

        let (sas_token, expires_at_millis) = self.generate_sas_token(account, container).await?;

        let mut storage_options = HashMap::new();
        // Use the standard key that object_store/lance-io expects
        storage_options.insert("azure_storage_sas_token".to_string(), sas_token.clone());
        storage_options.insert("azure_storage_account_name".to_string(), account.clone());
        storage_options.insert(
            "expires_at_millis".to_string(),
            expires_at_millis.to_string(),
        );

        info!(
            "Azure credentials vended: account={}, container={}, permission={}, expires_at={}, sas_token={}",
            account, container, self.config.permission, expires_at_millis, redact_credential(&sas_token)
        );

        Ok(VendedCredentials::new(storage_options, expires_at_millis))
    }

    fn provider_name(&self) -> &'static str {
        "azure"
    }

    fn permission(&self) -> VendedPermission {
        self.config.permission
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = AzureCredentialVendorConfig::new()
            .with_tenant_id("my-tenant-id")
            .with_account_name("myaccount")
            .with_duration_millis(7200000);

        assert_eq!(config.tenant_id, Some("my-tenant-id".to_string()));
        assert_eq!(config.account_name, Some("myaccount".to_string()));
        assert_eq!(config.duration_millis, 7200000);
    }

    #[test]
    fn test_build_sas_permissions_read() {
        let permissions = AzureCredentialVendor::build_sas_permissions(VendedPermission::Read);

        assert!(permissions.read, "Read permission should have read=true");
        assert!(permissions.list, "Read permission should have list=true");
        assert!(
            !permissions.write,
            "Read permission should have write=false"
        );
        assert!(!permissions.add, "Read permission should have add=false");
        assert!(
            !permissions.create,
            "Read permission should have create=false"
        );
        assert!(
            !permissions.delete,
            "Read permission should have delete=false"
        );
    }

    #[test]
    fn test_build_sas_permissions_write() {
        let permissions = AzureCredentialVendor::build_sas_permissions(VendedPermission::Write);

        assert!(permissions.read, "Write permission should have read=true");
        assert!(permissions.list, "Write permission should have list=true");
        assert!(permissions.write, "Write permission should have write=true");
        assert!(permissions.add, "Write permission should have add=true");
        assert!(
            permissions.create,
            "Write permission should have create=true"
        );
        assert!(
            !permissions.delete,
            "Write permission should have delete=false"
        );
    }

    #[test]
    fn test_build_sas_permissions_admin() {
        let permissions = AzureCredentialVendor::build_sas_permissions(VendedPermission::Admin);

        assert!(permissions.read, "Admin permission should have read=true");
        assert!(permissions.list, "Admin permission should have list=true");
        assert!(permissions.write, "Admin permission should have write=true");
        assert!(permissions.add, "Admin permission should have add=true");
        assert!(
            permissions.create,
            "Admin permission should have create=true"
        );
        assert!(
            permissions.delete,
            "Admin permission should have delete=true"
        );
    }
}
