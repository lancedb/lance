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
use azure_storage::shared_access_signature::service_sas::{BlobSharedAccessSignature, SasKey};
use azure_storage_blobs::prelude::*;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use lance_core::{Error, Result};
use lance_io::object_store::uri_to_url;
use lance_namespace::models::Identity;
use log::{debug, info, warn};
use sha2::{Digest, Sha256};

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
    /// Used to generate SAS permissions for all credential flows.
    pub permission: VendedPermission,

    /// Client ID of the Azure AD App Registration for Workload Identity Federation.
    /// Required when using auth_token identity for OIDC token exchange.
    pub federated_client_id: Option<String>,

    /// Salt for API key hashing.
    /// Required when using API key authentication.
    /// API keys are hashed as: SHA256(api_key + ":" + salt)
    pub api_key_salt: Option<String>,

    /// Map of SHA256(api_key + ":" + salt) -> permission level.
    /// When an API key is provided, its hash is looked up in this map.
    /// If found, the mapped permission is used instead of the default permission.
    pub api_key_hash_permissions: HashMap<String, VendedPermission>,
}

impl Default for AzureCredentialVendorConfig {
    fn default() -> Self {
        Self {
            tenant_id: None,
            account_name: None,
            duration_millis: DEFAULT_CREDENTIAL_DURATION_MILLIS,
            permission: VendedPermission::default(),
            federated_client_id: None,
            api_key_salt: None,
            api_key_hash_permissions: HashMap::new(),
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

    /// Set the federated client ID for Workload Identity Federation.
    pub fn with_federated_client_id(mut self, client_id: impl Into<String>) -> Self {
        self.federated_client_id = Some(client_id.into());
        self
    }

    /// Set the API key salt for hashing.
    pub fn with_api_key_salt(mut self, salt: impl Into<String>) -> Self {
        self.api_key_salt = Some(salt.into());
        self
    }

    /// Add an API key hash to permission mapping.
    pub fn with_api_key_hash_permission(
        mut self,
        key_hash: impl Into<String>,
        permission: VendedPermission,
    ) -> Self {
        self.api_key_hash_permissions
            .insert(key_hash.into(), permission);
        self
    }

    /// Set the entire API key hash permissions map.
    pub fn with_api_key_hash_permissions(
        mut self,
        permissions: HashMap<String, VendedPermission>,
    ) -> Self {
        self.api_key_hash_permissions = permissions;
        self
    }
}

/// Azure credential vendor that generates SAS tokens.
#[derive(Debug)]
pub struct AzureCredentialVendor {
    config: AzureCredentialVendorConfig,
    http_client: reqwest::Client,
}

impl AzureCredentialVendor {
    /// Create a new Azure credential vendor with the specified configuration.
    pub fn new(config: AzureCredentialVendorConfig) -> Self {
        Self {
            config,
            http_client: reqwest::Client::new(),
        }
    }

    /// Hash an API key using SHA-256 with salt (Polaris pattern).
    /// Format: SHA256(api_key + ":" + salt) as hex string.
    pub fn hash_api_key(api_key: &str, salt: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(format!("{}:{}", api_key, salt));
        format!("{:x}", hasher.finalize())
    }

    /// Extract a session name from a JWT token (best effort, no validation).
    /// Decodes the payload and extracts 'sub' or 'email' claim.
    /// Falls back to "lance-azure-identity" if parsing fails.
    fn derive_session_name_from_token(token: &str) -> String {
        // JWT format: header.payload.signature
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            return "lance-azure-identity".to_string();
        }

        // Decode the payload (second part)
        let payload = match URL_SAFE_NO_PAD.decode(parts[1]) {
            Ok(bytes) => bytes,
            Err(_) => {
                // Try standard base64 as fallback
                match base64::engine::general_purpose::STANDARD_NO_PAD.decode(parts[1]) {
                    Ok(bytes) => bytes,
                    Err(_) => return "lance-azure-identity".to_string(),
                }
            }
        };

        // Parse as JSON and extract 'sub' or 'email'
        let json: serde_json::Value = match serde_json::from_slice(&payload) {
            Ok(v) => v,
            Err(_) => return "lance-azure-identity".to_string(),
        };

        let subject = json
            .get("sub")
            .or_else(|| json.get("email"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        // Sanitize: keep only alphanumeric, @, -, .
        let sanitized: String = subject
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '@' || *c == '-' || *c == '.')
            .collect();

        format!("lance-{}", sanitized)
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

    /// Generate a SAS token with a specific permission level.
    async fn generate_sas_token_with_permission(
        &self,
        account: &str,
        container: &str,
        permission: VendedPermission,
    ) -> Result<(String, u64)> {
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

        let now = time::OffsetDateTime::now_utc();
        let duration_millis = self.config.duration_millis as i64;
        let end_time = now + time::Duration::milliseconds(duration_millis);

        let max_key_end = now + time::Duration::days(7) - time::Duration::seconds(60);
        let key_end_time = if end_time > max_key_end {
            max_key_end
        } else {
            end_time
        };

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

        let permissions = Self::build_sas_permissions(permission);
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

    /// Generate a directory-scoped SAS token.
    ///
    /// Unlike container-level SAS tokens, this restricts access to a specific directory
    /// path within the container. This is more secure for multi-tenant scenarios.
    ///
    /// # Arguments
    /// * `account` - Storage account name
    /// * `container` - Container name
    /// * `path` - Directory path within the container (e.g., "tenant-a/tables/my-table")
    /// * `permission` - Permission level for the SAS token
    async fn generate_directory_sas_token(
        &self,
        account: &str,
        container: &str,
        path: &str,
        permission: VendedPermission,
    ) -> Result<(String, u64)> {
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

        let now = time::OffsetDateTime::now_utc();
        let duration_millis = self.config.duration_millis as i64;
        let end_time = now + time::Duration::milliseconds(duration_millis);

        let max_key_end = now + time::Duration::days(7) - time::Duration::seconds(60);
        let key_end_time = if end_time > max_key_end {
            max_key_end
        } else {
            end_time
        };

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

        // Normalize path: remove leading/trailing slashes
        let normalized_path = path.trim_matches('/');
        let depth = if normalized_path.is_empty() {
            0
        } else {
            normalized_path.split('/').count()
        };

        // Build canonical resource path for directory-level SAS
        let canonical_resource = format!("/blob/{}/{}/{}", account, container, normalized_path);

        // Convert user delegation key to SasKey
        let sas_key = SasKey::UserDelegationKey(user_delegation_key.user_deligation_key);

        let permissions = Self::build_sas_permissions(permission);

        // Create directory-scoped SAS signature
        let sas = BlobSharedAccessSignature::new(
            sas_key,
            canonical_resource,
            permissions,
            end_time,
            BlobSignedResource::Directory,
        )
        .signed_directory_depth(depth as u8);

        let token = sas.token().map_err(|e| Error::IO {
            source: Box::new(std::io::Error::other(format!(
                "Failed to generate directory SAS token: {}",
                e
            ))),
            location: snafu::location!(),
        })?;

        let expires_at_millis =
            (end_time.unix_timestamp() * 1000 + end_time.millisecond() as i64) as u64;

        info!(
            "Azure directory-scoped SAS generated: account={}, container={}, path={}, depth={}, permission={}",
            account, container, normalized_path, depth, permission
        );

        Ok((token, expires_at_millis))
    }

    /// Exchange an OIDC token for Azure AD access token using Workload Identity Federation.
    ///
    /// This requires:
    /// 1. An Azure AD App Registration with Federated Credentials configured
    /// 2. The OIDC token's issuer and subject to match the Federated Credential configuration
    async fn exchange_oidc_for_azure_token(&self, oidc_token: &str) -> Result<String> {
        let tenant_id = self
            .config
            .tenant_id
            .as_ref()
            .ok_or_else(|| Error::InvalidInput {
                source: "azure_tenant_id must be configured for OIDC token exchange".into(),
                location: snafu::location!(),
            })?;

        let client_id =
            self.config
                .federated_client_id
                .as_ref()
                .ok_or_else(|| Error::InvalidInput {
                    source: "azure_federated_client_id must be configured for OIDC token exchange"
                        .into(),
                    location: snafu::location!(),
                })?;

        let token_url = format!(
            "https://login.microsoftonline.com/{}/oauth2/v2.0/token",
            tenant_id
        );

        let params = [
            ("grant_type", "client_credentials"),
            (
                "client_assertion_type",
                "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
            ),
            ("client_assertion", oidc_token),
            ("client_id", client_id),
            ("scope", "https://storage.azure.com/.default"),
        ];

        let response = self
            .http_client
            .post(&token_url)
            .form(&params)
            .send()
            .await
            .map_err(|e| Error::IO {
                source: Box::new(std::io::Error::other(format!(
                    "Failed to exchange OIDC token for Azure AD token: {}",
                    e
                ))),
                location: snafu::location!(),
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::IO {
                source: Box::new(std::io::Error::other(format!(
                    "Azure AD token exchange failed with status {}: {}",
                    status, body
                ))),
                location: snafu::location!(),
            });
        }

        let token_response: serde_json::Value = response.json().await.map_err(|e| Error::IO {
            source: Box::new(std::io::Error::other(format!(
                "Failed to parse Azure AD token response: {}",
                e
            ))),
            location: snafu::location!(),
        })?;

        token_response
            .get("access_token")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| Error::IO {
                source: Box::new(std::io::Error::other(
                    "Azure AD token response missing access_token",
                )),
                location: snafu::location!(),
            })
    }

    /// Generate a SAS token using a federated Azure AD token.
    ///
    /// Uses directory-scoped SAS when path is provided, container-level otherwise.
    async fn generate_sas_with_azure_token(
        &self,
        azure_token: &str,
        account: &str,
        container: &str,
        path: &str,
        permission: VendedPermission,
    ) -> Result<(String, u64)> {
        // Create a custom TokenCredential that uses our Azure AD token
        let credential = FederatedTokenCredential::new(azure_token.to_string());
        let credential: Arc<dyn TokenCredential> = Arc::new(credential);

        let blob_service_client = BlobServiceClient::new(account, credential.clone());

        let now = time::OffsetDateTime::now_utc();
        let duration_millis = self.config.duration_millis as i64;
        let end_time = now + time::Duration::milliseconds(duration_millis);

        let max_key_end = now + time::Duration::days(7) - time::Duration::seconds(60);
        let key_end_time = if end_time > max_key_end {
            max_key_end
        } else {
            end_time
        };

        let user_delegation_key = blob_service_client
            .get_user_deligation_key(now, key_end_time)
            .await
            .map_err(|e| Error::IO {
                source: Box::new(std::io::Error::other(format!(
                    "Failed to get user delegation key with federated token: {}",
                    e
                ))),
                location: snafu::location!(),
            })?;

        let permissions = Self::build_sas_permissions(permission);

        let expires_at_millis =
            (end_time.unix_timestamp() * 1000 + end_time.millisecond() as i64) as u64;

        // Use directory-scoped SAS when path is provided
        let normalized_path = path.trim_matches('/');
        let token = if normalized_path.is_empty() {
            // Container-level SAS
            let container_client = blob_service_client.container_client(container);
            let sas_token = container_client
                .user_delegation_shared_access_signature(
                    permissions,
                    &user_delegation_key.user_deligation_key,
                )
                .await
                .map_err(|e| Error::IO {
                    source: Box::new(std::io::Error::other(format!(
                        "Failed to generate SAS token with federated token: {}",
                        e
                    ))),
                    location: snafu::location!(),
                })?;

            sas_token.token().map_err(|e| Error::IO {
                source: Box::new(std::io::Error::other(format!(
                    "Failed to get SAS token: {}",
                    e
                ))),
                location: snafu::location!(),
            })?
        } else {
            // Directory-scoped SAS
            let depth = normalized_path.split('/').count();
            let canonical_resource = format!("/blob/{}/{}/{}", account, container, normalized_path);
            let sas_key = SasKey::UserDelegationKey(user_delegation_key.user_deligation_key);

            let sas = BlobSharedAccessSignature::new(
                sas_key,
                canonical_resource,
                permissions,
                end_time,
                BlobSignedResource::Directory,
            )
            .signed_directory_depth(depth as u8);

            sas.token().map_err(|e| Error::IO {
                source: Box::new(std::io::Error::other(format!(
                    "Failed to generate directory SAS token with federated token: {}",
                    e
                ))),
                location: snafu::location!(),
            })?
        };

        Ok((token, expires_at_millis))
    }

    /// Vend credentials using Workload Identity Federation (for auth_token).
    async fn vend_with_web_identity(
        &self,
        account: &str,
        container: &str,
        path: &str,
        auth_token: &str,
    ) -> Result<VendedCredentials> {
        let session_name = Self::derive_session_name_from_token(auth_token);
        debug!(
            "Azure vend_with_web_identity: account={}, container={}, path={}, session={}",
            account, container, path, session_name
        );

        // Exchange OIDC token for Azure AD token
        let azure_token = self.exchange_oidc_for_azure_token(auth_token).await?;

        // Generate SAS token using the Azure AD token
        // Use directory-scoped SAS when path is provided
        let (sas_token, expires_at_millis) = self
            .generate_sas_with_azure_token(
                &azure_token,
                account,
                container,
                path,
                self.config.permission,
            )
            .await?;

        let mut storage_options = HashMap::new();
        storage_options.insert("azure_storage_sas_token".to_string(), sas_token.clone());
        storage_options.insert(
            "azure_storage_account_name".to_string(),
            account.to_string(),
        );
        storage_options.insert(
            "expires_at_millis".to_string(),
            expires_at_millis.to_string(),
        );

        info!(
            "Azure credentials vended (web identity): account={}, container={}, path={}, permission={}, expires_at={}, sas_token={}",
            account, container, path, self.config.permission, expires_at_millis, redact_credential(&sas_token)
        );

        Ok(VendedCredentials::new(storage_options, expires_at_millis))
    }

    /// Vend credentials using API key validation.
    async fn vend_with_api_key(
        &self,
        account: &str,
        container: &str,
        path: &str,
        api_key: &str,
    ) -> Result<VendedCredentials> {
        let salt = self
            .config
            .api_key_salt
            .as_ref()
            .ok_or_else(|| Error::InvalidInput {
                source: "api_key_salt must be configured to use API key authentication".into(),
                location: snafu::location!(),
            })?;

        let key_hash = Self::hash_api_key(api_key, salt);

        // Look up permission from hash mapping
        let permission = self
            .config
            .api_key_hash_permissions
            .get(&key_hash)
            .copied()
            .ok_or_else(|| {
                warn!(
                    "Invalid API key: hash {} not found in permissions map",
                    &key_hash[..8]
                );
                Error::InvalidInput {
                    source: "Invalid API key".into(),
                    location: snafu::location!(),
                }
            })?;

        debug!(
            "Azure vend_with_api_key: account={}, container={}, path={}, permission={}",
            account, container, path, permission
        );

        // Use directory-scoped SAS when path is provided, container-level otherwise
        let (sas_token, expires_at_millis) = if path.is_empty() {
            self.generate_sas_token_with_permission(account, container, permission)
                .await?
        } else {
            self.generate_directory_sas_token(account, container, path, permission)
                .await?
        };

        let mut storage_options = HashMap::new();
        storage_options.insert("azure_storage_sas_token".to_string(), sas_token.clone());
        storage_options.insert(
            "azure_storage_account_name".to_string(),
            account.to_string(),
        );
        storage_options.insert(
            "expires_at_millis".to_string(),
            expires_at_millis.to_string(),
        );

        info!(
            "Azure credentials vended (api_key): account={}, container={}, path={}, permission={}, expires_at={}, sas_token={}",
            account, container, path, permission, expires_at_millis, redact_credential(&sas_token)
        );

        Ok(VendedCredentials::new(storage_options, expires_at_millis))
    }
}

/// A custom TokenCredential that wraps a pre-obtained Azure AD access token.
#[derive(Debug)]
struct FederatedTokenCredential {
    token: String,
}

impl FederatedTokenCredential {
    fn new(token: String) -> Self {
        Self { token }
    }
}

#[async_trait]
impl TokenCredential for FederatedTokenCredential {
    async fn get_token(
        &self,
        _scopes: &[&str],
    ) -> std::result::Result<azure_core::auth::AccessToken, azure_core::Error> {
        // Return the pre-obtained token with a 1-hour expiry (conservative estimate)
        let expires_on = time::OffsetDateTime::now_utc() + time::Duration::hours(1);
        Ok(azure_core::auth::AccessToken::new(
            azure_core::auth::Secret::new(self.token.clone()),
            expires_on,
        ))
    }

    async fn clear_cache(&self) -> std::result::Result<(), azure_core::Error> {
        Ok(())
    }
}

#[async_trait]
impl CredentialVendor for AzureCredentialVendor {
    async fn vend_credentials(
        &self,
        table_location: &str,
        identity: Option<&Identity>,
    ) -> Result<VendedCredentials> {
        debug!(
            "Azure credential vending: location={}, permission={}, identity={:?}",
            table_location,
            self.config.permission,
            identity.map(|i| format!(
                "api_key={}, auth_token={}",
                i.api_key.is_some(),
                i.auth_token.is_some()
            ))
        );

        let url = uri_to_url(table_location)?;

        let container = url.host_str().ok_or_else(|| Error::InvalidInput {
            source: format!("Azure URI '{}' missing container", table_location).into(),
            location: snafu::location!(),
        })?;

        // Extract path for directory-scoped SAS
        let path = url.path().trim_start_matches('/');

        let account =
            self.config
                .account_name
                .as_ref()
                .ok_or_else(|| Error::InvalidInput {
                    source: "Azure credential vending requires 'credential_vendor.azure_account_name' to be set in configuration".into(),
                    location: snafu::location!(),
                })?;

        // Dispatch based on identity
        match identity {
            Some(id) if id.auth_token.is_some() => {
                let auth_token = id.auth_token.as_ref().unwrap();
                self.vend_with_web_identity(account, container, path, auth_token)
                    .await
            }
            Some(id) if id.api_key.is_some() => {
                let api_key = id.api_key.as_ref().unwrap();
                self.vend_with_api_key(account, container, path, api_key)
                    .await
            }
            Some(_) => Err(Error::InvalidInput {
                source: "Identity provided but neither auth_token nor api_key is set".into(),
                location: snafu::location!(),
            }),
            None => {
                // Static credential vending using DefaultAzureCredential
                // Use directory-scoped SAS when path is provided, container-level otherwise
                let (sas_token, expires_at_millis) = if path.is_empty() {
                    self.generate_sas_token(account, container).await?
                } else {
                    self.generate_directory_sas_token(
                        account,
                        container,
                        path,
                        self.config.permission,
                    )
                    .await?
                };

                let mut storage_options = HashMap::new();
                storage_options.insert("azure_storage_sas_token".to_string(), sas_token.clone());
                storage_options.insert("azure_storage_account_name".to_string(), account.clone());
                storage_options.insert(
                    "expires_at_millis".to_string(),
                    expires_at_millis.to_string(),
                );

                info!(
                    "Azure credentials vended (static): account={}, container={}, path={}, permission={}, expires_at={}, sas_token={}",
                    account, container, path, self.config.permission, expires_at_millis, redact_credential(&sas_token)
                );

                Ok(VendedCredentials::new(storage_options, expires_at_millis))
            }
        }
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
