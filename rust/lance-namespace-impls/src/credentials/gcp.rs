// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! GCP credential vending using downscoped OAuth2 tokens.
//!
//! This module provides credential vending for GCP Cloud Storage by obtaining
//! OAuth2 access tokens and downscoping them using Credential Access Boundaries (CAB).
//!
//! ## Authentication
//!
//! This module uses [Application Default Credentials (ADC)][adc] for authentication.
//! ADC automatically finds credentials based on the environment:
//!
//! 1. **`GOOGLE_APPLICATION_CREDENTIALS` environment variable**: Set this to the path
//!    of a service account key file (JSON format) before starting the application.
//! 2. **Well-known file locations**: `~/.config/gcloud/application_default_credentials.json`
//!    on Linux/macOS, or the equivalent on Windows.
//! 3. **Metadata server**: When running on GCP (Compute Engine, Cloud Run, GKE, etc.),
//!    credentials are automatically obtained from the metadata server.
//!
//! For production deployments on GCP, using the metadata server (option 3) is recommended
//! as it doesn't require managing key files.
//!
//! [adc]: https://cloud.google.com/docs/authentication/application-default-credentials
//!
//! ## Service Account Impersonation
//!
//! For multi-tenant scenarios, you can configure `service_account` to impersonate a
//! different service account. The base credentials (from ADC) must have the
//! `roles/iam.serviceAccountTokenCreator` role on the target service account.
//!
//! ## Permission Scoping
//!
//! Permissions are enforced using GCP's Credential Access Boundaries:
//! - **Read**: `roles/storage.legacyObjectReader` + `roles/storage.objectViewer` (read and list)
//! - **Write**: Read permissions + `roles/storage.legacyBucketWriter` + `roles/storage.objectCreator`
//! - **Admin**: Write permissions + `roles/storage.objectAdmin` (includes delete)
//!
//! The downscoped token is restricted to the specific bucket and path prefix.
//!
//! Note: Legacy roles are used because modern roles like `storage.objectCreator` lack
//! `storage.buckets.get` which many client libraries require.

use std::collections::HashMap;

use async_trait::async_trait;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use google_cloud_auth::credentials;
use lance_core::{Error, Result};
use lance_io::object_store::uri_to_url;
use lance_namespace::models::Identity;
use log::{debug, info, warn};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{redact_credential, CredentialVendor, VendedCredentials, VendedPermission};

/// GCP STS token exchange endpoint for downscoping credentials.
const STS_TOKEN_EXCHANGE_URL: &str = "https://sts.googleapis.com/v1/token";

/// Configuration for GCP credential vending.
#[derive(Debug, Clone, Default)]
pub struct GcpCredentialVendorConfig {
    /// Optional service account to impersonate.
    ///
    /// When set, the vendor will impersonate this service account using the
    /// IAM Credentials API's generateAccessToken endpoint before downscoping.
    /// This is useful for multi-tenant scenarios where you want to issue tokens
    /// on behalf of different service accounts.
    ///
    /// The base credentials (from ADC) must have the `roles/iam.serviceAccountTokenCreator`
    /// role on this service account.
    ///
    /// Format: `my-sa@project.iam.gserviceaccount.com`
    pub service_account: Option<String>,

    /// Permission level for vended credentials.
    /// Default: Read
    /// Permissions are enforced via Credential Access Boundaries (CAB).
    ///
    /// Note: GCP token duration cannot be configured; the token lifetime
    /// is determined by the STS endpoint (typically 1 hour).
    pub permission: VendedPermission,

    /// Workload Identity Provider resource name for OIDC token exchange.
    /// Required when using auth_token identity for Workload Identity Federation.
    ///
    /// Format: `projects/{project_number}/locations/global/workloadIdentityPools/{pool_id}/providers/{provider_id}`
    ///
    /// The OIDC token's issuer must match the provider's configuration.
    pub workload_identity_provider: Option<String>,

    /// Service account to impersonate after Workload Identity Federation.
    /// Optional - if set, the exchanged token will be used to generate an
    /// access token for this service account.
    ///
    /// Format: `my-sa@project.iam.gserviceaccount.com`
    pub impersonation_service_account: Option<String>,

    /// Salt for API key hashing.
    /// Required when using API key authentication.
    /// API keys are hashed as: SHA256(api_key + ":" + salt)
    pub api_key_salt: Option<String>,

    /// Map of SHA256(api_key + ":" + salt) -> permission level.
    /// When an API key is provided, its hash is looked up in this map.
    /// If found, the mapped permission is used instead of the default permission.
    pub api_key_hash_permissions: HashMap<String, VendedPermission>,
}

impl GcpCredentialVendorConfig {
    /// Create a new default config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the service account to impersonate.
    ///
    /// When set, the vendor uses the IAM Credentials API to generate an access
    /// token for this service account, then downscopes it with CAB.
    ///
    /// The base credentials (from ADC) must have the `roles/iam.serviceAccountTokenCreator`
    /// role on this service account.
    pub fn with_service_account(mut self, service_account: impl Into<String>) -> Self {
        self.service_account = Some(service_account.into());
        self
    }

    /// Set the permission level for vended credentials.
    pub fn with_permission(mut self, permission: VendedPermission) -> Self {
        self.permission = permission;
        self
    }

    /// Set the Workload Identity Provider for OIDC token exchange.
    pub fn with_workload_identity_provider(mut self, provider: impl Into<String>) -> Self {
        self.workload_identity_provider = Some(provider.into());
        self
    }

    /// Set the service account to impersonate after Workload Identity Federation.
    pub fn with_impersonation_service_account(
        mut self,
        service_account: impl Into<String>,
    ) -> Self {
        self.impersonation_service_account = Some(service_account.into());
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

/// Access boundary rule for a single resource.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AccessBoundaryRule {
    available_resource: String,
    available_permissions: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    availability_condition: Option<AvailabilityCondition>,
}

/// Condition for access boundary rule.
#[derive(Debug, Clone, Serialize)]
struct AvailabilityCondition {
    expression: String,
}

/// Credential Access Boundary structure.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CredentialAccessBoundary {
    access_boundary: AccessBoundaryInner,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AccessBoundaryInner {
    access_boundary_rules: Vec<AccessBoundaryRule>,
}

/// Response from STS token exchange.
#[derive(Debug, Deserialize)]
struct TokenExchangeResponse {
    access_token: String,
    #[serde(default)]
    expires_in: Option<u64>,
}

/// Response from IAM generateAccessToken API.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerateAccessTokenResponse {
    access_token: String,
    #[allow(dead_code)]
    expire_time: String,
}

/// GCP credential vendor that provides downscoped OAuth2 tokens.
pub struct GcpCredentialVendor {
    config: GcpCredentialVendorConfig,
    http_client: Client,
    credential: credentials::Credential,
}

impl std::fmt::Debug for GcpCredentialVendor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GcpCredentialVendor")
            .field("config", &self.config)
            .field("credential", &"[credential]")
            .finish()
    }
}

impl GcpCredentialVendor {
    /// Create a new GCP credential vendor with the specified configuration.
    ///
    /// Uses [Application Default Credentials (ADC)][adc] for authentication.
    /// To use a service account key file, set the `GOOGLE_APPLICATION_CREDENTIALS`
    /// environment variable to the file path before starting the application.
    ///
    /// [adc]: https://cloud.google.com/docs/authentication/application-default-credentials
    pub async fn new(config: GcpCredentialVendorConfig) -> Result<Self> {
        let credential = credentials::create_access_token_credential()
            .await
            .map_err(|e| Error::IO {
                source: Box::new(std::io::Error::other(format!(
                    "Failed to create GCP credentials: {}",
                    e
                ))),
                location: snafu::location!(),
            })?;

        Ok(Self {
            config,
            http_client: Client::new(),
            credential,
        })
    }

    /// Parse a GCS URI to extract bucket and prefix.
    fn parse_gcs_uri(uri: &str) -> Result<(String, String)> {
        let url = uri_to_url(uri)?;

        if url.scheme() != "gs" {
            return Err(Error::InvalidInput {
                source: format!(
                    "Unsupported GCS URI scheme '{}', expected 'gs'",
                    url.scheme()
                )
                .into(),
                location: snafu::location!(),
            });
        }

        let bucket = url
            .host_str()
            .ok_or_else(|| Error::InvalidInput {
                source: format!("GCS URI '{}' missing bucket", uri).into(),
                location: snafu::location!(),
            })?
            .to_string();

        let prefix = url.path().trim_start_matches('/').to_string();

        Ok((bucket, prefix))
    }

    /// Get a source token for downscoping.
    ///
    /// If service_account is configured, impersonates that service account
    /// using the IAM Credentials API. Otherwise, uses the configured credential
    /// directly.
    async fn get_source_token(&self) -> Result<String> {
        let base_token = self.credential.get_token().await.map_err(|e| Error::IO {
            source: Box::new(std::io::Error::other(format!(
                "Failed to get GCP token: {}",
                e
            ))),
            location: snafu::location!(),
        })?;

        // If service account impersonation is configured, use generateAccessToken API
        if let Some(ref service_account) = self.config.service_account {
            return self
                .impersonate_service_account(&base_token.token, service_account)
                .await;
        }

        Ok(base_token.token)
    }

    /// Impersonate a service account using the IAM Credentials API.
    ///
    /// Uses the base token to call generateAccessToken for the target service account.
    async fn impersonate_service_account(
        &self,
        base_token: &str,
        service_account: &str,
    ) -> Result<String> {
        let url = format!(
            "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/{}:generateAccessToken",
            service_account
        );

        // Request body with cloud-platform scope (required for GCS access)
        let body = serde_json::json!({
            "scope": ["https://www.googleapis.com/auth/cloud-platform"]
        });

        let response = self
            .http_client
            .post(&url)
            .bearer_auth(base_token)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::IO {
                source: Box::new(std::io::Error::other(format!(
                    "Failed to call IAM generateAccessToken: {}",
                    e
                ))),
                location: snafu::location!(),
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".to_string());
            return Err(Error::IO {
                source: Box::new(std::io::Error::other(format!(
                    "IAM generateAccessToken failed for '{}' with status {}: {}",
                    service_account, status, body
                ))),
                location: snafu::location!(),
            });
        }

        let token_response: GenerateAccessTokenResponse =
            response.json().await.map_err(|e| Error::IO {
                source: Box::new(std::io::Error::other(format!(
                    "Failed to parse generateAccessToken response: {}",
                    e
                ))),
                location: snafu::location!(),
            })?;

        Ok(token_response.access_token)
    }

    /// Build Credential Access Boundary for the specified bucket/prefix and permission.
    fn build_access_boundary(
        bucket: &str,
        prefix: &str,
        permission: VendedPermission,
    ) -> CredentialAccessBoundary {
        let bucket_resource = format!("//storage.googleapis.com/projects/_/buckets/{}", bucket);

        let mut rules = vec![];

        // Build condition expression for path restriction
        let condition = if prefix.is_empty() {
            None
        } else {
            let prefix_trimmed = prefix.trim_end_matches('/');
            // CEL expression to restrict access to the specific path prefix.
            // We append '/' to ensure exact prefix matching - without it, prefix "data"
            // would incorrectly match "data-other/file.txt".
            //
            // For object access: resource.name must start with "prefix/"
            // For list operations: listPrefix must equal "prefix" OR start with "prefix/"
            let list_prefix_attr =
                "api.getAttribute('storage.googleapis.com/objectListPrefix', '')";
            let expr = format!(
                "resource.name.startsWith('projects/_/buckets/{}/objects/{}/') || \
                 {list_attr} == '{prefix}' || {list_attr}.startsWith('{prefix}/')",
                bucket,
                prefix_trimmed,
                list_attr = list_prefix_attr,
                prefix = prefix_trimmed
            );
            Some(AvailabilityCondition { expression: expr })
        };

        // Read permissions: legacyObjectReader for read + objectViewer for list
        // Using legacy roles because modern roles lack storage.buckets.get
        rules.push(AccessBoundaryRule {
            available_resource: bucket_resource.clone(),
            available_permissions: vec![
                "inRole:roles/storage.legacyObjectReader".to_string(),
                "inRole:roles/storage.objectViewer".to_string(),
            ],
            availability_condition: condition.clone(),
        });

        // Write permission: legacyBucketWriter + objectCreator for create/update
        if permission.can_write() {
            rules.push(AccessBoundaryRule {
                available_resource: bucket_resource.clone(),
                available_permissions: vec![
                    "inRole:roles/storage.legacyBucketWriter".to_string(),
                    "inRole:roles/storage.objectCreator".to_string(),
                ],
                availability_condition: condition.clone(),
            });
        }

        // Admin permission: objectAdmin for delete
        if permission.can_delete() {
            rules.push(AccessBoundaryRule {
                available_resource: bucket_resource,
                available_permissions: vec!["inRole:roles/storage.objectAdmin".to_string()],
                availability_condition: condition,
            });
        }

        CredentialAccessBoundary {
            access_boundary: AccessBoundaryInner {
                access_boundary_rules: rules,
            },
        }
    }

    /// Exchange source token for a downscoped token using STS.
    async fn downscope_token(
        &self,
        source_token: &str,
        access_boundary: &CredentialAccessBoundary,
    ) -> Result<(String, u64)> {
        let options_json = serde_json::to_string(access_boundary).map_err(|e| Error::IO {
            source: Box::new(std::io::Error::other(format!(
                "Failed to serialize access boundary: {}",
                e
            ))),
            location: snafu::location!(),
        })?;

        let params = [
            (
                "grant_type",
                "urn:ietf:params:oauth:grant-type:token-exchange",
            ),
            (
                "subject_token_type",
                "urn:ietf:params:oauth:token-type:access_token",
            ),
            (
                "requested_token_type",
                "urn:ietf:params:oauth:token-type:access_token",
            ),
            ("subject_token", source_token),
            ("options", &options_json),
        ];

        let response = self
            .http_client
            .post(STS_TOKEN_EXCHANGE_URL)
            .form(&params)
            .send()
            .await
            .map_err(|e| Error::IO {
                source: Box::new(std::io::Error::other(format!(
                    "Failed to call STS token exchange: {}",
                    e
                ))),
                location: snafu::location!(),
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".to_string());
            return Err(Error::IO {
                source: Box::new(std::io::Error::other(format!(
                    "STS token exchange failed with status {}: {}",
                    status, body
                ))),
                location: snafu::location!(),
            });
        }

        let token_response: TokenExchangeResponse =
            response.json().await.map_err(|e| Error::IO {
                source: Box::new(std::io::Error::other(format!(
                    "Failed to parse STS response: {}",
                    e
                ))),
                location: snafu::location!(),
            })?;

        // Calculate expiration time
        // Use expires_in from response if available, otherwise default to 1 hour
        let expires_in_secs = token_response.expires_in.unwrap_or(3600);
        let expires_at_millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time went backwards")
            .as_millis() as u64
            + expires_in_secs * 1000;

        Ok((token_response.access_token, expires_at_millis))
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
    /// Falls back to "lance-gcp-identity" if parsing fails.
    fn derive_session_name_from_token(token: &str) -> String {
        // JWT format: header.payload.signature
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            return "lance-gcp-identity".to_string();
        }

        // Decode the payload (second part)
        let payload = match URL_SAFE_NO_PAD.decode(parts[1]) {
            Ok(bytes) => bytes,
            Err(_) => {
                // Try standard base64 as fallback
                match base64::engine::general_purpose::STANDARD_NO_PAD.decode(parts[1]) {
                    Ok(bytes) => bytes,
                    Err(_) => return "lance-gcp-identity".to_string(),
                }
            }
        };

        // Parse as JSON and extract 'sub' or 'email'
        let json: serde_json::Value = match serde_json::from_slice(&payload) {
            Ok(v) => v,
            Err(_) => return "lance-gcp-identity".to_string(),
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

    /// Normalize the Workload Identity Provider to the full audience format expected by GCP STS.
    ///
    /// GCP STS expects audience in the format:
    /// `//iam.googleapis.com/projects/{project}/locations/global/workloadIdentityPools/{pool}/providers/{provider}`
    ///
    /// This function accepts either:
    /// - Full format: `//iam.googleapis.com/projects/...`
    /// - Short format: `projects/...` (will be prefixed with `//iam.googleapis.com/`)
    fn normalize_workload_identity_audience(provider: &str) -> String {
        const IAM_PREFIX: &str = "//iam.googleapis.com/";
        if provider.starts_with(IAM_PREFIX) {
            provider.to_string()
        } else {
            format!("{}{}", IAM_PREFIX, provider)
        }
    }

    /// Exchange an OIDC token for GCP access token using Workload Identity Federation.
    ///
    /// This requires:
    /// 1. A Workload Identity Pool and Provider configured in GCP
    /// 2. The OIDC token's issuer to match the provider's configuration
    /// 3. Optionally, a service account to impersonate after token exchange
    async fn exchange_oidc_for_gcp_token(&self, oidc_token: &str) -> Result<String> {
        let workload_identity_provider = self
            .config
            .workload_identity_provider
            .as_ref()
            .ok_or_else(|| Error::InvalidInput {
                source: "gcp_workload_identity_provider must be configured for OIDC token exchange"
                    .into(),
                location: snafu::location!(),
            })?;

        // Normalize audience to full format expected by GCP STS
        let audience = Self::normalize_workload_identity_audience(workload_identity_provider);

        // Exchange OIDC token for GCP federated token via STS
        let params = [
            (
                "grant_type",
                "urn:ietf:params:oauth:grant-type:token-exchange",
            ),
            ("subject_token_type", "urn:ietf:params:oauth:token-type:jwt"),
            (
                "requested_token_type",
                "urn:ietf:params:oauth:token-type:access_token",
            ),
            ("subject_token", oidc_token),
            ("audience", audience.as_str()),
            ("scope", "https://www.googleapis.com/auth/cloud-platform"),
        ];

        let response = self
            .http_client
            .post(STS_TOKEN_EXCHANGE_URL)
            .form(&params)
            .send()
            .await
            .map_err(|e| Error::IO {
                source: Box::new(std::io::Error::other(format!(
                    "Failed to exchange OIDC token for GCP token: {}",
                    e
                ))),
                location: snafu::location!(),
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::IO {
                source: Box::new(std::io::Error::other(format!(
                    "GCP STS token exchange failed with status {}: {}",
                    status, body
                ))),
                location: snafu::location!(),
            });
        }

        let token_response: TokenExchangeResponse =
            response.json().await.map_err(|e| Error::IO {
                source: Box::new(std::io::Error::other(format!(
                    "Failed to parse GCP STS token response: {}",
                    e
                ))),
                location: snafu::location!(),
            })?;

        let federated_token = token_response.access_token;

        // If impersonation is configured, use the federated token to get an impersonated token
        if let Some(ref service_account) = self.config.impersonation_service_account {
            return self
                .impersonate_service_account(&federated_token, service_account)
                .await;
        }

        Ok(federated_token)
    }

    /// Vend credentials using Workload Identity Federation (for auth_token).
    async fn vend_with_web_identity(
        &self,
        bucket: &str,
        prefix: &str,
        auth_token: &str,
    ) -> Result<VendedCredentials> {
        let session_name = Self::derive_session_name_from_token(auth_token);
        debug!(
            "GCP vend_with_web_identity: bucket={}, prefix={}, session={}",
            bucket, prefix, session_name
        );

        // Exchange OIDC token for GCP token
        let gcp_token = self.exchange_oidc_for_gcp_token(auth_token).await?;

        // Build access boundary and downscope
        let access_boundary = Self::build_access_boundary(bucket, prefix, self.config.permission);
        let (downscoped_token, expires_at_millis) =
            self.downscope_token(&gcp_token, &access_boundary).await?;

        let mut storage_options = HashMap::new();
        storage_options.insert("google_storage_token".to_string(), downscoped_token.clone());
        storage_options.insert(
            "expires_at_millis".to_string(),
            expires_at_millis.to_string(),
        );

        info!(
            "GCP credentials vended (web identity): bucket={}, prefix={}, permission={}, expires_at={}, token={}",
            bucket, prefix, self.config.permission, expires_at_millis, redact_credential(&downscoped_token)
        );

        Ok(VendedCredentials::new(storage_options, expires_at_millis))
    }

    /// Vend credentials using API key validation.
    async fn vend_with_api_key(
        &self,
        bucket: &str,
        prefix: &str,
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
            "GCP vend_with_api_key: bucket={}, prefix={}, permission={}",
            bucket, prefix, permission
        );

        // Get source token using ADC and downscope with the API key's permission
        let source_token = self.get_source_token().await?;
        let access_boundary = Self::build_access_boundary(bucket, prefix, permission);
        let (downscoped_token, expires_at_millis) = self
            .downscope_token(&source_token, &access_boundary)
            .await?;

        let mut storage_options = HashMap::new();
        storage_options.insert("google_storage_token".to_string(), downscoped_token.clone());
        storage_options.insert(
            "expires_at_millis".to_string(),
            expires_at_millis.to_string(),
        );

        info!(
            "GCP credentials vended (api_key): bucket={}, prefix={}, permission={}, expires_at={}, token={}",
            bucket, prefix, permission, expires_at_millis, redact_credential(&downscoped_token)
        );

        Ok(VendedCredentials::new(storage_options, expires_at_millis))
    }
}

#[async_trait]
impl CredentialVendor for GcpCredentialVendor {
    async fn vend_credentials(
        &self,
        table_location: &str,
        identity: Option<&Identity>,
    ) -> Result<VendedCredentials> {
        debug!(
            "GCP credential vending: location={}, permission={}, identity={:?}",
            table_location,
            self.config.permission,
            identity.map(|i| format!(
                "api_key={}, auth_token={}",
                i.api_key.is_some(),
                i.auth_token.is_some()
            ))
        );

        let (bucket, prefix) = Self::parse_gcs_uri(table_location)?;

        // Dispatch based on identity
        match identity {
            Some(id) if id.auth_token.is_some() => {
                let auth_token = id.auth_token.as_ref().unwrap();
                self.vend_with_web_identity(&bucket, &prefix, auth_token)
                    .await
            }
            Some(id) if id.api_key.is_some() => {
                let api_key = id.api_key.as_ref().unwrap();
                self.vend_with_api_key(&bucket, &prefix, api_key).await
            }
            Some(_) => Err(Error::InvalidInput {
                source: "Identity provided but neither auth_token nor api_key is set".into(),
                location: snafu::location!(),
            }),
            None => {
                // Static credential vending using ADC
                let source_token = self.get_source_token().await?;
                let access_boundary =
                    Self::build_access_boundary(&bucket, &prefix, self.config.permission);
                let (downscoped_token, expires_at_millis) = self
                    .downscope_token(&source_token, &access_boundary)
                    .await?;

                let mut storage_options = HashMap::new();
                storage_options
                    .insert("google_storage_token".to_string(), downscoped_token.clone());
                storage_options.insert(
                    "expires_at_millis".to_string(),
                    expires_at_millis.to_string(),
                );

                info!(
                    "GCP credentials vended (static): bucket={}, prefix={}, permission={}, expires_at={}, token={}",
                    bucket, prefix, self.config.permission, expires_at_millis, redact_credential(&downscoped_token)
                );

                Ok(VendedCredentials::new(storage_options, expires_at_millis))
            }
        }
    }

    fn provider_name(&self) -> &'static str {
        "gcp"
    }

    fn permission(&self) -> VendedPermission {
        self.config.permission
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gcs_uri() {
        let (bucket, prefix) = GcpCredentialVendor::parse_gcs_uri("gs://my-bucket/path/to/table")
            .expect("should parse");
        assert_eq!(bucket, "my-bucket");
        assert_eq!(prefix, "path/to/table");

        let (bucket, prefix) =
            GcpCredentialVendor::parse_gcs_uri("gs://my-bucket/").expect("should parse");
        assert_eq!(bucket, "my-bucket");
        assert_eq!(prefix, "");

        let (bucket, prefix) =
            GcpCredentialVendor::parse_gcs_uri("gs://my-bucket").expect("should parse");
        assert_eq!(bucket, "my-bucket");
        assert_eq!(prefix, "");
    }

    #[test]
    fn test_parse_gcs_uri_invalid() {
        // Wrong scheme - should fail
        let result = GcpCredentialVendor::parse_gcs_uri("s3://bucket/path");
        assert!(result.is_err());

        // Missing bucket
        let result = GcpCredentialVendor::parse_gcs_uri("gs:///path");
        assert!(result.is_err());

        // Invalid URI format
        let result = GcpCredentialVendor::parse_gcs_uri("not-a-uri");
        assert!(result.is_err());

        // Empty string
        let result = GcpCredentialVendor::parse_gcs_uri("");
        assert!(result.is_err());
    }

    #[test]
    fn test_config_builder() {
        let config = GcpCredentialVendorConfig::new()
            .with_service_account("my-sa@project.iam.gserviceaccount.com")
            .with_permission(VendedPermission::Write);

        assert_eq!(
            config.service_account,
            Some("my-sa@project.iam.gserviceaccount.com".to_string())
        );
        assert_eq!(config.permission, VendedPermission::Write);
    }

    #[test]
    fn test_build_access_boundary_read() {
        let boundary = GcpCredentialVendor::build_access_boundary(
            "my-bucket",
            "path/to/data",
            VendedPermission::Read,
        );

        let rules = &boundary.access_boundary.access_boundary_rules;
        assert_eq!(rules.len(), 1, "Read should have 1 rule");

        let permissions = &rules[0].available_permissions;
        assert!(permissions.contains(&"inRole:roles/storage.legacyObjectReader".to_string()));
        assert!(permissions.contains(&"inRole:roles/storage.objectViewer".to_string()));
        assert!(rules[0].availability_condition.is_some());
    }

    #[test]
    fn test_build_access_boundary_write() {
        let boundary = GcpCredentialVendor::build_access_boundary(
            "my-bucket",
            "path/to/data",
            VendedPermission::Write,
        );

        let rules = &boundary.access_boundary.access_boundary_rules;
        assert_eq!(rules.len(), 2, "Write should have 2 rules");

        let permissions: Vec<_> = rules
            .iter()
            .flat_map(|r| r.available_permissions.iter())
            .collect();
        assert!(permissions.contains(&&"inRole:roles/storage.legacyObjectReader".to_string()));
        assert!(permissions.contains(&&"inRole:roles/storage.objectViewer".to_string()));
        assert!(permissions.contains(&&"inRole:roles/storage.legacyBucketWriter".to_string()));
        assert!(permissions.contains(&&"inRole:roles/storage.objectCreator".to_string()));
    }

    #[test]
    fn test_build_access_boundary_admin() {
        let boundary = GcpCredentialVendor::build_access_boundary(
            "my-bucket",
            "path/to/data",
            VendedPermission::Admin,
        );

        let rules = &boundary.access_boundary.access_boundary_rules;
        assert_eq!(rules.len(), 3, "Admin should have 3 rules");

        let permissions: Vec<_> = rules
            .iter()
            .flat_map(|r| r.available_permissions.iter())
            .collect();
        assert!(permissions.contains(&&"inRole:roles/storage.legacyObjectReader".to_string()));
        assert!(permissions.contains(&&"inRole:roles/storage.objectViewer".to_string()));
        assert!(permissions.contains(&&"inRole:roles/storage.legacyBucketWriter".to_string()));
        assert!(permissions.contains(&&"inRole:roles/storage.objectCreator".to_string()));
        assert!(permissions.contains(&&"inRole:roles/storage.objectAdmin".to_string()));
    }

    #[test]
    fn test_build_access_boundary_no_prefix() {
        let boundary =
            GcpCredentialVendor::build_access_boundary("my-bucket", "", VendedPermission::Read);

        let rules = &boundary.access_boundary.access_boundary_rules;
        assert_eq!(rules.len(), 1);
        // No condition when prefix is empty (full bucket access)
        assert!(rules[0].availability_condition.is_none());
    }

    #[test]
    fn test_normalize_workload_identity_audience() {
        // Short format should be prefixed
        let short =
            "projects/123456/locations/global/workloadIdentityPools/my-pool/providers/my-provider";
        let normalized = GcpCredentialVendor::normalize_workload_identity_audience(short);
        assert_eq!(
            normalized,
            "//iam.googleapis.com/projects/123456/locations/global/workloadIdentityPools/my-pool/providers/my-provider"
        );

        // Full format should be unchanged
        let full = "//iam.googleapis.com/projects/123456/locations/global/workloadIdentityPools/my-pool/providers/my-provider";
        let normalized = GcpCredentialVendor::normalize_workload_identity_audience(full);
        assert_eq!(normalized, full);

        // Edge case: already has prefix (idempotent)
        let normalized_again =
            GcpCredentialVendor::normalize_workload_identity_audience(&normalized);
        assert_eq!(normalized_again, full);
    }
}
