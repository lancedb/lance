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
use google_cloud_auth::credentials;
use lance_core::{Error, Result};
use lance_io::object_store::uri_to_url;
use log::{debug, info};
use reqwest::Client;
use serde::{Deserialize, Serialize};

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
}

#[async_trait]
impl CredentialVendor for GcpCredentialVendor {
    async fn vend_credentials(&self, table_location: &str) -> Result<VendedCredentials> {
        debug!(
            "GCP credential vending: location={}, permission={}",
            table_location, self.config.permission
        );

        let (bucket, prefix) = Self::parse_gcs_uri(table_location)?;

        // Get source token from default credentials
        let source_token = self.get_source_token().await?;

        // Build access boundary for this location and permission
        let access_boundary = Self::build_access_boundary(&bucket, &prefix, self.config.permission);

        // Exchange for downscoped token
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
            "GCP credentials vended: bucket={}, prefix={}, permission={}, expires_at={}, token={}",
            bucket,
            prefix,
            self.config.permission,
            expires_at_millis,
            redact_credential(&downscoped_token)
        );

        Ok(VendedCredentials::new(storage_options, expires_at_millis))
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
}
