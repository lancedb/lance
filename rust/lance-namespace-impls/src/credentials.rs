// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Credential vending for cloud storage access.
//!
//! This module provides credential vending functionality that generates
//! temporary, scoped credentials for accessing cloud storage. Similar to
//! Apache Polaris's credential vending, it supports:
//!
//! - **AWS**: STS AssumeRole with scoped IAM policies (requires `credential-vendor-aws` feature)
//! - **GCP**: OAuth2 tokens with access boundaries (requires `credential-vendor-gcp` feature)
//! - **Azure**: SAS tokens with user delegation keys (requires `credential-vendor-azure` feature)
//!
//! The appropriate vendor is automatically selected based on the table location URI scheme:
//! - `s3://` for AWS
//! - `gs://` for GCP
//! - `az://` for Azure
//!
//! ## Configuration via Properties
//!
//! Credential vendors are configured via properties with the `credential_vendor.` prefix.
//!
//! ### Properties format:
//!
//! ```text
//! # Required to enable credential vending
//! credential_vendor.enabled = "true"
//!
//! # Common properties (apply to all providers)
//! credential_vendor.permission = "read"          # read, write, or admin (default: read)
//!
//! # AWS-specific properties (for s3:// locations)
//! credential_vendor.aws_role_arn = "arn:aws:iam::123456789012:role/MyRole"  # required for AWS
//! credential_vendor.aws_external_id = "my-external-id"
//! credential_vendor.aws_region = "us-west-2"
//! credential_vendor.aws_role_session_name = "my-session"
//! credential_vendor.aws_duration_millis = "3600000"  # 1 hour (default, range: 15min-12hrs)
//!
//! # GCP-specific properties (for gs:// locations)
//! # Note: GCP token duration cannot be configured; it's determined by the STS endpoint
//! # To use a service account key file, set GOOGLE_APPLICATION_CREDENTIALS env var before starting
//! credential_vendor.gcp_service_account = "my-sa@project.iam.gserviceaccount.com"
//!
//! # Azure-specific properties (for az:// locations)
//! credential_vendor.azure_account_name = "mystorageaccount"  # required for Azure
//! credential_vendor.azure_tenant_id = "my-tenant-id"
//! credential_vendor.azure_duration_millis = "3600000"  # 1 hour (default, up to 7 days)
//! ```
//!
//! ### Example using ConnectBuilder:
//!
//! ```ignore
//! ConnectBuilder::new("dir")
//!     .property("root", "s3://bucket/path")
//!     .property("credential_vendor.enabled", "true")
//!     .property("credential_vendor.aws_role_arn", "arn:aws:iam::123456789012:role/MyRole")
//!     .property("credential_vendor.permission", "read")
//!     .connect()
//!     .await?;
//! ```

#[cfg(feature = "credential-vendor-aws")]
pub mod aws;

#[cfg(feature = "credential-vendor-azure")]
pub mod azure;

#[cfg(feature = "credential-vendor-gcp")]
pub mod gcp;

use std::collections::HashMap;
use std::str::FromStr;

use async_trait::async_trait;
use lance_core::Result;
use lance_io::object_store::uri_to_url;

/// Default credential duration: 1 hour (3600000 milliseconds)
pub const DEFAULT_CREDENTIAL_DURATION_MILLIS: u64 = 3600 * 1000;

/// Redact a credential string for logging, showing first and last few characters.
///
/// This is useful for debugging while avoiding exposure of sensitive data.
/// Format: `AKIAIOSF***MPLE` (first 8 + "***" + last 4)
///
/// Shows 8 characters at the start (useful since AWS keys always start with AKIA/ASIA)
/// and 4 characters at the end. For short strings, shows only the first few with "***".
///
/// # Security Note
///
/// This function should only be used for identifiers and tokens, never for secrets
/// like `aws_secret_access_key` which should never be logged even in redacted form.
pub fn redact_credential(credential: &str) -> String {
    const SHOW_START: usize = 8;
    const SHOW_END: usize = 4;
    const MIN_LENGTH_FOR_BOTH_ENDS: usize = SHOW_START + SHOW_END + 4; // Need at least 16 chars

    if credential.is_empty() {
        return "[empty]".to_string();
    }

    if credential.len() < MIN_LENGTH_FOR_BOTH_ENDS {
        // For short credentials, just show beginning
        let show = credential.len().min(SHOW_START);
        format!("{}***", &credential[..show])
    } else {
        // Show first 8 and last 4 characters
        format!(
            "{}***{}",
            &credential[..SHOW_START],
            &credential[credential.len() - SHOW_END..]
        )
    }
}

/// Permission level for vended credentials.
///
/// This determines what access the vended credentials will have:
/// - `Read`: Read-only access to all table content
/// - `Write`: Full read and write access (no delete)
/// - `Admin`: Full read, write, and delete access
///
/// Permission enforcement by cloud provider:
/// - **AWS**: Permissions are enforced via scoped IAM policies attached to the AssumeRole request
/// - **Azure**: Permissions are enforced via SAS token permissions
/// - **GCP**: Permissions are enforced via Credential Access Boundaries (CAB) that downscope
///   the OAuth2 token to specific GCS IAM roles
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VendedPermission {
    /// Read-only access to all table content (metadata, indices, data files)
    #[default]
    Read,
    /// Full read and write access (no delete)
    /// This is intended ONLY for testing purposes to generate a write-only permission set.
    /// Technically, any user with write permission could "delete" the file by
    /// overwriting the file with empty content.
    /// So this cannot really prevent malicious use cases.
    Write,
    /// Full read, write, and delete access
    Admin,
}

impl VendedPermission {
    /// Returns true if this permission allows writing
    pub fn can_write(&self) -> bool {
        matches!(self, Self::Write | Self::Admin)
    }

    /// Returns true if this permission allows deleting
    pub fn can_delete(&self) -> bool {
        matches!(self, Self::Admin)
    }
}

impl FromStr for VendedPermission {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "read" => Ok(Self::Read),
            "write" => Ok(Self::Write),
            "admin" => Ok(Self::Admin),
            _ => Err(format!(
                "Invalid permission '{}'. Must be one of: read, write, admin",
                s
            )),
        }
    }
}

impl std::fmt::Display for VendedPermission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read => write!(f, "read"),
            Self::Write => write!(f, "write"),
            Self::Admin => write!(f, "admin"),
        }
    }
}

/// Property key prefix for credential vendor properties.
/// Properties with this prefix are stripped when using `from_properties`.
pub const PROPERTY_PREFIX: &str = "credential_vendor.";

/// Common property key to explicitly enable credential vending (short form).
pub const ENABLED: &str = "enabled";

/// Common property key for permission level (short form).
pub const PERMISSION: &str = "permission";

/// AWS-specific property keys (short form, without prefix)
#[cfg(feature = "credential-vendor-aws")]
pub mod aws_props {
    pub const ROLE_ARN: &str = "aws_role_arn";
    pub const EXTERNAL_ID: &str = "aws_external_id";
    pub const REGION: &str = "aws_region";
    pub const ROLE_SESSION_NAME: &str = "aws_role_session_name";
    /// AWS credential duration in milliseconds.
    /// Default: 3600000 (1 hour). Range: 900000 (15 min) to 43200000 (12 hours).
    pub const DURATION_MILLIS: &str = "aws_duration_millis";
}

/// GCP-specific property keys (short form, without prefix)
#[cfg(feature = "credential-vendor-gcp")]
pub mod gcp_props {
    pub const SERVICE_ACCOUNT: &str = "gcp_service_account";
}

/// Azure-specific property keys (short form, without prefix)
#[cfg(feature = "credential-vendor-azure")]
pub mod azure_props {
    pub const TENANT_ID: &str = "azure_tenant_id";
    /// Azure storage account name. Required for credential vending.
    pub const ACCOUNT_NAME: &str = "azure_account_name";
    /// Azure credential duration in milliseconds.
    /// Default: 3600000 (1 hour). Azure SAS tokens can be valid up to 7 days.
    pub const DURATION_MILLIS: &str = "azure_duration_millis";
}

/// Vended credentials with expiration information.
#[derive(Clone)]
pub struct VendedCredentials {
    /// Storage options map containing credential keys.
    /// - For AWS: `aws_access_key_id`, `aws_secret_access_key`, `aws_session_token`
    /// - For GCP: `google_storage_token`
    /// - For Azure: `azure_storage_sas_token`, `azure_storage_account_name`
    pub storage_options: HashMap<String, String>,

    /// Expiration time in milliseconds since Unix epoch.
    pub expires_at_millis: u64,
}

impl std::fmt::Debug for VendedCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VendedCredentials")
            .field(
                "storage_options",
                &format!("[{} keys redacted]", self.storage_options.len()),
            )
            .field("expires_at_millis", &self.expires_at_millis)
            .finish()
    }
}

impl VendedCredentials {
    /// Create new vended credentials.
    pub fn new(storage_options: HashMap<String, String>, expires_at_millis: u64) -> Self {
        Self {
            storage_options,
            expires_at_millis,
        }
    }

    /// Check if the credentials have expired.
    pub fn is_expired(&self) -> bool {
        let now_millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time went backwards")
            .as_millis() as u64;
        now_millis >= self.expires_at_millis
    }
}

/// Trait for credential vendors that generate temporary credentials.
///
/// Each cloud provider has its own configuration passed via the vendor
/// implementation. The permission level is configured at vendor creation time
/// via [`VendedPermission`].
#[async_trait]
pub trait CredentialVendor: Send + Sync + std::fmt::Debug {
    /// Vend credentials for accessing the specified table location.
    ///
    /// The permission level (read/write/admin) is determined by the vendor's
    /// configuration, not per-request.
    ///
    /// # Arguments
    ///
    /// * `table_location` - The table URI to vend credentials for
    ///
    /// # Returns
    ///
    /// Returns vended credentials with expiration information.
    async fn vend_credentials(&self, table_location: &str) -> Result<VendedCredentials>;

    /// Returns the cloud provider name (e.g., "aws", "gcp", "azure").
    fn provider_name(&self) -> &'static str;

    /// Returns the permission level configured for this vendor.
    fn permission(&self) -> VendedPermission;
}

/// Detect the cloud provider from a URI scheme.
///
/// Supported schemes for credential vending:
/// - AWS S3: `s3://`
/// - GCP GCS: `gs://`
/// - Azure Blob: `az://`
///
/// Returns "aws", "gcp", "azure", or "unknown".
pub fn detect_provider_from_uri(uri: &str) -> &'static str {
    let Ok(url) = uri_to_url(uri) else {
        return "unknown";
    };

    match url.scheme() {
        "s3" => "aws",
        "gs" => "gcp",
        "az" => "azure",
        _ => "unknown",
    }
}

/// Check if credential vending is enabled.
///
/// Returns true only if the `enabled` property is set to "true".
/// This expects properties with short names (prefix already stripped).
pub fn has_credential_vendor_config(properties: &HashMap<String, String>) -> bool {
    properties
        .get(ENABLED)
        .map(|v| v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Create a credential vendor for the specified table location based on its URI scheme.
///
/// This function automatically detects the cloud provider from the table location
/// and creates the appropriate credential vendor using the provided properties.
///
/// # Arguments
///
/// * `table_location` - The table URI to create a vendor for (e.g., "s3://bucket/path")
/// * `properties` - Configuration properties for credential vendors
///
/// # Returns
///
/// Returns `Some(vendor)` if the provider is detected and configured, `None` if:
/// - The provider cannot be detected from the URI (e.g., local file path)
/// - The required feature is not enabled for the detected provider
///
/// # Errors
///
/// Returns an error if the provider is detected but required configuration is missing:
/// - AWS: `credential_vendor.aws_role_arn` is required
/// - Azure: `credential_vendor.azure_account_name` is required
#[allow(unused_variables)]
pub async fn create_credential_vendor_for_location(
    table_location: &str,
    properties: &HashMap<String, String>,
) -> Result<Option<Box<dyn CredentialVendor>>> {
    let provider = detect_provider_from_uri(table_location);

    match provider {
        #[cfg(feature = "credential-vendor-aws")]
        "aws" => create_aws_vendor(properties).await,

        #[cfg(feature = "credential-vendor-gcp")]
        "gcp" => create_gcp_vendor(properties).await,

        #[cfg(feature = "credential-vendor-azure")]
        "azure" => create_azure_vendor(properties),

        _ => Ok(None),
    }
}

/// Parse permission from properties, defaulting to Read
fn parse_permission(properties: &HashMap<String, String>) -> VendedPermission {
    properties
        .get(PERMISSION)
        .and_then(|s| s.parse().ok())
        .unwrap_or_default()
}

/// Parse duration from properties using a vendor-specific key, defaulting to DEFAULT_CREDENTIAL_DURATION_MILLIS
fn parse_duration_millis(properties: &HashMap<String, String>, key: &str) -> u64 {
    properties
        .get(key)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_CREDENTIAL_DURATION_MILLIS)
}

#[cfg(feature = "credential-vendor-aws")]
async fn create_aws_vendor(
    properties: &HashMap<String, String>,
) -> Result<Option<Box<dyn CredentialVendor>>> {
    use aws::{AwsCredentialVendor, AwsCredentialVendorConfig};
    use lance_core::Error;

    // AWS requires role_arn to be configured
    let role_arn = properties
        .get(aws_props::ROLE_ARN)
        .ok_or_else(|| Error::InvalidInput {
            source: "AWS credential vending requires 'credential_vendor.aws_role_arn' to be set"
                .into(),
            location: snafu::location!(),
        })?;

    let duration_millis = parse_duration_millis(properties, aws_props::DURATION_MILLIS);

    let permission = parse_permission(properties);

    let mut config = AwsCredentialVendorConfig::new(role_arn)
        .with_duration_millis(duration_millis)
        .with_permission(permission);

    if let Some(external_id) = properties.get(aws_props::EXTERNAL_ID) {
        config = config.with_external_id(external_id);
    }
    if let Some(region) = properties.get(aws_props::REGION) {
        config = config.with_region(region);
    }
    if let Some(session_name) = properties.get(aws_props::ROLE_SESSION_NAME) {
        config = config.with_role_session_name(session_name);
    }

    let vendor = AwsCredentialVendor::new(config).await?;
    Ok(Some(Box::new(vendor)))
}

#[cfg(feature = "credential-vendor-gcp")]
async fn create_gcp_vendor(
    properties: &HashMap<String, String>,
) -> Result<Option<Box<dyn CredentialVendor>>> {
    use gcp::{GcpCredentialVendor, GcpCredentialVendorConfig};

    let permission = parse_permission(properties);

    let mut config = GcpCredentialVendorConfig::new().with_permission(permission);

    if let Some(sa) = properties.get(gcp_props::SERVICE_ACCOUNT) {
        config = config.with_service_account(sa);
    }

    let vendor = GcpCredentialVendor::new(config).await?;
    Ok(Some(Box::new(vendor)))
}

#[cfg(feature = "credential-vendor-azure")]
fn create_azure_vendor(
    properties: &HashMap<String, String>,
) -> Result<Option<Box<dyn CredentialVendor>>> {
    use azure::{AzureCredentialVendor, AzureCredentialVendorConfig};
    use lance_core::Error;

    // Azure requires account_name to be configured
    let account_name =
        properties
            .get(azure_props::ACCOUNT_NAME)
            .ok_or_else(|| {
                Error::InvalidInput {
            source:
                "Azure credential vending requires 'credential_vendor.azure_account_name' to be set"
                    .into(),
            location: snafu::location!(),
        }
            })?;

    let duration_millis = parse_duration_millis(properties, azure_props::DURATION_MILLIS);
    let permission = parse_permission(properties);

    let mut config = AzureCredentialVendorConfig::new()
        .with_account_name(account_name)
        .with_duration_millis(duration_millis)
        .with_permission(permission);

    if let Some(tenant_id) = properties.get(azure_props::TENANT_ID) {
        config = config.with_tenant_id(tenant_id);
    }

    let vendor = AzureCredentialVendor::new(config);
    Ok(Some(Box::new(vendor)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_provider_from_uri() {
        // AWS (supported scheme: s3://)
        assert_eq!(detect_provider_from_uri("s3://bucket/path"), "aws");
        assert_eq!(detect_provider_from_uri("S3://bucket/path"), "aws");

        // GCP (supported scheme: gs://)
        assert_eq!(detect_provider_from_uri("gs://bucket/path"), "gcp");
        assert_eq!(detect_provider_from_uri("GS://bucket/path"), "gcp");

        // Azure (supported scheme: az://)
        assert_eq!(detect_provider_from_uri("az://container/path"), "azure");

        // Unknown (unsupported schemes)
        assert_eq!(detect_provider_from_uri("/local/path"), "unknown");
        assert_eq!(detect_provider_from_uri("file:///local/path"), "unknown");
        assert_eq!(detect_provider_from_uri("memory://test"), "unknown");
        // Hadoop-style schemes not supported by lance-io
        assert_eq!(detect_provider_from_uri("s3a://bucket/path"), "unknown");
        assert_eq!(
            detect_provider_from_uri("abfss://container@account.dfs.core.windows.net/path"),
            "unknown"
        );
        assert_eq!(
            detect_provider_from_uri("wasbs://container@account.blob.core.windows.net/path"),
            "unknown"
        );
    }

    #[test]
    fn test_vended_permission_from_str() {
        // Valid values (case-insensitive)
        assert_eq!(
            "read".parse::<VendedPermission>().unwrap(),
            VendedPermission::Read
        );
        assert_eq!(
            "READ".parse::<VendedPermission>().unwrap(),
            VendedPermission::Read
        );
        assert_eq!(
            "write".parse::<VendedPermission>().unwrap(),
            VendedPermission::Write
        );
        assert_eq!(
            "WRITE".parse::<VendedPermission>().unwrap(),
            VendedPermission::Write
        );
        assert_eq!(
            "admin".parse::<VendedPermission>().unwrap(),
            VendedPermission::Admin
        );
        assert_eq!(
            "Admin".parse::<VendedPermission>().unwrap(),
            VendedPermission::Admin
        );

        // Invalid values should return error
        let err = "invalid".parse::<VendedPermission>().unwrap_err();
        assert!(err.contains("Invalid permission"));
        assert!(err.contains("invalid"));

        let err = "".parse::<VendedPermission>().unwrap_err();
        assert!(err.contains("Invalid permission"));

        let err = "readwrite".parse::<VendedPermission>().unwrap_err();
        assert!(err.contains("Invalid permission"));
    }

    #[test]
    fn test_vended_permission_display() {
        assert_eq!(VendedPermission::Read.to_string(), "read");
        assert_eq!(VendedPermission::Write.to_string(), "write");
        assert_eq!(VendedPermission::Admin.to_string(), "admin");
    }

    #[test]
    fn test_parse_permission_with_invalid_values() {
        // Invalid permission should default to Read
        let mut props = HashMap::new();
        props.insert(PERMISSION.to_string(), "invalid".to_string());
        assert_eq!(parse_permission(&props), VendedPermission::Read);

        // Empty permission should default to Read
        props.insert(PERMISSION.to_string(), "".to_string());
        assert_eq!(parse_permission(&props), VendedPermission::Read);

        // Missing permission should default to Read
        let empty_props: HashMap<String, String> = HashMap::new();
        assert_eq!(parse_permission(&empty_props), VendedPermission::Read);
    }

    #[test]
    fn test_parse_duration_millis_with_invalid_values() {
        const TEST_KEY: &str = "test_duration_millis";

        // Invalid duration should default to DEFAULT_CREDENTIAL_DURATION_MILLIS
        let mut props = HashMap::new();
        props.insert(TEST_KEY.to_string(), "not_a_number".to_string());
        assert_eq!(
            parse_duration_millis(&props, TEST_KEY),
            DEFAULT_CREDENTIAL_DURATION_MILLIS
        );

        // Negative number (parsed as u64 fails)
        props.insert(TEST_KEY.to_string(), "-1000".to_string());
        assert_eq!(
            parse_duration_millis(&props, TEST_KEY),
            DEFAULT_CREDENTIAL_DURATION_MILLIS
        );

        // Empty string should default
        props.insert(TEST_KEY.to_string(), "".to_string());
        assert_eq!(
            parse_duration_millis(&props, TEST_KEY),
            DEFAULT_CREDENTIAL_DURATION_MILLIS
        );

        // Missing duration should default
        let empty_props: HashMap<String, String> = HashMap::new();
        assert_eq!(
            parse_duration_millis(&empty_props, TEST_KEY),
            DEFAULT_CREDENTIAL_DURATION_MILLIS
        );

        // Valid duration should work
        props.insert(TEST_KEY.to_string(), "7200000".to_string());
        assert_eq!(parse_duration_millis(&props, TEST_KEY), 7200000);
    }

    #[test]
    fn test_has_credential_vendor_config() {
        // enabled = true
        let mut props = HashMap::new();
        props.insert(ENABLED.to_string(), "true".to_string());
        assert!(has_credential_vendor_config(&props));

        // enabled = TRUE (case-insensitive)
        props.insert(ENABLED.to_string(), "TRUE".to_string());
        assert!(has_credential_vendor_config(&props));

        // enabled = false
        props.insert(ENABLED.to_string(), "false".to_string());
        assert!(!has_credential_vendor_config(&props));

        // enabled = invalid value
        props.insert(ENABLED.to_string(), "yes".to_string());
        assert!(!has_credential_vendor_config(&props));

        // enabled missing
        let empty_props: HashMap<String, String> = HashMap::new();
        assert!(!has_credential_vendor_config(&empty_props));
    }

    #[test]
    fn test_vended_credentials_debug_redacts_secrets() {
        let mut storage_options = HashMap::new();
        storage_options.insert(
            "aws_access_key_id".to_string(),
            "AKIAIOSFODNN7EXAMPLE".to_string(),
        );
        storage_options.insert(
            "aws_secret_access_key".to_string(),
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".to_string(),
        );
        storage_options.insert(
            "aws_session_token".to_string(),
            "FwoGZXIvYXdzE...".to_string(),
        );

        let creds = VendedCredentials::new(storage_options, 1234567890);
        let debug_output = format!("{:?}", creds);

        // Should NOT contain actual secrets
        assert!(!debug_output.contains("AKIAIOSFODNN7EXAMPLE"));
        assert!(!debug_output.contains("wJalrXUtnFEMI"));
        assert!(!debug_output.contains("FwoGZXIvYXdzE"));

        // Should contain redacted message
        assert!(debug_output.contains("redacted"));
        assert!(debug_output.contains("3 keys"));

        // Should contain expiration time
        assert!(debug_output.contains("1234567890"));
    }

    #[test]
    fn test_vended_credentials_is_expired() {
        // Create credentials that expired in the past
        let past_millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
            - 1000; // 1 second ago

        let expired_creds = VendedCredentials::new(HashMap::new(), past_millis);
        assert!(expired_creds.is_expired());

        // Create credentials that expire in the future
        let future_millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
            + 3600000; // 1 hour from now

        let valid_creds = VendedCredentials::new(HashMap::new(), future_millis);
        assert!(!valid_creds.is_expired());
    }

    #[test]
    fn test_redact_credential() {
        // Long credential: shows first 8 and last 4
        assert_eq!(redact_credential("AKIAIOSFODNN7EXAMPLE"), "AKIAIOSF***MPLE");

        // Exactly 16 chars: shows first 8 and last 4
        assert_eq!(redact_credential("1234567890123456"), "12345678***3456");

        // Short credential (< 16 chars): shows only first few
        assert_eq!(redact_credential("short1234567"), "short123***");
        assert_eq!(redact_credential("short123"), "short123***");
        assert_eq!(redact_credential("tiny"), "tiny***");
        assert_eq!(redact_credential("ab"), "ab***");
        assert_eq!(redact_credential("a"), "a***");

        // Empty string
        assert_eq!(redact_credential(""), "[empty]");

        // Real-world examples
        // AWS access key ID (20 chars) - shows AKIA + 4 more chars which helps identify the key
        assert_eq!(redact_credential("AKIAIOSFODNN7EXAMPLE"), "AKIAIOSF***MPLE");

        // GCP token (typically very long)
        let long_token = "ya29.a0AfH6SMBx1234567890abcdefghijklmnopqrstuvwxyz";
        assert_eq!(redact_credential(long_token), "ya29.a0A***wxyz");

        // Azure SAS token
        let sas_token = "sv=2021-06-08&ss=b&srt=sco&sp=rwdlacuiytfx&se=2024-12-31";
        assert_eq!(redact_credential(sas_token), "sv=2021-***2-31");
    }
}
