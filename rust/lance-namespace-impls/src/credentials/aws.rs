// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! AWS credential vending using STS AssumeRole.
//!
//! This module provides credential vending for AWS S3 storage by assuming
//! an IAM role using AWS STS (Security Token Service).

use std::collections::HashMap;

use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_sdk_sts::Client as StsClient;
use lance_core::{Error, Result};
use lance_io::object_store::uri_to_url;
use log::{debug, info};

use super::{
    redact_credential, CredentialVendor, VendedCredentials, VendedPermission,
    DEFAULT_CREDENTIAL_DURATION_MILLIS,
};

/// Configuration for AWS credential vending.
#[derive(Debug, Clone)]
pub struct AwsCredentialVendorConfig {
    /// The IAM role ARN to assume.
    pub role_arn: String,

    /// Optional external ID for the assume role request.
    pub external_id: Option<String>,

    /// Duration for vended credentials in milliseconds.
    /// Default: 3600000 (1 hour).
    /// AWS STS allows 900-43200 seconds (15 min - 12 hours).
    /// Values outside this range will be clamped.
    pub duration_millis: u64,

    /// Optional role session name. Defaults to "lance-credential-vending".
    pub role_session_name: Option<String>,

    /// Optional AWS region for the STS client.
    pub region: Option<String>,

    /// Permission level for vended credentials.
    /// Default: Read (full read access)
    pub permission: VendedPermission,
}

impl AwsCredentialVendorConfig {
    /// Create a new config with the specified role ARN.
    pub fn new(role_arn: impl Into<String>) -> Self {
        Self {
            role_arn: role_arn.into(),
            external_id: None,
            duration_millis: DEFAULT_CREDENTIAL_DURATION_MILLIS,
            role_session_name: None,
            region: None,
            permission: VendedPermission::default(),
        }
    }

    /// Set the external ID for the assume role request.
    pub fn with_external_id(mut self, external_id: impl Into<String>) -> Self {
        self.external_id = Some(external_id.into());
        self
    }

    /// Set the credential duration in milliseconds.
    pub fn with_duration_millis(mut self, millis: u64) -> Self {
        self.duration_millis = millis;
        self
    }

    /// Set the role session name.
    pub fn with_role_session_name(mut self, name: impl Into<String>) -> Self {
        self.role_session_name = Some(name.into());
        self
    }

    /// Set the AWS region for the STS client.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Set the permission level for vended credentials.
    pub fn with_permission(mut self, permission: VendedPermission) -> Self {
        self.permission = permission;
        self
    }
}

/// AWS credential vendor that uses STS AssumeRole.
#[derive(Debug)]
pub struct AwsCredentialVendor {
    config: AwsCredentialVendorConfig,
    sts_client: StsClient,
}

impl AwsCredentialVendor {
    /// Create a new AWS credential vendor with the specified configuration.
    pub async fn new(config: AwsCredentialVendorConfig) -> Result<Self> {
        let mut aws_config_loader = aws_config::defaults(BehaviorVersion::latest());

        if let Some(ref region) = config.region {
            aws_config_loader = aws_config_loader.region(aws_config::Region::new(region.clone()));
        }

        let aws_config = aws_config_loader.load().await;
        let sts_client = StsClient::new(&aws_config);

        Ok(Self { config, sts_client })
    }

    /// Create a new AWS credential vendor with an existing STS client.
    pub fn with_sts_client(config: AwsCredentialVendorConfig, sts_client: StsClient) -> Self {
        Self { config, sts_client }
    }

    /// Parse an S3 URI to extract bucket and prefix.
    fn parse_s3_uri(uri: &str) -> Result<(String, String)> {
        let url = uri_to_url(uri)?;

        let bucket = url
            .host_str()
            .ok_or_else(|| Error::InvalidInput {
                source: format!("S3 URI '{}' missing bucket", uri).into(),
                location: snafu::location!(),
            })?
            .to_string();

        let prefix = url.path().trim_start_matches('/').to_string();

        Ok((bucket, prefix))
    }

    /// Build a scoped IAM policy for the specified location and permission level.
    ///
    /// Permission levels:
    /// - `Read`: Full read access to all content (metadata, indices, data files)
    /// - `Write`: Full read and write access (no delete)
    /// - `Admin`: Full read, write, and delete access
    fn build_policy(bucket: &str, prefix: &str, permission: VendedPermission) -> String {
        let prefix_trimmed = prefix.trim_end_matches('/');
        let base_path = if prefix.is_empty() {
            format!("arn:aws:s3:::{}/*", bucket)
        } else {
            format!("arn:aws:s3:::{}/{}/*", bucket, prefix_trimmed)
        };
        let bucket_arn = format!("arn:aws:s3:::{}", bucket);

        let mut statements = vec![];

        // List bucket permission (always needed)
        statements.push(serde_json::json!({
            "Effect": "Allow",
            "Action": "s3:ListBucket",
            "Resource": bucket_arn,
            "Condition": {
                "StringLike": {
                    "s3:prefix": if prefix.is_empty() {
                        "*".to_string()
                    } else {
                        format!("{}/*", prefix_trimmed)
                    }
                }
            }
        }));

        // Get bucket location (always needed)
        statements.push(serde_json::json!({
            "Effect": "Allow",
            "Action": "s3:GetBucketLocation",
            "Resource": bucket_arn
        }));

        // Read access (all permission levels have full read)
        statements.push(serde_json::json!({
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:GetObjectVersion"],
            "Resource": base_path
        }));

        // Write access (Write and Admin)
        if permission.can_write() {
            statements.push(serde_json::json!({
                "Effect": "Allow",
                "Action": "s3:PutObject",
                "Resource": base_path
            }));
        }

        // Delete access (Admin only)
        if permission.can_delete() {
            statements.push(serde_json::json!({
                "Effect": "Allow",
                "Action": "s3:DeleteObject",
                "Resource": base_path
            }));
        }

        let policy = serde_json::json!({
            "Version": "2012-10-17",
            "Statement": statements
        });

        policy.to_string()
    }
}

#[async_trait]
impl CredentialVendor for AwsCredentialVendor {
    async fn vend_credentials(&self, table_location: &str) -> Result<VendedCredentials> {
        debug!(
            "AWS credential vending: location={}, permission={}",
            table_location, self.config.permission
        );

        let (bucket, prefix) = Self::parse_s3_uri(table_location)?;
        let policy = Self::build_policy(&bucket, &prefix, self.config.permission);

        let role_session_name = self
            .config
            .role_session_name
            .clone()
            .unwrap_or_else(|| "lance-credential-vending".to_string());

        // Cap session name to 64 chars (AWS limit)
        let role_session_name = if role_session_name.len() > 64 {
            role_session_name[..64].to_string()
        } else {
            role_session_name
        };

        // Convert millis to seconds for AWS API (rounding up to ensure at least the requested duration)
        // AWS STS allows 900-43200 seconds (15 min - 12 hours), clamp to valid range
        let duration_secs = self.config.duration_millis.div_ceil(1000).clamp(900, 43200) as i32;

        let mut request = self
            .sts_client
            .assume_role()
            .role_arn(&self.config.role_arn)
            .role_session_name(&role_session_name)
            .policy(&policy)
            .duration_seconds(duration_secs);

        if let Some(ref external_id) = self.config.external_id {
            request = request.external_id(external_id);
        }

        let response = request.send().await.map_err(|e| Error::IO {
            source: Box::new(std::io::Error::other(format!(
                "Failed to assume role '{}': {}",
                self.config.role_arn, e
            ))),
            location: snafu::location!(),
        })?;

        let credentials = response.credentials().ok_or_else(|| Error::IO {
            source: Box::new(std::io::Error::other(
                "AssumeRole response missing credentials",
            )),
            location: snafu::location!(),
        })?;

        let access_key_id = credentials.access_key_id().to_string();
        let secret_access_key = credentials.secret_access_key().to_string();
        let session_token = credentials.session_token().to_string();

        let expiration = credentials.expiration();
        let expires_at_millis =
            (expiration.secs() as u64) * 1000 + (expiration.subsec_nanos() / 1_000_000) as u64;

        info!(
            "AWS credentials vended: bucket={}, prefix={}, permission={}, expires_at={}, access_key_id={}",
            bucket, prefix, self.config.permission, expires_at_millis, redact_credential(&access_key_id)
        );

        let mut storage_options = HashMap::new();
        storage_options.insert("aws_access_key_id".to_string(), access_key_id);
        storage_options.insert("aws_secret_access_key".to_string(), secret_access_key);
        storage_options.insert("aws_session_token".to_string(), session_token);
        storage_options.insert(
            "expires_at_millis".to_string(),
            expires_at_millis.to_string(),
        );

        // Include region if configured
        if let Some(ref region) = self.config.region {
            storage_options.insert("aws_region".to_string(), region.clone());
        }

        Ok(VendedCredentials::new(storage_options, expires_at_millis))
    }

    fn provider_name(&self) -> &'static str {
        "aws"
    }

    fn permission(&self) -> VendedPermission {
        self.config.permission
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_s3_uri() {
        let (bucket, prefix) = AwsCredentialVendor::parse_s3_uri("s3://my-bucket/path/to/table")
            .expect("should parse");
        assert_eq!(bucket, "my-bucket");
        assert_eq!(prefix, "path/to/table");

        let (bucket, prefix) =
            AwsCredentialVendor::parse_s3_uri("s3://my-bucket/").expect("should parse");
        assert_eq!(bucket, "my-bucket");
        assert_eq!(prefix, "");

        let (bucket, prefix) =
            AwsCredentialVendor::parse_s3_uri("s3://my-bucket").expect("should parse");
        assert_eq!(bucket, "my-bucket");
        assert_eq!(prefix, "");
    }

    #[test]
    fn test_build_policy_read() {
        let policy =
            AwsCredentialVendor::build_policy("my-bucket", "path/to/table", VendedPermission::Read);
        let parsed: serde_json::Value = serde_json::from_str(&policy).expect("valid json");

        let statements = parsed["Statement"].as_array().expect("statements array");
        assert_eq!(statements.len(), 3); // ListBucket, GetBucketLocation, GetObject

        // Verify no write actions
        for stmt in statements {
            let actions = stmt["Action"].clone();
            let action_list: Vec<String> = if actions.is_array() {
                actions
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|a| a.as_str().unwrap().to_string())
                    .collect()
            } else {
                vec![actions.as_str().unwrap().to_string()]
            };
            assert!(!action_list.contains(&"s3:PutObject".to_string()));
            assert!(!action_list.contains(&"s3:DeleteObject".to_string()));
        }
    }

    #[test]
    fn test_build_policy_write() {
        let policy = AwsCredentialVendor::build_policy(
            "my-bucket",
            "path/to/table",
            VendedPermission::Write,
        );
        let parsed: serde_json::Value = serde_json::from_str(&policy).expect("valid json");

        let statements = parsed["Statement"].as_array().expect("statements array");
        // ListBucket, GetBucketLocation, GetObject, PutObject
        assert_eq!(statements.len(), 4);

        // Verify PutObject is present
        let write_stmt = statements
            .iter()
            .find(|s| {
                let action = &s["Action"];
                action.as_str() == Some("s3:PutObject")
            })
            .expect("should have PutObject statement");
        assert!(write_stmt["Effect"].as_str() == Some("Allow"));

        // Verify DeleteObject is NOT present (Write doesn't have delete)
        let delete_stmt = statements.iter().find(|s| {
            let action = &s["Action"];
            action.as_str() == Some("s3:DeleteObject")
        });
        assert!(delete_stmt.is_none(), "Write should not have DeleteObject");

        // Verify no Deny statements
        let deny_stmt = statements
            .iter()
            .find(|s| s["Effect"].as_str() == Some("Deny"));
        assert!(deny_stmt.is_none(), "Write should not have Deny statements");
    }

    #[test]
    fn test_build_policy_admin() {
        let policy = AwsCredentialVendor::build_policy(
            "my-bucket",
            "path/to/table",
            VendedPermission::Admin,
        );
        let parsed: serde_json::Value = serde_json::from_str(&policy).expect("valid json");

        let statements = parsed["Statement"].as_array().expect("statements array");
        // ListBucket, GetBucketLocation, GetObject, PutObject, DeleteObject
        assert_eq!(statements.len(), 5);

        // Verify read actions
        let read_stmt = statements
            .iter()
            .find(|s| {
                let actions = s["Action"].clone();
                if actions.is_array() {
                    actions
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|a| a.as_str().unwrap() == "s3:GetObject")
                } else {
                    false
                }
            })
            .expect("should have read statement");
        assert!(read_stmt["Effect"].as_str() == Some("Allow"));

        // Verify PutObject
        let write_stmt = statements
            .iter()
            .find(|s| s["Action"].as_str() == Some("s3:PutObject"))
            .expect("should have PutObject statement");
        assert!(write_stmt["Effect"].as_str() == Some("Allow"));

        // Verify DeleteObject (Admin only)
        let delete_stmt = statements
            .iter()
            .find(|s| s["Action"].as_str() == Some("s3:DeleteObject"))
            .expect("should have DeleteObject statement");
        assert!(delete_stmt["Effect"].as_str() == Some("Allow"));

        // Verify no Deny statements
        let deny_stmt = statements
            .iter()
            .find(|s| s["Effect"].as_str() == Some("Deny"));
        assert!(deny_stmt.is_none(), "Admin should not have Deny statements");
    }

    #[test]
    fn test_config_builder() {
        let config = AwsCredentialVendorConfig::new("arn:aws:iam::123456789012:role/MyRole")
            .with_external_id("my-external-id")
            .with_duration_millis(7200000)
            .with_role_session_name("my-session")
            .with_region("us-west-2");

        assert_eq!(config.role_arn, "arn:aws:iam::123456789012:role/MyRole");
        assert_eq!(config.external_id, Some("my-external-id".to_string()));
        assert_eq!(config.duration_millis, 7200000);
        assert_eq!(config.role_session_name, Some("my-session".to_string()));
        assert_eq!(config.region, Some("us-west-2".to_string()));
    }

    // ============================================================================
    // Integration Tests
    // ============================================================================

    /// Integration tests for AWS credential vending.
    ///
    /// These tests require:
    /// - Valid AWS credentials (via environment, IAM role, or credential file)
    /// - The `LANCE_TEST_AWS_ROLE_ARN` environment variable set to a role ARN that
    ///   can be assumed by the current credentials
    /// - Access to the S3 bucket `jack-lancedb-devland-us-east-1`
    ///
    /// Run with: `cargo test --features credential-vendor-aws -- --ignored`
    #[cfg(test)]
    mod integration {
        use super::*;
        use crate::DirectoryNamespaceBuilder;
        use arrow::array::{Int32Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::ipc::writer::StreamWriter;
        use arrow::record_batch::RecordBatch;
        use bytes::Bytes;
        use lance_namespace::models::*;
        use lance_namespace::LanceNamespace;
        use std::sync::Arc;

        const TEST_BUCKET: &str = "jack-lancedb-devland-us-east-1";

        /// Helper to create Arrow IPC data for testing
        fn create_test_arrow_data() -> Bytes {
            let schema = Schema::new(vec![
                Field::new("id", DataType::Int32, false),
                Field::new("name", DataType::Utf8, false),
            ]);

            let batch = RecordBatch::try_new(
                Arc::new(schema),
                vec![
                    Arc::new(Int32Array::from(vec![1, 2, 3])),
                    Arc::new(StringArray::from(vec!["alice", "bob", "charlie"])),
                ],
            )
            .unwrap();

            let mut buffer = Vec::new();
            {
                let mut writer = StreamWriter::try_new(&mut buffer, &batch.schema()).unwrap();
                writer.write(&batch).unwrap();
                writer.finish().unwrap();
            }

            Bytes::from(buffer)
        }

        /// Generate a unique test path for each test run to avoid conflicts
        fn unique_test_path() -> String {
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis();
            format!("lance-test/credential-vending-{}", timestamp)
        }

        /// Get the role ARN from environment variable
        fn get_test_role_arn() -> Option<String> {
            std::env::var("LANCE_TEST_AWS_ROLE_ARN").ok()
        }

        #[tokio::test]
        #[ignore = "requires AWS credentials and LANCE_TEST_AWS_ROLE_ARN env var"]
        async fn test_aws_credential_vending_basic() {
            let role_arn = get_test_role_arn()
                .expect("LANCE_TEST_AWS_ROLE_ARN must be set for integration tests");

            let test_path = unique_test_path();
            let table_location = format!("s3://{}/{}/test_table", TEST_BUCKET, test_path);

            // Test Read permission
            let read_config = AwsCredentialVendorConfig::new(&role_arn)
                .with_duration_millis(900_000) // 15 minutes (minimum)
                .with_region("us-east-1")
                .with_permission(VendedPermission::Read);

            let read_vendor = AwsCredentialVendor::new(read_config)
                .await
                .expect("should create read vendor");

            let read_creds = read_vendor
                .vend_credentials(&table_location)
                .await
                .expect("should vend read credentials");

            assert!(
                read_creds.storage_options.contains_key("aws_access_key_id"),
                "should have access key id"
            );
            assert!(
                read_creds
                    .storage_options
                    .contains_key("aws_secret_access_key"),
                "should have secret access key"
            );
            assert!(
                read_creds.storage_options.contains_key("aws_session_token"),
                "should have session token"
            );
            assert!(
                !read_creds.is_expired(),
                "credentials should not be expired"
            );
            assert_eq!(
                read_vendor.permission(),
                VendedPermission::Read,
                "permission should be Read"
            );

            // Test Admin permission
            let admin_config = AwsCredentialVendorConfig::new(&role_arn)
                .with_duration_millis(900_000)
                .with_region("us-east-1")
                .with_permission(VendedPermission::Admin);

            let admin_vendor = AwsCredentialVendor::new(admin_config)
                .await
                .expect("should create admin vendor");

            let admin_creds = admin_vendor
                .vend_credentials(&table_location)
                .await
                .expect("should vend admin credentials");

            assert!(
                admin_creds
                    .storage_options
                    .contains_key("aws_access_key_id"),
                "should have access key id"
            );
            assert!(
                !admin_creds.is_expired(),
                "credentials should not be expired"
            );
            assert_eq!(
                admin_vendor.permission(),
                VendedPermission::Admin,
                "permission should be Admin"
            );
        }

        #[tokio::test]
        #[ignore = "requires AWS credentials and LANCE_TEST_AWS_ROLE_ARN env var"]
        async fn test_directory_namespace_with_aws_credential_vending() {
            let role_arn = get_test_role_arn()
                .expect("LANCE_TEST_AWS_ROLE_ARN must be set for integration tests");

            let test_path = unique_test_path();
            let root = format!("s3://{}/{}", TEST_BUCKET, test_path);

            // Build DirectoryNamespace with credential vending using short property names
            let namespace = DirectoryNamespaceBuilder::new(&root)
                .manifest_enabled(true)
                .credential_vendor_property("enabled", "true")
                .credential_vendor_property("aws_role_arn", &role_arn)
                .credential_vendor_property("aws_duration_millis", "900000") // 15 minutes
                .credential_vendor_property("aws_region", "us-east-1")
                .credential_vendor_property("permission", "admin")
                .build()
                .await
                .expect("should build namespace");

            // Create a child namespace
            let create_ns_req = CreateNamespaceRequest {
                id: Some(vec!["test_ns".to_string()]),
                properties: None,
                mode: None,
            };
            namespace
                .create_namespace(create_ns_req)
                .await
                .expect("should create namespace");

            // Create a table with data
            let table_data = create_test_arrow_data();
            let create_table_req = CreateTableRequest {
                id: Some(vec!["test_ns".to_string(), "test_table".to_string()]),
                mode: Some("Create".to_string()),
            };
            let create_response = namespace
                .create_table(create_table_req, table_data)
                .await
                .expect("should create table");

            assert!(
                create_response.location.is_some(),
                "should have location in response"
            );
            assert_eq!(create_response.version, Some(1), "should be version 1");

            // Describe the table (this should use vended credentials)
            let describe_req = DescribeTableRequest {
                id: Some(vec!["test_ns".to_string(), "test_table".to_string()]),
                ..Default::default()
            };
            let describe_response = namespace
                .describe_table(describe_req)
                .await
                .expect("should describe table");

            assert!(describe_response.location.is_some(), "should have location");
            assert!(
                describe_response.storage_options.is_some(),
                "should have storage_options with vended credentials"
            );

            let storage_options = describe_response.storage_options.unwrap();
            assert!(
                storage_options.contains_key("aws_access_key_id"),
                "should have vended aws_access_key_id"
            );
            assert!(
                storage_options.contains_key("aws_secret_access_key"),
                "should have vended aws_secret_access_key"
            );
            assert!(
                storage_options.contains_key("aws_session_token"),
                "should have vended aws_session_token"
            );
            assert!(
                storage_options.contains_key("expires_at_millis"),
                "should have expires_at_millis"
            );

            // Verify expiration is in the future
            let expires_at: u64 = storage_options
                .get("expires_at_millis")
                .unwrap()
                .parse()
                .expect("should parse expires_at_millis");
            let now_millis = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;
            assert!(
                expires_at > now_millis,
                "expiration should be in the future"
            );

            // List tables to verify the table was created
            let list_req = ListTablesRequest {
                id: Some(vec!["test_ns".to_string()]),
                page_token: None,
                limit: None,
            };
            let list_response = namespace
                .list_tables(list_req)
                .await
                .expect("should list tables");
            assert!(
                list_response.tables.contains(&"test_table".to_string()),
                "should contain test_table"
            );

            // Clean up: drop the table
            let drop_req = DropTableRequest {
                id: Some(vec!["test_ns".to_string(), "test_table".to_string()]),
            };
            namespace
                .drop_table(drop_req)
                .await
                .expect("should drop table");

            // Clean up: drop the namespace
            let mut drop_ns_req = DropNamespaceRequest::new();
            drop_ns_req.id = Some(vec!["test_ns".to_string()]);
            namespace
                .drop_namespace(drop_ns_req)
                .await
                .expect("should drop namespace");
        }

        #[tokio::test]
        #[ignore = "requires AWS credentials and LANCE_TEST_AWS_ROLE_ARN env var"]
        async fn test_credential_refresh_on_expiration() {
            let role_arn = get_test_role_arn()
                .expect("LANCE_TEST_AWS_ROLE_ARN must be set for integration tests");

            let test_path = unique_test_path();
            let table_location = format!("s3://{}/{}/refresh_test", TEST_BUCKET, test_path);

            // Create vendor with minimum duration and Admin permission
            let config = AwsCredentialVendorConfig::new(&role_arn)
                .with_duration_millis(900_000) // 15 minutes
                .with_region("us-east-1")
                .with_permission(VendedPermission::Admin);

            let vendor = AwsCredentialVendor::new(config)
                .await
                .expect("should create vendor");

            // Vend credentials multiple times to verify consistent behavior
            let creds1 = vendor
                .vend_credentials(&table_location)
                .await
                .expect("should vend credentials first time");

            let creds2 = vendor
                .vend_credentials(&table_location)
                .await
                .expect("should vend credentials second time");

            // Both should be valid (not expired)
            assert!(!creds1.is_expired(), "first credentials should be valid");
            assert!(!creds2.is_expired(), "second credentials should be valid");

            // Both should have access keys (they may be different due to new STS calls)
            assert!(
                creds1.storage_options.contains_key("aws_access_key_id"),
                "first creds should have access key"
            );
            assert!(
                creds2.storage_options.contains_key("aws_access_key_id"),
                "second creds should have access key"
            );
        }

        #[tokio::test]
        #[ignore = "requires AWS credentials and LANCE_TEST_AWS_ROLE_ARN env var"]
        async fn test_scoped_policy_permissions() {
            let role_arn = get_test_role_arn()
                .expect("LANCE_TEST_AWS_ROLE_ARN must be set for integration tests");

            let test_path = unique_test_path();

            // Create two different table locations
            let table1_location = format!("s3://{}/{}/table1", TEST_BUCKET, test_path);
            let table2_location = format!("s3://{}/{}/table2", TEST_BUCKET, test_path);

            let config = AwsCredentialVendorConfig::new(&role_arn)
                .with_duration_millis(900_000)
                .with_region("us-east-1")
                .with_permission(VendedPermission::Admin);

            let vendor = AwsCredentialVendor::new(config)
                .await
                .expect("should create vendor");

            // Vend credentials for table1
            let creds1 = vendor
                .vend_credentials(&table1_location)
                .await
                .expect("should vend credentials for table1");

            // Vend credentials for table2
            let creds2 = vendor
                .vend_credentials(&table2_location)
                .await
                .expect("should vend credentials for table2");

            // Both should be valid
            assert!(!creds1.is_expired(), "table1 credentials should be valid");
            assert!(!creds2.is_expired(), "table2 credentials should be valid");

            // The credentials are scoped to their respective paths via IAM policy
            // (the policy restricts access to specific S3 paths)
        }

        #[tokio::test]
        #[ignore = "requires AWS credentials and LANCE_TEST_AWS_ROLE_ARN env var"]
        async fn test_from_properties_builder() {
            let role_arn = get_test_role_arn()
                .expect("LANCE_TEST_AWS_ROLE_ARN must be set for integration tests");

            let test_path = unique_test_path();
            let root = format!("s3://{}/{}", TEST_BUCKET, test_path);

            // Build namespace using from_properties (simulating config from external source)
            // Properties use the "credential_vendor." prefix which gets stripped
            let mut properties = HashMap::new();
            properties.insert("root".to_string(), root.clone());
            properties.insert("manifest_enabled".to_string(), "true".to_string());
            properties.insert("credential_vendor.enabled".to_string(), "true".to_string());
            properties.insert(
                "credential_vendor.aws_role_arn".to_string(),
                role_arn.clone(),
            );
            properties.insert(
                "credential_vendor.aws_duration_millis".to_string(),
                "900000".to_string(),
            );
            properties.insert(
                "credential_vendor.aws_region".to_string(),
                "us-east-1".to_string(),
            );
            properties.insert(
                "credential_vendor.permission".to_string(),
                "admin".to_string(),
            );

            let namespace = DirectoryNamespaceBuilder::from_properties(properties, None)
                .expect("should parse properties")
                .build()
                .await
                .expect("should build namespace");

            // Verify namespace works
            let create_ns_req = CreateNamespaceRequest {
                id: Some(vec!["props_test".to_string()]),
                properties: None,
                mode: None,
            };
            namespace
                .create_namespace(create_ns_req)
                .await
                .expect("should create namespace");

            // Clean up
            let mut drop_ns_req = DropNamespaceRequest::new();
            drop_ns_req.id = Some(vec!["props_test".to_string()]);
            namespace
                .drop_namespace(drop_ns_req)
                .await
                .expect("should drop namespace");
        }
    }
}
