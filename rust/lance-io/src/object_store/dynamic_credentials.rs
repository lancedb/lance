// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use lance_core::error::{Error, Result};
use object_store::{CredentialProvider, Result as ObjectStoreResult};

use crate::object_store::{StorageOptionsAccessor, StorageOptionsProvider};

#[cfg(feature = "aws")]
use object_store::aws::AwsCredential as ObjectStoreAwsCredential;
#[cfg(feature = "azure")]
use object_store::azure::{AzureAccessKey, AzureCredential};
#[cfg(feature = "gcp")]
use object_store::gcp::GcpCredential;

/// Raw dynamic storage options fetched from a credential-vending source.
///
/// Callers must convert this bag into a cloud-specific credential type via
/// `TryFrom<DynamicCredentials>`.
#[derive(Clone)]
pub struct DynamicCredentials(pub HashMap<String, String>);

impl fmt::Debug for DynamicCredentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DynamicCredentials")
            .field(&format_args!("[{} keys redacted]", self.0.len()))
            .finish()
    }
}

#[derive(Clone)]
pub struct NamespaceCredentialsProvider<T> {
    accessor: Arc<StorageOptionsAccessor>,
    _credential: PhantomData<T>,
}

impl<T> fmt::Debug for NamespaceCredentialsProvider<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NamespaceCredentialsProvider")
            .field("accessor", &self.accessor)
            .field("credential_type", &std::any::type_name::<T>())
            .finish()
    }
}

impl<T> NamespaceCredentialsProvider<T> {
    pub fn new(accessor: Arc<StorageOptionsAccessor>) -> Self {
        Self {
            accessor,
            _credential: PhantomData,
        }
    }

    pub fn from_provider(provider: Arc<dyn StorageOptionsProvider>) -> Self {
        Self::new(Arc::new(StorageOptionsAccessor::with_provider(provider)))
    }

    pub fn from_provider_with_initial(
        provider: Arc<dyn StorageOptionsProvider>,
        initial_options: HashMap<String, String>,
    ) -> Self {
        Self::new(Arc::new(StorageOptionsAccessor::with_initial_and_provider(
            initial_options,
            provider,
        )))
    }
}

pub async fn supports_dynamic_credentials<T>(accessor: &Arc<StorageOptionsAccessor>) -> Result<bool>
where
    T: TryFrom<DynamicCredentials, Error = Error>,
{
    if let Some(initial) = accessor.initial_storage_options() {
        return Ok(T::try_from(DynamicCredentials(initial.clone())).is_ok());
    }

    // If there is no initial snapshot, trust that the accessor's provider is
    // being used for credential vending for this store type. This avoids an
    // eager fetch during store construction, but means callers should only use
    // this helper when the provider is expected to vend credentials compatible
    // with `T`.
    Ok(accessor.has_provider())
}

#[async_trait]
impl<T> CredentialProvider for NamespaceCredentialsProvider<T>
where
    T: TryFrom<DynamicCredentials, Error = Error> + fmt::Debug + Send + Sync + 'static,
{
    type Credential = T;

    async fn get_credential(&self) -> ObjectStoreResult<Arc<Self::Credential>> {
        let storage_options = self.accessor.get_storage_options().await.map_err(|e| {
            object_store::Error::Generic {
                store: "NamespaceCredentialsProvider",
                source: Box::new(e),
            }
        })?;

        let credential = T::try_from(DynamicCredentials(storage_options.0)).map_err(|e| {
            object_store::Error::Generic {
                store: "NamespaceCredentialsProvider",
                source: Box::new(e),
            }
        })?;

        Ok(Arc::new(credential))
    }
}

fn missing_dynamic_credential(kind: &str) -> Error {
    Error::invalid_input(format!(
        "Missing required {kind} credential fields in dynamic storage options"
    ))
}

#[cfg(feature = "azure")]
fn split_azure_sas(sas: &str) -> Result<Vec<(String, String)>> {
    let pairs = url::form_urlencoded::parse(sas.trim_start_matches('?').as_bytes())
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect::<Vec<_>>();

    if pairs.is_empty() {
        return Err(Error::invalid_input(
            "Azure SAS token is empty or invalid in dynamic storage options",
        ));
    }

    Ok(pairs)
}

#[cfg(feature = "aws")]
impl TryFrom<DynamicCredentials> for ObjectStoreAwsCredential {
    type Error = Error;

    fn try_from(credentials: DynamicCredentials) -> Result<Self> {
        let key_id = credentials.0.get("aws_access_key_id").cloned();
        let secret_key = credentials.0.get("aws_secret_access_key").cloned();
        let token = credentials.0.get("aws_session_token").cloned();

        match (key_id, secret_key) {
            (Some(key_id), Some(secret_key)) => Ok(Self {
                key_id,
                secret_key,
                token,
            }),
            _ => Err(missing_dynamic_credential("AWS")),
        }
    }
}

#[cfg(feature = "azure")]
impl TryFrom<DynamicCredentials> for AzureCredential {
    type Error = Error;

    fn try_from(credentials: DynamicCredentials) -> Result<Self> {
        if let Some(sas) = credentials
            .0
            .get("azure_storage_sas_token")
            .or_else(|| credentials.0.get("azure_storage_sas_key"))
        {
            return Ok(Self::SASToken(split_azure_sas(sas)?));
        }

        if let Some(token) = credentials
            .0
            .get("azure_storage_token")
            .or_else(|| credentials.0.get("bearer_token"))
        {
            return Ok(Self::BearerToken(token.clone()));
        }

        if let Some(access_key) = credentials
            .0
            .get("azure_storage_account_key")
            .or_else(|| credentials.0.get("azure_storage_access_key"))
            .or_else(|| credentials.0.get("azure_storage_master_key"))
            .or_else(|| credentials.0.get("access_key"))
            .or_else(|| credentials.0.get("master_key"))
            .or_else(|| credentials.0.get("account_key"))
        {
            return Ok(Self::AccessKey(
                AzureAccessKey::try_new(access_key).map_err(|source| {
                    Error::invalid_input(format!("Invalid Azure access key: {source}"))
                })?,
            ));
        }

        Err(missing_dynamic_credential("Azure"))
    }
}

#[cfg(feature = "gcp")]
impl TryFrom<DynamicCredentials> for GcpCredential {
    type Error = Error;

    fn try_from(credentials: DynamicCredentials) -> Result<Self> {
        let bearer = credentials
            .0
            .get("google_storage_token")
            .cloned()
            .ok_or_else(|| missing_dynamic_credential("GCP"))?;

        Ok(Self { bearer })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use super::*;
    use crate::object_store::test_utils::StaticMockStorageOptionsProvider;

    #[cfg(feature = "aws")]
    #[tokio::test]
    async fn test_dynamic_aws_credentials() {
        let provider = Arc::new(StaticMockStorageOptionsProvider {
            options: HashMap::from([
                ("aws_access_key_id".to_string(), "AKID".to_string()),
                ("aws_secret_access_key".to_string(), "SECRET".to_string()),
                ("aws_session_token".to_string(), "TOKEN".to_string()),
            ]),
        });

        let credentials =
            NamespaceCredentialsProvider::<ObjectStoreAwsCredential>::from_provider(provider)
                .get_credential()
                .await
                .expect("aws credentials should convert");

        assert_eq!(credentials.key_id, "AKID");
        assert_eq!(credentials.secret_key, "SECRET");
        assert_eq!(credentials.token.as_deref(), Some("TOKEN"));
    }

    #[cfg(feature = "azure")]
    #[tokio::test]
    async fn test_dynamic_azure_credentials() {
        let provider = Arc::new(StaticMockStorageOptionsProvider {
            options: HashMap::from([(
                "azure_storage_sas_token".to_string(),
                "?sv=2022-11-02&sp=rl&sig=test".to_string(),
            )]),
        });

        let credentials = NamespaceCredentialsProvider::<AzureCredential>::from_provider(provider)
            .get_credential()
            .await
            .expect("azure credentials should convert");

        match credentials.as_ref() {
            AzureCredential::SASToken(pairs) => {
                assert!(
                    pairs
                        .iter()
                        .any(|(key, value)| key == "sv" && value == "2022-11-02")
                );
                assert!(
                    pairs
                        .iter()
                        .any(|(key, value)| key == "sig" && value == "test")
                );
            }
            other => panic!("expected SAS token, got {other:?}"),
        }
    }

    #[cfg(feature = "gcp")]
    #[tokio::test]
    async fn test_dynamic_gcp_credentials() {
        let provider = Arc::new(StaticMockStorageOptionsProvider {
            options: HashMap::from([("google_storage_token".to_string(), "gcp-token".to_string())]),
        });

        let credentials = NamespaceCredentialsProvider::<GcpCredential>::from_provider(provider)
            .get_credential()
            .await
            .expect("gcp credentials should convert");

        assert_eq!(credentials.bearer, "gcp-token");
    }
}
