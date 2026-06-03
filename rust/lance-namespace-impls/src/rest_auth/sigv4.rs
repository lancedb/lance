// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! AWS SigV4 authentication provider for REST Namespace.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;

use async_trait::async_trait;
use aws_credential_types::Credentials;
use aws_credential_types::provider::{ProvideCredentials, SharedCredentialsProvider};
use aws_sigv4::http_request::{
    SignableBody, SignableRequest, SigningParams, SigningSettings, sign,
};
use aws_sigv4::sign::v4;
use lance_core::Result;
use lance_namespace::error::NamespaceError;
use tokio::sync::OnceCell;
use url::Url;

pub const REGION_KEY: &str = "rest.auth.sigv4.region";
pub const SERVICE_KEY: &str = "rest.auth.sigv4.service";
const DEFAULT_SERVICE: &str = "execute-api";

/// Injectable time source; tests use a fixed clock.
pub trait Clock: Send + Sync + std::fmt::Debug {
    fn now(&self) -> SystemTime;
}

#[derive(Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> SystemTime {
        SystemTime::now()
    }
}

pub struct SigV4AuthProvider {
    region: String,
    service: String,
    credentials_provider: OnceCell<SharedCredentialsProvider>,
    clock: Arc<dyn Clock>,
}

impl std::fmt::Debug for SigV4AuthProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SigV4AuthProvider")
            .field("region", &self.region)
            .field("service", &self.service)
            .field(
                "credentials_provider",
                &self.credentials_provider.get().map(|_| "resolved"),
            )
            .finish()
    }
}

impl SigV4AuthProvider {
    pub fn from_properties(properties: &HashMap<String, String>) -> Result<Self> {
        let region =
            properties
                .get(REGION_KEY)
                .cloned()
                .ok_or_else(|| NamespaceError::InvalidInput {
                    message: format!("{REGION_KEY} is required for SigV4 authentication"),
                })?;
        let service = properties
            .get(SERVICE_KEY)
            .cloned()
            .unwrap_or_else(|| DEFAULT_SERVICE.to_string());
        Ok(Self {
            region,
            service,
            credentials_provider: OnceCell::new(),
            clock: Arc::new(SystemClock),
        })
    }

    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }

    pub fn with_credentials_provider(self, provider: SharedCredentialsProvider) -> Self {
        let cell = OnceCell::new();
        cell.set(provider)
            .expect("freshly constructed OnceCell never returns Err");
        Self {
            credentials_provider: cell,
            ..self
        }
    }

    async fn ensure_credentials_provider(&self) -> Result<&SharedCredentialsProvider> {
        self.credentials_provider
            .get_or_try_init(|| async {
                // aws_config::load panics inside an existing tokio runtime.
                let region = self.region.clone();
                let provider = tokio::task::spawn_blocking(move || {
                    let rt = tokio::runtime::Handle::current();
                    rt.block_on(async {
                        aws_config::defaults(aws_config::BehaviorVersion::latest())
                            .region(aws_config::Region::new(region))
                            .load()
                            .await
                    })
                })
                .await
                .map_err(|e| {
                    lance_core::Error::from(NamespaceError::Internal {
                        message: format!("failed to load AWS config: {e}"),
                    })
                })?;
                provider.credentials_provider().ok_or_else(|| {
                    lance_core::Error::from(NamespaceError::Internal {
                        message: "AWS config did not yield a credentials provider".to_string(),
                    })
                })
            })
            .await
    }

    async fn resolve_credentials(&self) -> Result<Credentials> {
        let provider = self.ensure_credentials_provider().await?;
        provider.provide_credentials().await.map_err(|e| {
            NamespaceError::Unauthenticated {
                message: format!("failed to resolve AWS credentials: {e}"),
            }
            .into()
        })
    }
}

#[async_trait]
impl super::RestAuthProvider for SigV4AuthProvider {
    async fn authenticate(&self, ctx: &super::RequestContext) -> Result<HashMap<String, String>> {
        let creds = self.resolve_credentials().await?;
        let identity = creds.into();

        let mut signing_settings = SigningSettings::default();
        signing_settings.payload_checksum_kind =
            aws_sigv4::http_request::PayloadChecksumKind::XAmzSha256;
        let v4_params = v4::SigningParams::builder()
            .identity(&identity)
            .region(&self.region)
            .name(&self.service)
            .time(self.clock.now())
            .settings(signing_settings)
            .build()
            .map_err(|e| NamespaceError::Internal {
                message: format!("failed to build SigV4 signing params: {e}"),
            })?;
        let params: SigningParams = v4_params.into();

        let parsed_url = Url::parse(&ctx.url).map_err(|_| NamespaceError::InvalidInput {
            message: format!("SigV4 requires a valid URL: {}", ctx.url),
        })?;
        if parsed_url.host_str().is_none() {
            return Err(NamespaceError::InvalidInput {
                message: format!("SigV4 requires a URL with a host: {}", ctx.url),
            }
            .into());
        }
        let host = parsed_url[url::Position::BeforeHost..url::Position::AfterPort].to_string();

        let other_headers = ctx
            .headers
            .iter()
            .filter(|(k, _)| !k.eq_ignore_ascii_case("host"));
        let header_iter = std::iter::once(("host", host.as_str()))
            .chain(other_headers.map(|(k, v)| (k.as_str(), v.as_str())));

        let body = match ctx.body_sha256.as_deref() {
            Some(hash) => SignableBody::Precomputed(hash.to_string()),
            None => SignableBody::UnsignedPayload,
        };

        let signable =
            SignableRequest::new(&ctx.method, &ctx.url, header_iter, body).map_err(|e| {
                NamespaceError::Internal {
                    message: format!("failed to construct SigV4 signable request: {e}"),
                }
            })?;

        let (instructions, _signature) = sign(signable, &params)
            .map_err(|e| NamespaceError::Internal {
                message: format!("SigV4 signing failed: {e}"),
            })?
            .into_parts();

        Ok(instructions
            .headers()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect())
    }

    async fn initialize(&self) -> Result<()> {
        self.resolve_credentials().await.map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rest_auth::{RequestContext, RestAuthProvider};
    use std::time::{Duration, UNIX_EPOCH};

    // AWS SigV4 test vector credentials (botocore cross-verified).
    const VECTOR_ACCESS_KEY: &str = "AKIDEXAMPLE";
    const VECTOR_SECRET_KEY: &str = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";
    const VECTOR_REGION: &str = "us-east-1";
    const VECTOR_SERVICE: &str = "service";
    const VECTOR_UNIX_SECS: u64 = 1_440_938_160; // 2015-08-30T12:36:00Z
    const VECTOR_EXPECTED_AUTHORIZATION: &str = "AWS4-HMAC-SHA256 \
        Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, \
        SignedHeaders=host;x-amz-content-sha256;x-amz-date, \
        Signature=726c5c4879a6b4ccbbd3b24edbd6b8826d34f87450fbbf4e85546fc7ba9c1642";

    #[derive(Debug)]
    struct FixedClock(SystemTime);

    impl Clock for FixedClock {
        fn now(&self) -> SystemTime {
            self.0
        }
    }

    fn vector_provider() -> SigV4AuthProvider {
        let creds = Credentials::new(
            VECTOR_ACCESS_KEY,
            VECTOR_SECRET_KEY,
            None,
            None,
            "lance-sigv4-test",
        );
        let mut props = HashMap::new();
        props.insert(REGION_KEY.to_string(), VECTOR_REGION.to_string());
        props.insert(SERVICE_KEY.to_string(), VECTOR_SERVICE.to_string());
        SigV4AuthProvider::from_properties(&props)
            .unwrap()
            .with_clock(Arc::new(FixedClock(
                UNIX_EPOCH + Duration::from_secs(VECTOR_UNIX_SECS),
            )))
            .with_credentials_provider(SharedCredentialsProvider::new(creds))
    }

    #[test]
    fn from_properties_requires_region() {
        let err = SigV4AuthProvider::from_properties(&HashMap::new()).unwrap_err();
        assert!(err.to_string().contains(REGION_KEY));
    }

    #[test]
    fn from_properties_defaults_service_to_execute_api() {
        let mut props = HashMap::new();
        props.insert(REGION_KEY.to_string(), "us-west-2".to_string());
        let provider = SigV4AuthProvider::from_properties(&props).unwrap();
        assert_eq!(provider.service, DEFAULT_SERVICE);
        assert_eq!(provider.region, "us-west-2");
    }

    #[test]
    fn from_properties_accepts_explicit_service() {
        let mut props = HashMap::new();
        props.insert(REGION_KEY.to_string(), "us-east-1".to_string());
        props.insert(SERVICE_KEY.to_string(), "s3".to_string());
        let provider = SigV4AuthProvider::from_properties(&props).unwrap();
        assert_eq!(provider.service, "s3");
    }

    #[tokio::test]
    async fn reproduces_aws_get_vanilla_reference_vector() {
        let provider = vector_provider();
        let ctx = RequestContext {
            method: "GET".to_string(),
            url: "https://example.amazonaws.com/".to_string(),
            headers: HashMap::new(),
            body_sha256: Some(crate::rest::EMPTY_BODY_SHA256.to_string()),
        };
        let headers = provider.authenticate(&ctx).await.unwrap();
        let actual = headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("authorization"))
            .map(|(_, v)| v.as_str())
            .expect("authorization header must be produced");
        assert_eq!(actual, VECTOR_EXPECTED_AUTHORIZATION);
    }

    #[tokio::test]
    async fn initialize_resolves_injected_credentials() {
        vector_provider().initialize().await.unwrap();
    }

    #[tokio::test]
    async fn authenticate_rejects_url_without_host() {
        let provider = vector_provider();
        let ctx = RequestContext {
            method: "GET".to_string(),
            url: "file:///nowhere".to_string(),
            headers: HashMap::new(),
            body_sha256: Some(crate::rest::EMPTY_BODY_SHA256.to_string()),
        };
        let err = provider.authenticate(&ctx).await.unwrap_err();
        assert!(err.to_string().contains("host"));
    }

    #[tokio::test]
    async fn authenticate_overrides_preexisting_host_header() {
        let provider = vector_provider();
        let mut headers = HashMap::new();
        headers.insert("Host".to_string(), "wrong.example.com".to_string());
        let ctx = RequestContext {
            method: "GET".to_string(),
            url: "https://example.amazonaws.com/".to_string(),
            headers,
            body_sha256: Some(crate::rest::EMPTY_BODY_SHA256.to_string()),
        };
        let result = provider.authenticate(&ctx).await.unwrap();
        let actual = result
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("authorization"))
            .map(|(_, v)| v.as_str())
            .expect("authorization header must be produced");
        assert_eq!(
            actual, VECTOR_EXPECTED_AUTHORIZATION,
            "pre-existing Host header must be replaced by the URL-derived host"
        );
    }

    /// AWS test vector: percent-encoded path (%3D → double-encoded %253D).
    #[tokio::test]
    async fn reproduces_aws_double_encode_path_vector() {
        let creds = Credentials::new(
            "ANOTREAL",
            "notrealrnrELgWzOk3IfjzDKtFBhDby",
            None,
            None,
            "lance-sigv4-test",
        );
        let mut props = HashMap::new();
        props.insert(REGION_KEY.to_string(), "us-east-1".to_string());
        props.insert(SERVICE_KEY.to_string(), "service".to_string());
        let provider = SigV4AuthProvider::from_properties(&props)
            .unwrap()
            .with_clock(Arc::new(FixedClock(
                UNIX_EPOCH + Duration::from_secs(VECTOR_UNIX_SECS),
            )))
            .with_credentials_provider(SharedCredentialsProvider::new(creds));

        let ctx = RequestContext {
            method: "POST".to_string(),
            url: "https://tj9n5r0m12.execute-api.us-east-1.amazonaws.com/test/@connections/JBDvjfGEIAMCERw%3D".to_string(),
            headers: HashMap::new(),
            body_sha256: Some(crate::rest::EMPTY_BODY_SHA256.to_string()),
        };
        let headers = provider.authenticate(&ctx).await.unwrap();
        let auth = headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("authorization"))
            .map(|(_, v)| v.as_str())
            .expect("authorization header must be produced");
        assert_eq!(
            auth,
            "AWS4-HMAC-SHA256 Credential=ANOTREAL/20150830/us-east-1/service/aws4_request, \
             SignedHeaders=host;x-amz-content-sha256;x-amz-date, \
             Signature=ed434df8a348089a1188defcfcc1aa24049990a7e82021d0418cfa0eb05e4d99",
            "double-encode-path: signature must match botocore cross-verification"
        );
    }

    #[tokio::test]
    async fn authenticate_with_unsigned_payload_still_produces_signature() {
        let provider = vector_provider();
        let ctx = RequestContext {
            method: "GET".to_string(),
            url: "https://example.amazonaws.com/".to_string(),
            headers: HashMap::new(),
            body_sha256: None,
        };
        let headers = provider.authenticate(&ctx).await.unwrap();
        let auth = headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("authorization"))
            .map(|(_, v)| v.as_str())
            .expect("authorization header must be produced");
        assert!(auth.starts_with("AWS4-HMAC-SHA256 "));
        assert!(auth.contains("Credential="));
        assert!(auth.contains("SignedHeaders="));
        assert!(auth.contains("Signature="));
    }

    #[tokio::test]
    async fn authenticate_with_session_token_produces_correct_signature() {
        let creds = Credentials::new(
            VECTOR_ACCESS_KEY,
            VECTOR_SECRET_KEY,
            Some("FakeSessionToken123".to_string()),
            None,
            "lance-sigv4-test",
        );
        let mut props = HashMap::new();
        props.insert(REGION_KEY.to_string(), VECTOR_REGION.to_string());
        props.insert(SERVICE_KEY.to_string(), VECTOR_SERVICE.to_string());
        let provider = SigV4AuthProvider::from_properties(&props)
            .unwrap()
            .with_clock(Arc::new(FixedClock(
                UNIX_EPOCH + Duration::from_secs(VECTOR_UNIX_SECS),
            )))
            .with_credentials_provider(SharedCredentialsProvider::new(creds));

        let ctx = RequestContext {
            method: "GET".to_string(),
            url: "https://example.amazonaws.com/".to_string(),
            headers: HashMap::new(),
            body_sha256: Some(crate::rest::EMPTY_BODY_SHA256.to_string()),
        };
        let headers = provider.authenticate(&ctx).await.unwrap();

        let token_header = headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("x-amz-security-token"))
            .map(|(_, v)| v.as_str());
        assert_eq!(
            token_header,
            Some("FakeSessionToken123"),
            "session token must be included in output headers"
        );

        let auth = headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("authorization"))
            .map(|(_, v)| v.as_str())
            .unwrap();
        assert!(
            auth.contains("x-amz-security-token"),
            "session token must be in SignedHeaders: {}",
            auth
        );
        assert_eq!(
            auth,
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, \
             SignedHeaders=host;x-amz-content-sha256;x-amz-date;x-amz-security-token, \
             Signature=d690ca83bd782879e22797e35b2e25958c0d19696a92cfb479b73428e4d950f4",
            "session token signature must match botocore cross-verification"
        );
    }
}
