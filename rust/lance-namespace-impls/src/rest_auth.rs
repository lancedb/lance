// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Authentication providers for REST Namespace HTTP requests.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use lance_core::Result;
use lance_namespace::error::NamespaceError;

#[cfg(feature = "rest-auth-sigv4")]
pub mod sigv4;

pub const AUTH_TYPE_KEY: &str = "rest.auth.type";
pub const AUTH_PROPERTY_PREFIX: &str = "rest.auth.";
pub const AUTH_TYPE_NONE: &str = "none";
#[cfg(feature = "rest-auth-sigv4")]
pub const AUTH_TYPE_SIGV4: &str = "sigv4";

#[derive(Debug, Clone)]
pub struct RequestContext {
    pub method: String,
    pub url: String,
    pub headers: HashMap<String, String>,
    /// `None` for streaming bodies.
    pub body_sha256: Option<String>,
}

#[async_trait]
pub trait RestAuthProvider: Send + Sync + std::fmt::Debug {
    async fn authenticate(&self, ctx: &RequestContext) -> Result<HashMap<String, String>>;

    /// Connect-time init; default no-op.
    async fn initialize(&self) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct NoopAuthProvider;

#[async_trait]
impl RestAuthProvider for NoopAuthProvider {
    async fn authenticate(&self, _ctx: &RequestContext) -> Result<HashMap<String, String>> {
        Ok(HashMap::new())
    }
}

pub fn create_auth_provider(
    properties: &HashMap<String, String>,
) -> Result<Arc<dyn RestAuthProvider>> {
    let auth_type = properties
        .get(AUTH_TYPE_KEY)
        .map(|s| s.as_str())
        .unwrap_or(AUTH_TYPE_NONE);
    match auth_type {
        AUTH_TYPE_NONE => Ok(Arc::new(NoopAuthProvider)),
        #[cfg(feature = "rest-auth-sigv4")]
        AUTH_TYPE_SIGV4 => Ok(Arc::new(sigv4::SigV4AuthProvider::from_properties(
            properties,
        )?)),
        other => Err(NamespaceError::InvalidInput {
            message: format!(
                "unsupported {AUTH_TYPE_KEY} '{other}' (supported: {})",
                supported_auth_types()
            ),
        }
        .into()),
    }
}

fn supported_auth_types() -> &'static str {
    #[cfg(feature = "rest-auth-sigv4")]
    {
        "none, sigv4"
    }
    #[cfg(not(feature = "rest-auth-sigv4"))]
    {
        "none"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_ctx() -> RequestContext {
        RequestContext {
            method: "GET".to_string(),
            url: "http://example.com/v1/test".to_string(),
            headers: HashMap::new(),
            body_sha256: None,
        }
    }

    #[tokio::test]
    async fn noop_returns_empty_headers() {
        assert!(
            NoopAuthProvider
                .authenticate(&empty_ctx())
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn noop_initialize_is_ok() {
        NoopAuthProvider.initialize().await.unwrap();
    }

    #[test]
    fn factory_accepts_missing_auth_type() {
        assert!(create_auth_provider(&HashMap::new()).is_ok());
    }

    #[test]
    fn factory_accepts_explicit_none() {
        let mut props = HashMap::new();
        props.insert(AUTH_TYPE_KEY.to_string(), AUTH_TYPE_NONE.to_string());
        assert!(create_auth_provider(&props).is_ok());
    }

    #[test]
    fn factory_rejects_unknown_with_helpful_error() {
        let mut props = HashMap::new();
        props.insert(AUTH_TYPE_KEY.to_string(), "sigv4-typo".to_string());
        let msg = create_auth_provider(&props).unwrap_err().to_string();
        assert!(msg.contains("sigv4-typo"));
        assert!(msg.contains(AUTH_TYPE_NONE));
    }
}
