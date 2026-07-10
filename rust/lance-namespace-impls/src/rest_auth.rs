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

/// Property key selecting the authentication provider (see [`create_auth_provider`]).
pub const AUTH_TYPE_KEY: &str = "rest.auth.type";
/// Common prefix for all `rest.auth.*` configuration properties.
pub const AUTH_PROPERTY_PREFIX: &str = "rest.auth.";
/// [`AUTH_TYPE_KEY`] value selecting [`NoopAuthProvider`] (the default when unset).
pub const AUTH_TYPE_NONE: &str = "none";
/// [`AUTH_TYPE_KEY`] value selecting [`sigv4::SigV4AuthProvider`].
#[cfg(feature = "rest-auth-sigv4")]
pub const AUTH_TYPE_SIGV4: &str = "sigv4";

/// The per-request information a [`RestAuthProvider`] needs to authenticate an
/// outgoing REST Namespace HTTP request.
///
/// A provider inspects these fields (e.g. to compute a signature) and returns
/// the headers to add to the request.
///
/// # Examples
///
/// ```
/// use lance_namespace_impls::RequestContext;
/// use std::collections::HashMap;
///
/// let ctx = RequestContext {
///     method: "GET".to_string(),
///     url: "https://example.com/v1/tables".to_string(),
///     headers: HashMap::new(),
///     body_sha256: None,
/// };
/// assert_eq!(ctx.method, "GET");
/// ```
#[derive(Debug, Clone)]
pub struct RequestContext {
    /// The HTTP method (e.g. `"GET"`, `"POST"`).
    pub method: String,
    /// The fully-qualified request URL, including scheme, host, and path.
    pub url: String,
    /// Headers already set on the request, keyed by header name.
    pub headers: HashMap<String, String>,
    /// The hex-encoded SHA-256 digest of the request body.
    ///
    /// `Some` carries the lowercase hex digest of the full body, used by
    /// signing schemes that hash the payload (e.g. AWS SigV4). `None` means the
    /// body is streaming or otherwise unavailable for hashing, and providers
    /// that require it should fall back to their unsigned-payload behavior.
    pub body_sha256: Option<String>,
}

/// A pluggable strategy for authenticating REST Namespace HTTP requests.
///
/// Implementors receive a [`RequestContext`] per request and return the headers
/// to add before the request is sent. Selection is driven by the
/// [`AUTH_TYPE_KEY`] property via [`create_auth_provider`].
#[async_trait]
pub trait RestAuthProvider: Send + Sync + std::fmt::Debug {
    /// Authenticates a single request, returning the headers to add to it.
    ///
    /// The returned map is merged into the outgoing request's headers. An empty
    /// map means no authentication headers are required.
    async fn authenticate(&self, ctx: &RequestContext) -> Result<HashMap<String, String>>;

    /// Connect-time initialization hook for fail-fast credential validation.
    ///
    /// Called once before the namespace is used (see
    /// [`RestNamespace::warm_up_auth`](crate::RestNamespace::warm_up_auth)).
    /// Defaults to a no-op.
    async fn initialize(&self) -> Result<()> {
        Ok(())
    }
}

/// A [`RestAuthProvider`] that adds no authentication headers.
///
/// Used when `rest.auth.type` is unset or `none`.
#[derive(Debug, Default)]
pub struct NoopAuthProvider;

#[async_trait]
impl RestAuthProvider for NoopAuthProvider {
    async fn authenticate(&self, _ctx: &RequestContext) -> Result<HashMap<String, String>> {
        Ok(HashMap::new())
    }
}

/// Builds a [`RestAuthProvider`] from `rest.auth.*` properties, selected by
/// [`AUTH_TYPE_KEY`]. Defaults to [`NoopAuthProvider`] when the key is unset.
///
/// # Errors
///
/// Returns [`NamespaceError::InvalidInput`] if [`AUTH_TYPE_KEY`] names an
/// unsupported provider, or if the selected provider's own validation fails
/// (e.g. missing required SigV4 properties).
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
        let err = create_auth_provider(&props).unwrap_err();

        let message = match &err {
            lance_core::Error::Namespace { source, .. } => {
                match source.downcast_ref::<NamespaceError>() {
                    Some(NamespaceError::InvalidInput { message }) => message.clone(),
                    other => panic!("expected NamespaceError::InvalidInput, got {other:?}"),
                }
            }
            other => panic!("expected lance_core::Error::Namespace, got {other:?}"),
        };
        assert!(message.contains("sigv4-typo"));
        assert!(message.contains(AUTH_TYPE_NONE));
    }
}
