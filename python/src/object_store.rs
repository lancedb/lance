// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use object_store::aws::AwsCredential;
use object_store::CredentialProvider;
use reqwest::Client;
use serde::Deserialize;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::Mutex;

pub struct UrlBasedCredentialProvider {
    url: String,
    client: Client,
    lock: Mutex<()>,
    state: Mutex<CredentialState>,
}

#[derive(Deserialize, Debug)]
struct CredentialState {
    #[serde(rename = "AccessKeyId")]
    access_key_id: Option<String>,

    #[serde(rename = "SecretAccessKey")]
    secret_access_key: Option<String>,

    #[serde(rename = "SessionToken")]
    session_token: Option<String>,

    #[serde(rename = "ExpiredTime")]
    expired_time: Option<SystemTime>,
}

impl UrlBasedCredentialProvider {
    pub fn new(url: String) -> Self {
        Self {
            url,
            client: Client::new(),
            lock: Mutex::new(()),
            state: Mutex::new(CredentialState {
                access_key_id: None,
                secret_access_key: None,
                session_token: None,
                expired_time: None,
            }),
        }
    }

    async fn try_get_credentials(&self) -> Option<AwsCredential> {
        let state = self.state.lock().await;
        if let Some(expiration) = state.expired_time {
            if SystemTime::now() < expiration - Duration::from_secs(600) {
                return Some(AwsCredential {
                    key_id: state.access_key_id.clone().unwrap_or_default(),
                    secret_key: state.secret_access_key.clone().unwrap_or_default(),
                    token: state.session_token.clone(),
                });
            }
        }
        None
    }
}

impl Debug for UrlBasedCredentialProvider {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "UrlBasedCredentialProvider {{ url: {} }}", self.url)
    }
}

#[async_trait]
impl CredentialProvider for UrlBasedCredentialProvider {
    type Credential = AwsCredential;

    async fn get_credential(&self) -> object_store::Result<Arc<Self::Credential>> {
        if let Some(credentials) = self.try_get_credentials().await {
            return Ok(Arc::from(credentials));
        }

        let _guard = self.lock.lock().await;
        if let Some(credentials) = self.try_get_credentials().await {
            return Ok(Arc::from(credentials));
        }

        let Ok(response) = self.client.get(&self.url).send().await else {
            return Err(object_store::Error::Generic {
                store: "Request credential error.",
                source: Box::from(""),
            });
        };

        let credential_state: CredentialState = match response.json().await {
            Ok(state) => state,
            Err(_) => {
                return Err(object_store::Error::Generic {
                    store: "Parse response JSON error.",
                    source: Box::from(""),
                })
            }
        };

        let expiration_time: DateTime<Utc> = match credential_state.expired_time {
            Some(exp_time) => exp_time.into(),
            None => {
                return Err(object_store::Error::Generic {
                    store: "Parse expire time error.",
                    source: Box::from(""),
                })
            }
        };

        let mut state = self.state.lock().await;
        state.expired_time = Some(expiration_time.into());
        state.access_key_id = credential_state.access_key_id.clone();
        state.secret_access_key = credential_state.secret_access_key.clone();
        state.session_token = credential_state.session_token.clone();

        Ok(Arc::from(AwsCredential {
            key_id: state.access_key_id.clone().unwrap_or_default(),
            secret_key: state.secret_access_key.clone().unwrap_or_default(),
            token: state.session_token.clone(),
        }))
    }
}
