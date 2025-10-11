// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use jni::objects::{JMap, JObject, JString};
use jni::JNIEnv;
use lance_io::object_store::CredentialVendor;

use crate::error::Result;

/// Java-implemented credential vendor
///
/// This wraps a Java object that implements the CredentialVendor interface
/// and forwards get_credentials() calls to the Java implementation.
pub struct JavaCredentialVendor {
    /// GlobalRef to the Java CredentialVendor object
    java_vendor: jni::objects::GlobalRef,
    /// JavaVM for making JNI calls
    jvm: Arc<jni::JavaVM>,
}

impl JavaCredentialVendor {
    pub fn new(env: &mut JNIEnv, java_vendor: JObject) -> Result<Self> {
        // Create a global reference to the Java object so it persists
        let java_vendor = env.new_global_ref(java_vendor)?;

        // Get the JavaVM for later JNI calls
        let jvm = Arc::new(env.get_java_vm()?);

        Ok(Self { java_vendor, jvm })
    }
}

#[async_trait]
impl CredentialVendor for JavaCredentialVendor {
    async fn get_credentials(&self) -> lance_core::Result<(HashMap<String, String>, u64)> {
        // Spawn blocking task to call Java method
        let java_vendor = self.java_vendor.clone();
        let jvm = self.jvm.clone();

        tokio::task::spawn_blocking(move || {
            // Attach current thread to JVM
            let mut env = jvm
                .attach_current_thread()
                .map_err(|e| lance_core::Error::IO {
                    source: Box::new(std::io::Error::other(format!(
                        "Failed to attach to JVM: {}",
                        e
                    ))),
                    location: snafu::location!(),
                })?;

            // Call getCredentials() method on Java object
            // Returns Map<String, String> with all credentials including "expires_at_millis"
            let result = env
                .call_method(&java_vendor, "getCredentials", "()Ljava/util/Map;", &[])
                .map_err(|e| lance_core::Error::IO {
                    source: Box::new(std::io::Error::other(format!(
                        "Failed to call getCredentials: {}",
                        e
                    ))),
                    location: snafu::location!(),
                })?;

            let result_map = result.l().map_err(|e| lance_core::Error::IO {
                source: Box::new(std::io::Error::other(format!(
                    "getCredentials result is not an object: {}",
                    e
                ))),
                location: snafu::location!(),
            })?;

            // Convert Java Map to Rust HashMap
            let credentials_map =
                JMap::from_env(&mut env, &result_map).map_err(|e| lance_core::Error::IO {
                    source: Box::new(std::io::Error::other(format!(
                        "getCredentials result is not a Map: {}",
                        e
                    ))),
                    location: snafu::location!(),
                })?;

            let mut credentials = HashMap::new();
            let mut iter = credentials_map
                .iter(&mut env)
                .map_err(|e| lance_core::Error::IO {
                    source: Box::new(std::io::Error::other(format!(
                        "Failed to iterate credentials: {}",
                        e
                    ))),
                    location: snafu::location!(),
                })?;

            while let Some((key, value)) =
                iter.next(&mut env).map_err(|e| lance_core::Error::IO {
                    source: Box::new(std::io::Error::other(format!(
                        "Failed to get next credential entry: {}",
                        e
                    ))),
                    location: snafu::location!(),
                })?
            {
                let key_str: String = env
                    .get_string(&JString::from(key))
                    .map_err(|e| lance_core::Error::IO {
                        source: Box::new(std::io::Error::other(format!(
                            "credential key is not a string: {}",
                            e
                        ))),
                        location: snafu::location!(),
                    })?
                    .into();

                let value_str: String = env
                    .get_string(&JString::from(value))
                    .map_err(|e| lance_core::Error::IO {
                        source: Box::new(std::io::Error::other(format!(
                            "credential value is not a string: {}",
                            e
                        ))),
                        location: snafu::location!(),
                    })?
                    .into();

                credentials.insert(key_str, value_str);
            }

            // Extract and parse expires_at_millis
            let expires_at_millis_str = credentials.get("expires_at_millis").ok_or_else(|| {
                lance_core::Error::InvalidInput {
                    source: "getCredentials() result must contain 'expires_at_millis' key".into(),
                    location: snafu::location!(),
                }
            })?;

            let expires_at_millis: u64 =
                expires_at_millis_str
                    .parse()
                    .map_err(|e| lance_core::Error::InvalidInput {
                        source: format!("expires_at_millis must be a valid integer string: {}", e)
                            .into(),
                        location: snafu::location!(),
                    })?;

            // Remove expires_at_millis from credentials map as it's returned separately
            credentials.remove("expires_at_millis");

            Ok((credentials, expires_at_millis))
        })
        .await
        .map_err(|e| lance_core::Error::IO {
            source: Box::new(std::io::Error::other(format!(
                "Failed to spawn blocking task: {}",
                e
            ))),
            location: snafu::location!(),
        })?
    }
}
