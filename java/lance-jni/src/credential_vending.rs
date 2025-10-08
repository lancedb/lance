// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use jni::objects::{JMap, JObject, JString, JValue};
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
    async fn get_credentials(
        &self,
    ) -> lance_core::Result<(HashMap<String, String>, u64)> {
        // Spawn blocking task to call Java method
        let java_vendor = self.java_vendor.clone();
        let jvm = self.jvm.clone();

        tokio::task::spawn_blocking(move || {
            // Attach current thread to JVM
            let mut env = jvm.attach_current_thread().map_err(|e| {
                lance_core::Error::IO {
                    source: Box::new(std::io::Error::other(format!(
                        "Failed to attach to JVM: {}",
                        e
                    ))),
                    location: snafu::location!(),
                }
            })?;

            // Call getCredentials() method on Java object
            // Returns Map<String, Object> with "storage_options" (Map<String, String>) and "expires_at_millis" (Long)
            let result = env
                .call_method(
                    &java_vendor,
                    "getCredentials",
                    "()Ljava/util/Map;",
                    &[],
                )
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

            // Extract storage_options from the map
            let storage_options_key = env
                .new_string("storage_options")
                .map_err(|e| lance_core::Error::IO {
                    source: Box::new(std::io::Error::other(format!(
                        "Failed to create 'storage_options' string: {}",
                        e
                    ))),
                    location: snafu::location!(),
                })?;

            let storage_options_obj = env
                .call_method(
                    &result_map,
                    "get",
                    "(Ljava/lang/Object;)Ljava/lang/Object;",
                    &[JValue::Object(&storage_options_key)],
                )
                .map_err(|e| lance_core::Error::IO {
                    source: Box::new(std::io::Error::other(format!(
                        "Failed to get storage_options from result: {}",
                        e
                    ))),
                    location: snafu::location!(),
                })?
                .l()
                .map_err(|e| lance_core::Error::IO {
                    source: Box::new(std::io::Error::other(format!(
                        "storage_options is not an object: {}",
                        e
                    ))),
                    location: snafu::location!(),
                })?;

            // Convert Java Map to Rust HashMap
            let storage_options_map = JMap::from_env(&mut env, &storage_options_obj)
                .map_err(|e| lance_core::Error::IO {
                    source: Box::new(std::io::Error::other(format!(
                        "storage_options is not a Map: {}",
                        e
                    ))),
                    location: snafu::location!(),
                })?;

            let mut storage_options = HashMap::new();
            let mut iter = storage_options_map.iter(&mut env).map_err(|e| {
                lance_core::Error::IO {
                    source: Box::new(std::io::Error::other(format!(
                        "Failed to iterate storage_options: {}",
                        e
                    ))),
                    location: snafu::location!(),
                }
            })?;

            while let Some((key, value)) = iter.next(&mut env).map_err(|e| {
                lance_core::Error::IO {
                    source: Box::new(std::io::Error::other(format!(
                        "Failed to get next entry: {}",
                        e
                    ))),
                    location: snafu::location!(),
                }
            })? {
                let key_str: String = env
                    .get_string(&JString::from(key))
                    .map_err(|e| lance_core::Error::IO {
                        source: Box::new(std::io::Error::other(format!(
                            "storage_options key is not a string: {}",
                            e
                        ))),
                        location: snafu::location!(),
                    })?
                    .into();

                let value_str: String = env
                    .get_string(&JString::from(value))
                    .map_err(|e| lance_core::Error::IO {
                        source: Box::new(std::io::Error::other(format!(
                            "storage_options value is not a string: {}",
                            e
                        ))),
                        location: snafu::location!(),
                    })?
                    .into();

                storage_options.insert(key_str, value_str);
            }

            // Extract expires_at_millis from the map
            let expires_key = env
                .new_string("expires_at_millis")
                .map_err(|e| lance_core::Error::IO {
                    source: Box::new(std::io::Error::other(format!(
                        "Failed to create 'expires_at_millis' string: {}",
                        e
                    ))),
                    location: snafu::location!(),
                })?;

            let expires_obj = env
                .call_method(
                    &result_map,
                    "get",
                    "(Ljava/lang/Object;)Ljava/lang/Object;",
                    &[JValue::Object(&expires_key)],
                )
                .map_err(|e| lance_core::Error::IO {
                    source: Box::new(std::io::Error::other(format!(
                        "Failed to get expires_at_millis from result: {}",
                        e
                    ))),
                    location: snafu::location!(),
                })?
                .l()
                .map_err(|e| lance_core::Error::IO {
                    source: Box::new(std::io::Error::other(format!(
                        "expires_at_millis is not an object: {}",
                        e
                    ))),
                    location: snafu::location!(),
                })?;

            // Call longValue() to get the primitive long
            let expires_at_millis = env
                .call_method(&expires_obj, "longValue", "()J", &[])
                .map_err(|e| lance_core::Error::IO {
                    source: Box::new(std::io::Error::other(format!(
                        "Failed to call longValue on expires_at_millis: {}",
                        e
                    ))),
                    location: snafu::location!(),
                })?
                .j()
                .map_err(|e| lance_core::Error::IO {
                    source: Box::new(std::io::Error::other(format!(
                        "expires_at_millis longValue is not a long: {}",
                        e
                    ))),
                    location: snafu::location!(),
                })? as u64;

            Ok((storage_options, expires_at_millis))
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
