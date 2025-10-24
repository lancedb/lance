// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use jni::objects::{JMap, JObject, JString};
use jni::JNIEnv;
use lance_io::object_store::StorageOptionsProvider;

use crate::error::Result;

/// Java-implemented storage options provider
///
/// This wraps a Java object that implements the StorageOptionsProvider interface
/// and forwards get_storage_options() calls to the Java implementation.
#[derive(Debug)]
pub struct JavaStorageOptionsProvider {
    /// GlobalRef to the Java StorageOptionsProvider object
    java_provider: jni::objects::GlobalRef,
    /// JavaVM for making JNI calls
    jvm: Arc<jni::JavaVM>,
}

impl JavaStorageOptionsProvider {
    pub fn new(env: &mut JNIEnv, java_provider: JObject) -> Result<Self> {
        // Create a global reference to the Java object so it persists
        let java_provider = env.new_global_ref(java_provider)?;

        // Get the JavaVM for later JNI calls
        let jvm = Arc::new(env.get_java_vm()?);

        Ok(Self { java_provider, jvm })
    }
}

#[async_trait]
impl StorageOptionsProvider for JavaStorageOptionsProvider {
    async fn fetch_storage_options(&self) -> lance_core::Result<Option<HashMap<String, String>>> {
        // Spawn blocking task to call Java method
        let java_provider = self.java_provider.clone();
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

            // Call fetchStorageOptions() method on Java object
            // Returns Map<String, String> with all storage options including optional EXPIRES_AT_MILLIS_KEY
            // Or null if no storage options are available
            let result = env
                .call_method(
                    &java_provider,
                    "fetchStorageOptions",
                    "()Ljava/util/Map;",
                    &[],
                )
                .map_err(|e| lance_core::Error::IO {
                    source: Box::new(std::io::Error::other(format!(
                        "Failed to call fetchStorageOptions: {}",
                        e
                    ))),
                    location: snafu::location!(),
                })?;

            let result_obj = result.l().map_err(|e| lance_core::Error::IO {
                source: Box::new(std::io::Error::other(format!(
                    "fetchStorageOptions result is not an object: {}",
                    e
                ))),
                location: snafu::location!(),
            })?;

            // Check if result is null
            if result_obj.is_null() {
                return Ok(None);
            }

            // Convert Java Map to Rust HashMap
            let storage_options_map =
                JMap::from_env(&mut env, &result_obj).map_err(|e| lance_core::Error::IO {
                    source: Box::new(std::io::Error::other(format!(
                        "fetchStorageOptions result is not a Map: {}",
                        e
                    ))),
                    location: snafu::location!(),
                })?;

            let mut storage_options = HashMap::new();
            let mut iter =
                storage_options_map
                    .iter(&mut env)
                    .map_err(|e| lance_core::Error::IO {
                        source: Box::new(std::io::Error::other(format!(
                            "Failed to iterate storage options: {}",
                            e
                        ))),
                        location: snafu::location!(),
                    })?;

            while let Some((key, value)) =
                iter.next(&mut env).map_err(|e| lance_core::Error::IO {
                    source: Box::new(std::io::Error::other(format!(
                        "Failed to get next storage option entry: {}",
                        e
                    ))),
                    location: snafu::location!(),
                })?
            {
                let key_str: String = env
                    .get_string(&JString::from(key))
                    .map_err(|e| lance_core::Error::IO {
                        source: Box::new(std::io::Error::other(format!(
                            "storage option key is not a string: {}",
                            e
                        ))),
                        location: snafu::location!(),
                    })?
                    .into();

                let value_str: String = env
                    .get_string(&JString::from(value))
                    .map_err(|e| lance_core::Error::IO {
                        source: Box::new(std::io::Error::other(format!(
                            "storage option value is not a string: {}",
                            e
                        ))),
                        location: snafu::location!(),
                    })?
                    .into();

                storage_options.insert(key_str, value_str);
            }

            Ok(Some(storage_options))
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
