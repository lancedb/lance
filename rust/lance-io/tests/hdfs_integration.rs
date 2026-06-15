// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Integration tests for HDFS object store provider
//!
//! These tests need an existing HDFS cluster and are skipped unless
//! `HDFS_NAME_NODE` or `HDFS_TEST_ENABLED` is set.
//!
//! Set `HDFS_NAME_NODE` to the name node or nameservice address
//! (e.g., `hdfs://namenode:9000` or `hdfs://mycluster`). Set
//! `HDFS_TEST_ENABLED` to any value to run the tests when the name node is
//! provided by Hadoop configuration files instead. `HDFS_TEST_ENABLED` only
//! controls whether the tests are skipped; it does not configure HDFS or
//! verify connectivity.
//!
//! Example:
//!   HDFS_NAME_NODE=hdfs://localhost:9000 cargo test --features hdfs hdfs_integration_test
//!   HDFS_TEST_ENABLED=1 cargo test --features hdfs hdfs_integration_test
//!

#[cfg(feature = "hdfs")]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use lance_io::object_store::{
        ObjectStore, ObjectStoreParams, ObjectStoreRegistry, StorageOptionsAccessor,
    };
    use object_store::ObjectStoreExt;
    use object_store::path::Path;

    /// Returns whether the environment explicitly enables the HDFS integration tests.
    fn hdfs_available() -> bool {
        std::env::var("HDFS_NAME_NODE").is_ok() || std::env::var("HDFS_TEST_ENABLED").is_ok()
    }

    #[tokio::test]
    async fn test_hdfs_store_creation() {
        if !hdfs_available() {
            return;
        }

        let registry = Arc::new(ObjectStoreRegistry::default());
        let url = "hdfs://localhost:9000/test/path";
        let params = ObjectStoreParams::default();

        let (store, path) = ObjectStore::from_uri_and_params(registry, url, &params)
            .await
            .unwrap();

        assert_eq!(store.scheme(), "hdfs");
        assert_eq!(path, Path::from("test/path"));
    }

    #[tokio::test]
    async fn test_hdfs_store_with_custom_config() {
        if !hdfs_available() {
            return;
        }

        let registry = Arc::new(ObjectStoreRegistry::default());
        let url = "hdfs://namenode:9000/user/test";

        let mut storage_options = HashMap::new();
        storage_options.insert("hdfs_user".to_string(), "testuser".to_string());
        storage_options.insert(
            "hdfs_name_node".to_string(),
            "hdfs://namenode:9000".to_string(),
        );

        let params = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                storage_options,
            ))),
            ..Default::default()
        };

        let (store, path) = ObjectStore::from_uri_and_params(registry, url, &params)
            .await
            .unwrap();

        assert_eq!(store.scheme(), "hdfs");
        assert_eq!(path, Path::from("user/test"));
    }

    #[tokio::test]
    async fn test_hdfs_basic_operations() {
        if !hdfs_available() {
            return;
        }

        let registry = Arc::new(ObjectStoreRegistry::default());
        let url = "hdfs://localhost:9000/test";
        let params = ObjectStoreParams::default();

        let (store, _) = ObjectStore::from_uri_and_params(registry, url, &params)
            .await
            .unwrap();
        let test_path = Path::from("test_file.txt");
        let test_data = bytes::Bytes::from("Hello, HDFS!");

        store
            .inner
            .put(&test_path, test_data.clone().into())
            .await
            .unwrap();
        let read_data = store
            .inner
            .get(&test_path)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(read_data, test_data);
        store.inner.delete(&test_path).await.unwrap();
    }

    #[tokio::test]
    async fn test_hdfs_ha_configuration() {
        if !hdfs_available() {
            return;
        }

        // Test HA configuration like hdfs://ht-hdfsqa
        let registry = Arc::new(ObjectStoreRegistry::default());
        let url = "hdfs://ht-hdfsqa/user/test";
        let params = ObjectStoreParams::default();

        let (store, path) = ObjectStore::from_uri_and_params(registry, url, &params)
            .await
            .unwrap();

        assert_eq!(store.scheme(), "hdfs");
        assert_eq!(path, Path::from("user/test"));
    }
}
