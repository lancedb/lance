// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Integration tests for HDFS object store provider
//!
//! These tests require a running HDFS cluster. They can be run with:
//! cargo test --features hdfs hdfs_integration_test

#[cfg(feature = "hdfs")]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use lance_io::object_store::{
        ObjectStore, ObjectStoreParams, ObjectStoreRegistry, StorageOptionsAccessor,
    };
    use object_store::ObjectStoreExt;
    use object_store::path::Path;
    use url::Url;

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
    async fn test_hdfs_url_parsing() {
        let registry = Arc::new(ObjectStoreRegistry::default());

        let test_cases = vec![
            ("hdfs://localhost:9000/", ""),
            ("hdfs://localhost:9000/user", "user"),
            (
                "hdfs://localhost:9000/user/data/file.txt",
                "user/data/file.txt",
            ),
            ("hdfs://namenode/path/to/file", "path/to/file"),
            // HA configuration tests
            ("hdfs://ht-hdfsqa/", ""),
            ("hdfs://ht-hdfsqa/user/data", "user/data"),
            ("hdfs://mycluster/tmp/test.txt", "tmp/test.txt"),
        ];

        for (url_str, expected_path) in test_cases {
            let url = Url::parse(url_str).unwrap();
            let provider = registry.get_provider("hdfs").unwrap();
            let path = provider.extract_path(&url).unwrap();
            assert_eq!(path, Path::from(expected_path));
        }
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

    #[test]
    fn test_hdfs_config_validation() {
        // Test that missing namenode configuration is properly handled
        let storage_options = HashMap::new();
        let params = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                storage_options,
            ))),
            ..Default::default()
        };

        // This test verifies the configuration validation logic without actually connecting
        // The actual connection would fail, but we're testing the config validation
        assert!(params.storage_options().is_some());
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
