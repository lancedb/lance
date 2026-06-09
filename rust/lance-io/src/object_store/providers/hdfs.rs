// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use object_store_opendal::OpendalStore;
use opendal::{Operator, services::Hdfs};
use url::Url;

use crate::object_store::{
    DEFAULT_CLOUD_BLOCK_SIZE, DEFAULT_CLOUD_IO_PARALLELISM, DEFAULT_MAX_IOP_SIZE, ObjectStore,
    ObjectStoreParams, ObjectStoreProvider, StorageOptions,
};
use lance_core::error::{Error, Result};

/// HDFS object store provider backed by OpenDAL.
#[derive(Default, Debug)]
pub struct HdfsStoreProvider;

impl HdfsStoreProvider {
    fn build_config<I, K, V>(
        base_path: &Url,
        storage_options: &StorageOptions,
        env_vars: I,
    ) -> Result<HashMap<String, String>>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: Into<String>,
    {
        base_path
            .host_str()
            .ok_or_else(|| Error::invalid_input("HDFS URI must contain namenode host"))?;

        let env_vars = env_vars
            .into_iter()
            .filter_map(|(key, value)| {
                let value = value.into();
                if value.is_empty() {
                    None
                } else {
                    Some((key.as_ref().to_string(), value))
                }
            })
            .collect::<HashMap<_, _>>();

        let name_node = storage_options
            .0
            .get("hdfs_name_node")
            .cloned()
            .or_else(|| env_vars.get("HDFS_NAME_NODE").cloned())
            .unwrap_or_else(|| format!("hdfs://{}", base_path.authority()));

        let mut config = HashMap::from([
            ("name_node".to_string(), name_node),
            ("root".to_string(), "/".to_string()),
        ]);

        let user = storage_options
            .0
            .get("hdfs_user")
            .cloned()
            .or_else(|| env_vars.get("HADOOP_USER_NAME").cloned())
            .or_else(|| env_vars.get("HDFS_USER").cloned());
        if let Some(user) = user {
            config.insert("user".to_string(), user);
        }

        for (storage_key, config_key) in [
            (
                "hdfs_kerberos_ticket_cache_path",
                "kerberos_ticket_cache_path",
            ),
            ("hdfs_atomic_write_dir", "atomic_write_dir"),
        ] {
            if let Some(value) = storage_options.0.get(storage_key).filter(|v| !v.is_empty()) {
                config.insert(config_key.to_string(), value.clone());
            }
        }

        Ok(config)
    }
}

#[async_trait::async_trait]
impl ObjectStoreProvider for HdfsStoreProvider {
    async fn new_store(&self, base_path: Url, params: &ObjectStoreParams) -> Result<ObjectStore> {
        let block_size = params.block_size.unwrap_or(DEFAULT_CLOUD_BLOCK_SIZE);
        let storage_options = StorageOptions(params.storage_options().cloned().unwrap_or_default());
        let config = Self::build_config(&base_path, &storage_options, std::env::vars())?;

        let operator = Operator::from_iter::<Hdfs>(config)
            .map_err(|e| Error::invalid_input(format!("Failed to create HDFS operator: {e:?}")))?
            .finish();
        let opendal_store = Arc::new(OpendalStore::new(operator));

        Ok(ObjectStore {
            scheme: "hdfs".to_string(),
            inner: opendal_store,
            block_size,
            max_iop_size: *DEFAULT_MAX_IOP_SIZE,
            use_constant_size_upload_parts: params.use_constant_size_upload_parts,
            list_is_lexically_ordered: params.list_is_lexically_ordered.unwrap_or(true),
            io_parallelism: DEFAULT_CLOUD_IO_PARALLELISM,
            download_retry_count: storage_options.download_retry_count(),
            io_tracker: Default::default(),
            store_prefix: self
                .calculate_object_store_prefix(&base_path, params.storage_options())?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object_store::ObjectStoreProvider;
    use object_store::path::Path;

    #[test]
    fn test_hdfs_store_paths() {
        let provider = HdfsStoreProvider;
        let cases = [
            ("hdfs://namenode:9000/path/to/file", "path/to/file"),
            ("hdfs://namenode/path/to/file", "path/to/file"),
            ("hdfs://namenode:9000/", ""),
            (
                "hdfs://namenode:9000/user/data/dataset/file.parquet",
                "user/data/dataset/file.parquet",
            ),
            ("hdfs://ht-hdfsqa/user/data/file.txt", "user/data/file.txt"),
        ];

        for (url, expected_path) in cases {
            let path = provider.extract_path(&Url::parse(url).unwrap()).unwrap();
            assert_eq!(path, Path::from(expected_path));
        }
    }

    #[test]
    fn test_hdfs_config_from_url() {
        let url = Url::parse("hdfs://namenode:9000/test").unwrap();
        let config = HdfsStoreProvider::build_config(
            &url,
            &StorageOptions::default(),
            Vec::<(&str, &str)>::new(),
        )
        .unwrap();

        assert_eq!(config.get("name_node").unwrap(), "hdfs://namenode:9000");
        assert_eq!(config.get("root").unwrap(), "/");
    }

    #[test]
    fn test_hdfs_storage_options_override_environment_and_url() {
        let url = Url::parse("hdfs://url-namenode:9000/test").unwrap();
        let storage_options = StorageOptions(HashMap::from([
            (
                "hdfs_name_node".to_string(),
                "hdfs://option-namenode:8020".to_string(),
            ),
            ("hdfs_user".to_string(), "option-user".to_string()),
            (
                "hdfs_kerberos_ticket_cache_path".to_string(),
                "/tmp/krb5cc".to_string(),
            ),
            (
                "hdfs_atomic_write_dir".to_string(),
                "/tmp/atomic".to_string(),
            ),
        ]));
        let env_vars = [
            ("HDFS_NAME_NODE", "hdfs://env-namenode:9000"),
            ("HADOOP_USER_NAME", "env-user"),
        ];

        let config = HdfsStoreProvider::build_config(&url, &storage_options, env_vars).unwrap();

        assert_eq!(
            config.get("name_node").unwrap(),
            "hdfs://option-namenode:8020"
        );
        assert_eq!(config.get("user").unwrap(), "option-user");
        assert_eq!(
            config.get("kerberos_ticket_cache_path").unwrap(),
            "/tmp/krb5cc"
        );
        assert_eq!(config.get("atomic_write_dir").unwrap(), "/tmp/atomic");
    }

    #[test]
    fn test_hdfs_config_from_environment() {
        let url = Url::parse("hdfs://url-namenode:9000/test").unwrap();
        let env_vars = [
            ("HDFS_NAME_NODE", "hdfs://env-namenode:9000"),
            ("HADOOP_USER_NAME", "env-user"),
        ];

        let config =
            HdfsStoreProvider::build_config(&url, &StorageOptions::default(), env_vars).unwrap();

        assert_eq!(config.get("name_node").unwrap(), "hdfs://env-namenode:9000");
        assert_eq!(config.get("user").unwrap(), "env-user");
    }

    #[test]
    fn test_hdfs_config_rejects_url_without_host() {
        let url = Url::parse("hdfs:///test").unwrap();
        let error = HdfsStoreProvider::build_config(
            &url,
            &StorageOptions::default(),
            Vec::<(&str, &str)>::new(),
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("namenode host"));
    }
}
