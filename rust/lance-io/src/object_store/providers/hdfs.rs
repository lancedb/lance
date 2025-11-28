// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use lance_core::Error;
use object_store::path::Path;
use object_store_opendal::OpendalStore;
use opendal::{services::HdfsNative, Operator};
use snafu::location;

use crate::object_store::{
    ObjectStore, ObjectStoreParams, ObjectStoreProvider, StorageOptions,
    DEFAULT_CLOUD_IO_PARALLELISM,
};

#[derive(Debug, Clone)]
pub struct HdfsStoreProvider;

#[async_trait::async_trait]
impl ObjectStoreProvider for HdfsStoreProvider {
    async fn new_store(
        &self,
        base_path: url::Url,
        params: &ObjectStoreParams,
    ) -> Result<ObjectStore, lance_core::Error> {
        let mut storage_options =
            StorageOptions(params.storage_options.clone().unwrap_or_default());

        let download_retry_count = storage_options.download_retry_count();
        storage_options.0.insert(
            "name_node".to_string(),
            format!("{}://{}", base_path.scheme(), base_path.authority()),
        );

        let operator = Operator::from_iter::<HdfsNative>(storage_options.0.into_iter())
            .map_err(|e| {
                Error::invalid_input(
                    format!("Failed to create HDFS native operator: {:?}", e),
                    location!(),
                )
            })?
            .finish();

        let opendal_store = Arc::new(OpendalStore::new(operator));
        Ok(ObjectStore::new(
            opendal_store,
            base_path,
            params.block_size,
            None,
            params.use_constant_size_upload_parts,
            true,
            DEFAULT_CLOUD_IO_PARALLELISM,
            download_retry_count,
            params.storage_options.as_ref(),
        ))
    }

    fn extract_path(&self, url: &url::Url) -> Result<Path, Error> {
        if let Ok(file_path) = url.to_file_path() {
            if let Ok(path) = Path::from_absolute_path(&file_path) {
                return Ok(path);
            }
        }

        Path::parse(url.path()).map_err(|e| {
            Error::invalid_input(
                format!("Failed to parse path '{}': {}", url.path(), e),
                location!(),
            )
        })
    }
}
#[cfg(test)]
mod tests {
    use url::Url;

    use super::*;

    #[test]
    fn test_hdfs_store_path() {
        let provider = HdfsStoreProvider;

        let url = Url::parse("hdfs://hdfs-server/path/to/file").unwrap();
        let path = provider.extract_path(&url).unwrap();
        let expected_path = Path::from("/path/to/file");
        assert_eq!(path, expected_path);
    }
}
