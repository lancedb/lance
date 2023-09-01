// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Trait for external manifest handler.
//!
//! This trait abstracts an external storage with put_if_not_exists semantics.

use std::sync::Arc;

use async_trait::async_trait;
use object_store::path::Path;

use crate::format::{Index, Manifest};
use crate::io::commit::{CommitError, CommitHandler, ManifestWriter};
use crate::io::ObjectStore;

use crate::{Error, Result};

use super::{latest_manifest_path, manifest_path, write_latest_manifest};

/// External manifest store
///
/// This trait abstracts an external storage for source of truth for manifests.
/// The storge is expected to remember (uri, version) -> manifest_path
/// and able to run transactions on the manifest_path.
///
/// This trait is called an **External** manifest store because the store is
/// expected to work in tandem with the object store. We are only leveraging
/// the external store for concurrent commit. Any manifest committed thru this
/// trait should ultimately be materialized in the object store.
#[async_trait]
pub trait ExternalManifestStore: std::fmt::Debug + Send + Sync {
    /// Get the manifest path for a given base_uri and version
    async fn get(&self, base_uri: &str, version: u64) -> Result<String>;

    /// Get the latest version of a dataset at the base_uri
    async fn get_latest_version(&self, base_uri: &str) -> Result<Option<u64>>;

    /// Put the manifest path for a given base_uri and version, should fail if the version already exists
    async fn put_if_not_exists(&self, base_uri: &str, version: u64, path: String) -> Result<()>;
}

/// External manifest commit handler
/// This handler is used to commit a manifest to an external store
/// for detailed design, see https://github.com/lancedb/lance/issues/1183
#[derive(Debug)]
pub struct ExternalManifestCommitHandler {
    external_manifest_store: Arc<dyn ExternalManifestStore>,
}

fn make_staging_manifest_path(base: &Path) -> Result<Path> {
    let id = uuid::Uuid::new_v4().to_string();
    Path::parse(format!("{base}-{id}")).map_err(|e| Error::IO {
        message: format!("failed to parse path: {}", e),
    })
}

#[async_trait]
impl CommitHandler for ExternalManifestCommitHandler {
    /// Get the latest version of a dataset at the path
    async fn resolve_latest_version(
        &self,
        base_path: &Path,
        object_store: &ObjectStore,
    ) -> std::result::Result<Path, crate::Error> {
        let version = self
            .external_manifest_store
            .get_latest_version(base_path.as_ref())
            .await?;

        match version {
            Some(version) => {
                let object_store_manifest_path = manifest_path(base_path, version);

                // external store and object store are out of sync
                // try to sync them
                // if sync fails, return error
                if !object_store.exists(&object_store_manifest_path).await? {
                    let manifest_location = self
                        .external_manifest_store
                        .get(base_path.as_ref(), version)
                        .await?;

                    let manifest_path = Path::parse(manifest_location)?;
                    let staging = make_staging_manifest_path(&manifest_path)?;
                    object_store.inner.copy(&manifest_path, &staging).await?;
                    object_store
                        .inner
                        .rename(&staging, &object_store_manifest_path)
                        .await?;

                    write_latest_manifest(self, &manifest_path, base_path, object_store).await?;
                    // Also copy for _latest.manifest
                    object_store.inner.copy(&manifest_path, &staging).await?;
                    object_store
                        .inner
                        .rename(&staging, &latest_manifest_path(base_path))
                        .await?;
                }

                Ok(object_store_manifest_path)
            }
            // Dataset not found in the external store, this could be because the dataset did not
            // use external store for commit before. In this case, we use the _latest.manifest file
            None => Ok(latest_manifest_path(base_path)),
        }
    }

    async fn resolve_version(
        &self,
        base_path: &Path,
        version: u64,
        object_store: &ObjectStore,
    ) -> std::result::Result<Path, crate::Error> {
        let manifest_path = manifest_path(base_path, version);
        // found in object store
        if object_store.exists(&manifest_path).await? {
            return Ok(manifest_path);
        }

        // not found in object store, try to get from external store
        let physical_path = self
            .external_manifest_store
            .get(base_path.as_ref(), version)
            .await
            .map_err(|_| Error::NotFound {
                uri: manifest_path.to_string(),
            })?;

        // try to materialize the manifest from external store to object store
        // multiple writers could try to copy at the same time, this is okay
        // as the content is immutable and copy is atomic
        // We can't use `copy_if_not_exists` here because not all store supports it
        object_store
            .inner
            .copy(&Path::parse(physical_path)?, &manifest_path)
            .await?;

        Ok(manifest_path)
    }

    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<Index>>,
        base_path: &Path,
        object_store: &ObjectStore,
        manifest_writer: ManifestWriter,
    ) -> std::result::Result<(), CommitError> {
        // path we get here is the path to the manifest we want to write
        // use object_store.base_path.as_ref() for getting the root of the dataset

        // step 1: Write the manifest we want to commit to object store with a temporary name
        let path = manifest_path(base_path, manifest.version);
        let staging_path = make_staging_manifest_path(&path)?;
        manifest_writer(object_store, manifest, indices, &staging_path).await?;

        // step 2 & 3: Try to commit this version to external store, return err on failure
        // TODO: add logic to clean up orphaned staged manifests, the ones that failed to commit to external store
        // https://github.com/lancedb/lance/issues/1201
        self.external_manifest_store
            .put_if_not_exists(
                object_store.base_path().as_ref(),
                manifest.version,
                staging_path.to_string(),
            )
            .await
            .map_err(|_| CommitError::CommitConflict {})?;

        // step 4: copy the manifest to the final location
        object_store.inner.copy(
            &staging_path,
            &path,
        ).await.map_err(|e| CommitError::OtherError(
            Error::IO { message: format!("commit to external store is successful, but could not copy manifest to object store, with error: {}.", e) }
        ))?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use std::{collections::HashMap, time::Duration};

    use futures::future::join_all;
    use object_store::local::LocalFileSystem;
    use tokio::sync::Mutex;

    use crate::{
        dataset::{ReadParams, WriteMode, WriteParams},
        io::object_store::ObjectStoreParams,
        utils::datagen::{BatchGenerator, IncrementingInt32},
        Dataset,
    };

    use super::*;

    // sleep for 1 second to simulate a slow external store on write
    #[derive(Debug)]
    struct SleepyExternalManifestStore {
        store: Arc<Mutex<HashMap<(String, u64), String>>>,
    }

    impl SleepyExternalManifestStore {
        fn new() -> Self {
            Self {
                store: Arc::new(Mutex::new(HashMap::new())),
            }
        }
    }

    #[async_trait]
    impl ExternalManifestStore for SleepyExternalManifestStore {
        /// Get the manifest path for a given uri and version
        async fn get(&self, uri: &str, version: u64) -> Result<String> {
            let store = self.store.lock().await;
            match store.get(&(uri.to_string(), version)) {
                Some(path) => Ok(path.clone()),
                None => Err(Error::IO {
                    message: format!("manifest not found for uri: {}, version: {}", uri, version),
                }),
            }
        }

        /// Get the latest version of a dataset at the path
        async fn get_latest_version(&self, uri: &str) -> Result<Option<u64>> {
            let store = self.store.lock().await;
            Ok(store
                .keys()
                .filter_map(|(stored_uri, version)| {
                    if stored_uri == uri {
                        Some(version)
                    } else {
                        None
                    }
                })
                .max()
                .copied())
        }

        /// Put the manifest path for a given uri and version, should fail if the version already exists
        async fn put_if_not_exists(&self, uri: &str, version: u64, path: String) -> Result<()> {
            tokio::time::sleep(Duration::from_secs(1)).await;

            let mut store = self.store.lock().await;
            match store.get(&(uri.to_string(), version)) {
                Some(_) => Err(Error::IO {
                    message: format!(
                        "manifest already exists for uri: {}, version: {}",
                        uri, version
                    ),
                }),
                None => {
                    store.insert((uri.to_string(), version), path);
                    Ok(())
                }
            }
        }
    }

    fn read_params(handler: Arc<dyn CommitHandler>) -> ReadParams {
        ReadParams {
            store_options: Some(ObjectStoreParams {
                commit_handler: Some(handler),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn write_params(handler: Arc<dyn CommitHandler>) -> WriteParams {
        WriteParams {
            store_params: Some(ObjectStoreParams {
                commit_handler: Some(handler),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_dataset_can_onboard_external_store() {
        // First write a dataset WITHOUT external store
        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("x".to_owned())));
        let reader = data_gen.batch(100).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let ds_uri = dir.path().to_str().unwrap();
        Dataset::write(reader, ds_uri, None).await.unwrap();

        // Then try to load the dataset with external store handler set
        let sleepy_store = SleepyExternalManifestStore::new();
        let handler = ExternalManifestCommitHandler {
            external_manifest_store: Arc::new(sleepy_store),
        };
        let options = read_params(Arc::new(handler));
        Dataset::open_with_params(ds_uri, &options).await.expect(
            "If this fails, it means the external store handler does not correctly handle the case when a dataset exist, but it has never used external store before."
        );
    }

    #[tokio::test]
    async fn test_can_create_dataset_with_external_store() {
        let sleepy_store = SleepyExternalManifestStore::new();
        let handler = ExternalManifestCommitHandler {
            external_manifest_store: Arc::new(sleepy_store),
        };
        let handler = Arc::new(handler);

        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("x".to_owned())));
        let reader = data_gen.batch(100).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let ds_uri = dir.path().to_str().unwrap();
        Dataset::write(reader, ds_uri, Some(write_params(handler.clone())))
            .await
            .unwrap();

        // load the data and check the content
        let ds = Dataset::open_with_params(ds_uri, &read_params(handler))
            .await
            .unwrap();
        assert_eq!(ds.count_rows().await.unwrap(), 100);
    }

    #[tokio::test]
    async fn test_concurrent_commits_are_okay() {
        let sleepy_store = SleepyExternalManifestStore::new();
        let handler = ExternalManifestCommitHandler {
            external_manifest_store: Arc::new(sleepy_store),
        };
        let handler = Arc::new(handler);

        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("x".to_owned())));
        let dir = tempfile::tempdir().unwrap();
        let ds_uri = dir.path().to_str().unwrap();

        Dataset::write(
            data_gen.batch(10).unwrap(),
            ds_uri,
            Some(write_params(handler.clone())),
        )
        .await
        .unwrap();

        // we have 5 retries by default, more than this will just fail
        let write_futs = (0..5)
            .map(|_| data_gen.batch(10).unwrap())
            .map(|data| {
                let mut params = write_params(handler.clone());
                params.mode = WriteMode::Append;
                Dataset::write(data, ds_uri, Some(params))
            })
            .collect::<Vec<_>>();

        let res = join_all(write_futs).await;

        let errors = res
            .into_iter()
            .filter(|r| r.is_err())
            .map(|r| r.unwrap_err())
            .collect::<Vec<_>>();

        assert!(errors.is_empty(), "{:?}", errors);

        // load the data and check the content
        let ds = Dataset::open_with_params(ds_uri, &read_params(handler))
            .await
            .unwrap();
        assert_eq!(ds.count_rows().await.unwrap(), 60);
    }

    #[tokio::test]
    async fn test_out_of_sync_dataset_can_recover() {
        let sleepy_store = SleepyExternalManifestStore::new();
        let handler = ExternalManifestCommitHandler {
            external_manifest_store: Arc::new(sleepy_store),
        };
        let handler = Arc::new(handler);

        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("x".to_owned())));
        let dir = tempfile::tempdir().unwrap();
        let ds_uri = dir.path().to_str().unwrap();

        let mut ds = Dataset::write(
            data_gen.batch(10).unwrap(),
            ds_uri,
            Some(write_params(handler.clone())),
        )
        .await
        .unwrap();

        for _ in 0..5 {
            let data = data_gen.batch(10).unwrap();
            let mut params = write_params(handler.clone());
            params.mode = WriteMode::Append;
            ds = Dataset::write(data, ds_uri, Some(params)).await.unwrap();
        }

        // manually simulate last version is out of sync
        let localfs: Box<dyn object_store::ObjectStore> = Box::new(LocalFileSystem::new());
        localfs.delete(&manifest_path(&ds.base, 6)).await.unwrap();
        localfs
            .copy(&manifest_path(&ds.base, 5), &latest_manifest_path(&ds.base))
            .await
            .unwrap();

        // Open without external store handler, should not see the out-of-sync commit
        let params = ReadParams::default();
        let ds = Dataset::open_with_params(ds_uri, &params).await.unwrap();
        assert_eq!(ds.version().version, 5);
        assert_eq!(ds.count_rows().await.unwrap(), 50);

        // Open with external store handler, should sync the out-of-sync commit on open
        let ds = Dataset::open_with_params(ds_uri, &read_params(handler))
            .await
            .unwrap();
        assert_eq!(ds.version().version, 6);
        assert_eq!(ds.count_rows().await.unwrap(), 60);

        // Open without external store handler again, should see the newly sync'd commit
        let params = ReadParams::default();
        let ds = Dataset::open_with_params(ds_uri, &params).await.unwrap();
        assert_eq!(ds.version().version, 6);
        assert_eq!(ds.count_rows().await.unwrap(), 60);
    }
}
