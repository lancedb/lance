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
use log::warn;
use object_store::path::Path;
use snafu::{location, Location};

use crate::format::{Index, Manifest};
use crate::io::commit::{CommitError, CommitHandler, ManifestWriter};
use crate::io::ObjectStore;

use crate::{Error, Result};

use super::{
    current_manifest_path, make_staging_manifest_path, manifest_path, parse_version_from_path,
    write_latest_manifest, MANIFEST_EXTENSION,
};

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
/// For a visual explaination of the commit loop see
/// https://github.com/lancedb/lance/assets/12615154/b0822312-0826-432a-b554-3965f8d48d04
#[async_trait]
pub trait ExternalManifestStore: std::fmt::Debug + Send + Sync {
    /// Get the manifest path for a given base_uri and version
    async fn get(&self, base_uri: &str, version: u64) -> Result<String>;

    /// Get the latest version of a dataset at the base_uri, and the path to the manifest.
    /// The path is provided as an optimization. The path is deterministic based on
    /// the version and the store should not customize it.
    async fn get_latest_version(&self, base_uri: &str) -> Result<Option<(u64, String)>>;

    /// Put the manifest path for a given base_uri and version, should fail if the version already exists
    async fn put_if_not_exists(&self, base_uri: &str, version: u64, path: &str) -> Result<()>;

    /// Put the manifest path for a given base_uri and version, should fail if the version **does not** already exist
    async fn put_if_exists(&self, base_uri: &str, version: u64, path: &str) -> Result<()>;
}

/// External manifest commit handler
/// This handler is used to commit a manifest to an external store
/// for detailed design, see https://github.com/lancedb/lance/issues/1183
#[derive(Debug)]
pub struct ExternalManifestCommitHandler {
    pub external_manifest_store: Arc<dyn ExternalManifestStore>,
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
            Some((version, path)) => {
                // The path is finalized, no need to check object store
                if path.ends_with(&format!(".{MANIFEST_EXTENSION}")) {
                    return Ok(Path::parse(path)?);
                }
                // path is not finalized yet, we should try to finalize the path before loading
                // if sync/finalize fails, return error
                //
                // step 1: copy path -> object_store_manifest_path
                let object_store_manifest_path = manifest_path(base_path, version);
                let manifest_path = Path::parse(path)?;
                let staging = make_staging_manifest_path(&manifest_path)?;
                // TODO: remove copy-rename once we upgrade object_store crate
                object_store.inner.copy(&manifest_path, &staging).await?;
                object_store
                    .inner
                    .rename(&staging, &object_store_manifest_path)
                    .await?;

                // step 2: write _latest.manifest
                write_latest_manifest(&manifest_path, base_path, object_store).await?;

                // step 3: update external store to finalize path
                self.external_manifest_store
                    .put_if_exists(
                        base_path.as_ref(),
                        version,
                        object_store_manifest_path.as_ref(),
                    )
                    .await?;

                Ok(object_store_manifest_path)
            }
            // Dataset not found in the external store, this could be because the dataset did not
            // use external store for commit before. In this case, we search for the latest manifest
            None => current_manifest_path(object_store, base_path).await,
        }
    }

    async fn resolve_latest_version_id(
        &self,
        base_path: &Path,
        object_store: &ObjectStore,
    ) -> std::result::Result<u64, crate::Error> {
        let version = self
            .external_manifest_store
            .get_latest_version(base_path.as_ref())
            .await?;

        match version {
            Some((version, _)) => Ok(version),
            None => parse_version_from_path(&current_manifest_path(object_store, base_path).await?),
        }
    }

    async fn resolve_version(
        &self,
        base_path: &Path,
        version: u64,
        object_store: &ObjectStore,
    ) -> std::result::Result<Path, crate::Error> {
        let path_res = self
            .external_manifest_store
            .get(base_path.as_ref(), version)
            .await;

        let path = match path_res {
            Ok(p) => p,
            // not board external manifest yet, direct to object store
            Err(Error::NotFound { .. }) => {
                let path = manifest_path(base_path, version);
                // if exist update external manifest store
                if object_store.exists(&path).await? {
                    // best effort put, if it fails, it's okay
                    match self
                        .external_manifest_store
                        .put_if_not_exists(base_path.as_ref(), version, path.as_ref())
                        .await
                    {
                        Ok(_) => {}
                        Err(e) => {
                            warn!("could up update external manifest store during load, with error: {}", e);
                        }
                    }
                } else {
                    return Err(Error::NotFound {
                        uri: path.to_string(),
                        location: location!(),
                    });
                }
                return Ok(manifest_path(base_path, version));
            }
            Err(e) => return Err(e),
        };

        // finalized path, just return
        if path.ends_with(&format!(".{MANIFEST_EXTENSION}")) {
            return Ok(Path::parse(path)?);
        }

        let manifest_path = manifest_path(base_path, version);
        let staging_path = make_staging_manifest_path(&manifest_path)?;

        // step1: try to materialize the manifest from external store to object store
        // multiple writers could try to copy at the same time, this is okay
        // as the content is immutable and copy is atomic
        // We can't use `copy_if_not_exists` here because not all store supports it
        object_store
            .inner
            .copy(&Path::parse(path)?, &staging_path)
            .await?;
        object_store
            .inner
            .rename(&staging_path, &manifest_path)
            .await?;

        // finalize the external store
        self.external_manifest_store
            .put_if_exists(base_path.as_ref(), version, manifest_path.as_ref())
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
            .put_if_not_exists(base_path.as_ref(), manifest.version, staging_path.as_ref())
            .await
            .map_err(|_| CommitError::CommitConflict {})?;

        // step 4: copy the manifest to the final location
        object_store.inner.copy(
            &staging_path,
            &path,
        ).await.map_err(|e| CommitError::OtherError(
            Error::IO {
                message: format!("commit to external store is successful, but could not copy manifest to object store, with error: {}.", e),
                location: location!(),
            }
        ))?;

        // update the _latest.manifest pointer
        write_latest_manifest(&path, base_path, object_store).await?;

        // step 5: flip the external store to point to the final location
        self.external_manifest_store
            .put_if_exists(base_path.as_ref(), manifest.version, path.as_ref())
            .await?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use std::{collections::HashMap, time::Duration};

    use futures::{future::join_all, StreamExt, TryStreamExt};
    use lance_testing::datagen::{BatchGenerator, IncrementingInt32};
    use object_store::local::LocalFileSystem;
    use tokio::sync::Mutex;

    use crate::{
        dataset::{ReadParams, WriteMode, WriteParams},
        io::{commit::latest_manifest_path, object_store::ObjectStoreParams},
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
                None => Err(Error::NotFound {
                    uri: uri.to_string(),
                    location: location!(),
                }),
            }
        }

        /// Get the latest version of a dataset at the path
        async fn get_latest_version(&self, uri: &str) -> Result<Option<(u64, String)>> {
            let store = self.store.lock().await;
            let max_version = store
                .iter()
                .filter_map(|((stored_uri, version), manifest_uri)| {
                    if stored_uri == uri {
                        Some((version, manifest_uri))
                    } else {
                        None
                    }
                })
                .max_by_key(|v| v.0);

            Ok(max_version.map(|(version, uri)| (*version, uri.clone())))
        }

        /// Put the manifest path for a given uri and version, should fail if the version already exists
        async fn put_if_not_exists(&self, uri: &str, version: u64, path: &str) -> Result<()> {
            tokio::time::sleep(Duration::from_millis(100)).await;

            let mut store = self.store.lock().await;
            match store.get(&(uri.to_string(), version)) {
                Some(_) => Err(Error::IO {
                    message: format!(
                        "manifest already exists for uri: {}, version: {}",
                        uri, version
                    ),
                    location: location!(),
                }),
                None => {
                    store.insert((uri.to_string(), version), path.to_string());
                    Ok(())
                }
            }
        }

        /// Put the manifest path for a given uri and version, should fail if the version already exists
        async fn put_if_exists(&self, uri: &str, version: u64, path: &str) -> Result<()> {
            tokio::time::sleep(Duration::from_millis(100)).await;

            let mut store = self.store.lock().await;
            match store.get(&(uri.to_string(), version)) {
                Some(_) => {
                    store.insert((uri.to_string(), version), path.to_string());
                    Ok(())
                }
                None => Err(Error::IO {
                    message: format!(
                        "manifest already exists for uri: {}, version: {}",
                        uri, version
                    ),
                    location: location!(),
                }),
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
        let reader = data_gen.batch(100);
        let dir = tempfile::tempdir().unwrap();
        let ds_uri = dir.path().to_str().unwrap();
        Dataset::write(reader, ds_uri, None).await.unwrap();

        // Then try to load the dataset with external store handler set
        let sleepy_store = SleepyExternalManifestStore::new();
        let handler = Arc::new(ExternalManifestCommitHandler {
            external_manifest_store: Arc::new(sleepy_store),
        });
        let options = read_params(handler.clone());
        Dataset::open_with_params(ds_uri, &options).await.expect(
            "If this fails, it means the external store handler does not correctly handle the case when a dataset exist, but it has never used external store before."
        );

        Dataset::write(
            data_gen.batch(100),
            ds_uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                store_params: Some(ObjectStoreParams {
                    commit_handler: Some(handler),
                    ..Default::default()
                }),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
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
        let reader = data_gen.batch(100);
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

    #[cfg(not(windows))]
    #[tokio::test]
    async fn test_concurrent_commits_are_okay() {
        // Run test 20 times to have a higher chance of catching race conditions
        for _ in 0..20 {
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
                data_gen.batch(10),
                ds_uri,
                Some(write_params(handler.clone())),
            )
            .await
            .unwrap();

            // we have 5 retries by default, more than this will just fail
            let write_futs = (0..5)
                .map(|_| data_gen.batch(10))
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
    }

    #[tokio::test]
    async fn test_out_of_sync_dataset_can_recover() {
        let sleepy_store = SleepyExternalManifestStore::new();
        let inner_store = sleepy_store.store.clone();
        let handler = ExternalManifestCommitHandler {
            external_manifest_store: Arc::new(sleepy_store),
        };
        let handler = Arc::new(handler);

        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("x".to_owned())));
        let dir = tempfile::tempdir().unwrap();
        let ds_uri = dir.path().to_str().unwrap();

        let mut ds = Dataset::write(
            data_gen.batch(10),
            ds_uri,
            Some(write_params(handler.clone())),
        )
        .await
        .unwrap();

        for _ in 0..5 {
            let data = data_gen.batch(10);
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
        // set the store back to dataset path with -{uuid} suffix
        let mut version_six = localfs
            .list(Some(&ds.base))
            .await
            .unwrap()
            .try_filter(|p| {
                let p = p.clone();
                async move { p.location.filename().unwrap().starts_with("6.manifest-") }
            })
            .take(1)
            .collect::<Vec<_>>()
            .await;
        assert_eq!(version_six.len(), 1);
        let version_six_staging_location = version_six.pop().unwrap().unwrap().location;
        {
            inner_store.lock().await.insert(
                (ds.base.to_string(), 6),
                version_six_staging_location.to_string(),
            );
        }

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
