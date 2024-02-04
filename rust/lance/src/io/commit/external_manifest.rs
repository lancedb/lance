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

/// Keep the tests in `lance` crate because it has dependency on [Dataset].
#[cfg(test)]
mod test {
    use std::sync::Arc;
    use std::{collections::HashMap, time::Duration};

    use async_trait::async_trait;
    use futures::{future::join_all, StreamExt, TryStreamExt};
    use lance_core::{Error, Result};
    use lance_table::io::commit::external_manifest::{
        ExternalManifestCommitHandler, ExternalManifestStore,
    };
    use lance_table::io::commit::{latest_manifest_path, manifest_path, CommitHandler};
    use lance_testing::datagen::{BatchGenerator, IncrementingInt32};
    use object_store::local::LocalFileSystem;
    use snafu::{location, Location};
    use tokio::sync::Mutex;

    use crate::dataset::builder::DatasetBuilder;
    use crate::{
        dataset::{ReadParams, WriteMode, WriteParams},
        Dataset,
    };

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
            commit_handler: Some(handler),
            ..Default::default()
        }
    }

    fn write_params(handler: Arc<dyn CommitHandler>) -> WriteParams {
        WriteParams {
            commit_handler: Some(handler),
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
        DatasetBuilder::from_uri(ds_uri)
            .with_read_params(options)
            .load()
            .await
            .unwrap();

        Dataset::write(
            data_gen.batch(100),
            ds_uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                commit_handler: Some(handler),
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
        let ds = DatasetBuilder::from_uri(ds_uri)
            .with_read_params(read_params(handler))
            .load()
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
            let ds = DatasetBuilder::from_uri(ds_uri)
                .with_read_params(read_params(handler))
                .load()
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
        let ds = DatasetBuilder::from_uri(ds_uri).load().await.unwrap();
        assert_eq!(ds.version().version, 5);
        assert_eq!(ds.count_rows().await.unwrap(), 50);

        // Open with external store handler, should sync the out-of-sync commit on open
        let ds = DatasetBuilder::from_uri(ds_uri)
            .with_commit_handler(handler.clone())
            .load()
            .await
            .unwrap();
        assert_eq!(ds.version().version, 6);
        assert_eq!(ds.count_rows().await.unwrap(), 60);

        // Open without external store handler again, should see the newly sync'd commit
        let ds = DatasetBuilder::from_uri(ds_uri).load().await.unwrap();
        assert_eq!(ds.version().version, 6);
        assert_eq!(ds.count_rows().await.unwrap(), 60);
    }
}
