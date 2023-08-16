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

//! A task to clean up a lance dataset

use std::collections::HashSet;

use chrono::{DateTime, Duration, Utc};
use futures::TryStreamExt;
use object_store::path::Path;

use crate::{
    format::Manifest,
    index::vector::index_dir,
    io::{deletion_file_path, read_manifest, reader::read_manifest_indexes},
    utils::temporal::utc_now,
    Dataset, Error, Result,
};

#[derive(Default)]
struct InvalidFiles {
    unref_data_paths: HashSet<String>,
    unref_delete_paths: HashSet<String>,
    unref_index_uuids: HashSet<String>,
    old_manifest_paths: Vec<String>,
}

impl InvalidFiles {
    fn new(all_files: &FilesList) -> Self {
        Self {
            unref_data_paths: HashSet::from_iter(all_files.data_file_paths.iter().cloned()),
            unref_delete_paths: HashSet::from_iter(all_files.delete_file_paths.iter().cloned()),
            unref_index_uuids: HashSet::from_iter(all_files.index_uuids.iter().cloned()),
            old_manifest_paths: Vec::new(),
        }
    }
}

impl InvalidFiles {
    fn finish(self, dataset: &Dataset) -> (Vec<String>, Vec<String>) {
        let mut all_paths = self.old_manifest_paths;
        all_paths.extend(self.unref_data_paths);
        all_paths.extend(self.unref_delete_paths);
        let all_dirs = Vec::from_iter(
            self.unref_index_uuids
                .into_iter()
                .map(|uuid| index_dir(dataset, &uuid).to_string()),
        );
        (all_paths, all_dirs)
    }
}

#[derive(Default)]
struct FilesList {
    data_file_paths: Vec<String>,
    manifest_file_paths: Vec<String>,
    index_uuids: HashSet<String>,
    delete_file_paths: Vec<String>,
}

fn join_paths(base: &Path, child: &Path) -> Path {
    return child
        .parts()
        .fold(base.clone(), |joined, part| joined.child(part));
}

fn remove_prefix(path: &Path, prefix: &Path) -> Path {
    let relative_parts = path.prefix_match(prefix);
    if relative_parts.is_none() {
        return path.clone();
    }
    Path::from_iter(relative_parts.unwrap())
}

#[derive(Debug, Clone)]
struct CleanupTask<'a> {
    dataset: &'a Dataset,
    /// Cleanup all versions before this time
    before: DateTime<Utc>,
}

const MINIMUM_CLEANUP_DAYS: i64 = 7;

impl<'a> CleanupTask<'a> {
    fn new(dataset: &'a Dataset, before: DateTime<Utc>) -> Self {
        Self { dataset, before }
    }

    async fn run(&mut self) -> Result<()> {
        self.validate()?;
        let all_paths: Vec<Path> = self
            .dataset
            .object_store
            .read_dir_all(&self.dataset.base)
            .await?
            .try_collect()
            .await?;
        let mut files_list: FilesList = Default::default();
        for path in all_paths {
            self.discover_path(&path, &mut files_list)?;
        }
        let mut invalid_files = InvalidFiles::new(&files_list);
        let mut has_valid_manifest = false;
        for manifest in &files_list.manifest_file_paths {
            self.process_manifest_file(manifest, &mut invalid_files, &mut has_valid_manifest)
                .await?;
        }
        if !has_valid_manifest {
            // If there are no valid manifest files then err on the side of
            // not deleting anything.
            return Ok(());
        }
        // For paths we get relative paths.  For index dirs we get fully qualified
        // paths.
        let (paths_to_del, dirs_to_del) = invalid_files.finish(self.dataset);
        for path in &paths_to_del {
            let full_path = join_paths(&self.dataset.base, &Path::parse(path)?);
            self.dataset.object_store.remove(full_path).await?;
        }
        for dir in &dirs_to_del {
            self.dataset
                .object_store
                .remove_dir_all(dir.to_owned())
                .await?;
        }
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        if (utc_now() - self.before) < Duration::days(MINIMUM_CLEANUP_DAYS) {
            return Err(Error::invalid_input(format!(
                "Cannot cleanup data less than {} days old",
                MINIMUM_CLEANUP_DAYS
            )));
        }
        Ok(())
    }

    async fn process_manifest_file(
        &self,
        path: &String,
        invalid_files: &mut InvalidFiles,
        has_valid_manifest: &mut bool,
    ) -> Result<bool> {
        // TODO: We can't cleanup invalid manifests.  There is no way to distinguish
        // between an invalid manifest and a temporary I/O error.  It's also not safe
        // to ignore a manifest error because if it is a temporary I/O error and we
        // ignore it then we might delete valid data files thinking they are not
        // referenced.

        let manifest_path = join_paths(&self.dataset.base, &Path::parse(path)?);
        let manifest = read_manifest(&self.dataset.object_store, &manifest_path).await?;
        let dataset_version = self.dataset.version().version;
        // Don't delete the latest version, even if it is old
        let is_latest = dataset_version == manifest.version;
        if is_latest || manifest.timestamp() >= self.before {
            println!(
                "Skipping manifest {:?} because it is newer than the cutoff",
                path
            );
            *has_valid_manifest = true;
            self.process_valid_manifest(&manifest, path, invalid_files)
                .await?;
            return Ok(true);
        } else {
            println!(
                "Found old manifest {:?} with timestamp {:?} but the cutoff is {:?}",
                path,
                manifest.timestamp(),
                self.before
            );
            invalid_files.old_manifest_paths.push(path.clone());
        }
        Ok(false)
    }

    async fn process_valid_manifest(
        &self,
        manifest: &Manifest,
        manifest_path: &str,
        invalid_files: &mut InvalidFiles,
    ) -> Result<()> {
        for fragment in manifest.fragments.iter() {
            for file in fragment.files.iter() {
                let full_data_path = self.dataset.data_dir().child(file.path.as_str());
                let relative_data_path = remove_prefix(&full_data_path, &self.dataset.base);
                invalid_files
                    .unref_data_paths
                    .remove(relative_data_path.as_ref());
            }
            let delpath = fragment
                .deletion_file
                .as_ref()
                .map(|delfile| deletion_file_path(&self.dataset.base, fragment.id, delfile));
            if delpath.is_some() {
                let relative_path = remove_prefix(&delpath.unwrap(), &self.dataset.base);
                invalid_files
                    .unref_delete_paths
                    .remove(relative_path.as_ref());
            }
        }
        let full_path = join_paths(&self.dataset.base, &Path::from(manifest_path));
        let indexes =
            read_manifest_indexes(&self.dataset.object_store, &full_path, manifest).await?;
        for index in indexes {
            let uuid_str = index.uuid.to_string();
            invalid_files.unref_index_uuids.remove(&uuid_str);
        }
        Ok(())
    }

    fn discover_path(&mut self, path: &Path, files_list: &mut FilesList) -> Result<()> {
        let relative_path = remove_prefix(path, &self.dataset.base);
        if relative_path.as_ref().starts_with("_indices") {
            // For indices we just grab the UUID because we want to delete the entire
            // folder if we determine an index is expired.
            if let Some(uuid) = relative_path.parts().nth(1) {
                files_list
                    .index_uuids
                    .insert(Path::parse(uuid)?.as_ref().to_owned());
            }
            // Intentionally returning early here.  We don't want data files in index directories
            // to be treated like normal data files.
            return Ok(());
        }
        match path.extension() {
            Some("lance") => {
                if relative_path.as_ref().starts_with("data") {
                    files_list
                        .data_file_paths
                        .push(relative_path.as_ref().to_string());
                }
            }
            Some("manifest") => {
                // We intentionally ignore _latest.manifest.  We should never delete
                // it and we can assume that it is a clone of one of the _versions/... files
                if relative_path.as_ref().starts_with("_versions") {
                    files_list
                        .manifest_file_paths
                        .push(relative_path.as_ref().to_string())
                }
            }
            Some("arrow") => {
                if relative_path.as_ref().starts_with("_deletions") {
                    files_list
                        .delete_file_paths
                        .push(relative_path.as_ref().to_string());
                }
            }
            Some("bin") => {
                if relative_path.as_ref().starts_with("_deletions") {
                    files_list
                        .delete_file_paths
                        .push(relative_path.as_ref().to_string());
                }
            }
            _ => (),
        };
        Ok(())
    }
}

pub async fn cleanup_old_versions(dataset: &Dataset, before: DateTime<Utc>) -> Result<()> {
    let mut cleanup = CleanupTask::new(dataset, before);
    cleanup.run().await
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use arrow_array::RecordBatchReader;
    use chrono::Duration;

    use crate::{
        dataset::{ReadParams, WriteMode, WriteParams},
        index::{
            vector::{MetricType, StageParams, VectorIndexParams},
            DatasetIndexExt, IndexType,
        },
        io::{
            object_store::{ObjectStoreParams, WrappingObjectStore},
            ObjectStore,
        },
        utils::{
            datagen::{some_batch, BatchGenerator, IncrementingInt32},
            temporal::utc_now,
            testing::{assert_err_containing, MockClock, ProxyObjectStore, ProxyObjectStorePolicy},
        },
        Error, Result,
    };
    use all_asserts::{assert_gt, assert_lt};
    use tempfile::{tempdir, TempDir};

    use super::*;

    #[derive(Debug)]
    struct MockObjectStore {
        policy: Arc<Mutex<ProxyObjectStorePolicy>>,
    }

    impl WrappingObjectStore for MockObjectStore {
        fn wrap(
            &self,
            original: Arc<dyn object_store::ObjectStore>,
        ) -> Arc<dyn object_store::ObjectStore> {
            Arc::new(ProxyObjectStore::new(original, self.policy.clone()))
        }
    }

    impl MockObjectStore {
        pub(crate) fn new() -> Self {
            Self {
                policy: Arc::new(Mutex::new(ProxyObjectStorePolicy::new())),
            }
        }
    }

    #[derive(Debug, PartialEq)]
    struct FileCounts {
        num_data_files: usize,
        num_manifest_files: usize,
        num_index_files: usize,
        num_delete_files: usize,
    }

    struct MockDatasetFixture<'a> {
        // This is a temporary directory that will be deleted when the fixture
        // is dropped
        _tmpdir: TempDir,
        tmpdir_str: String,
        mock_store: Arc<MockObjectStore>,
        pub clock: MockClock<'a>,
    }

    impl<'a> MockDatasetFixture<'a> {
        fn try_new() -> Result<Self> {
            let tmpdir = tempdir()?;
            let maybe_tmpdir_str = tmpdir.path().to_owned().into_os_string().into_string();
            if maybe_tmpdir_str.is_err() {
                return Err(Error::invalid_input(format!(
                    "Temporary directory {:?} could not be converted to string",
                    tmpdir.path()
                )));
            }
            Ok(Self {
                _tmpdir: tmpdir,
                tmpdir_str: maybe_tmpdir_str.unwrap(),
                mock_store: Arc::new(MockObjectStore::new()),
                clock: MockClock::new(),
            })
        }

        fn dataset_uri(&self) -> String {
            format!("file://{}/my_db", self.tmpdir_str)
        }

        fn os_params(&self) -> ObjectStoreParams {
            ObjectStoreParams {
                object_store_wrapper: Some(self.mock_store.clone()),
                ..Default::default()
            }
        }

        async fn write_data_impl(
            &self,
            data: impl RecordBatchReader + Send + 'static,
            mode: WriteMode,
        ) -> Result<()> {
            Dataset::write(
                data,
                &self.dataset_uri(),
                Some(WriteParams {
                    store_params: Some(self.os_params()),
                    mode,
                    ..Default::default()
                }),
            )
            .await?;
            Ok(())
        }

        async fn write_some_data_impl(&self, mode: WriteMode) -> Result<()> {
            self.write_data_impl(some_batch().unwrap(), mode).await?;
            Ok(())
        }

        async fn create_some_data(&self) -> Result<()> {
            self.write_some_data_impl(WriteMode::Create).await
        }

        async fn overwrite_some_data(&self) -> Result<()> {
            self.write_some_data_impl(WriteMode::Overwrite).await
        }

        async fn append_some_data(&self) -> Result<()> {
            self.write_some_data_impl(WriteMode::Append).await
        }

        async fn create_with_data(
            &self,
            data: impl RecordBatchReader + Send + 'static,
        ) -> Result<()> {
            self.write_data_impl(data, WriteMode::Create).await
        }

        async fn append_data(&self, data: impl RecordBatchReader + Send + 'static) -> Result<()> {
            self.write_data_impl(data, WriteMode::Append).await
        }

        async fn overwrite_data(
            &self,
            data: impl RecordBatchReader + Send + 'static,
        ) -> Result<()> {
            self.write_data_impl(data, WriteMode::Overwrite).await
        }

        async fn delete_data(&self, predicate: &str) -> Result<()> {
            let mut db = self.open().await?;
            db.delete(predicate).await?;
            Ok(())
        }

        async fn create_some_index(&self) -> Result<()> {
            let db = self.open().await?;
            let index_params = Box::new(VectorIndexParams {
                stages: vec![StageParams::DiskANN(Default::default())],
                metric_type: MetricType::L2,
            });
            db.create_index(
                &["indexable"],
                IndexType::Vector,
                Some("some_index".to_owned()),
                &*index_params,
                false,
            )
            .await?;
            Ok(())
        }

        fn block_commits(&mut self) {
            let mut policy = self.mock_store.policy.lock().unwrap();
            policy.set_before_policy("block_commit", |op, _| -> Result<()> {
                if op.contains("copy") {
                    return Err(Error::Internal {
                        message: "Copy blocked".to_string(),
                    });
                }
                Ok(())
            });
        }

        async fn run_cleanup(&self, before: DateTime<Utc>) -> Result<()> {
            let db = self.open().await?;
            let mut cleanup = CleanupTask::new(&db, before);
            cleanup.run().await?;
            Ok(())
        }

        async fn open(&self) -> Result<Box<Dataset>> {
            Ok(Box::new(
                Dataset::open_with_params(
                    &self.dataset_uri(),
                    &ReadParams {
                        store_options: Some(self.os_params()),
                        ..Default::default()
                    },
                )
                .await?,
            ))
        }

        async fn count_files(&self) -> Result<FileCounts> {
            let (os, path) =
                ObjectStore::from_uri_and_params(&self.dataset_uri(), self.os_params()).await?;
            let mut file_stream = os.read_dir_all(&path).await?;
            let mut file_count = FileCounts {
                num_data_files: 0,
                num_delete_files: 0,
                num_index_files: 0,
                num_manifest_files: 0,
            };
            while let Some(path) = file_stream.try_next().await? {
                match path.extension() {
                    Some("lance") => file_count.num_data_files += 1,
                    Some("manifest") => file_count.num_manifest_files += 1,
                    Some("arrow") => file_count.num_delete_files += 1,
                    Some("bin") => file_count.num_delete_files += 1,
                    Some("idx") => file_count.num_index_files += 1,
                    _ => (),
                }
            }
            Ok(file_count)
        }

        async fn count_rows(&self) -> Result<usize> {
            let db = self.open().await?;
            let count = db.count_rows().await?;
            Ok(count)
        }

        async fn debug_print_files(&self) -> Result<()> {
            let (os, path) =
                ObjectStore::from_uri_and_params(&self.dataset_uri(), self.os_params()).await?;
            let mut file_stream = os.read_dir_all(&path).await?;
            while let Some(path) = file_stream.try_next().await.unwrap() {
                println!("File: {:?}", path);
            }
            Ok(())
        }
    }

    #[tokio::test]
    async fn cleanup_unreferenced_data_files() {
        // We should clean up data files that are only referenced
        // by old versions.  This can happen, for example, due to
        // an overwrite
        let fixture = MockDatasetFixture::try_new().unwrap();
        fixture.create_some_data().await.unwrap();
        fixture.overwrite_some_data().await.unwrap();

        fixture.clock.set_system_time(Duration::days(10));

        let before_count = fixture.count_files().await.unwrap();

        fixture
            .run_cleanup(utc_now() - Duration::days(8))
            .await
            .unwrap();

        let after_count = fixture.count_files().await.unwrap();
        // There should be one less data file
        assert_lt!(after_count.num_data_files, before_count.num_data_files);
        // And one less manifest file
        assert_lt!(
            after_count.num_manifest_files,
            before_count.num_manifest_files
        );

        // The latest manifest should still be there, even if it is older than
        // the given time.
        assert_gt!(after_count.num_manifest_files, 0);
        assert_gt!(after_count.num_data_files, 0);
    }

    #[tokio::test]
    async fn do_not_cleanup_newer_data() {
        // Only manifest files older than the `before` parameter should
        // be removed
        let fixture = MockDatasetFixture::try_new().unwrap();
        fixture.create_some_data().await.unwrap();
        fixture.clock.set_system_time(Duration::days(10));
        fixture.append_some_data().await.unwrap();
        fixture.append_some_data().await.unwrap();

        let before_count = fixture.count_files().await.unwrap();

        // 3 versions (plus one extra latest.manifest)
        assert_eq!(before_count.num_data_files, 3);
        assert_eq!(before_count.num_manifest_files, 4);

        let before = utc_now() - Duration::days(7);
        fixture.run_cleanup(before).await.unwrap();

        let after_count = fixture.count_files().await.unwrap();
        // The data files should all remain since they are referenced by
        // the latest version
        assert_eq!(after_count.num_data_files, 3);
        // Only the oldest manifest file should be removed
        assert_eq!(after_count.num_manifest_files, 3);
    }

    #[tokio::test]
    async fn reject_before_date_close_to_now() {
        let fixture = MockDatasetFixture::try_new().unwrap();
        fixture.create_some_data().await.unwrap();
        assert_err_containing!(
            fixture.run_cleanup(utc_now()).await,
            "Cannot cleanup data less than"
        );
    }

    #[tokio::test]
    async fn cleanup_old_index() {
        let fixture = MockDatasetFixture::try_new().unwrap();
        fixture.create_some_data().await.unwrap();
        fixture.create_some_index().await.unwrap();
        fixture.clock.set_system_time(Duration::days(10));
        fixture.overwrite_some_data().await.unwrap();

        fixture.debug_print_files().await.unwrap();

        let before_count = fixture.count_files().await.unwrap();
        assert_eq!(before_count.num_index_files, 1);
        // Two user data files and one lance file for the index
        assert_eq!(before_count.num_data_files, 3);
        // Creating an index creates a new manifest so there are 4 total
        assert_eq!(before_count.num_manifest_files, 4);

        let before = utc_now() - Duration::days(8);
        fixture.run_cleanup(before).await.unwrap();

        fixture.debug_print_files().await.unwrap();

        let after_count = fixture.count_files().await.unwrap();
        assert_eq!(after_count.num_index_files, 0);
        assert_eq!(after_count.num_data_files, 1);
        assert_eq!(after_count.num_manifest_files, 2);
    }

    #[tokio::test]
    async fn clean_old_delete_files() {
        let fixture = MockDatasetFixture::try_new().unwrap();
        let mut data_gen = BatchGenerator::new().col(Box::new(
            IncrementingInt32::new().named("filter_me".to_owned()),
        ));

        fixture
            .create_with_data(data_gen.batch(16).unwrap())
            .await
            .unwrap();
        fixture
            .append_data(data_gen.batch(16).unwrap())
            .await
            .unwrap();
        // This will keep some data from the appended file and should
        // completely remove the first file
        fixture.delete_data("filter_me < 20").await.unwrap();
        fixture.clock.set_system_time(Duration::days(10));
        fixture
            .overwrite_data(data_gen.batch(16).unwrap())
            .await
            .unwrap();
        // This will delete half of the last fragment
        fixture.delete_data("filter_me >= 40").await.unwrap();

        let before_count = fixture.count_files().await.unwrap();
        assert_eq!(before_count.num_data_files, 3);
        assert_eq!(before_count.num_delete_files, 2);
        assert_eq!(before_count.num_manifest_files, 6);

        let before = utc_now() - Duration::days(8);
        fixture.run_cleanup(before).await.unwrap();

        let after_count = fixture.count_files().await.unwrap();
        assert_eq!(after_count.num_data_files, 1);
        assert_eq!(after_count.num_delete_files, 1);
        assert_eq!(after_count.num_manifest_files, 3);

        // Ensure we can still read the dataset
        let row_count_after = fixture.count_rows().await.unwrap();
        assert_eq!(row_count_after, 8);
    }

    #[tokio::test]
    async fn dont_clean_index_data_files() {
        // Indexes have .lance files in them that are not referenced
        // by any fragment.  We need to make sure the cleanup routine
        // doesn't over-zealously delete these
        let fixture = MockDatasetFixture::try_new().unwrap();
        fixture.clock.set_system_time(Duration::days(10));
        fixture.create_some_data().await.unwrap();
        fixture.create_some_index().await.unwrap();

        fixture.debug_print_files().await.unwrap();

        let before_count = fixture.count_files().await.unwrap();
        let before = utc_now() - Duration::days(8);
        fixture.run_cleanup(before).await.unwrap();
        let after_count = fixture.count_files().await.unwrap();

        fixture.debug_print_files().await.unwrap();

        assert_eq!(before_count, after_count);
    }

    #[tokio::test]
    async fn cleanup_failed_commit_data_file() {
        // We should clean up data files that are written but the commit failed
        // for whatever reason

        let mut fixture = MockDatasetFixture::try_new().unwrap();
        fixture.clock.set_system_time(Duration::days(10));
        fixture.create_some_data().await.unwrap();
        fixture.block_commits();
        assert!(fixture.append_some_data().await.is_err());

        let before_count = fixture.count_files().await.unwrap();
        // This append will fail since the commit is blocked but it should have
        // deposited a data file
        assert_eq!(before_count.num_data_files, 2);
        assert_eq!(before_count.num_manifest_files, 2);

        // All of our manifests are newer than the threshold but temp files
        // should still be deleted.
        fixture
            .run_cleanup(utc_now() - Duration::days(7))
            .await
            .unwrap();

        let after_count = fixture.count_files().await.unwrap();
        assert_eq!(after_count.num_data_files, 1);
        assert_eq!(after_count.num_manifest_files, 2);
    }
}
